"""Checks for the Kumo acquisition adapter; no model downloads required."""

import importlib.util
import sys
import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from bayesopt.models.kumo import KumoEUV, KumoYield, _normal_approximation
from bayesopt.optimization import run_optimization


class QuantileApproximationTests(unittest.TestCase):
    def test_normal_interval_recovers_its_scale(self):
        z = norm.ppf(.975)
        center, scale = _normal_approximation(
            np.array([5.0 - 2*z]), np.array([5.0]), np.array([5.0 + 2*z])
        )
        np.testing.assert_allclose(center, [5.0])
        np.testing.assert_allclose(scale, [2.0])

    def test_asymmetric_interval_keeps_median(self):
        center, _ = _normal_approximation(np.array([0.]), np.array([1.]), np.array([8.]))
        np.testing.assert_array_equal(center, [1.])

    def test_invalid_quantiles_fail_before_acquisition(self):
        for lower, median, upper in ((0., float('nan'), 2.), (2., 1., 3.), (0., 3., 2.)):
            with self.subTest(values=(lower, median, upper)), self.assertRaises(ValueError):
                _normal_approximation(*(np.array([v]) for v in (lower, median, upper)))

    def test_invalid_backend_options_fail_before_measurement(self):
        for kwargs in ({'euv_model': 'typo'}, {'euv_model': 'kumo', 'kumo_batch_size': 0},
                       {'kumo_checkpoint': 'unused.pt'},
                       {'surrogate_scope': 'typo'}, {'warmup': -1},
                       {'euv_model': 'kumo', 'surrogate_scope': 'all', 'dy_model': 2}):
            with self.subTest(kwargs=kwargs), patch('bayesopt.optimization.srcmdl') as source:
                with self.assertRaises(ValueError):
                    run_optimization(n=1, **kwargs)
                source.assert_not_called()


@unittest.skipUnless(importlib.util.find_spec('sdm'), 'optional SDM dependencies not installed')
class KumoContextTests(unittest.TestCase):
    def test_checkpoint_loads_strictly_without_pretrained_download(self):
        fake = Mock()
        weights = {'example': object()}
        with patch('sdm.models.KumoTabular', return_value=fake) as construct, \
             patch('torch.load', return_value=weights) as load:
            model = KumoEUV(size='large', checkpoint='custom.pt')
        construct.assert_called_once_with(
            task='regression', size='large', pretrained=False, device='cpu')
        load.assert_called_once_with('custom.pt', map_location='cpu', weights_only=True)
        fake.load_state_dict.assert_called_once_with(weights, strict=True)
        fake.eval.assert_called_once()
        self.assertIs(model.model, fake)

    def test_all_responses_load_checkpoint_only_once(self):
        fake = Mock()
        with tempfile.NamedTemporaryFile(suffix='.pt') as checkpoint, \
             patch('sdm.models.KumoTabular', return_value=fake) as construct, \
             patch('torch.load', return_value={}) as load:
            euv = KumoEUV(size='large', checkpoint=checkpoint.name)
            others = [KumoYield(size='large', shared_model=euv.model)
                      for _ in range(4)]
        construct.assert_called_once()
        load.assert_called_once()
        self.assertTrue(all(model.model is euv.model for model in others))

    def test_yield_transform_is_bounded_at_zero_and_one(self):
        from scipy.special import expit
        x = np.full((3, 9), .5)
        fake = Mock()
        with patch('sdm.models.KumoTabular', return_value=fake):
            model = KumoYield()
            model.fit(x, np.array([0., .5, 1.]), seed=1)
            targets = fake.fit.call_args.kwargs['y'].to_pandas().iloc[:, 0]
            np.testing.assert_allclose(expit(targets), [1e-4, .5, 1-1e-4], rtol=1e-5)
            with patch.object(model, 'predict_quantiles', return_value=(
                np.array([-10.]), np.array([0.]), np.array([10.])
            )):
                center, lower, upper = model.predict(x[:1])
                self.assertTrue(0 < lower[0] < center[0] < upper[0] < 1)

    def test_all_responses_share_weights_and_do_not_call_gp(self):
        import pandas as pd
        import sdm
        fake = Mock()
        targets_seen = []
        contexts = []
        current_target = None

        def fit(**kwargs):
            nonlocal current_target
            current_target = kwargs['y'].to_pandas().columns[0]
            targets_seen.append(current_target)
            contexts.append(len(kwargs['x'].to_pandas()))

        def predict(table):
            n = len(table.to_pandas())
            center = {'euv_residual_mJ': 1., 'rx_deg': 0., 'ry_deg': 17.,
                      'target_size': .5, 'logit_yield': 5.}[current_target]
            frame = pd.DataFrame({'q025': [center-.1]*n, 'q500': [center]*n, 'q975': [center+.1]*n})
            return sdm.TableTensor.from_pandas(frame, stypes={c: 'numerical' for c in frame})

        fake.fit.side_effect = fit
        fake.predict.side_effect = predict
        state = Mock()
        measurement = (state, np.array([6.]), 0., 17., .5, 1., 71, 0., 6., .999)
        with patch('sdm.models.KumoTabular', return_value=fake) as load, \
             patch('bayesopt.optimization.srcmdl', return_value=measurement) as source, \
             patch('bayesopt.optimization._fit_gpr') as gp, \
             patch('bayesopt.optimization._fit_yield_model') as yield_gp:
            result = run_optimization(n=30, n_test=8, warmup=0, dy_model=1,
                                      euv_model='kumo', surrogate_scope='all')
        load.assert_called_once()
        gp.assert_not_called()
        yield_gp.assert_not_called()
        self.assertEqual(source.call_count, 30)  # Eight candidates are not eight evaluations.
        self.assertEqual(set(targets_seen), {'logit_yield', 'euv_residual_mJ', 'rx_deg', 'ry_deg', 'target_size'})
        self.assertEqual(contexts, [27]*6 + [28]*6 + [29]*6)
        self.assertEqual(current_target, 'logit_yield')  # Final predictions and dose solve use yield.
        self.assertEqual(result.selected_by_optimizer.sum(), 3)
        self.assertTrue(result.kumo_used)
        self.assertTrue(np.isfinite(result.predicted_die_yield).all())

    def test_load_once_refresh_context_and_batch_candidates(self):
        import pandas as pd
        import sdm

        fake_model = Mock()
        def predict(table):
            # Queries contain only the nine setting columns, never measured EUV.
            self.assertEqual(list(table.to_pandas().columns),
                             ['x', 'y', 'z', 'e', 'd', 'p', 'k', 'b', 'dose_target_mJ'])
            n = len(table.to_pandas())
            frame = pd.DataFrame({'q025': [3.]*n, 'q500': [4.]*n, 'q975': [5.]*n})
            return sdm.TableTensor.from_pandas(frame, stypes={c: 'numerical' for c in frame})
        fake_model.predict.side_effect = predict
        x = np.full((4, 9), .5)
        x[:, -1] = 55.
        measured = np.array([6., 7., 8., 9.])
        with patch('sdm.models.KumoTabular', return_value=fake_model) as load:
            model = KumoEUV(batch_size=2)
            with self.assertRaises(RuntimeError):
                model.predict(x)
            model.fit(x, measured, seed=7)
            supplied = fake_model.fit.call_args.kwargs
            np.testing.assert_allclose(supplied['y'].to_pandas().iloc[:, 0], measured - 5.5)
            self.assertEqual(supplied['generator'].initial_seed(), 7)
            center, scale = model.predict(x)
            np.testing.assert_allclose(center, [9.5]*4)  # Restore fixed dose basis.
            np.testing.assert_allclose(scale, [1/norm.ppf(.975)]*4)
            self.assertEqual(fake_model.predict.call_count, 2)
            model.fit(x[:3], measured[:3], seed=8)
            self.assertEqual(len(fake_model.fit.call_args.kwargs['x'].to_pandas()), 3)
            self.assertEqual(fake_model.clear.call_count, 2)
            load.assert_called_once()


if __name__ == '__main__':
    unittest.main()
