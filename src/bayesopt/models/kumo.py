"""Optional Kumo Tabular response surrogates for the existing BO loop.

Kumo's pretrained weights stay fixed; fit() refreshes its measured context.
The existing acquisition function expects a central value and a standard
deviation. We use the predictive median and a normal-equivalent scale from
the 2.5% and 97.5% quantiles. This is an approximation, not a GP posterior
or a claim that Kumo's uncertainty is calibrated for Bayesian optimization.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import norm
from scipy.special import expit

from ..types import FloatArray


SETTINGS = ("x", "y", "z", "e", "d", "p", "k", "b", "dose_target_mJ")


def _normal_approximation(
    lower: FloatArray, median: FloatArray, upper: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Translate Kumo quantiles into the acquisition function's input format."""
    values = np.column_stack((lower, median, upper))
    if not np.isfinite(values).all():
        raise ValueError("Kumo returned non-finite EUV predictions")
    if np.any(lower > median) or np.any(median > upper):
        raise ValueError("Kumo returned unordered EUV prediction quantiles")
    scale = (upper - lower) / (2.0 * norm.ppf(0.975))
    return median, scale


class KumoRegressor:
    """Load once, refresh context per iteration, and predict in bounded batches."""

    def __init__(
        self, *, size: str = "small", device: str = "cpu",
        num_estimators: int = 4, batch_size: int = 256,
        target_name: str = "response", dose_slope: float = 0.0,
        shared_model=None, checkpoint: str | Path | None = None,
    ) -> None:
        # Optional imports keep the default GP path independent of SDM/PyTorch.
        try:
            import sdm
            import torch
        except ImportError as error:
            raise ImportError(
                "Kumo requires its optional dependencies. Run: "
                "python -m pip install -r requirements-kumo.txt"
            ) from error
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable; select kumo_device='cpu'")
        if size not in ("small", "medium", "large") or device not in ("cpu", "cuda"):
            raise ValueError("Select a valid Kumo size and cpu/cuda device")
        if num_estimators < 1 or batch_size < 1:
            raise ValueError("Kumo estimator count and batch size must be positive")
        self.sdm = sdm
        self.torch = torch
        self.device = device
        self.num_estimators = num_estimators
        self.batch_size = batch_size
        if shared_model is not None and checkpoint is not None:
            raise ValueError("Load a checkpoint once, then share the loaded model")
        if shared_model is not None:
            self.model = shared_model
        elif checkpoint is None:
            self.model = sdm.models.KumoTabular(task="regression", size=size, device=device)
        else:
            # Construct without downloading the original pretrained weights.
            self.model = sdm.models.KumoTabular(
                task="regression", size=size, pretrained=False, device=device,
            )
            weights = torch.load(checkpoint, map_location=device, weights_only=True)
            self.model.load_state_dict(weights, strict=True)
            self.model.eval()
        self.fitted = False
        self.dose_slope = dose_slope
        self.target_name = target_name

    def _table(self, x: FloatArray):
        import pandas as pd

        frame = pd.DataFrame(x, columns=SETTINGS)
        return self.sdm.TableTensor.from_pandas(
            frame, stypes={name: "numerical" for name in frame}, device=self.device,
        )

    def fit(self, x: FloatArray, euv: FloatArray, *, seed: int) -> None:
        """Use only the retained measured settings and EUV as context."""
        import pandas as pd

        residual = euv - self.dose_slope * x[:, -1]
        frame = pd.DataFrame({self.target_name: residual})
        target = self.sdm.TableTensor.from_pandas(
            frame, stypes={self.target_name: "numerical"}, device=self.device,
        )
        self.model.clear()  # Release the preceding iteration's cached context.
        self.fitted = False
        with self.torch.inference_mode():
            self.model.fit(
                x=self._table(x), y=target,
                num_estimators=self.num_estimators,
                generator=self.torch.Generator(device=self.device).manual_seed(seed),
            )
        self.fitted = True

    def predict_quantiles(self, x: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Predict with the currently fitted context; shared users must fit first."""
        if not self.fitted:
            raise RuntimeError("Fit Kumo's measured context before predicting")
        values = np.empty((3, len(x)))
        with self.torch.inference_mode():
            for start in range(0, len(x), self.batch_size):
                stop = min(start + self.batch_size, len(x))
                quantiles = self.model.predict(self._table(x[start:stop]))
                lower, median, upper = (
                    quantiles[name].numerical.squeeze(-1).cpu().numpy().astype(float)
                    for name in ("q025", "q500", "q975")
                )
                _normal_approximation(lower, median, upper)  # Validate before use.
                values[:, start:stop] = np.stack((lower, median, upper))
        values += self.dose_slope * x[:, -1]
        return tuple(values)

    def predict(self, x: FloatArray) -> tuple[FloatArray, FloatArray]:
        return _normal_approximation(*self.predict_quantiles(x))


class KumoEUV(KumoRegressor):
    def __init__(self, **kwargs):
        super().__init__(target_name="euv_residual_mJ", dose_slope=0.1, **kwargs)


class KumoYield(KumoRegressor):
    """Regress clipped logit yield; transform quantiles back to probabilities.

    Like the logit GP, this treats each observed fraction as one regression
    target, not a binomial likelihood. Intervals are not calibrated guarantees.
    """

    def __init__(self, **kwargs):
        super().__init__(target_name="logit_yield", **kwargs)

    def fit(self, x: FloatArray, yield_fraction: FloatArray, *, seed: int) -> None:
        safe = np.clip(yield_fraction, 1e-4, 1 - 1e-4)
        super().fit(x, np.log(safe / (1 - safe)), seed=seed)

    def predict(self, x: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        lower, median, upper = self.predict_quantiles(x)
        return expit(median), expit(lower), expit(upper)
