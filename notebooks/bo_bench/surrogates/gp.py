"""BoTorch Gaussian-process baseline (SingleTaskGP, ARD Matern-5/2, fitted by maximum marginal likelihood)."""
from __future__ import annotations
import warnings
import numpy as np
from scipy.stats import norm
from .base import Backend, LEVELS

_Z = norm.ppf(LEVELS)


class GPBackend(Backend):
    """Hyperparameters are warm-started from the previous fit of the same output (tag) and the
    optimiser is capped at `maxiter` steps; an uncapped cold fit costs ~50x more for the same result."""
    name = "gp"

    def __init__(self, maxiter: int = 50):
        self.maxiter, self._state = maxiter, {}

    def regress(self, Xc, yc, Xq, tag=None):
        import torch
        from botorch.models import SingleTaskGP
        from botorch.models.transforms import Normalize, Standardize
        from botorch.optim.fit import fit_gpytorch_mll_scipy
        from gpytorch.mlls import ExactMarginalLogLikelihood
        X = torch.as_tensor(np.asarray(Xc), dtype=torch.double)
        y = torch.as_tensor(np.asarray(yc), dtype=torch.double).unsqueeze(-1)
        if float(y.std()) < 1e-12:                         # constant target: nothing to fit
            m = np.full(len(Xq), float(y.mean()))
            Q = np.repeat(m[:, None], len(LEVELS), 1)
            return m, Q, Q
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp = SingleTaskGP(X, y, input_transform=Normalize(X.shape[1]), outcome_transform=Standardize(1))
            key = (tag, X.shape[1])
            if tag is not None and key in self._state:
                try:
                    gp.load_state_dict({k: v for k, v in self._state[key].items() if "transform" not in k}, strict=False)
                except Exception:
                    pass
            try:
                mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
                mll.train()
                # one L-BFGS run; botorch's default wrapper restarts from scratch up to five times
                # whenever the step cap is hit, which costs far more than it gains here
                fit_gpytorch_mll_scipy(mll, options={"maxiter": self.maxiter})
                if tag is not None:
                    self._state[key] = {k: v.detach().clone() for k, v in gp.state_dict().items()}
            except Exception as err:                        # keep the run alive on a failed fit
                print(f"  [gp] fit failed ({type(err).__name__}); using previous/prior hyperparameters")
            gp.eval()
            means, sds, sds_obs = [], [], []
            Xt = torch.as_tensor(np.asarray(Xq), dtype=torch.double)
            with torch.no_grad():
                for i in range(0, len(Xt), 2048):
                    post = gp.posterior(Xt[i:i + 2048])
                    obs = gp.posterior(Xt[i:i + 2048], observation_noise=True)
                    means.append(post.mean.squeeze(-1).numpy())
                    sds.append(post.variance.clamp_min(0).sqrt().squeeze(-1).numpy())
                    sds_obs.append(obs.variance.clamp_min(0).sqrt().squeeze(-1).numpy())
        m, s, so = np.concatenate(means), np.concatenate(sds), np.concatenate(sds_obs)
        return m, m[:, None] + s[:, None] * _Z, m[:, None] + so[:, None] * _Z

    def classify(self, Xc, yc, Xq, n_classes, weights=None, tag=None):
        raise NotImplementedError("the GP baseline models yield with a logit GP, not a classifier")
