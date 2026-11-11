"""Model adapters. Each returns a predictor(X_cur, y_cur, k_cur, n_cur, X_ctx, y_ctx, X_query) -> common format.

Common format (common.finalize_output): mean, median, sorted quantiles on config.QUANTILE_LEVELS (logit scale), sd.
GPs never receive context rows. TFMs receive context rows concatenated in front of the current day's revealed rows,
with no dataset-ID column.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

import config as C
from common import finalize_output, interp_quantiles

LEVELS = C.QUANTILE_LEVELS
Z = norm.ppf(LEVELS)


def _gauss_quantiles(mean, sd):
    return mean[:, None] + sd[:, None] * Z[None, :]


def _stack_ctx(X_cur, y_cur, X_ctx, y_ctx):
    if len(y_ctx):
        return np.vstack([X_ctx, X_cur]), np.concatenate([y_ctx, y_cur])
    return X_cur, y_cur


# ----------------------------------------------------------------------------------------------------------------------
# GPs
# ----------------------------------------------------------------------------------------------------------------------
def make_gpr_laplace():
    from gp_laplace import LaplaceGPC

    def predict(X, y, k, n, Xc, yc, Xq):
        m = LaplaceGPC(grad=C.LAPLACE_GRAD).fit(X, k, n)
        mu, var = m.predict_latent(Xq)
        sd = np.sqrt(np.maximum(var, 1e-12))
        return finalize_output(mu, _gauss_quantiles(mu, sd), sd)

    return predict


def make_gpr_botorch():
    """BoTorch SingleTaskGP on y = logit((k+.5)/(n+1)).

    Inputs are z-scored with the training mean/std (as MATLAB 'Standardize',1). Kernel: ScaleKernel(MaternKernel(
    nu=2.5, ARD)). Everything else is the SingleTaskGP default (constant mean, Standardize outcome transform, Gaussian
    likelihood with BoTorch's default noise prior/constraint), fitted with fit_gpytorch_mll from scratch every step.
    """
    import torch
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from gpytorch.kernels import MaternKernel, ScaleKernel
    from gpytorch.mlls import ExactMarginalLogLikelihood

    dt = torch.float64  # local to BoTorch; never change torch's global default (it leaks into TabPFN)

    def predict(X, y, k, n, Xc, yc, Xq):
        mu_x, sd_x = X.mean(0), X.std(0, ddof=1)
        sd_x[sd_x == 0] = 1
        tx = torch.as_tensor((X - mu_x) / sd_x, dtype=dt)
        ty = torch.as_tensor(y, dtype=dt).unsqueeze(-1)
        torch.manual_seed(C.SEED)
        model = SingleTaskGP(tx, ty, covar_module=ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=X.shape[1])))
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        try:
            fit_gpytorch_mll(mll)
        except Exception as exc:  # keep going with the initial hyperparameters, like MATLAB on failure
            print(f"    botorch fit failed ({exc}); using initial hyperparameters")
        model.eval()
        with torch.no_grad():
            post = model.posterior(torch.as_tensor((Xq - mu_x) / sd_x, dtype=dt),
                                   observation_noise=C.BOTORCH_OBSERVATION_NOISE)
            mu = post.mean.squeeze(-1).numpy()
            sd = post.variance.clamp_min(1e-12).sqrt().squeeze(-1).numpy()
        return finalize_output(mu, _gauss_quantiles(mu, sd), sd)

    return predict


# ----------------------------------------------------------------------------------------------------------------------
# TFMs (regression heads, default settings, fixed seeds)
# ----------------------------------------------------------------------------------------------------------------------
def _device():
    if C.TFM_DEVICE == "cuda":
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    return C.TFM_DEVICE


def make_causilo():
    """causilo.CausiloRegressor. predict(output_type='quantiles') interpolates its 999 native quantiles
    (levels i/1000) onto the requested levels; output_type='mean' is the model's own mean."""
    from causilo import CausiloRegressor

    reg = CausiloRegressor(random_state=C.SEED, device=_device())

    def predict(X, y, k, n, Xc, yc, Xq):
        Xt, yt = _stack_ctx(X, y, Xc, yc)
        reg.fit(Xt.astype(np.float32), yt.astype(np.float32))
        q = reg.predict(Xq.astype(np.float32), output_type="quantiles", quantiles=[float(v) for v in LEVELS])
        mean = reg.predict(Xq.astype(np.float32), output_type="mean")
        return finalize_output(mean, q)

    return predict


def _counts_from_logit(y, n):
    """Invert y = logit((k+0.5)/(n+1)) back to the integer count k (context rows only carry y)."""
    p = 1 / (1 + np.exp(-np.asarray(y, float)))
    return np.clip(np.rint(p * (n + 1) - 0.5), 0, n).astype(int)


def _binomial_target_quantiles(p, n):
    """Quantiles (on config.QUANTILE_LEVELS) and mean of the measured target logit((K+0.5)/(n+1)), K ~ Bin(n, p).

    This is the predictive distribution of the NEXT MEASUREMENT given a predicted yield p: it contains the counting
    noise but no extra model uncertainty (a classifier gives one p, not a spread of p's). Discrete, so quantiles are
    step values; mean = expectation over K."""
    from scipy.stats import binom
    kk = np.arange(n + 1)
    yy = np.log((kk + 0.5) / (n + 1 - kk - 0.5))
    Q = np.empty((len(p), len(LEVELS)))
    mu = np.empty(len(p))
    for i, pp in enumerate(p):
        w = binom.pmf(kk, n, pp)
        cdf = np.cumsum(w)
        Q[i] = yy[np.minimum(np.searchsorted(cdf, LEVELS - 1e-12), n)]
        mu[i] = (w * yy).sum()
    return mu, Q


def make_causilo_trials():
    """Causilo as a binomial CLASSIFIER with frequency weights (needs the causilo frequency-weights build, PR #19).

    Each setpoint becomes at most two context rows: label 1 with weight k and label 0 with weight n - k (zero-weight
    rows dropped). Mathematically the same as one row per die, without the 70x larger context. The predicted
    P(in spec) is turned into the benchmark's common format via the exact binomial distribution of a 70-try
    measurement (_binomial_target_quantiles). Context days only carry y, so their counts are recovered by inverting
    the logit (exact for n = 70)."""
    import inspect

    from causilo import CausiloClassifier

    if "sample_weight" not in inspect.signature(CausiloClassifier.fit).parameters:
        raise RuntimeError("this causilo build has no frequency weights: install the PR #19 branch "
                           "(pip install --force-reinstall --no-deps "
                           "git+https://github.com/nums-ai/causilo@06b3fd0642a29d355581d6894899a3b5839686df)")
    clf = CausiloClassifier(random_state=C.SEED, device=_device())

    def predict(X, y, k, n, Xc, yc, Xq):
        n_cur = np.asarray(n, int)
        n_q = int(np.median(n_cur))
        rows_x, rows_k, rows_n = [np.asarray(X, float)], [np.asarray(k, int)], [n_cur]
        if len(yc):
            rows_x.append(np.asarray(Xc, float))
            rows_k.append(_counts_from_logit(yc, n_q))
            rows_n.append(np.full(len(yc), n_q))
        Xa, ka, na = np.vstack(rows_x), np.concatenate(rows_k), np.concatenate(rows_n)
        Xr = np.repeat(Xa, 2, axis=0)
        yr = np.tile([1, 0], len(Xa))
        wr = np.column_stack([ka, na - ka]).ravel()
        keep = wr > 0
        clf.fit(Xr[keep].astype(np.float32), yr[keep], sample_weight=wr[keep])
        proba = clf.predict_proba(np.asarray(Xq, np.float32))
        col = int(np.flatnonzero(np.asarray(clf.classes_) == 1)[0]) if 1 in list(clf.classes_) else proba.shape[1] - 1
        p = np.clip(proba[:, col], 1e-6, 1 - 1e-6)
        mu, Q = _binomial_target_quantiles(p, n_q)
        return finalize_output(mu, Q)

    return predict


KUMO_NATIVE_LEVELS = np.arange(1, 1000) / 1000.0  # q001..q999


def make_kumo(size: str = "large"):
    """NVIDIA Kumo Tabular (sdm.models.KumoTabular, size='large' by default; 'small'/'medium' via kumo_small/kumo_medium). Returns 999 fixed quantiles q001..q999; they are
    linearly interpolated onto the shared grid (every shared level is a native level, so this is exact). Kumo exposes
    no mean, so the mean is the average of the 999 native quantiles. num_estimators is left at the library default
    (None -> one estimator for 2-D inputs) unless config.KUMO_NUM_ESTIMATORS is set."""
    import sdm
    import torch

    dev = _device()
    model = sdm.models.KumoTabular(task="regression", size=size, device=dev)
    model.eval()
    n_est = getattr(C, "KUMO_NUM_ESTIMATORS", None)

    def predict(X, y, k, n, Xc, yc, Xq):
        Xt, yt = _stack_ctx(X, y, Xc, yc)
        x = torch.as_tensor(Xt, dtype=torch.float32, device=dev)
        t = torch.as_tensor(yt, dtype=torch.float32, device=dev).unsqueeze(-1)
        gen = torch.Generator(device=dev).manual_seed(C.SEED)
        kw = {} if n_est is None else dict(num_estimators=n_est)
        model.fit(x, t, generator=gen, **kw)
        out = model.predict(torch.as_tensor(Xq, dtype=torch.float32, device=dev))
        nat = out.numerical.detach().float().cpu().numpy().reshape(len(Xq), -1)
        assert nat.shape[1] == 999, nat.shape
        nat = np.sort(nat, axis=1)
        return finalize_output(nat.mean(1), interp_quantiles(KUMO_NATIVE_LEVELS, nat))

    return predict


def _make_tabpfn(version: str):
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    mv = {"v2": ModelVersion.V2, "v3.5": ModelVersion.V3_5}[version]
    reg = TabPFNRegressor.create_default_for_version(mv, random_state=C.SEED, device=_device())

    def predict(X, y, k, n, Xc, yc, Xq):
        import torch
        # TabPFN builds its network in torch's default dtype at every fit. If anything in this process changed the
        # default to float64, the weights become float64 while autocast feeds float16 activations -> dtype crash.
        prev = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        try:
            Xt, yt = _stack_ctx(X, y, Xc, yc)
            reg.fit(Xt, yt)
            out = reg.predict(Xq, output_type="main", quantiles=[float(v) for v in LEVELS])
        finally:
            torch.set_default_dtype(prev)
        q = np.stack([np.asarray(a).reshape(-1) for a in out["quantiles"]], axis=1)
        return finalize_output(np.asarray(out["mean"]), q)

    return predict


def make_tabpfn2():
    return _make_tabpfn("v2")


def make_tabpfn35():
    return _make_tabpfn("v3.5")


# ----------------------------------------------------------------------------------------------------------------------
# Stand-in TFM for testing the pipeline on CPU without weights (never part of the benchmark)
# ----------------------------------------------------------------------------------------------------------------------
def make_mock():
    """Distance-weighted kNN on z-scored inputs; quantiles from a Gaussian with the neighbours' weighted spread."""

    def predict(X, y, k, n, Xc, yc, Xq):
        Xt, yt = _stack_ctx(X, y, Xc, yc)
        mu, sd = Xt.mean(0), Xt.std(0)
        sd[sd == 0] = 1
        A, B = (Xt - mu) / sd, (Xq - mu) / sd
        d = np.sqrt(((B[:, None, :] - A[None, :, :]) ** 2).sum(-1))
        kk = min(15, len(yt))
        idx = np.argsort(d, axis=1)[:, :kk]
        w = 1 / (np.take_along_axis(d, idx, 1) + 1e-6)
        w /= w.sum(1, keepdims=True)
        m = (w * yt[idx]).sum(1)
        s = np.sqrt((w * (yt[idx] - m[:, None]) ** 2).sum(1)) + 0.3
        return finalize_output(m, _gauss_quantiles(m, s))

    return predict


FACTORIES = dict(gpr_laplace=make_gpr_laplace, gpr_botorch=make_gpr_botorch, causilo=make_causilo, kumo=make_kumo,
                 causilo_trials=make_causilo_trials,
                 kumo_small=lambda: make_kumo("small"), kumo_medium=lambda: make_kumo("medium"),
                 tabpfn2=make_tabpfn2, tabpfn35=make_tabpfn35, mock=make_mock)


def package_versions() -> dict:
    from importlib.metadata import PackageNotFoundError, version
    out = {}
    for p in ["numpy", "scipy", "pandas", "torch", "botorch", "gpytorch", "causilo", "structured-data-models", "tabpfn"]:
        try:
            out[p] = version(p)
        except PackageNotFoundError:
            out[p] = "not installed"
    return out
