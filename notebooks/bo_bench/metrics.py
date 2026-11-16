"""Every metric in the report, each computed from a model's actual output.

Count outputs (yield) are scored as a probability for every possible count 0..n.
Real outputs are scored from the 199 predicted quantiles.
Nothing here invents a width: a model that gives only a point is scored as the distribution it
implies (a binomial around that point) and is marked as such in the report."""
from __future__ import annotations
import numpy as np
from scipy.stats import binom, kendalltau
from .surrogates.base import LEVELS

COV_LEVELS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99])


def _floor(pmf, eps):
    return (1 - eps) * pmf + eps / pmf.shape[1]


def _pit_cover(lo, hi, level):
    """Share of a uniform [lo, hi] PIT interval that lies inside the central `level` band."""
    a, b = (1 - level) / 2, (1 + level) / 2
    w = np.maximum(hi - lo, 1e-12)
    return np.clip(np.minimum(hi, b) - np.maximum(lo, a), 0, None) / w


def _linear_fit(x, y):
    A = np.c_[x, np.ones_like(x)]
    (slope, icpt), *_ = np.linalg.lstsq(A, y, rcond=None)
    ss = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ((y - A @ [slope, icpt]) ** 2).sum() / ss if ss > 0 else np.nan
    return float(slope), float(icpt), float(r2)


def _spread_skill(pred_sd, err, bins=6):
    """Predicted spread vs the error that actually happened, in equal-count bins."""
    ok = np.isfinite(pred_sd) & np.isfinite(err)
    pred_sd, err = pred_sd[ok], err[ok]
    if len(err) < 2 * bins:
        return [], [], np.nan, np.nan
    order = np.argsort(pred_sd)
    xs, ys = [], []
    for chunk in np.array_split(order, bins):
        xs.append(float(np.sqrt(np.mean(pred_sd[chunk] ** 2))))
        ys.append(float(np.sqrt(np.mean(err[chunk] ** 2))))
    ratio = float(np.sqrt(np.mean(pred_sd ** 2)) / max(np.sqrt(np.mean(err ** 2)), 1e-12))
    slope = _linear_fit(np.array(xs), np.array(ys))[0] if np.ptp(xs) > 1e-12 else np.nan
    return xs, ys, ratio, slope


def ranking(pred, target, top=0.10):
    ok = np.isfinite(pred) & np.isfinite(target)
    pred, target = pred[ok], target[ok]
    if len(pred) < 5 or np.ptp(target) == 0:
        return {"kendall": np.nan, "top10_recall": np.nan}
    k = max(1, int(round(top * len(pred))))
    best, picked = set(np.argsort(-target)[:k]), set(np.argsort(-pred)[:k])
    tau = kendalltau(pred, target).statistic if np.ptp(pred) > 0 else np.nan
    return {"kendall": float(tau), "top10_recall": len(best & picked) / k}


def count_metrics(pmf, k, n, mean_p, true_p=None, eps=1e-3):
    """pmf [N, n+1] predicted count distribution, k observed successes, n trials (scalar),
    mean_p predicted mean yield, true_p the simulator's underlying yield when known."""
    pmf, k, n = np.asarray(pmf, float), np.rint(np.asarray(k)).astype(int), int(n)
    ok = np.isfinite(pmf).all(1) & np.isfinite(mean_p)
    if ok.sum() < 3:
        return {}
    pmf, k, mean_p = _floor(pmf[ok] / pmf[ok].sum(1, keepdims=True), eps), k[ok], np.clip(mean_p[ok], 1e-6, 1 - 1e-6)
    true_p = None if true_p is None else np.asarray(true_p, float)[ok]
    N, ks = len(k), np.arange(n + 1)
    rows = np.arange(N)
    cdf = np.cumsum(pmf, 1)
    nll = -np.log(pmf[rows, k])
    # entropy floor: what a model that knew the yield would still score, from counting noise alone
    p_ref = np.clip(k / n, 1e-9, 1 - 1e-9) if true_p is None else np.clip(true_p, 1e-9, 1 - 1e-9)
    ent = binom.entropy(n, p_ref)
    crps = ((cdf - (ks[None, :] >= k[:, None])) ** 2).sum(1)
    pit_hi = cdf[rows, k]
    pit_lo = np.where(k > 0, cdf[rows, np.maximum(k - 1, 0)], 0.0)
    cov = np.array([_pit_cover(pit_lo, pit_hi, L).mean() for L in COV_LEVELS])
    slope, icpt, r2 = _linear_fit(COV_LEVELS, cov)
    # overdispersion: observed deviance over the deviance a binomial with the predicted mean expects
    def dev(kk, p):
        kk = np.asarray(kk, float)
        with np.errstate(divide="ignore", invalid="ignore"):
            a = np.where(kk > 0, kk * np.log(kk / (n * p)), 0.0)
            b = np.where(kk < n, (n - kk) * np.log((n - kk) / (n * (1 - p))), 0.0)
        return 2 * (a + b)
    exp_dev = (binom.pmf(ks[None, :], n, mean_p[:, None]) * dev(ks[None, :], mean_p[:, None])).sum(1)
    mean_k = (pmf * ks).sum(1)
    sd_k = np.sqrt(np.maximum((pmf * ks ** 2).sum(1) - mean_k ** 2, 0))
    ssx, ssy, ratio, ss_slope = _spread_skill(sd_k, k - mean_k)
    q = lambda level: np.array([np.searchsorted(c, level) for c in cdf])
    width90 = (q(0.95) - q(0.05)).astype(float)
    resid = k - mean_k
    needed90 = float(np.quantile(resid, 0.95) - np.quantile(resid, 0.05))
    p_perfect, perfect = pmf[:, n], (k == n).astype(float)
    target = k / n if true_p is None else true_p
    out = {
        "nll": nll.mean(), "entropy_floor": float(np.mean(ent)), "excess": float(np.mean(nll - ent)),
        "overdisp": float(dev(k, mean_p).sum() / max(exp_dev.sum(), 1e-12)),
        "crps": crps.mean(), "cov90": float(cov[8]), "cov_err": float(np.abs(cov - COV_LEVELS).mean()),
        "rel_slope": slope, "rel_intercept": icpt, "rel_r2": r2,
        "spread_ratio": ratio, "spread_slope": ss_slope,
        "width90": float(width90.mean()), "needed90": needed90,
        "brier_perfect": float(np.mean((p_perfect - perfect) ** 2)),
        "rmse": float(np.sqrt(np.mean((mean_p - target) ** 2))), "mae": float(np.mean(np.abs(mean_p - target))),
        "n": int(N), "cov_curve": cov.tolist(), "ss_x": ssx, "ss_y": ssy,
        **ranking(mean_p, target),
    }
    return {k_: (float(v) if isinstance(v, (np.floating, float)) else v) for k_, v in out.items()}


def real_metrics(mean, Q, y, truth=None):
    """mean [N], Q [N,199] quantiles of a new measurement, y observed values."""
    mean, Q, y = np.asarray(mean, float), np.asarray(Q, float), np.asarray(y, float)
    ok = np.isfinite(mean) & np.isfinite(y) & np.isfinite(Q).all(1)
    if ok.sum() < 3:
        return {}
    mean, Q, y = mean[ok], np.sort(Q[ok], 1), y[ok]
    target = y if truth is None else np.asarray(truth, float)[ok]
    d = y[:, None] - Q
    crps = 2 * np.mean(np.where(d >= 0, LEVELS * d, (LEVELS - 1) * d), axis=1)      # pinball average ~ CRPS
    pit = np.array([np.interp(v, q + np.arange(len(q)) * 1e-12, LEVELS, left=0.0, right=1.0) for v, q in zip(y, Q)])
    cov = np.array([np.mean(np.abs(pit - 0.5) <= L / 2) for L in COV_LEVELS])
    slope, icpt, r2 = _linear_fit(COV_LEVELS, cov)
    sd = (Q[:, 167] - Q[:, 31]) / 2.0                                              # half the 16%-84% range
    ssx, ssy, ratio, ss_slope = _spread_skill(sd, y - mean)
    width90 = Q[:, 189] - Q[:, 9]
    resid = y - mean
    out = {
        "crps": crps.mean(), "cov90": float(cov[8]), "cov_err": float(np.abs(cov - COV_LEVELS).mean()),
        "rel_slope": slope, "rel_intercept": icpt, "rel_r2": r2, "spread_ratio": ratio, "spread_slope": ss_slope,
        "width90": float(width90.mean()), "needed90": float(np.quantile(resid, 0.95) - np.quantile(resid, 0.05)),
        "rmse": float(np.sqrt(np.mean((mean - target) ** 2))), "mae": float(np.mean(np.abs(mean - target))),
        "n": int(ok.sum()), "cov_curve": cov.tolist(), "ss_x": ssx, "ss_y": ssy, **ranking(mean, target),
    }
    return {k_: (float(v) if isinstance(v, (np.floating, float)) else v) for k_, v in out.items()}


def bo_curves(value, feasible, reference=None, targets=(0.95, 0.99)):
    """Best feasible true objective after each evaluation, regret, and evaluations needed to reach
    a fraction of the reference optimum."""
    value, feasible = np.asarray(value, float), np.asarray(feasible, float) > 0
    done = np.isfinite(value)
    v = np.where(feasible & done, value, -np.inf)
    best = np.maximum.accumulate(v)
    best = np.where(np.isfinite(best), best, np.nan)
    best[~done] = np.nan
    out = {"best": best.tolist(), "n_done": int(done.sum()), "n_feasible": int((feasible & done).sum()),
           "final_best": float(best[done][-1]) if done.any() else np.nan,
           "first_feasible": int(np.argmax(feasible) + 1) if feasible.any() else None}
    if reference is not None and np.isfinite(reference):
        reg = reference - best
        out["regret"] = reg.tolist()
        out["final_regret"] = float(reg[done][-1]) if done.any() else np.nan
        for t in targets:
            thr = reference * t if reference > 0 else reference - abs(reference) * (1 - t)
            hit = np.flatnonzero(np.nan_to_num(best, nan=-np.inf) >= thr)
            out[f"evals_to_{int(round(t * 1000)) / 10:g}"] = int(hit[0] + 1) if len(hit) else None
    return out
