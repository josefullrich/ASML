"""Metrics. Every function consumes only the common prediction format (y_true, k, n, mean, median, q_*, sd).

Choices that the spec leaves open (also listed in the README):
* Kendall's tau is tau-b (scipy default). logit DY has heavy ties (k=0 and k=n rows), and tau-a's ceiling then sits
  well below 1 and depends on the dataset's tie fraction.
* Top-10% recall: the true-top set is every row with y >= the 90th percentile of y (ties can make it larger than 10%);
  the predicted-top set has the same size, taken by descending predicted median.
* Big-error flagging and MAE use the predicted median; RMSE and binomial NLL use the predicted mean.
* Binomial NLL includes the correct log binomial coefficient, identical for all models.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.special import expit, gammaln
from scipy.stats import binom as binom_dist
from scipy.stats import kendalltau, norm

import config as C

LEVELS = C.BASE_QUANTILE_LEVELS  # CRPS / PIT always on the fixed base grid
I05 = int(np.flatnonzero(np.isclose(LEVELS, 0.05))[0])
I95 = int(np.flatnonzero(np.isclose(LEVELS, 0.95))[0])

# metric name -> (direction, category). direction: "lower" | "higher" | "target90"
METRIC_INFO = {
    "crps": ("lower", "uncertainty"),
    "coverage90": ("target90", "uncertainty"),
    "width90": ("lower", "uncertainty"),
    "calib_error": ("lower", "uncertainty"),
    "kendall_tau": ("higher", "ranking"),
    "top10_recall": ("higher", "ranking"),
    "binom_nll": ("lower", "point"),
    "rmse": ("lower", "point"),
    "nrmse": ("lower", "point"),
    "mae": ("lower", "point"),
    "rmse_mae_ratio": ("target125", "point"),
    # noise-aware, reported but NOT ranked (they are RMSE rescaled by the measurement noise; ranking them would double
    # count RMSE). noise_rmse is the same for every model on the same data.
    "rmse_over_noise": ("lower", "point"),
    "within_noise2": ("target95", "point"),
    "noise_rmse": ("info", "point"),
    # interval quality, reported but NOT ranked (see interval_quality)
    "coverage_curve_error": ("lower", "uncertainty"),
    "width_error": ("lower", "uncertainty"),
    "width_ratio90": ("target1", "uncertainty"),
    "width_adaptivity": ("higher", "uncertainty"),
    # more noise-aware metrics, reported but NOT ranked (see noise_aware)
    "nll_excess": ("lower", "point"),
    "overdispersion": ("target1", "point"),
    "overdispersion_top": ("target1", "point"),
    "crps_over_noise": ("lower", "uncertainty"),
    "mae_over_noise": ("lower", "point"),
    "rmse_noise_corrected": ("lower", "point"),
    "tau_frac": ("higher", "ranking"),
    "top10_frac": ("higher", "ranking"),
    "tau_ceiling": ("info", "ranking"),
    "top10_ceiling": ("info", "ranking"),
}
# Metrics that enter the rankings, per scope.
RANKED = {
    "all": ["crps", "coverage90", "width90", "calib_error", "kendall_tau", "top10_recall", "binom_nll", "rmse", "mae",
            "rmse_mae_ratio"],
    "last20": ["crps", "coverage90", "width90", "calib_error", "kendall_tau", "top10_recall", "binom_nll", "rmse",
               "nrmse", "mae"],
}


def qmat(df: pd.DataFrame) -> np.ndarray:
    return df[C.BASE_QCOLS].to_numpy(float)


def pinball(y, Q):
    """(rows, levels) pinball loss."""
    d = y[:, None] - Q
    return np.maximum(LEVELS * d, (LEVELS - 1) * d)


def crps_grid(y, Q):
    return 2 * pinball(y, Q).mean(1)


def crps_gaussian(y, mu, sd):
    z = (y - mu) / sd
    return sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def pit(y, Q):
    return (Q < y[:, None]).mean(1)


@lru_cache(maxsize=None)
def _noise_sd_nk(n: int, k: int) -> float:
    """Exact sd of the target logit((K+0.5)/(n+1)) when K ~ Binomial(n, p), p estimated as (k+0.5)/(n+1).

    This is the measurement (counting) noise of one row: even a model that knew the true yield would miss the target
    by about this much on average. ~0.25 at mid yields, ~0.6 near 0% or 100% (n = 70)."""
    p = (k + 0.5) / (n + 1)
    kk = np.arange(n + 1)
    w = binom_dist.pmf(kk, n, p)
    yy = np.log((kk + 0.5) / (n + 1 - kk - 0.5))
    m = (w * yy).sum()
    return float(np.sqrt((w * (yy - m) ** 2).sum()))


def noise_sd(k, n):
    return np.array([_noise_sd_nk(int(nn), int(kk)) for kk, nn in zip(np.asarray(k), np.asarray(n))])


def binom_nll(k, n, mean_logit):
    p = np.clip(expit(mean_logit), 1e-12, 1 - 1e-12)
    logc = gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
    return -(k * np.log(p) + (n - k) * np.log1p(-p) + logc)


def calib_error(pit_vals):
    if len(pit_vals) == 0:
        return np.nan
    obs = np.array([(pit_vals <= p).mean() for p in C.PIT_LEVELS])
    return float(np.abs(obs - C.PIT_LEVELS).mean())


def tau_b(a, b):
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return np.nan
    return float(kendalltau(a, b).statistic)


def top10_recall(y, pred):
    if len(y) < 10:
        return np.nan
    thr = np.quantile(y, 0.9)
    true_top = set(np.flatnonzero(y >= thr))
    m = len(true_top)
    pred_top = set(np.argsort(-pred, kind="stable")[:m])
    return len(true_top & pred_top) / m


def _level_col(df, level):
    c = C.qcol(level)
    return df[c].to_numpy(float) if c in df and df[c].notna().all() else None


def interval_quality(df: pd.DataFrame) -> dict:
    """Coverage AND width quality of the central intervals at C.COVERAGE_LEVELS.

    For each level a (e.g. 0.9 -> the 5%..95% interval):
      cov{a}     share of steps whose truth fell inside the interval (target = a)
      width{a}   mean interval width (logit units)
      wratio{a}  actual width / NEEDED width. The needed width is how much THIS model's intervals would have to be
                 stretched or shrunk (around its median, keeping their shape) to contain exactly a of the truths:
                 needed scale s* = the a-quantile of r_t, r_t = distance of the truth from the median in units of the
                 interval half-width on that side. 1.0 = widths exactly right for the errors the model actually made;
                 > 1 too wide (could be narrower and still honest); < 1 too narrow (overconfident). A model whose errors
                 are big is NOT penalised for wide intervals: its needed width is big too.
    Summaries: coverage_curve_error = mean |cov - a|; width_error = mean |log2(wratio)| (0 = right width at every
    level; 1 = off by 2x on average); width_ratio90 = wratio at 90%; width_adaptivity = Spearman correlation between
    each step's 90% width and its |error| (> 0: widens where it is actually wrong).
    """
    from scipy.stats import spearmanr
    y, med = df["y_true"].to_numpy(float), df["median"].to_numpy(float)
    out, cov_err, wlog = {}, [], []
    for a in C.COVERAGE_LEVELS:
        lo, hi = _level_col(df, (1 - a) / 2), _level_col(df, (1 + a) / 2)
        tag = f"{int(round(a * 100))}"
        if lo is None or hi is None or len(y) < 5:
            continue
        cov = float(((y >= lo) & (y <= hi)).mean())
        width = float((hi - lo).mean())
        up, dn = np.maximum(hi - med, 1e-9), np.maximum(med - lo, 1e-9)
        r = np.where(y >= med, (y - med) / up, (med - y) / dn)
        s_star = float(np.quantile(r, a))
        ratio = (1.0 / s_star) if s_star > 0 else np.nan
        out[f"cov{tag}"], out[f"width{tag}"], out[f"wratio{tag}"] = cov, width, ratio
        cov_err.append(abs(cov - a))
        if np.isfinite(ratio) and ratio > 0:
            wlog.append(abs(np.log2(ratio)))
    if cov_err:
        out["coverage_curve_error"] = float(np.mean(cov_err))
    if wlog:
        out["width_error"] = float(np.mean(wlog))
    if "wratio90" in out:
        out["width_ratio90"] = out["wratio90"]
    lo, hi = _level_col(df, 0.05), _level_col(df, 0.95)
    if lo is not None and len(y) >= 5 and np.ptp(hi - lo) > 0:
        out["width_adaptivity"] = float(spearmanr(hi - lo, np.abs(y - med)).statistic)
    return out


@lru_cache(maxsize=None)
def _row_noise_stats(n: int, k: int):
    """For a row with k of n, with the true yield approximated by p_hat = (k+0.5)/(n+1):
    entropy of Binomial(n, p_hat) (the NLL a perfect model averages), and the CRPS / MAE a perfect predictive
    distribution would average on the logit target. Cached: (n, k) pairs repeat a lot."""
    p = (k + 0.5) / (n + 1)
    kk = np.arange(n + 1)
    w = binom_dist.pmf(kk, n, p)
    yy = np.log((kk + 0.5) / (n + 1 - kk - 0.5))
    nz = w > 0
    H = float(-(w[nz] * np.log(w[nz])).sum())
    crps0 = float(0.5 * (w[:, None] * w[None, :] * np.abs(yy[:, None] - yy[None, :])).sum())
    med = yy[np.searchsorted(np.cumsum(w), 0.5)]
    mae0 = float((w * np.abs(yy - med)).sum())
    return H, crps0, mae0


def _deviance(k, n, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = np.where(k > 0, k * np.log(np.maximum(k, 1) / (n * p)), 0.0)
        t2 = np.where(n - k > 0, (n - k) * np.log(np.maximum(n - k, 1) / (n * (1 - p))), 0.0)
    return 2 * (t1 + t2)


def _expected_deviance(n, p):
    """E[deviance] when K ~ Binomial(n, p): what the deviance averages if the model's p were exactly right."""
    out = np.empty(len(p))
    for i, (nn, pp) in enumerate(zip(n, p)):
        kk = np.arange(int(nn) + 1)
        out[i] = (binom_dist.pmf(kk, int(nn), pp) * _deviance(kk, nn, pp)).sum()
    return out


def noise_aware(df: pd.DataFrame, s: pd.DataFrame, rmse: float, mae: float, crps: float, n_sim: int = 100) -> dict:
    """Scores relative to what binomial counting noise allows (all reported, not ranked).

    nll_excess        mean binomial NLL - mean entropy floor (0 = as good as physics allows; floor uses p_hat)
    overdispersion    mean deviance / mean deviance expected if the model's p were exactly right (1 = errors are
                      pure counting luck; >1 = model misses structure or is overconfident). _top: rows with p > 0.95.
    crps/mae_over_noise   score / the score a perfect predictive distribution would get (1 = at the floor)
    rmse_noise_corrected  sqrt(rmse^2 - noise_rmse^2): estimated RMSE against the TRUE yield
    tau_ceiling / top10_ceiling  best tau-b / top-10% recall a perfect predictor could get given the noise and ties
                      (simulated: fresh binomial draws around p_hat); *_frac = model score / ceiling
    """
    k, n = df["k"].to_numpy().astype(int), df["n"].to_numpy().astype(int)
    stats = np.array([_row_noise_stats(int(nn), int(kk)) for kk, nn in zip(k, n)])
    H, crps0, mae0 = stats[:, 0], stats[:, 1], stats[:, 2]
    r = {"nll_excess": float(s["binom_nll"].mean() - H.mean()),
         "crps_over_noise": float(crps / crps0.mean()) if crps0.mean() > 0 else np.nan,
         "mae_over_noise": float(mae / mae0.mean()) if mae0.mean() > 0 else np.nan}
    nr = float(np.sqrt((s["noise_sd"] ** 2).mean()))
    r["rmse_noise_corrected"] = float(np.sqrt(max(rmse ** 2 - nr ** 2, 0.0)))
    p = np.clip(expit(df["mean"].to_numpy(float)), 1e-12, 1 - 1e-12)
    dev, edev = _deviance(k, n, p), _expected_deviance(n, p)
    r["overdispersion"] = float(dev.mean() / edev.mean()) if edev.mean() > 0 else np.nan
    top = p > 0.95
    if top.sum() >= 10 and edev[top].mean() > 0:
        r["overdispersion_top"] = float(dev[top].mean() / edev[top].mean())
    # ceilings (depend only on the data; fixed seed per dataset so every model gets the same ceiling)
    y = df["y_true"].to_numpy(float)
    if len(y) >= 10:
        ph = (k + 0.5) / (n + 1)
        f = np.log(ph / (1 - ph))
        seed = int(abs(hash((len(y), int(k.sum()), int(n.sum())))) % (2 ** 31))
        rng = np.random.default_rng(seed)
        taus, recs = [], []
        for _ in range(n_sim):
            ks = rng.binomial(n, ph)
            ys = np.log((ks + 0.5) / (n + 1 - ks - 0.5))
            taus.append(tau_b(ys, f))
            recs.append(top10_recall(ys, f))
        tc, rc = float(np.nanmean(taus)), float(np.nanmean(recs))
        r["tau_ceiling"], r["top10_ceiling"] = tc, rc
        med = df["median"].to_numpy(float)
        tm, rm = tau_b(y, med), top10_recall(y, med)
        r["tau_frac"] = float(tm / tc) if tc and np.isfinite(tm) else np.nan
        r["top10_frac"] = float(rm / rc) if rc and np.isfinite(rm) else np.nan
    return r


def per_step(df: pd.DataFrame) -> pd.DataFrame:
    """One row per prediction with the point-wise quantities every other metric is built from."""
    y, Q = df["y_true"].to_numpy(float), qmat(df)
    out = pd.DataFrame({"iter": df["iter"].to_numpy(int)})
    out["crps"] = crps_grid(y, Q)
    out["width90"] = Q[:, I95] - Q[:, I05]
    out["covered90"] = ((y >= Q[:, I05]) & (y <= Q[:, I95])).astype(float)
    out["pit"] = pit(y, Q)
    out["err_mean"] = y - df["mean"].to_numpy(float)
    out["err_median"] = y - df["median"].to_numpy(float)
    out["abs_err"] = np.abs(out["err_median"])
    out["sq_err"] = out["err_mean"] ** 2
    out["binom_nll"] = binom_nll(df["k"].to_numpy(float), df["n"].to_numpy(float), df["mean"].to_numpy(float))
    out["noise_sd"] = noise_sd(df["k"].to_numpy(), df["n"].to_numpy())
    if "sd" in df and df["sd"].notna().all():
        out["crps_exact_gauss"] = crps_gaussian(y, df["mean"].to_numpy(float), df["sd"].to_numpy(float))
    return out


def step_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Long format: dataset, model, arm, iter, metric, value. Per-step values plus expanding-window values."""
    df = df.sort_values("iter").reset_index(drop=True)
    s = per_step(df)
    y, med = df["y_true"].to_numpy(float), df["median"].to_numpy(float)
    cnt = np.arange(1, len(s) + 1)
    wide = pd.DataFrame({"iter": s["iter"]})
    for m in ["crps", "width90", "binom_nll", "abs_err"]:
        wide[m] = s[m]
    wide["crps_running"] = s["crps"].cumsum() / cnt
    wide["binom_nll_running"] = s["binom_nll"].cumsum() / cnt
    wide["coverage90_expanding"] = s["covered90"].cumsum() / cnt
    wide["rmse_running"] = np.sqrt(s["sq_err"].cumsum() / cnt)
    wide["mae_running"] = s["abs_err"].cumsum() / cnt
    wide["kendall_tau_expanding"] = [tau_b(y[: i + 1], med[: i + 1]) for i in range(len(y))]
    if "crps_exact_gauss" in s:
        wide["crps_exact_gauss"] = s["crps_exact_gauss"]
    long = wide.melt(id_vars="iter", var_name="metric", value_name="value")
    for c in ["dataset", "model", "arm"]:
        long.insert(0, c, df[c].iloc[0])
    return long[["dataset", "model", "arm", "iter", "metric", "value"]]


def last20_mask(n_pred: int) -> np.ndarray:
    m = np.zeros(n_pred, bool)
    m[n_pred - int(np.ceil(C.LAST_FRAC * n_pred)):] = True
    return m


def _scope_metrics(df, s, scope):
    y, med, mean = df["y_true"].to_numpy(float), df["median"].to_numpy(float), df["mean"].to_numpy(float)
    rmse = float(np.sqrt(s["sq_err"].mean()))
    mae = float(s["abs_err"].mean())
    r = {
        "crps": float(s["crps"].mean()),
        "coverage90": float(s["covered90"].mean()),
        "coverage90_absdev": float(abs(s["covered90"].mean() - 0.9)),
        "width90": float(s["width90"].mean()),
        "calib_error": calib_error(s["pit"].to_numpy()),
        "kendall_tau": tau_b(y, med),
        "top10_recall": top10_recall(y, med),
        "binom_nll": float(s["binom_nll"].mean()),
        "rmse": rmse,
        "mae": mae,
        "rmse_mae_ratio": rmse / mae if mae > 0 else np.nan,
    }
    noise_rmse = float(np.sqrt((s["noise_sd"] ** 2).mean()))
    r["noise_rmse"] = noise_rmse
    r["rmse_over_noise"] = rmse / noise_rmse if noise_rmse > 0 else np.nan
    r["within_noise2"] = float((s["abs_err"] <= 2 * s["noise_sd"]).mean())
    r.update(interval_quality(df))
    r.update(noise_aware(df, s, rmse, mae, r["crps"]))
    if scope == "last20":
        sd = float(np.std(y))
        r["nrmse"] = rmse / sd if sd > 0 else np.nan
    if "crps_exact_gauss" in s:
        r["crps_exact_gauss"] = float(s["crps_exact_gauss"].mean())
        r["crps_grid_minus_exact"] = r["crps"] - r["crps_exact_gauss"]
    del mean
    return r


def final_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """dataset, model, arm, metric, scope (all | last20), value."""
    df = df.sort_values("iter").reset_index(drop=True)
    s = per_step(df)
    out = []
    mask = last20_mask(len(df))
    for scope, sel in [("all", np.ones(len(df), bool)), ("last20", mask)]:
        for m, v in _scope_metrics(df[sel].reset_index(drop=True), s[sel].reset_index(drop=True), scope).items():
            out.append(dict(dataset=df["dataset"].iloc[0], model=df["model"].iloc[0], arm=df["arm"].iloc[0],
                            metric=m, scope=scope, value=v))
    return pd.DataFrame(out)


def flag_big_errors(df: pd.DataFrame) -> pd.DataFrame:
    """Steps with |y - median| > BIG_ERROR_FACTOR x the model's median |error| on that dataset."""
    e = (df["y_true"] - df["median"]).abs()
    thr = C.BIG_ERROR_FACTOR * e.median()
    f = df.loc[e > thr, ["dataset", "model", "arm", "iter", "y_true", "median"]].copy()
    f = f.rename(columns={"median": "predicted"})
    f["error"] = f["y_true"] - f["predicted"]
    f["threshold"] = thr
    return f
