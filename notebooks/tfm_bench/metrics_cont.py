"""Metrics for a continuous target (EUV_mJ, Rx_deg, Ry_deg, TargetSize_um, or logit yield treated as a plain number).

Same predictions format as metrics.py, but nothing binomial: no k/n likelihood, no counting-noise floors. Every
quantity is in the target's own units, which is why comparing ACROSS targets needs normalization
(targets_report.py); within one target the raw values compare directly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
import metrics as M  # reuses the target-agnostic pieces: CRPS grid, PIT, calibration, tau, interval quality

# metric -> (direction, category)
METRIC_INFO = {
    "crps": ("lower", "uncertainty"),
    "coverage90": ("target90", "uncertainty"),
    "width90": ("lower", "uncertainty"),
    "width_ratio90": ("target1", "uncertainty"),
    "width_error": ("lower", "uncertainty"),
    "calib_error": ("lower", "uncertainty"),
    "coverage_curve_error": ("lower", "uncertainty"),
    "width_adaptivity": ("higher", "uncertainty"),
    "kendall_tau": ("higher", "ranking"),
    "top10_recall": ("higher", "ranking"),
    "rmse": ("lower", "point"),
    "mae": ("lower", "point"),
    "nrmse": ("lower", "point"),
}
RANKED = ["crps", "coverage90", "width90", "width_error", "calib_error", "kendall_tau", "rmse"]


def per_step(df: pd.DataFrame) -> pd.DataFrame:
    y, Q = df["y_true"].to_numpy(float), M.qmat(df)
    out = pd.DataFrame({"iter": df["iter"].to_numpy(int)})
    out["crps"] = M.crps_grid(y, Q)
    out["width90"] = Q[:, M.I95] - Q[:, M.I05]
    out["covered90"] = ((y >= Q[:, M.I05]) & (y <= Q[:, M.I95])).astype(float)
    out["pit"] = M.pit(y, Q)
    out["err_mean"] = y - df["mean"].to_numpy(float)
    out["abs_err"] = np.abs(y - df["median"].to_numpy(float))
    out["sq_err"] = out["err_mean"] ** 2
    return out


def step_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("iter").reset_index(drop=True)
    s = per_step(df)
    y, med = df["y_true"].to_numpy(float), df["median"].to_numpy(float)
    cnt = np.arange(1, len(s) + 1)
    wide = pd.DataFrame({"iter": s["iter"], "crps": s["crps"], "width90": s["width90"], "abs_err": s["abs_err"],
                         "crps_running": s["crps"].cumsum() / cnt,
                         "coverage90_expanding": s["covered90"].cumsum() / cnt,
                         "rmse_running": np.sqrt(s["sq_err"].cumsum() / cnt), "mae_running": s["abs_err"].cumsum() / cnt,
                         "kendall_tau_expanding": [M.tau_b(y[: i + 1], med[: i + 1]) for i in range(len(y))]})
    long = wide.melt(id_vars="iter", var_name="metric", value_name="value")
    for c in ["dataset", "model", "arm"]:
        long.insert(0, c, df[c].iloc[0])
    return long[["dataset", "model", "arm", "iter", "metric", "value"]]


def _scope(df, s):
    y, med = df["y_true"].to_numpy(float), df["median"].to_numpy(float)
    rmse = float(np.sqrt(s["sq_err"].mean()))
    sd = float(np.std(y))
    r = dict(crps=float(s["crps"].mean()), coverage90=float(s["covered90"].mean()), width90=float(s["width90"].mean()),
             calib_error=M.calib_error(s["pit"].to_numpy()), kendall_tau=M.tau_b(y, med),
             top10_recall=M.top10_recall(y, med), rmse=rmse, mae=float(s["abs_err"].mean()),
             nrmse=rmse / sd if sd > 0 else np.nan)
    r.update(M.interval_quality(df))  # covXX, widthXX, wratioXX, coverage_curve_error, width_error, ...
    return r


def final_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("iter").reset_index(drop=True)
    s = per_step(df)
    out = []
    for scope, sel in [("all", np.ones(len(df), bool)), ("last20", M.last20_mask(len(df)))]:
        for m, v in _scope(df[sel].reset_index(drop=True), s[sel].reset_index(drop=True)).items():
            out.append(dict(dataset=df["dataset"].iloc[0], model=df["model"].iloc[0], arm=df["arm"].iloc[0],
                            metric=m, scope=scope, value=v))
    return pd.DataFrame(out)
