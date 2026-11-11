"""Rebuild every metric, ranking and the interactive summary from saved predictions. No model is rerun.

    python analyze.py                          # uses config.NORMALIZATION (default rank, as on Kaggle)
    python analyze.py --norm z                 # z | robust_z | skill | rank | minmax
    python analyze.py --use-cached             # reuse final/step metrics already on disk (faster, may be stale)
    python analyze.py --include-mock           # also show the CPU stand-in (pipeline tests only)

Reads   results/<config>/results_<dataset>/predictions.csv     (the only thing a run must produce)
Writes  results/<config>/results_<dataset>/final_metrics.csv, step_metrics.csv   (recomputed)
        avg_everything/summary.html            interactive page; normalization can be switched in the page
        avg_everything/rankings.md             rankings under the chosen normalization
        avg_everything/rankings_all_norms.csv  overall + category scores under every normalization
        avg_everything/metrics_per_dataset.csv every metric, every config, every dataset (long format)
        avg_everything/flagged_errors.csv

How a ranking is built (same in Python and in the HTML):
  1. per dataset and config, every ranked metric becomes a badness b >= 0, lower = better:
       lower-better: b = value   higher-better: b = 1 - value   target t: b = |value - t|   target 1 (ratio): |log2 value|
  2. within each dataset, b is rescaled across ALL configs with the chosen normalization (see config.NORMALIZATION);
  3. averaged over the datasets every config has (so no config is scored on easier days than another);
  4. averaged over the metrics of each category (uncertainty / ranking / point), then over the categories (overall).
  Lower is better everywhere.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

import config as C
import metrics as M

NORMS = ["robust_z", "z", "skill", "rank", "minmax"]
CATS = ["uncertainty", "ranking", "point"]
SKILL_CLIP = 3.0  # skill = log2(b / b_ref) clipped to [-3, 3] (8x better / worse), so a near-zero reference can't dominate

MODEL_COLOR = {"gpr_laplace": "#2a78d6", "gpr_botorch": "#eb6834", "causilo": "#1baf7a", "causilo_trials": "#0d6e4c",
               "kumo": "#3f2fa8", "kumo_medium": "#6f5fe0", "kumo_small": "#a499f0", "tabpfn2": "#e87ba4",
               "tabpfn35": "#008300", "mock": "#8a8984"}
MODEL_LABEL = {"gpr_laplace": "GP Laplace (MATLAB port)", "gpr_botorch": "GP BoTorch", "causilo": "Causilo",
               "causilo_trials": "Causilo classifier", "kumo_small": "Kumo small", "kumo_medium": "Kumo medium",
               "kumo": "Kumo large", "tabpfn2": "TabPFN 2", "tabpfn35": "TabPFN 3.5", "mock": "mock"}
CURVES = {  # step metric -> title, direction text, log y, reference line, metric used to order the hover list
    "crps": ("CRPS (grid)", "lower is better", True, None, "crps"),
    "width90": ("90% interval width", "lower is better, given good coverage", False, None, "width90"),
    "binom_nll": ("Binomial NLL", "lower is better", True, None, "binom_nll"),
    "abs_err": ("|error| (median)", "lower is better", True, None, "mae"),
    "crps_running": ("Running mean CRPS", "lower is better", True, None, "crps"),
    "coverage90_expanding": ("90% coverage", "target 0.90", False, 0.9, "coverage90"),
    "rmse_running": ("Running RMSE", "lower is better", True, None, "rmse"),
    "mae_running": ("Running MAE", "lower is better", True, None, "mae"),
    "kendall_tau_expanding": ("Kendall's tau-b", "higher is better", False, None, "kendall_tau"),
    "binom_nll_running": ("Running binomial NLL", "lower is better", True, None, "binom_nll"),
}
PER_STEP = {"crps", "width90", "binom_nll", "abs_err"}  # smoothed with a rolling mean before averaging


# ----------------------------------------------------------------------------------------------------------------------
# folders
# ----------------------------------------------------------------------------------------------------------------------
def folder_model_arm(folder):
    for m in sorted(C.ALL_TFM_MODELS + ["mock"], key=len, reverse=True):  # longest first: kumo_small before kumo
        if folder.startswith(m + "_") and folder[len(m) + 1:] in C.ARMS:
            return m, folder[len(m) + 1:]
    return folder, "none"


def label(folder):
    m, a = folder_model_arm(folder)
    return f"kumo_large_{a}" if m == "kumo" else folder


def discover_folders(include_mock):
    present = {os.path.basename(p) for p in glob.glob(os.path.join(C.RESULTS_DIR, "*"))
               if glob.glob(os.path.join(p, "results_*", "predictions.csv"))}
    order = C.model_folders() + ([f"mock_{a}" for a in C.ARMS] if include_mock else [])
    return [f for f in order if f in present] + sorted(present - set(order) - ({f"mock_{a}" for a in C.ARMS}
                                                                                 if not include_mock else set()))


# ----------------------------------------------------------------------------------------------------------------------
# load + (re)compute per-dataset metrics
# ----------------------------------------------------------------------------------------------------------------------
def load(folders, use_cached):
    preds, steps, finals, flags = [], [], [], []
    for f in folders:
        for d in sorted(glob.glob(os.path.join(C.RESULTS_DIR, f, "results_*"))):
            p = os.path.join(d, "predictions.csv")
            if not os.path.exists(p):
                continue
            df = pd.read_csv(p)
            fp, sp = os.path.join(d, "final_metrics.csv"), os.path.join(d, "step_metrics.csv")
            fresh = use_cached and os.path.exists(fp) and os.path.exists(sp) and os.path.getmtime(fp) >= os.path.getmtime(p)
            if fresh:
                fm, sm = pd.read_csv(fp), pd.read_csv(sp)
            else:
                fm, sm = M.final_metrics(df), M.step_metrics(df)
                fm.to_csv(fp, index=False)
                sm.to_csv(sp, index=False)
            df, fm, sm = (t.assign(folder=f) for t in (df, fm, sm))
            fl = M.flag_big_errors(df)
            fl["folder"] = f
            keep = ["folder", "dataset", "iter", "y_true", "median"] + [c for c in C.BASE_QCOLS if c in df]
            preds.append(df[keep])
            steps.append(sm)
            finals.append(fm)
            flags.append(fl)
        print(f"  {f}: {sum(1 for x in finals if x['folder'].iloc[0] == f)} datasets", flush=True)
    cat = lambda xs: pd.concat(xs, ignore_index=True) if xs else pd.DataFrame()
    return cat(preds), cat(steps), cat(finals), cat(flags)


# ----------------------------------------------------------------------------------------------------------------------
# normalization (the HTML has a line-for-line JavaScript copy: keep them in sync)
# ----------------------------------------------------------------------------------------------------------------------
def badness(metric, v):
    d = M.METRIC_INFO[metric][0]
    v = np.asarray(v, float)
    if d == "lower":
        return v
    if d == "higher":
        return 1 - v
    if d.startswith("target"):
        t = {"target90": 0.9, "target95": 0.95, "target125": 1.25, "target1": 1.0}[d]
        return np.abs(np.log2(np.maximum(v, 1e-9))) if d == "target1" else np.abs(v - t)
    return np.full_like(v, np.nan)


def normalize(b, method, ref_idx=None):
    """b: badness of every config on ONE dataset (NaN = missing). Returns the normalized score, lower = better."""
    b = np.asarray(b, float)
    ok = np.isfinite(b)
    out = np.full_like(b, np.nan)
    if ok.sum() == 0:
        return out
    x = b[ok]
    if method == "z":
        sd = x.std(ddof=0)
        out[ok] = (x - x.mean()) / sd if sd > 0 else 0.0
    elif method == "robust_z":
        med = np.median(x)
        s = 1.4826 * np.median(np.abs(x - med))
        if not s > 0:
            s = x.std(ddof=0)
        out[ok] = (x - med) / s if s > 0 else 0.0
    elif method == "skill":
        r = b[ref_idx] if ref_idx is not None else np.nan
        if np.isfinite(r) and r > 1e-12:
            out[ok] = np.clip(np.log2(np.maximum(x, 1e-12) / r), -SKILL_CLIP, SKILL_CLIP)
    elif method == "rank":
        rk = pd.Series(x).rank(method="average").to_numpy()
        out[ok] = (rk - 1) / (len(x) - 1) if len(x) > 1 else 0.0
    elif method == "minmax":
        span = x.max() - x.min()
        out[ok] = (x - x.min()) / span if span > 0 else 0.0
    else:
        raise ValueError(method)
    return out


def scores(finals, folders, scope, method, ref, ranked=None):
    """-> (DataFrame folder x [metrics..., uncertainty, ranking, point, overall], list of datasets used)."""
    ranked = ranked or M.RANKED[scope]
    f = finals[(finals.scope == scope) & finals.folder.isin(folders)]
    have = f.groupby("folder")["dataset"].agg(lambda s: set(s))
    common = sorted(set.intersection(*have.reindex(folders).dropna().tolist())) if len(have) else []
    ref_idx = folders.index(ref) if ref in folders else None
    per_metric = {}
    for m in ranked:
        w = f[f.metric == m].pivot_table(index="folder", columns="dataset", values="value").reindex(index=folders,
                                                                                                    columns=common)
        if w.empty or w.isna().all().all():
            continue
        B = badness(m, w.to_numpy(float))
        Z = np.column_stack([normalize(B[:, j], method, ref_idx) for j in range(B.shape[1])]) if B.shape[1] else B
        per_metric[m] = np.nanmean(Z, axis=1) if Z.size else np.full(len(folders), np.nan)
    out = pd.DataFrame(per_metric, index=folders)
    for c in CATS:
        ms = [m for m in per_metric if M.METRIC_INFO[m][1] == c]
        out[c] = out[ms].mean(axis=1) if ms else np.nan
    out["overall"] = out[[c for c in CATS if out[c].notna().any()]].mean(axis=1)
    return out, common


# ----------------------------------------------------------------------------------------------------------------------
# summary pieces
# ----------------------------------------------------------------------------------------------------------------------
def curves(steps, folders):
    out = {}
    s = steps[steps.metric.isin(CURVES)].copy()
    for (f, ds, m), g in s.groupby(["folder", "dataset", "metric"]):
        if m in PER_STEP:
            idx = g.sort_values("iter").index
            s.loc[idx, "value"] = g.loc[idx, "value"].rolling(C.ROLLING_WINDOW, min_periods=1).mean().to_numpy()
    agg = s.groupby(["metric", "folder", "iter"])["value"].agg(["mean", "std", "count"]).reset_index()
    agg["se"] = (agg["std"] / np.sqrt(agg["count"])).fillna(0)
    for m, (title, dtext, log, ref, rank_as) in CURVES.items():
        series = {}
        for f in folders:
            g = agg[(agg.metric == m) & (agg.folder == f)].sort_values("iter")
            if len(g):
                series[f] = [g["iter"].astype(int).tolist(), np.round(g["mean"], 5).tolist(),
                             np.round(g["se"], 5).tolist()]
        out[m] = dict(title=title, dirText=dtext, log=log, band=True, ref=ref, rankAs=rank_as, series=series)
    return out


def reliability(preds, folders):
    levels = [float(x) for x in C.PIT_LEVELS]
    series = {}
    for f in folders:
        d = preds[preds.folder == f]
        if len(d):
            pv = M.pit(d["y_true"].to_numpy(float), d[C.BASE_QCOLS].to_numpy(float))
            series[f] = [round(float((pv <= p).mean()), 4) for p in levels]
    return dict(levels=levels, series=series)


def intervals(finals, folders, scope):
    f = finals[(finals.scope == scope) & finals.folder.isin(folders)]
    w = f.pivot_table(index="folder", columns="metric", values="value", aggfunc="mean")
    lv = C.COVERAGE_LEVELS
    tags = [str(int(round(a * 100))) for a in lv]
    pick = lambda pre, fo: [None if (pre + t) not in w or not np.isfinite(w.loc[fo, pre + t]) else
                            round(float(w.loc[fo, pre + t]), 4) for t in tags]
    fs = [fo for fo in folders if fo in w.index]
    return dict(levels=lv, cov={fo: pick("cov", fo) for fo in fs}, wratio={fo: pick("wratio", fo) for fo in fs},
                width={fo: pick("width", fo) for fo in fs})


def to_json_num(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 6)


def clean(o):
    """NaN / inf -> None everywhere, so the page gets valid JSON."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    return o


def build_data(preds, steps, finals, flags, folders, norm):
    datasets = sorted(finals.dataset.unique())
    tables, per_ds = {}, {}
    for scope in ["all", "last20"]:
        f = finals[(finals.scope == scope) & finals.folder.isin(folders)]
        if f.empty:
            continue
        mean_vals = f.pivot_table(index="folder", columns="metric", values="value", aggfunc="mean")
        shown = [m for m in M.METRIC_INFO if m in mean_vals.columns]
        ranked = [m for m in M.RANKED[scope] if m in mean_vals.columns]
        shown = ranked + [m for m in shown if m not in ranked]
        tables[scope] = dict(metrics=shown, ranked=ranked,
                             values={fo: {m: to_json_num(mean_vals.loc[fo, m]) for m in shown}
                                     for fo in folders if fo in mean_vals.index})
        # raw per-dataset values for every metric the page may rank on: [metric][folder] -> list over datasets
        per_ds[scope] = {}
        for m in shown:
            w = f[f.metric == m].pivot_table(index="folder", columns="dataset", values="value").reindex(
                index=folders, columns=datasets)
            per_ds[scope][m] = [[to_json_num(v) for v in row] for row in w.to_numpy(float)]
    n_pred = finals[(finals.scope == "all") & (finals.metric == "crps")]
    first = preds.groupby(["folder", "dataset"])["iter"].agg(["min", "max", "count"])
    shade = float(np.median([lo + (n - int(np.ceil(C.LAST_FRAC * n))) for lo, _, n in first.to_numpy()])) if len(first) else None
    fl = flags.sort_values(["folder", "dataset", "iter"]) if len(flags) else flags
    flag_rows = []
    if len(fl):
        fl = fl.assign(model=fl.folder.map(lambda x: folder_model_arm(x)[0]), arm=fl.folder.map(lambda x: folder_model_arm(x)[1]))
        cols = ["dataset", "model", "arm", "iter", "y_true", "predicted", "error", "threshold"]
        flag_rows = [[(round(float(v), 4) if isinstance(v, (float, np.floating)) else (int(v) if isinstance(v, (np.integer,)) else v))
                      for v in r] for r in fl[cols].head(3000).itertuples(index=False)]
    models_present = []
    for fo in folders:
        m = folder_model_arm(fo)[0]
        if m not in [x["id"] for x in models_present]:
            models_present.append(dict(id=m, label=MODEL_LABEL.get(m, m), color=MODEL_COLOR.get(m, "#52514e")))
    del n_pred
    return dict(
        title="TFM vs GP surrogates: die-yield prediction",
        note=(f"{len(datasets)} datasets · {len(folders)} model configs · logit scale · one-step-ahead, warm start "
              f"{C.N_WARMUP} rows. Colour = model (Kumo sizes share one hue: dark = large, light = small), line style "
              f"= context arm (solid = zero-shot/GP, dashed = 5 days, dotted = all days). Built "
              f"{time.strftime('%Y-%m-%d %H:%M')}."),
        models=models_present, arms=[a for a in C.ARMS if any(folder_model_arm(f)[1] == a for f in folders)],
        folders=[dict(id=f, model=folder_model_arm(f)[0], arm=folder_model_arm(f)[1], label=label(f)) for f in folders],
        metricInfo={m: dict(dir=d, cat=c) for m, (d, c) in M.METRIC_INFO.items()},
        tables=tables, datasets=datasets, perDataset=per_ds,
        norm=dict(default=norm, ref=C.NORMALIZATION_REF, skillClip=SKILL_CLIP, options=NORMS),
        curves=curves(steps, folders), shadeFrom=shade, reliability=reliability(preds, folders),
        intervals={s: intervals(finals, folders, s) for s in tables},
        flagged=dict(columns=["dataset", "model", "arm", "iter", "y_true", "predicted", "error", "threshold"],
                     total=int(len(flags)), rows=flag_rows,
                     note="|y − median| &gt; 2 × the model's median |error| on that dataset."),
        staticImages=[])


# ----------------------------------------------------------------------------------------------------------------------
# reports
# ----------------------------------------------------------------------------------------------------------------------
def md_table(df, fmt="{:.3g}"):
    cols = ["config"] + list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for i, r in df.iterrows():
        lines.append("| " + " | ".join([label(i)] + [fmt.format(v) if np.isfinite(v) else "" for v in r.values]) + " |")
    return "\n".join(lines)


def write_reports(finals, folders, norm):
    os.makedirs(C.AVG_DIR, exist_ok=True)
    rows = []
    lines = [f"# Rankings (normalization: `{norm}`)", "",
             "Scores are normalized badness (lower = better), averaged over the datasets every config has. "
             "`overall` = mean of the three categories. Switch normalization live in summary.html.", ""]
    for scope, title in [("all", "Whole run"), ("last20", "Last 20%")]:
        for nm in NORMS:
            sc, common = scores(finals, folders, scope, nm, C.NORMALIZATION_REF)
            for fo, r in sc.iterrows():
                rows.append(dict(scope=scope, norm=nm, config=fo, n_datasets=len(common),
                                 **{c: r.get(c, np.nan) for c in CATS + ["overall"]}))
            if nm == norm:
                sc = sc.sort_values("overall")
                lines += [f"## {title} ({len(common)} common datasets)", "",
                          md_table(sc[CATS + ["overall"]]), "", "Per metric:", "",
                          md_table(sc[[c for c in sc.columns if c not in CATS + ["overall"]]]), ""]
    pd.DataFrame(rows).to_csv(os.path.join(C.AVG_DIR, "rankings_all_norms.csv"), index=False)
    open(os.path.join(C.AVG_DIR, "rankings.md"), "w").write("\n".join(lines))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--norm", default=C.NORMALIZATION, choices=NORMS)
    ap.add_argument("--use-cached", action="store_true")
    ap.add_argument("--include-mock", action="store_true")
    a = ap.parse_args()
    folders = discover_folders(a.include_mock)
    assert folders, f"no results/*/results_*/predictions.csv under {C.RESULTS_DIR}"
    print(f"{len(folders)} configs under {C.RESULTS_DIR}; computing metrics ...", flush=True)
    t = time.time()
    preds, steps, finals, flags = load(folders, a.use_cached)
    os.makedirs(C.AVG_DIR, exist_ok=True)
    finals.to_csv(os.path.join(C.AVG_DIR, "metrics_per_dataset.csv"), index=False)
    flags.to_csv(os.path.join(C.AVG_DIR, "flagged_errors.csv"), index=False)
    md = write_reports(finals, folders, a.norm)
    data = build_data(preds, steps, finals, flags, folders, a.norm)
    tpl = open(os.path.join(C.CODE_DIR, "summary_template.html"), encoding="utf-8").read()
    js = json.dumps(clean(data), separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    out = os.path.join(C.AVG_DIR, "summary.html")
    open(out, "w", encoding="utf-8").write(tpl.replace("/*__DATA__*/null", js))
    print(f"done in {time.time() - t:.0f}s -> {out} ({os.path.getsize(out) / 1e6:.1f} MB)")
    i = md.find("## Whole run")
    print(md[i:md.find("Per metric", i)])


if __name__ == "__main__":
    main()
