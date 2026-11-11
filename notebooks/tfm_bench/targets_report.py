"""Rankings across the regressor outputs: EUV_mJ, Rx_deg, Ry_deg, TargetSize_um (+ logit die yield as a 5th).

    python targets_report.py                     # -> avg_everything/targets_comparison.html
    python targets_report.py --norm z            # default normalization: robust_z | z | skill | rank | minmax
    python targets_report.py --include-mock      # pipeline tests only

Only regressors and the regression GP: GP BoTorch and the TFM regressors (Causilo, Kumo x3, TabPFN 2 / 3.5) x context
arms. The binomial models (gpr_laplace, causilo_trials) are left out: they only model die yield and are compared in
binomial_comparison.html.

Every target is scored with the same continuous metrics (metrics_cont.py) in its own units. Within one target the raw
numbers compare directly. To rank ACROSS targets (mJ vs degrees vs um) they're normalized: per target and dataset,
each metric's badness is rescaled across configs (same five options as analyze.py), averaged over datasets -> metrics
-> categories = the target score; the overall score is the mean over the chosen targets.
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

import analyze as A
import config as C
import metrics as M
import metrics_cont as MC

warnings.filterwarnings("ignore")
TARGET_ORDER = ["EUV_mJ", "Rx_deg", "Ry_deg", "TargetSize_um", "dy"]


def badness(metric, v):
    d = MC.METRIC_INFO[metric][0]
    v = np.asarray(v, float)
    if d == "lower":
        return v
    if d == "higher":
        return 1 - v
    if d == "target1":
        return np.abs(np.log2(np.maximum(v, 1e-9)))
    return np.abs(v - {"target90": 0.9}[d])


def load_target(target, include_mock):
    root = C.results_dir_for(target)
    order = ["gpr_botorch"] + [f"{m}_{a}" for m in C.TFM_MODELS for a in C.ARMS]
    if include_mock:
        order += [f"mock_{a}" for a in C.ARMS]
    folders = [f for f in order if glob.glob(os.path.join(root, f, "results_*", "predictions.csv"))]
    finals, pits = [], {}
    for f in folders:
        pv = []
        for p in sorted(glob.glob(os.path.join(root, f, "results_*", "predictions.csv"))):
            df = pd.read_csv(p)
            fm = MC.final_metrics(df).assign(folder=f)
            finals.append(fm)
            pv.append(M.pit(df["y_true"].to_numpy(float), df[C.BASE_QCOLS].to_numpy(float)))
        pv = np.concatenate(pv)
        pits[f] = [round(float((pv <= p).mean()), 4) for p in C.PIT_LEVELS]
    return folders, (pd.concat(finals, ignore_index=True) if finals else pd.DataFrame()), pits


def target_scores(finals, folders, scope, method, ref, ranked):
    f = finals[(finals.scope == scope) & finals.folder.isin(folders)]
    have = f.groupby("folder")["dataset"].agg(set)
    common = sorted(set.intersection(*have.reindex(folders).dropna().tolist())) if len(have) else []
    ref_idx = folders.index(ref) if ref in folders else None
    per = {}
    for m in ranked:
        w = f[f.metric == m].pivot_table(index="folder", columns="dataset", values="value").reindex(index=folders,
                                                                                                    columns=common)
        if w.empty:
            continue
        B = badness(m, w.to_numpy(float))
        Z = np.column_stack([A.normalize(B[:, j], method, ref_idx) for j in range(B.shape[1])])
        per[m] = np.nanmean(Z, axis=1)
    out = pd.DataFrame(per, index=folders)
    cats = [c for c in A.CATS if any(MC.METRIC_INFO[m][1] == c for m in per)]
    for c in cats:
        out[c] = out[[m for m in per if MC.METRIC_INFO[m][1] == c]].mean(axis=1)
    out["score"] = out[cats].mean(axis=1)
    return out, common


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--norm", default=C.TARGETS_NORMALIZATION, choices=A.NORMS)
    ap.add_argument("--include-mock", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    targets, per_t = [], {}
    for T in TARGET_ORDER:
        folders, finals, pits = load_target(T, a.include_mock)
        if folders:
            targets.append(T)
            per_t[T] = (folders, finals, pits)
            print(f"  {T}: {len(folders)} configs, {finals.dataset.nunique()} datasets ({time.time() - t0:.0f}s)")
    assert targets, "no results for any target"
    all_folders = []
    for T in targets:
        all_folders += [f for f in per_t[T][0] if f not in all_folders]
    order = ["gpr_botorch"] + [f"{m}_{a}" for m in C.TFM_MODELS + ["mock"] for a in C.ARMS]
    all_folders.sort(key=lambda f: order.index(f) if f in order else 99)

    # ---- python-side rankings under every normalization (the page recomputes live)
    rows, lines = [], [f"# Rankings across outputs (default normalization `{a.norm}`)", ""]
    for scope in ["all", "last20"]:
        for nm in A.NORMS:
            sc = {}
            for T in targets:
                s, _ = target_scores(per_t[T][1], all_folders, scope, nm, C.NORMALIZATION_REF, MC.RANKED)
                sc[T] = s["score"]
            S = pd.DataFrame(sc)
            S["overall"] = S[targets].mean(axis=1, skipna=False)
            for f, r in S.iterrows():
                rows.append(dict(scope=scope, norm=nm, config=f, **r.to_dict()))
            if nm == a.norm:
                lines += [f"## {'Whole run' if scope == 'all' else 'Last 20%'}", "",
                          A.md_table(S.sort_values("overall")[targets + ["overall"]]), ""]
    os.makedirs(C.AVG_DIR, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(C.AVG_DIR, "targets_rankings_all_norms.csv"), index=False)
    open(os.path.join(C.AVG_DIR, "targets_rankings.md"), "w").write("\n".join(lines))
    pd.concat([per_t[T][1].assign(target=T) for T in targets]).to_csv(
        os.path.join(C.AVG_DIR, "targets_metrics_per_dataset.csv"), index=False)

    # ---- page payload
    metrics_shown = list(MC.METRIC_INFO)
    tdata = {}
    for T in targets:
        folders, finals, pits = per_t[T]
        datasets = sorted(finals.dataset.unique())
        d = dict(datasets=datasets, reliability=pits, per={}, values={}, intervals={})
        for scope in ["all", "last20"]:
            f = finals[finals.scope == scope]
            d["per"][scope] = {m: [[A.to_json_num(v) for v in row] for row in
                                   f[f.metric == m].pivot_table(index="folder", columns="dataset", values="value")
                                   .reindex(index=all_folders, columns=datasets).to_numpy(float)] for m in metrics_shown}
            mv = f.pivot_table(index="folder", columns="metric", values="value", aggfunc="mean")
            d["values"][scope] = {fo: {m: A.to_json_num(mv.loc[fo, m]) if m in mv else None for m in metrics_shown}
                                  for fo in folders if fo in mv.index}
            d["intervals"][scope] = A.intervals(finals, folders, scope)
        tdata[T] = d
    data = dict(title="Regressor outputs: TFMs vs GP, every output", generated=time.strftime("%Y-%m-%d %H:%M"),
                targets=targets, targetLabels=C.TARGETS, folders=[dict(id=f, model=A.folder_model_arm(f)[0],
                arm=A.folder_model_arm(f)[1], label=A.label(f)) for f in all_folders],
                models=[dict(id=m, label=A.MODEL_LABEL.get(m, m), color=A.MODEL_COLOR.get(m, "#52514e")) for m in
                        dict.fromkeys(A.folder_model_arm(f)[0] for f in all_folders)],
                arms=[x for x in C.ARMS if any(A.folder_model_arm(f)[1] == x for f in all_folders)],
                metricInfo={m: dict(dir=dd, cat=c) for m, (dd, c) in MC.METRIC_INFO.items()}, ranked=MC.RANKED,
                pitLevels=[float(x) for x in C.PIT_LEVELS],
                norm=dict(default=a.norm, ref=C.NORMALIZATION_REF, skillClip=A.SKILL_CLIP, options=["none"] + A.NORMS),
                t=tdata)
    tpl = open(os.path.join(C.CODE_DIR, "targets_template.html"), encoding="utf-8").read()
    out = os.path.join(C.AVG_DIR, "targets_comparison.html")
    from htmlutil import inline_plotly
    open(out, "w", encoding="utf-8").write(inline_plotly(tpl.replace("/*__DATA__*/null", json.dumps(A.clean(data), separators=(
        ",", ":"), allow_nan=False).replace("</", "<\\/"))))
    print(f"done in {time.time() - t0:.0f}s -> {out} ({os.path.getsize(out) / 1e6:.1f} MB)")
    print("\n".join(lines[2:lines.index("## Last 20%") if "## Last 20%" in lines else None]))


if __name__ == "__main__":
    main()
