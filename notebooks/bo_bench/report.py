"""Collect saved runs into one self-contained interactive HTML page per problem."""
from __future__ import annotations
import glob, json, os, time
import numpy as np
from .config import problem_dir, YIELD_ARMS, ACQS
from .loop import Store, plan, run_key
from . import bb, metrics as M
from .yield_arms import CLASSIFIER_ARMS

ARM_LABELS = {**YIELD_ARMS, "reg": "Regressors for every output", "feas_clf": "Regressor objective + feasibility classifier"}
MODEL_LABELS = {"gp": "GP (BoTorch)", "kumo": "Kumo", "causilo": "Causilo"}
CTX_LABELS = {"none": "no context", "machines": "context: other machines"}


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 6)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


def summarise_run(cfg, problem, path):
    st = Store(path)
    st.load()
    a, meta = st.a, st.meta
    T, n_init = meta["n_iter"], meta["n_init"]
    ref = problem.reference(meta["seed"]) or {}
    refv = ref.get("value")
    out = {k: meta[k] for k in ("model", "context", "arm", "acq", "seed")}
    out.update(id=meta["key"], done=meta.get("status") == "done", i_done=meta["i_done"], n_iter=T, n_init=n_init,
               context_rows=meta.get("context_rows", 0), yield_state=meta.get("yield_state", {}))
    bo = M.bo_curves(a["score_value"], a["score_feasible"], refv, cfg.euv_targets) if "score_value" in a else {}
    curves = {"best": bo.pop("best", []), "regret": bo.pop("regret", [])}
    if "score_true_cost" in a:
        c = np.where(np.isfinite(a["score_true_cost"]), a["score_true_cost"], -np.inf)
        if "score_shape_ok" in a:                      # srcmdl: yield enters the cost, shape constraints must hold
            c = np.where(a["score_shape_ok"] > 0, c, -np.inf)
        elif "score_feasible" in a:                     # other problems: only feasible points count
            c = np.where(a["score_feasible"] > 0, c, -np.inf)
        best_c = np.maximum.accumulate(c)
        curves["cost"] = np.where(np.isfinite(best_c), best_c, np.nan)
        bo["final_cost"] = float(curves["cost"][meta["i_done"] - 1]) if meta["i_done"] else np.nan
    if "score_true_yield" in a:
        curves["true_yield"] = a["score_true_yield"]
        ty = a["score_true_yield"][n_init:meta["i_done"]]
        bo["share_meeting_yield"] = float(np.mean(ty >= cfg.yield_target - 1e-9)) if len(ty) else np.nan
    if "score_violation" in a:
        curves["violation"] = a["score_violation"]
    sec = a["sec"][n_init:meta["i_done"]]
    bo["sec_per_step"] = float(np.nanmean(sec)) if len(sec) else np.nan
    bo["hours"] = float(np.nansum(a["sec"]) / 3600)
    out["bo"], out["curves"] = bo, curves
    # one-step-ahead prediction metrics along the trajectory (BO steps only)
    traj, shadow = {}, {}
    sl = slice(n_init, meta["i_done"])
    for o in problem.outputs:
        obs = a[f"obs_{o.name}"][sl]
        if o.kind == "binomial" and f"pmf_{o.name}" in a:
            n = int(np.nanmedian(a[f"trials_{o.name}"][: meta["i_done"]])) if meta["i_done"] else 0
            truth = a.get(f"truth_{o.name}")
            tr = None if truth is None else truth[sl]
            pm = a[f"pm_{o.name}"][sl]
            traj[o.name] = M.count_metrics(a[f"pmf_{o.name}"][sl], obs * n, n, pm, tr, cfg.pmf_floor)
            praw = a[f"praw_{o.name}"][sl]
            if np.isfinite(praw).sum() > 5:                      # shadow arms: same p, different distribution
                k = np.rint(obs * n)
                sh = {"raw_binomial": M.count_metrics(bb.binom_pmf(n, np.nan_to_num(praw, nan=0.5)), k, n, praw, tr, cfg.pmf_floor)}
                phis = np.full(len(praw), bb.PHI_MAX)
                for t in range(cfg.phi_min_fit, len(praw), 5):   # phi learned during the run, refitted every 5 steps
                    ok = np.isfinite(praw[:t])
                    phis[t:t + 5] = bb.fit_phi(praw[:t][ok], k[:t][ok], np.full(ok.sum(), n))
                sh["bb_during_run"] = M.count_metrics(bb.betabinom_pmf(n, np.nan_to_num(praw, nan=0.5), phis), k, n, praw, tr, cfg.pmf_floor)
                shadow[o.name] = sh
        elif f"q_{o.name}" in a:
            traj[o.name] = M.real_metrics(a[f"pm_{o.name}"][sl], a[f"q_{o.name}"][sl], obs)
    out["traj"], out["traj_shadow"] = traj, shadow
    return out


def build(cfg, problem):
    pdir = problem_dir(cfg)
    runs = [summarise_run(cfg, problem, p) for p in sorted(glob.glob(os.path.join(pdir, "runs", "*.npz"))) if not p.endswith(".tmp.npz")]
    pred_path = os.path.join(pdir, "pred_results.json")
    pred = list(json.load(open(pred_path)).values()) if os.path.exists(pred_path) else []
    planned = plan(cfg, problem)
    seeds = sorted({r["seed"] for r in planned})
    has_binomial = any(o.kind == "binomial" for o in problem.outputs)
    D = {
        "problem": cfg.problem, "generated": time.strftime("%Y-%m-%d %H:%M"), "standin": bool(cfg.standin),
        "planned": len(planned), "n_iter": runs[0]["n_iter"] if runs else None,
        "outputs": [{"name": o.name, "label": o.label or o.name, "kind": o.kind, "unit": o.unit} for o in problem.outputs],
        "labels": {"arm": ARM_LABELS, "model": MODEL_LABELS, "context": CTX_LABELS, "acq": ACQS},
        "point_arms": ["clf_raw"], "cov_levels": M.COV_LEVELS.tolist(),
        "reference": {str(s): problem.reference(s) for s in seeds},
        "targets": [f"evals_to_{int(round(t * 1000)) / 10:g}" for t in cfg.euv_targets],
        "objective_label": "Best true EUV with true yield ≥ %.1f%% (mJ)" % (100 * cfg.yield_target) if has_binomial else "Best feasible objective",
        "settings": {k: getattr(cfg, k) for k in ("n_candidates", "n_estimators", "kumo_size", "cost_option", "extra_scatter_rho",
                                                     "context_per_machine", "context_input_scaling", "weights_backend", "bocode_noise", "logit_target")},
        "runs": runs, "pred": pred,
    }
    here = os.path.dirname(os.path.abspath(__file__))
    html = open(os.path.join(here, "report_template.html"), encoding="utf-8").read()
    try:
        from plotly.offline import get_plotlyjs
        plotly = "<script>" + get_plotlyjs() + "</script>"
    except Exception:
        plotly = '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'
    html = html.replace("/*__PLOTLY__*/", "").replace("<!--__PLOTLY__-->", plotly)
    html = html.replace("__DATA__", json.dumps(_clean(D), separators=(",", ":")).replace("</", "<\\/"))
    out = os.path.join(pdir, "report.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    return out, D
