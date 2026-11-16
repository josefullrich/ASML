"""Prediction-only benchmark: every arm is fitted to the SAME black-box data (no acquisition
function involved) and scored on the SAME held-out points. This is what the report compares against
BO performance. It also holds the cheap post-processing variants ("shadow arms") and the quiz-size sweep."""
from __future__ import annotations
import json, os, time
import numpy as np
from .config import problem_dir
from .problems import lhs, scale_to
from .surrogates import make_backend
from .model import ArmSurrogate
from .yield_arms import YieldModel, CLASSIFIER_ARMS
from .loop import build_context, _stack, plan
from . import bb, metrics as M
from .surrogates.base import LEVELS


def make_data(cfg, problem, seed):
    """Training points (LHS, shared by all arms) and held-out test points with the truth."""
    path = os.path.join(problem_dir(cfg), f"pred_data_s{seed}.npz")
    if os.path.exists(path):
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    n_tr, n_te, d = max(cfg.pred_train_sizes), cfg.pred_n_test, problem.dim
    Xtr = scale_to(lhs(n_tr, d, 50_000 + seed), problem.bounds)
    Xte = scale_to(lhs(n_te, d, 60_000 + seed), problem.bounds)
    ref = problem.reference(seed)
    region = np.zeros(n_te, int)
    if "x" in ref and np.all(np.isfinite(ref["x"])):          # half the test set near the optimum, where BO works
        rng = np.random.default_rng(70_000 + seed)
        span = problem.bounds[:, 1] - problem.bounds[:, 0]
        half = n_te // 2
        near = np.array(ref["x"]) + 0.08 * span * rng.standard_normal((half, d))
        lo = np.minimum(problem.bounds[:, 0], np.array(ref["x"]) - 0.1 * span)
        hi = np.maximum(problem.bounds[:, 1], np.array(ref["x"]) + 0.1 * span)
        Xte[:half] = np.clip(near, lo, hi)
        region[:half] = 1
        near_tr = np.array(ref["x"]) + 0.15 * span * rng.standard_normal((n_tr // 2, d))
        Xtr[1::2][: len(near_tr)] = np.clip(near_tr, lo, hi)[: len(Xtr[1::2])]
    out = {"Xtr": Xtr, "Xte": Xte, "region": region}
    problem.reset(seed, 0)
    for tag, X in (("tr", Xtr), ("te", Xte)):
        evs = [problem.evaluate(x) for x in X]
        for o in problem.outputs:
            out[f"{tag}_obs_{o.name}"] = np.array([e.obs[o.name] for e in evs])
            if o.kind == "binomial":
                out[f"{tag}_trials_{o.name}"] = np.array([e.trials[o.name] for e in evs], float)
            key = o.name if o.name in evs[0].truth else None
            if key:
                out[f"{tag}_truth_{o.name}"] = np.array([e.truth[key] for e in evs])
    np.savez_compressed(path, **out)
    return out


def _score_count(pmf, mean, D, name, n, cfg, sel):
    truth = D.get(f"te_truth_{name}")
    return M.count_metrics(pmf[sel], D[f"te_obs_{name}"][sel] * n, n, mean[sel], None if truth is None else truth[sel], cfg.pmf_floor)


def run_pred(cfg, problem, deadline=None):
    path = os.path.join(problem_dir(cfg), "pred_results.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    backends = {}
    combos = sorted({(r["model"], r["context"], r["arm"], r["seed"]) for r in plan(cfg, problem)})
    for model, ctxmode, arm, seed in combos:
        D = make_data(cfg, problem, seed)
        regions = {"all": np.ones(len(D["Xte"]), bool)}
        if D["region"].any():
            regions.update(near_optimum=D["region"] == 1, global_=D["region"] == 0)
        for size in cfg.pred_train_sizes:
            key = f"{model}__{ctxmode}__{arm}__s{seed}__n{size}"
            if key in res:
                continue
            if deadline is not None and time.time() > deadline:
                json.dump(res, open(path, "w"))
                print("Budget reached during the prediction benchmark; saved. Run again to continue.")
                return False
            if model not in backends:
                backends[model] = make_backend(cfg, model)
            t0 = time.time()
            ctx_sets, aug = build_context(cfg, problem, ctxmode, seed)
            sur = ArmSurrogate(cfg, problem, backends[model], arm, bool(ctx_sets))
            sur.prepare(ctx_sets)
            ctx = _stack(ctx_sets, problem)
            names = [o.name for o in problem.outputs]
            binom = [o.name for o in problem.outputs if o.kind == "binomial"]
            X = aug(D["Xtr"][:size]); obs = {n: D[f"tr_obs_{n}"][:size] for n in names}
            tr = {n: D[f"tr_trials_{n}"][:size] for n in binom}
            if ctx is not None:
                X = np.concatenate([ctx[0], X]); obs = {n: np.concatenate([ctx[1][n], obs[n]]) for n in names}
                tr = {n: np.concatenate([ctx[2][n], tr[n]]) for n in binom}
            n_q = {n: int(np.median(D[f"te_trials_{n}"])) for n in binom}
            preds = sur.predict(X, obs, tr, aug(D["Xte"]), n_q)
            entry = {"model": model, "context": ctxmode, "arm": arm, "seed": seed, "size": size, "outputs": {}, "shadow": {}, "quiz": {}}
            for o in problem.outputs:
                p = preds[o.name]
                for rname, sel in regions.items():
                    if o.kind == "binomial":
                        pmf = YieldModel.pmf_at(p, np.arange(len(p.mean)), n_q[o.name])
                        m = _score_count(pmf, p.mean, D, o.name, n_q[o.name], cfg, sel)
                    else:
                        truth = D.get(f"te_truth_{o.name}")
                        m = M.real_metrics(p.mean[sel], p.obs_q()[sel], D[f"te_obs_{o.name}"][sel], None if truth is None else truth[sel])
                    entry["outputs"].setdefault(o.name, {})[rname] = m
            # ---- shadow arms: reprocess the classifier's p, no new BO run needed ----
            for nme in binom:
                p = preds[nme]
                if arm not in ("clf_raw", "clf_bb") or "p_raw" not in p.extra:
                    continue
                ym, n = sur.yield_models[nme], n_q[nme]
                praw = p.extra["p_raw"]
                data = [{"X": c["X"], "k": c["obs"][nme] * c["trials"][nme], "n": c["trials"][nme]} for c in ctx_sets]
                data.append({"X": aug(D["Xtr"][:size]), "k": D[f"tr_obs_{nme}"][:size] * D[f"tr_trials_{nme}"][:size], "n": D[f"tr_trials_{nme}"][:size]})
                clf = lambda Xa, ka, na, Xq: ym._classify(Xa, ka, na, Xq)
                variants = {}
                if arm == "clf_raw":
                    q = bb.quiz(clf, data, "fraction", 0.2, cfg.quiz_repeats, seed)       # 20% of all rows held back
                    z = np.log(praw / (1 - praw))
                    pl = bb.clip_p(1 / (1 + np.exp(-(q["a"] * z + q["b"]))))
                    variants["platt"] = (bb.binom_pmf(n, pl), pl)
                    variants["platt_bb"] = (bb.betabinom_pmf(n, pl, q["phi"]), pl)
                    variants["bb_quiz20"] = (bb.betabinom_pmf(n, praw, q["phi"]), praw)
                for nm, (pmf, mean) in variants.items():
                    entry["shadow"].setdefault(nme, {})[nm] = {r_: _score_count(pmf, mean, D, nme, n, cfg, s_) for r_, s_ in regions.items()}
                if arm == "clf_bb" and ctx_sets:                                           # quiz-size sweep
                    cdata = data[:-1]
                    sweep = []
                    for f in cfg.quiz_sweep:
                        phis = [bb.quiz(clf, cdata, "fraction", f, 1, 1000 * rep + seed)["phi"] for rep in range(cfg.quiz_repeats)]
                        phi = float(np.exp(np.mean(np.log(phis))))
                        m = _score_count(bb.betabinom_pmf(n, praw, phi), praw, D, nme, n, cfg, regions["all"])
                        sweep.append({"quiz": f"{int(f * 100)}% of rows", "frac": f, "phi": phi, "phi_lo": float(min(phis)), "phi_hi": float(max(phis)),
                                      "excess": m.get("excess"), "cov_err": m.get("cov_err"), "nll": m.get("nll")})
                    q = bb.quiz(clf, cdata, "lodo", 0.2, 1, seed)
                    m = _score_count(bb.betabinom_pmf(n, praw, q["phi"]), praw, D, nme, n, cfg, regions["all"])
                    sweep.append({"quiz": "leave one dataset out", "frac": None, "phi": q["phi"], "phi_lo": q["phi"], "phi_hi": q["phi"],
                                  "excess": m.get("excess"), "cov_err": m.get("cov_err"), "nll": m.get("nll")})
                    entry["quiz"][nme] = sweep
            entry["sec"] = time.time() - t0
            res[key] = entry
            print(f"  pred {key}: {entry['sec']:.1f}s", flush=True)
            json.dump(res, open(path, "w"))
    return True
