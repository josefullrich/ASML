"""Resumable closed-loop BO.

Every run is saved after each evaluation, so stopping (wall-clock budget, disconnect, crash) loses at
most one step. Re-running the same command continues from the saved files."""
from __future__ import annotations
import json, os, time
import numpy as np
from .config import problem_dir
from .problems import lhs, scale_to
from .surrogates import make_backend
from .model import ArmSurrogate
from .yield_arms import YieldModel, CLASSIFIER_ARMS
from . import acquisition as A


def run_key(r: dict) -> str:
    return f"{r['model']}__{r['context']}__{r['arm']}__{r['acq']}__s{r['seed']}"


def plan(cfg, problem) -> list:
    """Every run of the benchmark, ordered so that all arms finish a seed before the next seed starts."""
    has_binomial = any(o.kind == "binomial" for o in problem.outputs)
    arms = list(cfg.yield_arms) if has_binomial else ["reg", "feas_clf"]
    base_arm = "reg_logit" if has_binomial else "reg"
    is_bocode = cfg.problem.startswith("bocode:")
    seeds = cfg.bocode_seeds if is_bocode else cfg.seeds
    contexts = [c for c in cfg.contexts if c == "none" or not is_bocode]
    runs = []
    for seed in seeds:
        if cfg.gp_baseline:
            runs += [dict(model="gp", context="none", arm=base_arm, acq=a, seed=seed) for a in cfg.acqs]
        for model in cfg.models:
            for ctx in contexts:
                for arm in arms:
                    runs += [dict(model=model, context=ctx, arm=arm, acq=a, seed=seed) for a in cfg.acqs]
    return runs


def n_iter_for(cfg) -> int:
    return cfg.bocode_n_iter if cfg.problem.startswith("bocode:") else cfg.n_iter


class Store:
    """All arrays of one run, written atomically."""
    def __init__(self, path):
        self.path, self.a, self.meta = path, {}, {}
    def load(self):
        if os.path.exists(self.path):
            with np.load(self.path, allow_pickle=False) as z:
                self.a = {k: z[k] for k in z.files if k != "meta"}
                self.meta = json.loads(str(z["meta"]))
            return True
        return False
    def save(self):
        tmp = self.path + ".tmp.npz"
        np.savez_compressed(tmp, meta=json.dumps(self.meta), **self.a)
        os.replace(tmp, self.path)
    def ensure(self, name, shape, dtype=float, fill=np.nan):
        if name not in self.a:
            self.a[name] = np.full(shape, fill, dtype=dtype)
        return self.a[name]


def build_context(cfg, problem, mode, seed):
    sets = problem.context(mode, seed)
    use_col = bool(sets) and cfg.context_machine_column
    for c in sets:
        if use_col:
            c["X"] = np.c_[c["X"], np.full(len(c["X"]), c.get("machine_id", 1.0))]
    aug = (lambda X: np.c_[X, np.zeros(len(X))]) if use_col else (lambda X: X)
    return sets, aug


def _stack(sets, problem):
    if not sets:
        return None
    X = np.concatenate([c["X"] for c in sets])
    obs = {o.name: np.concatenate([c["obs"][o.name] for c in sets]) for o in problem.outputs}
    trials = {o.name: np.concatenate([c["trials"][o.name] for c in sets]) for o in problem.outputs if o.kind == "binomial"}
    return X, obs, trials


def run_one(cfg, problem, r: dict, deadline: float | None = None, backends: dict | None = None, max_steps: int | None = None) -> str:
    key, T, d = run_key(r), n_iter_for(cfg), problem.dim
    n_init = cfg.n_init or 3 * d
    store = Store(os.path.join(problem_dir(cfg), "runs", key + ".npz"))
    fresh = not store.load()
    if not fresh and store.meta.get("n_iter", T) > T:
        raise ValueError(f"{key}: saved run has {store.meta['n_iter']} evaluations but n_iter is now {T}; "
                         "use a new results_dir or raise n_iter")
    if not fresh and store.meta.get("n_iter") < T:            # a longer run was asked for: extend the saved one
        for k, v in list(store.a.items()):
            pad = np.full((T - len(v),) + v.shape[1:], -1 if v.dtype == np.int8 else np.nan, dtype=v.dtype)
            store.a[k] = np.concatenate([v, pad])
        store.meta.update(n_iter=T, status="running")
    if not fresh and store.meta.get("status") == "done":
        return "done"
    seed = r["seed"]
    backends = backends if backends is not None else {}
    if r["model"] not in backends:
        backends[r["model"]] = make_backend(cfg, r["model"])
    ctx_sets, aug = build_context(cfg, problem, r["context"], seed)
    sur = ArmSurrogate(cfg, problem, backends[r["model"]], r["arm"], bool(ctx_sets))
    ctx = _stack(ctx_sets, problem)
    names = [o.name for o in problem.outputs]
    binom = [o.name for o in problem.outputs if o.kind == "binomial"]
    n_q = dict(problem.default_trials)

    s = store
    X = s.ensure("X", (T, d))
    design = scale_to(lhs(n_init, d, seed), problem.bounds)   # same starting design for every arm of a seed
    for nme in names:
        s.ensure(f"obs_{nme}", T); s.ensure(f"pm_{nme}", T); s.ensure(f"plo_{nme}", T); s.ensure(f"phi_{nme}", T)
    for nme in binom:
        s.ensure(f"trials_{nme}", T); s.ensure(f"praw_{nme}", T); s.ensure(f"bbphi_{nme}", T)
    for k in ("sec", "target"):
        s.ensure(k, T)
    how = s.ensure("how", T, dtype=np.int8, fill=-1)          # 0 initial design, 1 acquisition, 2 fallback
    i0 = int(store.meta.get("i_done", 0))
    if fresh:
        store.meta = {"key": key, **r, "n_iter": T, "n_init": n_init, "i_done": 0, "status": "running",
                      "problem": cfg.problem, "context_groups": [c["group"] for c in ctx_sets],
                      "context_rows": int(sum(len(c["X"]) for c in ctx_sets))}
        t0 = time.time()
        sur.prepare(ctx_sets)                                  # the quiz happens here, before any evaluation
        store.meta["prepare_sec"] = time.time() - t0
        store.meta["yield_state"] = {n: {"phi": ym.phi, "phi_source": ym.phi_source, "quiz": ym.quiz_info} for n, ym in sur.yield_models.items()}
    else:                                                      # restore the yield model's state
        for nme, ym in sur.yield_models.items():
            st = store.meta.get("yield_state", {}).get(nme, {})
            ym.phi, ym.phi_source, ym.quiz_info = st.get("phi", ym.phi), st.get("phi_source", "none"), st.get("quiz", {})
            for j in range(i0):
                p = s.a[f"praw_{nme}"][j] if np.isfinite(s.a[f"praw_{nme}"][j]) else s.a[f"pm_{nme}"][j]
                if np.isfinite(p):
                    ym.observe(p, s.a[f"obs_{nme}"][j], s.a[f"trials_{nme}"][j])
    problem.reset(seed, i0)
    inc_x, inc_vals, steps = None, None, 0
    for i in range(i0, T):
        if deadline is not None and time.time() > deadline:
            store.save()
            return "paused"
        if max_steps is not None and steps >= max_steps:
            store.save()
            return "paused"
        t0 = time.time()
        if i < n_init:
            x = design[i].copy()
        else:                                                  # only used if the acquisition cannot choose
            x = scale_to(np.random.default_rng(1_000_003 * (seed + 1) + i).random(d), problem.bounds)
        chosen_pred, chosen_idx, mode, target = None, None, 0 if i < n_init else 2, np.nan
        obs_run = {n: s.a[f"obs_{n}"][:i] for n in names}
        ready = i >= n_init
        if ready or (ctx is not None and i >= 0):
            start = min(cfg.warmup_forget, max(i - n_init, 0)) if (cfg.warmup_forget and i > n_init) else 0
            Xtr, obs_tr = aug(X[start:i]), {n: obs_run[n][start:] for n in names}
            tr_tr = {n: s.a[f"trials_{n}"][start:i] for n in binom}
            if ctx is not None:
                Xtr = np.concatenate([ctx[0], Xtr])
                obs_tr = {n: np.concatenate([ctx[1][n], obs_tr[n]]) for n in names}
                tr_tr = {n: np.concatenate([ctx[2][n], tr_tr[n]]) for n in binom}
            if ready:
                cand = scale_to(lhs(cfg.n_candidates, d, 100_003 * (seed + 1) + i), problem.candidate_bounds(inc_x, inc_vals))
                preds = sur.predict(Xtr, obs_tr, tr_tr, aug(np.concatenate([cand, X[:i]])), n_q)
                cut = lambda p, a, b: type(p)(mean=p.mean[a:b], Q=p.Q[a:b], Qobs=None if p.Qobs is None else p.Qobs[a:b],
                                              extra={k: (v[a:b] if isinstance(v, np.ndarray) and len(v) == len(p.mean) else v) for k, v in p.extra.items()})
                pc = {n: cut(p, 0, len(cand)) for n, p in preds.items()}
                pt = {n: cut(p, len(cand), None) for n, p in preds.items()}
                target, inc_i, vals = A.incumbent(problem, obs_run, pt)
                if target is not None:
                    inc_x, inc_vals = X[inc_i].copy(), {n: float(vals[n][inc_i]) for n in names}
                    score = A.ACQ_FUNCS[r["acq"]](problem, pc, target, np.random.default_rng(7 * seed + i))
                    j = int(np.argmax(score))
                    if np.isfinite(score[j]):
                        x, chosen_pred, chosen_idx, mode = cand[j], pc, j, 1
                elif problem.infeasible_fallback == "min_violation":
                    j = int(np.argmax(A.min_violation(problem, pc)))
                    x, chosen_pred, chosen_idx, mode = cand[j], pc, j, 2
                target = np.nan if target is None else target
            if chosen_pred is None and len(Xtr) >= 5:           # still log a one-step-ahead prediction
                chosen_pred, chosen_idx = sur.predict(Xtr, obs_tr, tr_tr, aug(x[None, :]), n_q), 0
        ev = problem.evaluate(x)
        X[i] = x
        for n in names:
            s.a[f"obs_{n}"][i] = ev.obs[n]
        for n in binom:
            s.a[f"trials_{n}"][i] = ev.trials[n]
            n_q[n] = int(ev.trials[n])
        for k, v in ev.truth.items():
            s.ensure(f"truth_{k}", T)[i] = v
        for k, v in problem.true_score(x, ev).items():
            s.ensure(f"score_{k}", T)[i] = float(v)
        if chosen_pred is not None:
            for n in names:
                p = chosen_pred[n]
                s.a[f"pm_{n}"][i], s.a[f"plo_{n}"][i], s.a[f"phi_{n}"][i] = p.mean[chosen_idx], p.lo[chosen_idx], p.hi[chosen_idx]
                if n in binom:
                    pmf = YieldModel.pmf_at(p, chosen_idx, n_q[n])[0]
                    s.ensure(f"pmf_{n}", (T, len(pmf)), dtype=np.float32)[i] = pmf
                    praw = p.extra.get("p_raw")
                    s.a[f"praw_{n}"][i] = praw[chosen_idx] if praw is not None else np.nan
                    ph = p.extra.get("phi")
                    s.a[f"bbphi_{n}"][i] = ph[chosen_idx] if ph is not None else np.nan
                    sur.yield_models[n].observe(s.a[f"praw_{n}"][i] if praw is not None else p.mean[chosen_idx], ev.obs[n], ev.trials[n])
                else:
                    s.ensure(f"q_{n}", (T, p.Q.shape[1]), dtype=np.float32)[i] = p.obs_q()[chosen_idx]
        how[i], s.a["target"][i], s.a["sec"][i] = mode, target, time.time() - t0
        steps += 1
        store.meta["i_done"] = i + 1
        store.meta["yield_state"] = {n: {"phi": ym.phi, "phi_source": ym.phi_source, "quiz": ym.quiz_info} for n, ym in sur.yield_models.items()}
        if (i + 1) % cfg.save_every == 0 or i + 1 == T:
            store.save()
    store.meta["status"] = "done"
    store.save()
    return "done"


def run_all(cfg, problem, only=None, estimate=False):
    """Run every planned run until done or until the wall-clock budget is used."""
    runs = [r for r in plan(cfg, problem) if only is None or only(r)]
    deadline = time.time() + cfg.budget_hours * 3600
    backends, done, t_start = {}, 0, time.time()
    if estimate:
        return estimate_time(cfg, problem, runs, backends)
    for n, r in enumerate(runs):
        status = run_one(cfg, problem, r, deadline, backends)
        done += status == "done"
        print(f"[{n + 1}/{len(runs)}] {run_key(r)}: {status}   ({(time.time() - t_start) / 3600:.2f} h used)", flush=True)
        if status == "paused":
            print(f"\nTime budget reached. Everything is saved under {problem_dir(cfg)}.\n"
                  f"Run the same cell again (or `run.py run`) to continue. {done}/{len(runs)} runs complete.")
            return False
    print(f"\nAll {len(runs)} runs complete.")
    return True


def estimate_time(cfg, problem, runs, backends):
    """Time a few BO steps of one run per (model, context, arm, acq) and project the total."""
    import copy, shutil, tempfile
    tmp = tempfile.mkdtemp()
    c2 = copy.copy(cfg); c2.results_dir = tmp
    T = n_iter_for(cfg); n_init = cfg.n_init or 3 * problem.dim
    seen, total = {}, 0.0
    for r in runs:
        sig = (r["model"], r["context"], r["arm"], r["acq"])
        if sig not in seen:
            t0 = time.time()
            run_one(c2, problem, {**r, "seed": r["seed"]}, None, backends, max_steps=n_init + 3)
            st = Store(os.path.join(problem_dir(c2), "runs", run_key(r) + ".npz")); st.load()
            sec = st.a["sec"][n_init:n_init + 3]
            seen[sig] = float(np.nanmean(sec))
            print(f"  {run_key(r)}: {seen[sig]:.2f} s per BO step (measured at {n_init} rows; later steps are slower)")
        total += seen[sig] * (T - n_init)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\nProjected total: about {total / 3600:.1f} h for {len(runs)} runs of {T} evaluations "
          f"(lower bound: step time grows with the number of rows). Budget per session: {cfg.budget_hours} h.")
    return total
