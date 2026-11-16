"""The ASML source-model black box (srcmdl.py) as a Problem.

Nine inputs (eight normalised settings + dose target), five modelled outputs
(EUV, Rx, Ry, target size, die yield). The simulator also returns the true
underlying yield, which is used ONLY for scoring, never by a surrogate."""
from __future__ import annotations
import csv, glob, hashlib, json, os, sys
import numpy as np
from .base import Problem, OutputSpec, Evaluation

# CSV column -> simulator input, in simulator order (x, y, z, e, d, p, k, b, dt); from evaluate_model.m
CSV_INPUTS = ["PP2DX_um", "PP2DY_um", "PP2DZ_um", "PPE_mJ", "MP2PPDelay_ns", "MMPED_mJ", "FFWPED", "MPBS_um", "DoseTarget_mJ"]
INPUTS = ["l2dx", "l2dy", "l2dz", "pp_energy", "mp2pp_delay", "drive_energy", "ff_gain", "beam_size", "dose_target"]


def _import_srcmdl(folder: str):
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, "..", "..", ".."))
    tried = []
    for cand in [folder, os.path.join(repo, folder), os.path.join(repo, "src"), os.path.join(repo, "srcmdl"),
                 os.path.join(repo, "src", "srcmdl"), os.path.join(repo, "matlab"), os.getcwd()]:
        cand = os.path.abspath(cand)
        tried.append(cand)
        if os.path.isfile(os.path.join(cand, "srcmdl.py")):
            if cand not in sys.path:
                sys.path.insert(0, cand)
            import srcmdl as mod
            return mod
    raise FileNotFoundError("srcmdl.py not found. Set srcmdl_dir. Looked in:\n  " + "\n  ".join(tried))


class SrcmdlProblem(Problem):
    name = "srcmdl"
    input_names = INPUTS
    infeasible_fallback = "lhs"
    default_trials = {"dy": 71}

    def __init__(self, cfg):
        self.cfg = cfg
        self.mod = _import_srcmdl(cfg.srcmdl_dir)
        self.lbr, self.ubr = 50.0, 60.0
        self.bounds = np.array([[0.0, 1.0]] * 8 + [[self.lbr, self.ubr]])
        self.lbdy = cfg.yield_target
        self.outputs = [
            OutputSpec("euv", "real", "EUV", "mJ", in_cost=True),
            OutputSpec("rx", "real", "Rx", "deg", lower=-23.5, upper=23.5),
            OutputSpec("ry", "real", "Ry", "deg", lower=11.5, upper=23.5),
            OutputSpec("ts", "real", "Target size", "norm.", lower=0.1, upper=0.9),
            OutputSpec("dy", "binomial", "Die yield", "fraction", in_cost=True, noisy_incumbent=True),
        ]
        self.seed, self.state, self.n_done = 0, 0, 0

    # ---- black box ---------------------------------------------------
    def reset(self, seed: int, n_done: int = 0) -> None:
        self.seed, self.n_done = int(seed), int(n_done)
        self.state = int(seed) if n_done == 0 else self.mod.SourceState(seed=int(seed), drift=np.zeros(6), k=int(n_done))

    def _call(self, x, state):
        return self.mod.srcmdl(*[float(v) for v in x[:8]], float(x[8]), state)

    def evaluate(self, x: np.ndarray) -> Evaluation:
        state, euv, rx, ry, ts, dy, n, _de, euvol, dy_actual = self._call(x, self.state)
        state.drift.fill(0.0)                      # drift disabled, as in the MATLAB optimiser
        self.state, self.n_done = state, self.n_done + 1
        rho = float(self.cfg.extra_scatter_rho)
        if rho > 0:                                # optional run-to-run scatter around the true yield
            rng = np.random.default_rng(1_000_003 * self.seed + state.k)
            phi = 1.0 / rho - 1.0
            p = float(np.clip(dy_actual, 1e-6, 1 - 1e-6))
            dy = rng.binomial(int(n), rng.beta(p * phi, (1 - p) * phi)) / n
        return Evaluation(
            obs={"euv": float(np.mean(euv)), "rx": float(rx), "ry": float(ry), "ts": float(ts), "dy": float(dy)},
            trials={"dy": int(n)},
            truth={"dy": float(dy_actual), "euv_open_loop": float(euvol)},
        )

    # ---- objective ---------------------------------------------------
    def cost(self, v: dict) -> np.ndarray:
        euv, dy = np.asarray(v["euv"], float), np.asarray(v["dy"], float)
        tol, lb = 1e-4, self.lbdy
        opt = self.cfg.cost_option
        if opt == 0:
            return euv - 10.0 * np.abs(dy - lb)
        if opt == 1:
            return euv * (1.0 - np.abs(dy - lb) ** 2)
        d = np.clip(dy, tol, 1 - tol)
        return euv - np.abs(np.log(d / (1 - d)) - np.log(lb / (1 - lb))) / 10.0

    def candidate_bounds(self, inc_x, inc_values) -> np.ndarray:
        b = self.bounds.copy()
        if inc_x is not None:
            best_dt, best_dy = float(inc_x[8]), float(inc_values["dy"])
            if np.isfinite(best_dt) and np.isfinite(best_dy) and 0.98 * best_dt * best_dy > self.lbr:
                b[8] = [0.97 * best_dt * best_dy, 1.03 * best_dt * best_dy]   # +/-3% around best dose x yield
        return b

    def true_score(self, x, ev: Evaluation) -> dict:
        shape_ok = bool(self.feasible({k: np.array([ev.obs[k]]) for k in ("rx", "ry", "ts")})[0])
        ok = shape_ok and ev.truth["dy"] >= self.lbdy - 1e-9
        return {"value": ev.obs["euv"], "feasible": ok, "shape_ok": shape_ok, "true_yield": ev.truth["dy"],
                "true_cost": float(self.cost({"euv": ev.obs["euv"], "dy": ev.truth["dy"]}))}

    # ---- reference optimum (for regret) ------------------------------
    def reference(self, seed: int) -> dict:
        from ..config import problem_dir
        path = os.path.join(problem_dir(self.cfg), f"reference_seed{seed}.json")
        if os.path.exists(path):
            return json.load(open(path))
        from scipy.optimize import differential_evolution, brentq
        K = 200                                     # past the start-up transient of the simulator

        def call(x8, dt):
            st = self.mod.SourceState(seed=int(seed), drift=np.zeros(6), k=K - 1)
            return self._call(np.r_[x8, dt], st)

        def neg_open_loop(x8):
            _s, _e, rx, ry, ts, *_r, euvol, _dya = call(x8, 100.0)
            viol = max(0, -23.5 - rx, rx - 23.5) + max(0, 11.5 - ry, ry - 23.5) + 20 * max(0, 0.1 - ts, ts - 0.9)
            return -euvol + viol
        res = differential_evolution(neg_open_loop, [(0, 1)] * 8, seed=0, popsize=24, maxiter=120, tol=1e-9, polish=True)
        x8 = np.clip(res.x, 0, 1)
        g = lambda dt: call(x8, dt)[-1] - self.lbdy   # true yield minus target, decreasing in dose
        lo, hi = 20.0, 100.0
        dt_star = brentq(g, lo, hi, xtol=1e-6) if g(lo) > 0 > g(hi) else float("nan")
        out = call(x8, dt_star) if np.isfinite(dt_star) else None
        ref = {"x": [*map(float, x8), float(dt_star)], "open_loop_euv": float(-res.fun),
               "value": float(np.mean(out[1])) if out else float("nan"),
               "true_yield": float(out[-1]) if out else float("nan"),
               "note": "best EUV with true yield at the target and shape constraints met (numerical search)"}
        json.dump(ref, open(path, "w"), indent=1)
        return ref

    # ---- context from the real machines -------------------------------
    def context(self, mode: str, seed: int):
        if mode in ("none", "", None):
            return []
        if mode != "machines":
            raise ValueError(f"unknown context mode {mode}")
        return load_machine_context(self.cfg, seed)


def read_logs(data_dir: str):
    """Read opt_results_<machine>_<stamp>.csv, dropping byte-identical duplicate files."""
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, "..", "..", ".."))
    for cand in [data_dir, os.path.join(repo, data_dir)]:
        files = sorted(glob.glob(os.path.join(cand, "opt_results_*.csv")))
        if files:
            break
    else:
        raise FileNotFoundError(f"no opt_results_*.csv under {data_dir!r}; set data_dir")
    seen, logs = set(), []
    for f in files:
        digest = hashlib.md5(open(f, "rb").read()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        parts = os.path.basename(f)[:-4].split("_")
        machine = parts[2] if len(parts) >= 5 else "unknown"
        rows = list(csv.DictReader(open(f)))
        logs.append({"file": os.path.basename(f), "machine": machine,
                     "cols": {c: np.array([float(r[c]) for r in rows]) for c in rows[0]}})
    return logs


def _scale(col: np.ndarray, how: str) -> np.ndarray:
    if how == "none":
        return col.astype(float)
    if how == "log":
        col = np.log1p(col - col.min())
        how = "minmax"
    if how == "zscore":
        s = col.std()
        return (col - col.mean()) / (s if s > 0 else 1.0)
    if how == "minmax":
        span = col.max() - col.min()
        return (col - col.min()) / (span if span > 0 else 1.0)
    raise ValueError(f"unknown scaling {how}")


def load_machine_context(cfg, seed: int):
    """An equal number of datasets from every machine, mapped onto the simulator's inputs/outputs.

    Dose (mJ), EUV (mJ) and the angles (deg) share units with the simulator. Target size uses the
    simulator's documented (um - 300) / 180. The eight other inputs have no documented mapping, so
    each is rescaled per machine with `context_input_scaling`."""
    logs = read_logs(cfg.data_dir)
    machines = sorted({l["machine"] for l in logs})
    per = min(cfg.context_per_machine, min(sum(l["machine"] == m for l in logs) for m in machines))
    rng = np.random.default_rng(10_000 + seed)
    out = []
    for mi, m in enumerate(machines):
        mine = [l for l in logs if l["machine"] == m]
        stats_src = {c: np.concatenate([l["cols"][c] for l in mine]) for c in CSV_INPUTS[:8]}
        for idx in sorted(rng.choice(len(mine), size=per, replace=False)):
            c = mine[idx]["cols"]
            n, k = c["NrOfExposures"], c["NrOfInSpecDose"]
            keep = (n > 0) & (n <= cfg.context_max_trials * np.median(n))
            X = np.empty((len(n), 9))
            for j, name in enumerate(CSV_INPUTS[:8]):
                ref = stats_src[name]
                both = _scale(np.concatenate([ref, c[name]]), cfg.context_input_scaling)   # per-machine statistics
                X[:, j] = both[len(ref):]
            X[:, 8] = c["DoseTarget_mJ"]
            out.append({"group": mine[idx]["file"][12:-4], "machine": m, "machine_id": mi + 1, "X": X[keep],
                        "obs": {"euv": c["EUV_mJ"][keep], "rx": c["Rx_deg"][keep], "ry": c["Ry_deg"][keep],
                                "ts": (c["TargetSize_um"][keep] - 300.0) / 180.0, "dy": (k / np.maximum(n, 1))[keep]},
                        "trials": {"dy": n[keep]}})
    return out
