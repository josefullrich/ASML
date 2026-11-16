"""Any single-objective constrained BOCoDe problem as a Problem.

Use problem = "bocode:<ClassName>", e.g. "bocode:WeldedBeamSO". BOCoDe returns values to
MAXIMISE and constraints that are satisfied when <= 0. `bocode_noise` adds relative Gaussian
noise to every output for non-deterministic tests."""
from __future__ import annotations
import numpy as np
from .base import Problem, OutputSpec, Evaluation

# The "ASML-like" suite: 4 mechanical, 2 systems, 7 CEC2020. The seven CEC2020 ids are a GUESS
# (4-11 inputs, 1-9 constraints) -- replace with the ids your partner used.
ASML_LIKE_SUITE = ["WeldedBeamSO", "PressureVessel", "SpeedReducer", "HeatExchanger", "SatelliteDesign", "PEARL",
                   "CEC2020_p1", "CEC2020_p2", "CEC2020_p4", "CEC2020_p5", "CEC2020_p19", "CEC2020_p21", "CEC2020_p29"]


class BocodeProblem(Problem):
    infeasible_fallback = "min_violation"

    def __init__(self, cfg, class_name: str):
        import bocode, torch
        self.cfg, self.torch = cfg, torch
        self.name = f"bocode:{class_name}"
        self.p = getattr(bocode, class_name)()
        tb = torch.as_tensor(self.p.torch_bounds, dtype=torch.double).numpy()
        self.bounds = tb if tb.shape[1] == 2 else tb.T
        self.input_names = [f"x{i+1}" for i in range(self.bounds.shape[0])]
        self.outputs = [OutputSpec("f", "real", "Objective (maximised)", in_cost=True)]
        self.outputs += [OutputSpec(f"g{j+1}", "real", f"Constraint {j+1}", upper=0.0) for j in range(self.p.num_constraints)]
        self.seed, self.n_done, self.scale = 0, 0, None

    def reset(self, seed: int, n_done: int = 0) -> None:
        self.seed = int(seed)
        self.n_done = int(n_done)

    def _raw(self, X):
        v, c = self.p.evaluate(self.torch.as_tensor(np.atleast_2d(X), dtype=self.torch.double))
        return v.numpy()[:, 0], (c.numpy() if c is not None else np.zeros((len(v), 0)))

    def evaluate(self, x: np.ndarray) -> Evaluation:
        f, g = self._raw(x)
        truth = {"f": float(f[0]), **{f"g{j+1}": float(g[0, j]) for j in range(g.shape[1])}}
        obs = dict(truth)
        if self.cfg.bocode_noise > 0:
            if self.scale is None:                 # output scales from a fixed probe set
                from .base import lhs, scale_to
                F, G = self._raw(scale_to(lhs(256, self.dim, 12345), self.bounds))
                sd = lambda a: float(np.nanstd(a[np.isfinite(a)])) or 1.0
                self.scale = {"f": sd(F), **{f"g{j+1}": sd(G[:, j]) for j in range(G.shape[1])}}
            rng = np.random.default_rng(7_919 * self.seed + self.n_done)   # same noise for every arm
            for k in obs:
                obs[k] += self.cfg.bocode_noise * self.scale[k] * rng.standard_normal()
        self.n_done += 1
        return Evaluation(obs=obs, truth=truth)

    def cost(self, v: dict) -> np.ndarray:
        return np.asarray(v["f"], float)

    def true_score(self, x, ev: Evaluation) -> dict:
        g = np.array([ev.truth[o.name] for o in self.outputs[1:]])
        return {"value": ev.truth["f"], "feasible": bool(np.all(g <= 1e-9)), "true_cost": ev.truth["f"],
                "violation": float(np.sum(np.maximum(g, 0)))}

    def reference(self, seed: int) -> dict:
        opt = getattr(self.p, "optimum", None)
        if opt is None:
            return {}
        val = -float(self.torch.as_tensor(opt).flatten()[0])   # BOCoDe stores the minimisation optimum
        return {"value": val, "note": "optimum reported by BOCoDe (sign flipped to the maximised value)"}
