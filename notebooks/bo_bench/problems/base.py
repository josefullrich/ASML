"""Problem interface. A black box only has to describe its inputs and outputs and say
how outputs turn into a cost and into feasibility; the loop, acquisitions, metrics and
report never look at problem-specific names."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class OutputSpec:
    name: str
    kind: str = "real"            # "real" or "binomial" (successes out of trials)
    label: str = ""
    unit: str = ""
    lower: float = -np.inf        # hard feasibility bounds (constraint outputs)
    upper: float = np.inf
    in_cost: bool = False         # enters the cost function
    noisy_incumbent: bool = False # use the model's mean (not the raw measurement) when scoring past points


@dataclass
class Evaluation:
    obs: dict                      # name -> float (binomial: observed fraction)
    trials: dict = field(default_factory=dict)   # name -> number of trials (binomial outputs)
    truth: dict = field(default_factory=dict)    # simulation-only noise-free values


class Problem:
    name: str = "problem"
    input_names: list = []
    bounds: np.ndarray = None      # [d, 2]
    outputs: list = []             # list[OutputSpec]
    infeasible_fallback: str = "lhs"   # what to do before any feasible point exists: lhs | min_violation
    default_trials: dict = {}          # usual number of trials for each binomial output

    # --- required ---
    def reset(self, seed: int, n_done: int = 0) -> None: ...
    def evaluate(self, x: np.ndarray) -> Evaluation: ...
    def cost(self, values: dict) -> np.ndarray:
        """Objective to MAXIMISE from arrays of output values."""
        raise NotImplementedError

    # --- optional hooks ---
    def candidate_bounds(self, incumbent_x, incumbent_values) -> np.ndarray:
        return self.bounds
    def context(self, mode: str, seed: int):
        """Return a list of context datasets (dict X, obs, trials, group) or []."""
        return []
    def reference(self, seed: int) -> dict:
        """Known/estimated optimum for regret. {} if unknown."""
        return {}
    def true_score(self, x, ev: Evaluation) -> dict:
        """Per-evaluation truth metrics: {'value': objective, 'feasible': bool, ...}."""
        return {}

    # --- helpers ---
    def spec(self, name):
        return next(o for o in self.outputs if o.name == name)
    def feasible(self, values: dict) -> np.ndarray:
        ok = None
        for o in self.outputs:
            if np.isfinite(o.lower) or np.isfinite(o.upper):
                v = np.asarray(values[o.name], float)
                m = (v >= o.lower) & (v <= o.upper)
                ok = m if ok is None else ok & m
        if ok is None:
            ok = np.ones(np.shape(next(iter(values.values()))), bool)
        return ok
    def constraint_outputs(self):
        return [o for o in self.outputs if np.isfinite(o.lower) or np.isfinite(o.upper)]
    def cost_outputs(self):
        return [o for o in self.outputs if o.in_cost]
    @property
    def dim(self):
        return len(self.input_names)


def lhs(n: int, d: int, seed: int) -> np.ndarray:
    from scipy.stats import qmc
    return qmc.LatinHypercube(d=d, seed=seed).random(n)


def scale_to(u: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    return bounds[:, 0] + u * (bounds[:, 1] - bounds[:, 0])
