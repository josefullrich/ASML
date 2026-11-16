"""One surrogate "arm": a backend for the real-valued outputs plus a yield model (or, for problems
without binomial outputs, an optional feasibility classifier)."""
from __future__ import annotations
import numpy as np
from .surrogates.base import Pred
from .yield_arms import YieldModel


class ArmSurrogate:
    def __init__(self, cfg, problem, backend, arm: str, has_context: bool):
        self.cfg, self.problem, self.backend, self.arm = cfg, problem, backend, arm
        self.binomial = [o.name for o in problem.outputs if o.kind == "binomial"]
        self.yield_models = {name: YieldModel(cfg, backend, arm, has_context) for name in self.binomial}

    def prepare(self, context_sets):
        for name, ym in self.yield_models.items():
            sets = [{"X": c["X"], "frac": c["obs"][name], "n": c["trials"][name]} for c in context_sets]
            n_ref = int(np.median(np.concatenate([s["n"] for s in sets]))) if sets else 0
            ym.prepare(sets, n_ref)

    def predict(self, X, obs, trials, Xq, n_q: dict) -> dict:
        """X [n,d], obs name->[n], trials name->[n], Xq [m,d]. Returns name -> Pred (+ '_feas' if used)."""
        out = {}
        for o in self.problem.outputs:
            if o.kind == "binomial":
                out[o.name] = self.yield_models[o.name].predict(X, obs[o.name], trials[o.name], Xq, n_q[o.name])
            else:
                m, Q, Qobs = self.backend.regress(X, obs[o.name], Xq, tag=o.name)
                out[o.name] = Pred(mean=m, Q=Q, Qobs=Qobs)
        if self.arm == "feas_clf":                         # one classifier for "all constraints met"
            ok = self.problem.feasible(obs).astype(int)
            if ok.min() == ok.max():
                p = np.full(len(Xq), float(ok[0]))
            else:
                p = self.backend.classify(X, ok, Xq, 2)[:, 1]
            out["_feas"] = Pred(mean=p, Q=np.repeat(p[:, None], 199, 1))
        return out
