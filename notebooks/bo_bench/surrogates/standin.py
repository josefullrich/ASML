"""CPU stand-in for the foundation models (random forests). Only for testing the pipeline
without GPU weights; its numbers mean nothing about Kumo or Causilo."""
from __future__ import annotations
import numpy as np
from scipy.stats import norm
from .base import Backend, LEVELS

_Z = norm.ppf(LEVELS)


class StandinBackend(Backend):
    def __init__(self, name="standin", seed=0):
        self.name, self.seed = name, seed

    def regress(self, Xc, yc, Xq, tag=None):
        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(n_estimators=12, min_samples_leaf=2, random_state=self.seed, oob_score=len(yc) > 10,
                                   bootstrap=True, max_samples=min(1.0, 500 / max(len(yc), 1)) if len(yc) > 500 else None, n_jobs=1)
        rf.fit(Xc, yc)
        per_tree = np.stack([t.predict(Xq) for t in rf.estimators_], 1)
        m, s = per_tree.mean(1), per_tree.std(1)
        oob = rf.oob_prediction_ if len(yc) > 10 else None
        resid = float(np.nanstd(yc - oob)) if oob is not None and np.isfinite(oob).any() else float(np.std(yc)) * 0.3
        so = np.sqrt(s ** 2 + resid ** 2)
        return m, m[:, None] + s[:, None] * _Z, m[:, None] + so[:, None] * _Z

    def classify(self, Xc, yc, Xq, n_classes, weights=None):
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=12, min_samples_leaf=2, random_state=self.seed, n_jobs=1,
                                    max_samples=min(1.0, 800 / max(len(yc), 1)) if len(yc) > 800 else None)
        rf.fit(Xc, yc, sample_weight=weights)
        P = np.zeros((len(Xq), n_classes))
        P[:, rf.classes_.astype(int)] = rf.predict_proba(Xq)
        return P
