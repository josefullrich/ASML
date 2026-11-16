"""Common prediction container and helpers shared by every backend."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

LEVELS = np.linspace(0.005, 0.995, 199)          # quantile grid every backend reports on
ACQ_IDX = np.arange(2, 199, 5)                   # 40 equally spaced levels used as samples by the spread-aware EI
I_LO, I_HI = 4, 194                              # 2.5% and 97.5%


@dataclass
class Pred:
    """Prediction for one output at N query points.

    mean  [N]       point prediction
    Q     [N, 199]  quantiles of the model's belief about the underlying value (what acquisition uses)
    Qobs  [N, 199]  quantiles of a new measurement (used for scoring real outputs); defaults to Q
    pmf   [N, n+1]  count distribution of a new measurement (binomial outputs, scoring only)
    """
    mean: np.ndarray
    Q: np.ndarray
    Qobs: np.ndarray | None = None
    pmf: np.ndarray | None = None
    extra: dict = field(default_factory=dict)

    @property
    def lo(self):
        return self.Q[:, I_LO]
    @property
    def hi(self):
        return self.Q[:, I_HI]
    @property
    def samples(self):
        return self.Q[:, ACQ_IDX]
    def obs_q(self):
        return self.Q if self.Qobs is None else self.Qobs


def point_pred(mean: np.ndarray) -> Pred:
    mean = np.asarray(mean, float)
    return Pred(mean=mean, Q=np.repeat(mean[:, None], len(LEVELS), axis=1))


def interp_levels(raw: np.ndarray, raw_levels: np.ndarray) -> np.ndarray:
    """Re-grid sorted quantiles [N, R] at raw_levels onto LEVELS."""
    raw = np.sort(np.asarray(raw, float), axis=1)
    idx = np.interp(LEVELS, raw_levels, np.arange(len(raw_levels)))
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, len(raw_levels) - 1)
    w = idx - lo
    return raw[:, lo] * (1 - w) + raw[:, hi] * w


class Backend:
    """A model family. regress() returns (mean, Q, Qobs); classify() returns class probabilities."""
    name = "backend"
    supports_context = True
    def regress(self, Xc, yc, Xq, tag=None):
        raise NotImplementedError
    def classify(self, Xc, yc, Xq, n_classes: int, weights=None):
        raise NotImplementedError
