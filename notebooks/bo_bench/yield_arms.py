"""Models for binomial outputs (die yield): one baseline regressor and the classifier variants.

Every arm returns the same thing: a mean yield, quantiles of the underlying yield (what the
acquisition uses) and, on request, a distribution over counts (what the prediction metrics score)."""
from __future__ import annotations
import numpy as np
from scipy.special import expit
from . import bb
from .surrogates.base import Pred, LEVELS

CLASSIFIER_ARMS = ("clf_raw", "clf_bb", "clf_ess_bb", "clf_phix", "clf_binned")


def logit_target(frac, n, how):
    k = np.asarray(frac, float) * np.asarray(n, float)
    if how == "clip":                                   # MATLAB example: clip the fraction at 1e-4
        f = np.clip(frac, 1e-4, 1 - 1e-4)
        return np.log(f / (1 - f))
    return np.log((k + 0.5) / (np.asarray(n, float) - k + 0.5))


def deficit_bins(n_ref: int, max_classes: int = 10):
    """Count bins by number of failures, fine near a perfect run: 0, 1, 2, 3-4, 5-8, 9-16, ..."""
    edges, lo, width = [], 0, 1
    while lo <= n_ref and len(edges) < max_classes - 1:
        hi = min(lo + width - 1, n_ref)
        edges.append((lo, hi))
        lo = hi + 1
        if len(edges) >= 3:
            width *= 2
    if lo <= n_ref:
        edges.append((lo, n_ref))
    return edges


class YieldModel:
    def __init__(self, cfg, backend, arm: str, has_context: bool):
        self.cfg, self.backend, self.arm = cfg, backend, arm
        self.phi, self.platt = bb.PHI_MAX, (1.0, 0.0)
        self.phi_source = "none"
        self.hist = {"p": [], "k": [], "n": []}          # one-step-ahead history for the concurrent fit
        self.has_context = has_context
        self.quiz_info = {}

    # ---- classifier call on trial data --------------------------------
    def _classify(self, X, k, n, Xq, scale=1.0):
        k, n = np.asarray(k, float), np.asarray(n, float)
        if k.sum() <= 0 or (n - k).sum() <= 0:             # only one class observed so far
            return np.full(len(Xq), 1 - 1e-4 if k.sum() > 0 else 1e-4)
        Xc = np.concatenate([X, X])
        yc = np.r_[np.ones(len(X), int), np.zeros(len(X), int)]
        w = np.r_[k, n - k] / scale
        return bb.clip_p(self.backend.classify(Xc, yc, Xq, 2, weights=w)[:, 1])

    def needs_phi(self):
        return self.arm in ("clf_bb", "clf_ess_bb")

    # ---- context stage --------------------------------------------------
    def prepare(self, context_sets, n_ref):
        """Fit phi on held-back context (the quiz). Without context phi is learned during the run."""
        if not (self.needs_phi() and context_sets):
            return
        data = [{"X": c["X"], "k": c["frac"] * c["n"], "n": c["n"]} for c in context_sets]
        q = bb.quiz(lambda X, k, n, Xq: self._classify(X, k, n, Xq), data, self.cfg.quiz_mode,
                    self.cfg.quiz_fraction, self.cfg.quiz_repeats)
        self.phi, self.phi_source = q["phi"], f"quiz:{self.cfg.quiz_mode}"
        self.quiz_info = {k: v for k, v in q.items() if k in ("phi", "a", "b", "n_quiz")}

    def observe(self, p_pred, frac, n):
        """Record a one-step-ahead prediction and its outcome; refit phi when it is learned during the run."""
        self.hist["p"].append(float(p_pred)); self.hist["k"].append(float(frac) * n); self.hist["n"].append(float(n))
        if self.needs_phi() and not self.phi_source.startswith("quiz") and len(self.hist["p"]) >= self.cfg.phi_min_fit:
            self.phi = bb.fit_phi(self.hist["p"], self.hist["k"], self.hist["n"])
            self.phi_source = "during_run"

    # ---- prediction -----------------------------------------------------
    def predict(self, X, frac, n, Xq, n_q: int) -> Pred:
        arm, cfg = self.arm, self.cfg
        X, frac, n, Xq = np.asarray(X, float), np.asarray(frac, float), np.asarray(n, float), np.asarray(Xq, float)
        extra = {"arm": arm, "n_q": int(n_q)}
        if arm == "reg_logit" or arm == "clf_phix":
            m, Q, Qobs = self.backend.regress(X, logit_target(frac, n, cfg.logit_target), Xq, tag="yield_logit")
            reg = {"p": expit(m), "Q": expit(Q), "Qobs": expit(Qobs)}
        if arm == "reg_logit":
            extra["kind"] = "fraction_quantiles"
            return Pred(mean=reg["p"], Q=reg["Q"], Qobs=reg["Qobs"], extra=extra)
        if arm == "clf_binned":
            return self._predict_binned(X, frac, n, Xq, n_q, extra)
        scale = 1.0
        if arm == "clf_ess_bb":
            scale = bb.design_effect(self.phi, np.median(n))    # each run is worth n / design-effect trials
            extra["design_effect"] = scale
        p = self._classify(X, frac * n, n, Xq, scale)
        extra["p_raw"] = p
        if arm == "clf_raw":
            extra.update(kind="binomial")
            return Pred(mean=p, Q=np.repeat(p[:, None], len(LEVELS), 1), extra=extra)
        if arm == "clf_phix":
            var = np.var(reg["Qobs"], axis=1)                    # spread of the regressor's predicted fraction
            phi = bb.phi_from_variance(p, var, n_q)
        else:
            phi = np.full(len(p), self.phi)
        extra.update(kind="betabinomial", phi=phi)
        return Pred(mean=p, Q=bb.beta_quantiles(p, phi, LEVELS), extra=extra)

    def _predict_binned(self, X, frac, n, Xq, n_q, extra):
        bins = deficit_bins(int(n_q))
        deficit = np.rint((1 - frac) * n * (n_q / n)).astype(int)        # failures, rescaled to the usual trial count
        cls = np.array([next(i for i, (lo, hi) in enumerate(bins) if lo <= min(d, n_q) <= hi) for d in deficit])
        present = np.unique(cls)
        if len(present) == 1:
            P = np.zeros((len(Xq), len(bins))); P[:, present[0]] = 1.0
        else:
            remap = {c: i for i, c in enumerate(present)}                # classifier needs labels 0..K-1
            Pp = self.backend.classify(X, np.array([remap[c] for c in cls]), Xq, len(present))
            P = np.zeros((len(Xq), len(bins))); P[:, present] = Pp
        pmf = np.zeros((len(Xq), n_q + 1))
        for j, (lo, hi) in enumerate(bins):                              # spread a bin's mass evenly over its counts
            ks = n_q - np.arange(lo, hi + 1)
            pmf[:, ks] += P[:, [j]] / len(ks)
        pmf /= pmf.sum(1, keepdims=True)
        mean = (pmf * np.arange(n_q + 1)).sum(1) / n_q
        extra.update(kind="pmf", pmf=pmf)
        return Pred(mean=mean, Q=bb.quantiles_from_pmf(pmf, LEVELS), extra=extra)

    # ---- count distribution at chosen rows (scoring only) -----------------
    @staticmethod
    def pmf_at(pred: Pred, idx, n_q: int) -> np.ndarray:
        idx = np.atleast_1d(idx)
        kind = pred.extra["kind"]
        if kind == "pmf":
            return pred.extra["pmf"][idx]
        if kind == "binomial":
            return bb.binom_pmf(n_q, pred.mean[idx])
        if kind == "betabinomial":
            return bb.betabinom_pmf(n_q, pred.mean[idx], pred.extra["phi"][idx])
        return bb.pmf_from_fraction_quantiles(pred.obs_q()[idx], LEVELS, n_q)
