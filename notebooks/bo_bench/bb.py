"""Beta-binomial maths and the context "quiz".

Beta-binomial: a run first draws its own pass probability from Beta(p*phi, (1-p)*phi) around the
predicted mean p, then every die/exposure passes with that probability. Large phi = plain binomial.
The quiz fits phi on held-back context rows BEFORE the optimisation starts, so no BO steps are spent
learning it."""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import beta as beta_dist, betabinom, binom

PHI_MAX = 1e6
EPS_P = 1e-6


def clip_p(p):
    return np.clip(np.asarray(p, float), EPS_P, 1 - EPS_P)


def binom_pmf(n: int, p) -> np.ndarray:
    return binom.pmf(np.arange(n + 1)[None, :], n, clip_p(p)[:, None])


def betabinom_pmf(n: int, p, phi) -> np.ndarray:
    p = clip_p(p)
    phi = np.broadcast_to(np.asarray(phi, float), p.shape)
    out = betabinom.pmf(np.arange(n + 1)[None, :], n, (p * phi)[:, None], ((1 - p) * phi)[:, None])
    big = phi >= PHI_MAX * 0.999                    # numerically a binomial
    if big.any():
        out[big] = binom_pmf(n, p[big])
    return out


def beta_quantiles(p, phi, levels) -> np.ndarray:
    p = clip_p(p)
    phi = np.broadcast_to(np.asarray(phi, float), p.shape)
    Q = beta_dist.ppf(np.asarray(levels)[None, :], (p * phi)[:, None], ((1 - p) * phi)[:, None])
    big = (phi >= PHI_MAX * 0.999) | ~np.isfinite(Q).all(1)
    Q[big] = p[big, None]
    return Q


def fit_phi(p, k, n) -> float:
    """Maximum-likelihood phi given predicted means p, successes k and trials n."""
    p, k, n = clip_p(p), np.rint(np.asarray(k, float)), np.rint(np.asarray(n, float))
    if len(p) < 3:
        return PHI_MAX
    nll = lambda z: -betabinom.logpmf(k, n, p * np.exp(z), (1 - p) * np.exp(z)).sum()
    res = minimize_scalar(nll, bounds=(np.log(1e-2), np.log(PHI_MAX)), method="bounded")
    return float(np.exp(res.x))


def fit_platt(p, k, n):
    """Logit shift/scale (a, b) maximising the binomial likelihood: p' = sigmoid(a*logit(p)+b)."""
    from scipy.optimize import minimize
    z = np.log(clip_p(p) / (1 - clip_p(p)))
    k, n = np.rint(np.asarray(k, float)), np.rint(np.asarray(n, float))
    def nll(ab):
        q = clip_p(1 / (1 + np.exp(-(ab[0] * z + ab[1]))))
        return -(k * np.log(q) + (n - k) * np.log(1 - q)).sum()
    res = minimize(nll, [1.0, 0.0], method="Nelder-Mead")
    return float(res.x[0]), float(res.x[1])


def design_effect(phi: float, n: float) -> float:
    """How many times more a count varies than a binomial: 1 + (n-1)/(phi+1)."""
    return 1.0 + (float(n) - 1.0) / (float(phi) + 1.0)


def phi_from_variance(p, var_fraction, n) -> np.ndarray:
    """Per-point phi so that a beta-binomial has the given variance of the observed fraction."""
    p = clip_p(p)
    base = p * (1 - p) / n
    rho = (np.asarray(var_fraction, float) / np.maximum(base, 1e-12) - 1.0) / max(n - 1.0, 1.0)
    rho = np.clip(rho, 1e-5, 0.99)
    return 1.0 / rho - 1.0


def pmf_from_fraction_quantiles(Q: np.ndarray, levels: np.ndarray, n: int) -> np.ndarray:
    """Turn quantiles of an observed fraction into a distribution over counts 0..n by cutting
    the CDF halfway between neighbouring counts (as in the earlier binomial report)."""
    cuts = (np.arange(n) + 0.5) / n
    out = np.empty((len(Q), n + 1))
    lv = np.r_[0.0, levels, 1.0]
    for i, q in enumerate(Q):
        qq = np.r_[min(q[0] - 1e-9, -1e-9), q, max(q[-1] + 1e-9, 1 + 1e-9)]
        cdf = np.interp(cuts, qq, lv)
        out[i] = np.diff(np.r_[0.0, cdf, 1.0])
    return np.clip(out, 0, None)


def quantiles_from_pmf(pmf: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Quantiles of the fraction k/n implied by a count distribution (linear within a count)."""
    n = pmf.shape[1] - 1
    cdf = np.cumsum(pmf, 1)
    edges = np.r_[0.0, (np.arange(n) + 0.5) / n, 1.0]
    out = np.empty((len(pmf), len(levels)))
    for i in range(len(pmf)):
        c = np.r_[0.0, cdf[i]]
        c[-1] = 1.0
        out[i] = np.interp(levels, c + np.arange(n + 2) * 1e-12, edges)
    return out


def quiz(classify, datasets, mode="lodo", fraction=0.2, repeats=3, seed=0):
    """Hold back part of the context, predict it from the rest, return what the misses imply.

    classify(X_train, k_train, n_train, X_query) -> p.   datasets: list of dict(X, k, n).
    mode "lodo": leave one context dataset out in turn.  mode "fraction": hold back a random share of rows."""
    rng = np.random.default_rng(seed)
    P, K, N = [], [], []
    if mode == "lodo" and len(datasets) >= 2:
        for i, held in enumerate(datasets):
            rest = [d for j, d in enumerate(datasets) if j != i]
            X = np.concatenate([d["X"] for d in rest]); k = np.concatenate([d["k"] for d in rest]); n = np.concatenate([d["n"] for d in rest])
            P.append(classify(X, k, n, held["X"])); K.append(held["k"]); N.append(held["n"])
    else:
        X = np.concatenate([d["X"] for d in datasets]); k = np.concatenate([d["k"] for d in datasets]); n = np.concatenate([d["n"] for d in datasets])
        for _ in range(repeats):
            held = rng.random(len(X)) < fraction
            if held.sum() < 3 or (~held).sum() < 3:
                continue
            P.append(classify(X[~held], k[~held], n[~held], X[held])); K.append(k[held]); N.append(n[held])
    if not P:
        return {"phi": PHI_MAX, "a": 1.0, "b": 0.0, "n_quiz": 0}
    p, k, n = np.concatenate(P), np.concatenate(K), np.concatenate(N)
    a, b = fit_platt(p, k, n)
    return {"phi": fit_phi(p, k, n), "a": a, "b": b, "n_quiz": int(len(p)), "p": p, "k": k, "n": n}
