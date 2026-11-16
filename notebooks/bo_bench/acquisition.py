"""The two acquisition functions. Both are expected improvement on problem.cost and work with any
Problem; they differ only in how much of each output's distribution they use.

matlab  The MATLAB example's rule: improvement from the means, one width from the cost at the upper
        vs lower 95% bounds divided by 4, candidates filtered by mean feasibility.
spread  Averages the improvement over samples of every cost output's distribution and multiplies by
        the probability that the constraints hold.

A point prediction (no spread) is a valid input to both: zero width / a single spike."""
from __future__ import annotations
import numpy as np
from scipy.stats import norm


def incumbent(problem, obs: dict, pred_train: dict):
    """Best past point. Noisy outputs (yield) are scored by the model's mean, as in the MATLAB example;
    feasibility uses the measured constraint outputs."""
    vals = {o.name: (pred_train[o.name].mean if o.noisy_incumbent else np.asarray(obs[o.name], float)) for o in problem.outputs}
    ok = problem.feasible(obs)
    if not ok.any():
        return None, None, vals
    c = np.where(ok, problem.cost(vals), -np.inf)
    i = int(np.argmax(c))
    return float(c[i]), i, vals


def _mean_feasible(problem, preds):
    if "_feas" in preds:
        return preds["_feas"].mean >= 0.5
    return problem.feasible({o.name: preds[o.name].mean for o in problem.outputs})


def _prob_feasible(problem, preds):
    if "_feas" in preds:
        return preds["_feas"].mean
    p = np.ones(len(next(iter(preds.values())).mean))
    for o in problem.constraint_outputs():
        s = preds[o.name].samples
        p *= ((s >= o.lower) & (s <= o.upper)).mean(1)
    return p


def acq_matlab(problem, preds, target, rng=None):
    names = [o.name for o in problem.outputs]
    mean = {n: preds[n].mean for n in names}
    cost_names = {o.name for o in problem.cost_outputs()}
    up = {n: (preds[n].hi if n in cost_names else preds[n].mean) for n in names}
    lo = {n: (preds[n].lo if n in cost_names else preds[n].mean) for n in names}
    improvement = problem.cost(mean) - target
    sigma = np.maximum(np.abs(problem.cost(up) - problem.cost(lo)) / 4.0, 1e-4)
    g = improvement / sigma
    ei = improvement * norm.cdf(g) + sigma * norm.pdf(g)
    ei[~_mean_feasible(problem, preds)] = -np.inf
    return ei


def acq_spread(problem, preds, target, rng=None):
    rng = rng or np.random.default_rng(0)
    names = [o.name for o in problem.outputs]
    cost_names = [o.name for o in problem.cost_outputs()]
    S = preds[cost_names[0]].samples.shape[1]
    vals = {n: np.repeat(preds[n].mean[:, None], S, 1) for n in names}
    for j, n in enumerate(cost_names):                     # outputs treated as independent: shuffle sample order
        s = preds[n].samples
        vals[n] = s if j == 0 else s[:, rng.permutation(S)]
    ei = np.maximum(problem.cost(vals) - target, 0.0).mean(1)
    return ei * _prob_feasible(problem, preds)


def min_violation(problem, preds):
    """Before any feasible point exists: go where the constraints are most likely to hold."""
    if "_feas" in preds:
        return preds["_feas"].mean
    v = np.zeros(len(next(iter(preds.values())).mean))
    for o in problem.constraint_outputs():
        m = preds[o.name].mean
        v += np.maximum(o.lower - m, 0) + np.maximum(m - o.upper, 0)
    return -v


ACQ_FUNCS = {"matlab": acq_matlab, "spread": acq_spread}
