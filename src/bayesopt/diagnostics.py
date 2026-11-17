"""Simulation-only yield prediction diagnostics."""

import numpy as np
from scipy.special import xlogy

from .types import FloatArray, R2Method


def _yield_scores(
    actual: FloatArray, predicted: FloatArray, r2_method: R2Method = 1
) -> tuple[float, float]:
    """Return MATLAB's selected R² in percent and raw probability RMSE.

    Methods: 0 = SSE, 1 = binomial deviance, 2 = McFadden log-likelihood.
    R² uses probabilities clipped to [1e-4, 1-1e-4], without exposure weights,
    matching the intended MATLAB formulas. SciPy xlogy means x*log(y);
    ASML's MATLAB xlogy has a different meaning and must not be used as-is.
    Undefined SSE/deviance scores for constant yields are NaN.
    Non-finite inputs raise rather than silently dropping observations.
    """
    if r2_method not in (0, 1, 2):
        raise ValueError("r2_method must be 0 (SSE), 1 (deviance), or 2 (log-likelihood)")
    if actual.shape != predicted.shape:
        raise ValueError("actual and predicted yield arrays must have matching shapes")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("yield diagnostics require finite probabilities")
    if actual.size == 0:
        raise ValueError("yield diagnostics require at least one probability")

    squared_errors = (predicted - actual) ** 2
    # Keep RMSE unclipped so endpoint probability errors remain observable.
    rmse = float(np.sqrt(np.mean(squared_errors)))
    y = np.clip(actual, 1e-4, 1.0 - 1e-4)
    yhat = np.clip(predicted, 1e-4, 1.0 - 1e-4)
    ynull = float(np.mean(y))
    # A constant-probability null model has zero SSE/deviance: dividing by
    # its loss cannot define those scores, even for a perfect prediction.
    if r2_method in (0, 1) and (y.size < 2 or np.all(y == y.flat[0])):
        return float("nan"), rmse
    if r2_method == 0:
        model_loss = float(np.sum((yhat - y) ** 2))
        null_loss = float(np.sum((y - ynull) ** 2))
    elif r2_method == 1:
        # Binomial deviance compares the model with a saturated model whose
        # probability equals each actual yield. Smaller deviance is better.
        model_loss = float(2 * np.sum(xlogy(y, y / yhat) + xlogy(1 - y, (1 - y) / (1 - yhat))))
        null_loss = float(2 * np.sum(xlogy(y, y / ynull) + xlogy(1 - y, (1 - y) / (1 - ynull))))
    else:
        # McFadden compares log-likelihoods directly; unlike SSE/deviance,
        # matching fractional probabilities need not produce a 100% score.
        model_loss = float(np.sum(xlogy(y, yhat) + xlogy(1 - y, 1 - yhat)))
        null_loss = float(np.sum(xlogy(y, ynull) + xlogy(1 - y, 1 - ynull)))
    if null_loss == 0.0:
        return float("nan"), rmse
    r_squared = 100.0 * (1.0 - model_loss / null_loss)
    return r_squared, rmse
