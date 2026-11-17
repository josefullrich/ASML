"""Gaussian-process construction, fitting, and prediction."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

from ..types import FloatArray


@dataclass
class FittedGP:
    """Standardized-input GP, with an optional linear dose-target basis."""

    scaler: StandardScaler
    model: GaussianProcessRegressor
    dose_slope: float = 0.0


def _new_gpr(
    n_features: int,
    noise_sigma: float,
    response_scale: float,
) -> GaussianProcessRegressor:
    """Build an ARD Matérn-5/2 GP, matching MATLAB's ardmatern52 kernel."""
    # normalize_y rescales the response, so the physical noise standard
    # deviation must also be rescaled before setting the kernel variance.
    normalized_noise = noise_sigma / (
        response_scale if response_scale > np.finfo(float).eps else 1.0
    )
    kernel = (
        ConstantKernel(1.0, constant_value_bounds="fixed")
        * Matern(length_scale=np.ones(n_features), length_scale_bounds=(1e-2, 1e2), nu=2.5)
        + WhiteKernel(
            noise_level=normalized_noise**2,
            noise_level_bounds="fixed",
        )
    )
    return GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-10,
        normalize_y=True,
        n_restarts_optimizer=0,
        random_state=0,
    )


def _fit_gpr(
    x: FloatArray,
    y: FloatArray,
    noise_sigma: float,
    *,
    dose_trend: bool = False,
) -> FittedGP:
    # Subtract a fixed linear dose-target basis when requested. This mirrors
    # the MATLAB EUV GP basis that includes the final input variable.
    adjusted_y = np.asarray(y, dtype=float)
    dose_slope = 0.0
    if dose_trend:
        # MATLAB fixes the EUV GP basis coefficient Beta to 0.1 and uses the
        # dose target as its basis function.
        dose_slope = 0.1
        adjusted_y = adjusted_y - dose_slope * x[:, -1]

    scaler = StandardScaler().fit(x)
    model = _new_gpr(x.shape[1], noise_sigma, float(np.std(adjusted_y)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(scaler.transform(x), adjusted_y)
    return FittedGP(scaler, model, dose_slope)


def _predict(
    fitted: FittedGP,
    x: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    mean, standard_deviation = fitted.model.predict(
        fitted.scaler.transform(x),
        return_std=True,
    )
    mean = np.asarray(mean) + fitted.dose_slope * x[:, -1]
    return mean, np.asarray(standard_deviation)
