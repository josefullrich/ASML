"""Die-yield surrogates for the MATLAB Bayesian-optimization example.

Mode 0 uses a binomial polynomial GLM. Mode 2 uses scikit-learn's binary
GaussianProcessClassifier. Its fit API has no sample_weight argument, so
success/failure counts are represented exactly by repeated exposure rows.
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import GLMResultsWrapper


from ..types import DYModelOption, FloatArray
from .gp import FittedGP, _fit_gpr, _predict


def exposure_counts(yield_fraction: FloatArray, exposures: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Validate integer binomial counts without silently rounding observations."""
    if yield_fraction.shape != exposures.shape or yield_fraction.ndim != 1:
        raise ValueError("yield and exposures must be matching one-dimensional arrays")
    if not np.isfinite(yield_fraction).all() or not np.isfinite(exposures).all():
        raise ValueError("yield and exposures must be finite")
    if np.any((yield_fraction < 0) | (yield_fraction > 1)):
        raise ValueError("yield must be between zero and one")
    if np.any(exposures <= 0) or not np.allclose(exposures, np.rint(exposures), rtol=0, atol=1e-8):
        raise ValueError("exposures must be positive integers")
    raw_successes = yield_fraction * exposures
    successes = np.rint(raw_successes)
    if not np.allclose(raw_successes, successes, rtol=0, atol=1e-8):
        raise ValueError("yield times exposures must be an integer success count")
    return successes, np.rint(exposures) - successes


@dataclass
class YieldClassifier:
    scaler: StandardScaler
    model: GaussianProcessClassifier

    def predict(self, x: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        # Bound cross-covariance memory for the default 50,000 candidates.
        probability = np.empty(len(x))
        lower = np.empty(len(x))
        upper = np.empty(len(x))
        z = norm.ppf(0.975)
        for start in range(0, len(x), 256):
            stop = min(start + 256, len(x))
            query = self.scaler.transform(x[start:stop])
            probability[start:stop] = self.model.predict_proba(query)[:, 1]
            mean, variance = self.model.latent_mean_and_variance(query)
            std = np.sqrt(np.maximum(variance, 0.0))
            # Predict_proba integrates latent uncertainty. These intervals
            # instead transform the latent 95% credible bounds.
            lower[start:stop] = expit(mean - z * std)
            upper[start:stop] = expit(mean + z * std)
        return probability, lower, upper


def fit_yield_classifier(
    x: FloatArray,
    yield_fraction: FloatArray,
    exposures: FloatArray,
) -> YieldClassifier:
    successes, failures = exposure_counts(yield_fraction, exposures)
    if not np.any(successes) or not np.any(failures):
        raise ValueError("GaussianProcessClassifier requires observed successes and failures")
    if len(x) != len(exposures):
        raise ValueError("one input row is required per binomial observation")
    # Fit standardization on observations, not exposure-weighted repetitions,
    # to retain the MATLAB predictor standardization convention.
    scaler = StandardScaler().fit(x)
    standardized = scaler.transform(x)
    total_exposures = int(successes.sum() + failures.sum())
    if total_exposures > 1_000:
        matrix_gib = total_exposures**2 * 8 / 1024**3
        warnings.warn(
            f"DY classifier expands {len(x)} observations into {total_exposures} exposures; "
            f"each dense GP matrix uses approximately {matrix_gib:.2f} GiB. "
            "Fitting needs multiple matrices and cubic-time factorizations.",
            RuntimeWarning,
            stacklevel=2,
        )
    expanded_x = np.concatenate(
        (
            np.repeat(standardized, successes.astype(np.int64), axis=0),
            np.repeat(standardized, failures.astype(np.int64), axis=0),
        )
    )
    # Repeated Bernoulli labels produce p**successes * (1-p)**failures,
    # preserving the binomial likelihood up to a constant count factor.
    # GPC's sample_weight support is for score(), not fit(): two labels
    # without training weights would incorrectly encode 50% yield.
    labels = np.concatenate(
        (np.ones(int(successes.sum()), dtype=int), np.zeros(int(failures.sum()), dtype=int))
    )
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=np.ones(x.shape[1]), length_scale_bounds=(1e-2, 1e2), nu=2.5
    )
    # ARD learns one length scale per process input; the constant kernel
    # learns latent-function variance, not a mean/intercept term.
    model = GaussianProcessClassifier(kernel=kernel, random_state=0, max_iter_predict=100)
    model.fit(expanded_x, labels)
    return YieldClassifier(scaler, model)


@dataclass
class YieldGLM:
    polynomial: PolynomialFeatures
    columns: NDArray[np.bool_]
    model: GLMResultsWrapper

    def predict(self, x: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        design = self.polynomial.transform(x)[:, self.columns]
        prediction = self.model.get_prediction(design).summary_frame(alpha=0.05)
        return (
            prediction["mean"].to_numpy(),
            prediction["mean_ci_lower"].to_numpy(),
            prediction["mean_ci_upper"].to_numpy(),
        )


def fit_yield_glm(
    x: FloatArray,
    yield_fraction: FloatArray,
    exposures: FloatArray,
) -> YieldGLM:
    successes, failures = exposure_counts(yield_fraction, exposures)
    # MATLAB poly222222221: linear terms, pairwise interactions, and squares
    # except for the last (dose) variable; Intercept=false.
    polynomial = PolynomialFeatures(degree=2, include_bias=False).fit(x)
    columns = polynomial.powers_[:, -1] < 2
    design = polynomial.transform(x)[:, columns]
    model = sm.GLM(
        np.column_stack((successes, failures)), design, family=sm.families.Binomial()
    ).fit(maxiter=1_000)
    if not model.converged:
        raise RuntimeError("Binomial DY GLM did not converge within 1000 iterations")
    return YieldGLM(polynomial, columns, model)


def _fit_yield_model(
    x: FloatArray, dy: FloatArray, exposures: FloatArray, option: DYModelOption
) -> FittedGP | YieldGLM | YieldClassifier:
    # GLM and GPC use the observed success/failure counts. Logit regression
    # treats each measured fraction as one continuous response instead.
    if option == 0:
        return fit_yield_glm(x, dy, exposures)
    if option == 2:
        return fit_yield_classifier(x, dy, exposures)
    # The logit maps probabilities to the real line; clipping avoids infinite
    # regression targets at exactly zero or one.
    tolerance = 1e-4
    safe_dy = np.clip(dy, tolerance, 1.0 - tolerance)
    return _fit_gpr(x, np.log(safe_dy / (1.0 - safe_dy)), tolerance)


def _predict_yield(
    fitted: FittedGP | YieldGLM | YieldClassifier, x: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    # Keep a common (probability, lower, upper) interface so acquisition and
    # dose solving use the selected model rather than an implicit GPR.
    if isinstance(fitted, FittedGP):
        mean, std = _predict(fitted, x)
        z = norm.ppf(0.975)
        # Transforming the latent mean gives the logit-normal median, not
        # the integrated probability returned by GPC.predict_proba.
        return expit(mean), expit(mean - z * std), expit(mean + z * std)
    return fitted.predict(x)
