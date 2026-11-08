"""Bayesian optimization demonstration for the source model.

The optimizer selects nine process settings using target-shape constraints
and a selectable EUV/die-yield objective. Gaussian processes predict physical
responses; yield uses a binomial GLM, logit GP regression, or an
exposure-expanded GaussianProcessClassifier. Predicted uncertainty guides
expected improvement for each subsequent measurement.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Literal
import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar
from scipy.special import expit, xlogy
from scipy.stats import norm, qmc
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

from srcmdl import SourceState, srcmdl
from dy_models import YieldClassifier, YieldGLM, exposure_counts, fit_yield_classifier, fit_yield_glm


FloatArray = NDArray[np.float64]
CostOption = Literal[0, 1, 2]
DYModelOption = Literal[0, 1, 2]
R2Method = Literal[0, 1, 2]


@dataclass
class OptimizationResult:
    """Training measurements and summary values from optimization."""

    x_train: FloatArray
    selected_by_optimizer: NDArray[np.bool_]
    euv_train: FloatArray
    rx_train: FloatArray
    ry_train: FloatArray
    target_size_train: FloatArray
    die_yield_train: FloatArray
    exposures_train: FloatArray
    open_loop_euv_train: FloatArray
    actual_die_yield_train: FloatArray
    predicted_die_yield: FloatArray
    best_dt: float
    best_dy: float
    best_x: FloatArray
    r_squared: float
    dt_at_minimum_yield: float
    predicted_yield_at_dt: float
    yield_residual: float
    cost_option: CostOption
    yield_rmse: float
    dy_model: DYModelOption
    r2_method: R2Method


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


def _latin_hypercube(n: int, dimensions: int, seed: int) -> FloatArray:
    # SciPy's LHS has no correlation criterion; strength-1 randomized LHS is
    # used to preserve one sample per stratum in every dimension.
    sampler = qmc.LatinHypercube(d=dimensions, seed=seed)
    return sampler.random(n)


def _cost(euv: FloatArray, dy: FloatArray, option: CostOption, lbdy: float) -> FloatArray:
    """Score EUV and yield; the optimizer maximizes this objective.

    Option 0 is a linear yield penalty, option 1 is multiplicative, and
    option 2 uses distance in logit-yield space.
    """
    tolerance = 1e-4
    # Yield is a soft objective penalty, not a hard feasibility constraint.
    # Both too-low and too-high yield are penalized relative to lbdy.
    clipped_dy = np.clip(dy, tolerance, 1.0 - tolerance)
    clipped_lbdy = float(np.clip(lbdy, tolerance, 1.0 - tolerance))
    if option == 0:
        return euv - 10.0 * np.abs(dy - lbdy)
    if option == 1:
        return euv * (1.0 - np.abs(dy - lbdy) ** 2)
    logit_dy = np.log(clipped_dy / (1.0 - clipped_dy))
    logit_lbdy = np.log(clipped_lbdy / (1.0 - clipped_lbdy))
    return euv - np.abs(logit_dy - logit_lbdy) / 10.0


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


def run_optimization(
    n: int = 300,
    *,
    warmup: int = 90,
    n_test: int = 50_000,
    seed: int = 0,
    source_seed: int = 0,
    cost_option: CostOption = 2,
    dy_model: DYModelOption = 2,
    r2_method: R2Method = 1,
) -> OptimizationResult:
    """Run sequential Bayesian optimization using scikit-learn Gaussian processes.

    cost_option: 0 = linear, 1 = multiplicative, 2 = logit yield penalty.
    dy_model: 0 = binomial polynomial GLM, 1 = logit GP regression,
    2 = GaussianProcessClassifier with exact per-exposure row replication.
    Mode 2 preserves success/failure counts but requires dense matrices over
    all exposures, not just observations; full runs can be very expensive.
    source_seed selects the simulated source independently of the design seed.
    r2_method: 0 = SSE, 1 = deviance (MATLAB default), 2 = McFadden
    log-likelihood. This diagnostic choice does not affect optimization.
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    if warmup < 1:
        raise ValueError("warmup must be at least 1")
    if n_test < 1:
        raise ValueError("n_test must be at least 1")
    if cost_option not in (0, 1, 2):
        raise ValueError("cost_option must be 0 (linear), 1 (multiplicative), or 2 (logit)")
    if dy_model not in (0, 1, 2):
        raise ValueError("dy_model must be 0 (binomial GLM), 1 (logit GP), or 2 (classifier GP)")
    if r2_method not in (0, 1, 2):
        raise ValueError("r2_method must be 0 (SSE), 1 (deviance), or 2 (log-likelihood)")

    # Number of process settings and physical/quality constraints. Bounds
    # apply to target dose, target size, target angles, and die yield.
    num_variables = 9
    lbr, ubr = 50.0, 60.0
    lbt, ubt = 0.1, 0.9
    lbrx, ubrx = -23.5, 23.5
    lbry, ubry = 11.5, 23.5
    lbdy = 0.995
    minpnts = num_variables * 3
    tolerance = 1e-4

    # Bayesian-optimization settings: learn after at least three observations
    # per variable, use an exploration factor of zero, gradually forget up to
    # warmup initial observations, and evaluate 50,000 candidates by default.

    # Latin hypercube sampling spreads the initial experiments across the
    # normalized process-input space. Dose starts in a lower range to capture
    # yield changes.
    x_train = _latin_hypercube(n, num_variables, seed)
    x_train[:, -1] = lbr + (ubr - lbr) * x_train[:, -1]

    euv_train = np.empty(n)
    rx_train = np.empty(n)
    ry_train = np.empty(n)
    ts_train = np.empty(n)
    dy_train = np.empty(n)
    exp_train = np.empty(n)
    euv_ol_train = np.empty(n)
    dy_actual_train = np.empty(n)

    # Cost choices mirror the MATLAB example: 0 is linear, 1 is
    # multiplicative, and 2 penalizes distance in logit-yield space.
    # Initialize the best feasible measured point and fit diagnostic so the
    # summary remains defined if no feasible point/model is found.
    best_dt = float("nan")
    best_dy = float("nan")
    best_x = np.full(num_variables, np.nan)
    r_squared = float("nan")
    yield_rmse = float("nan")
    dy_gp: FittedGP | YieldGLM | YieldClassifier | None = None
    state: SourceState | int = source_seed
    selected_by_optimizer = np.zeros(n, dtype=bool)

    for i in range(n):
        print(f"Measurement {i + 1} of {n}")
        # Bayesian optimization starts once enough observations exist for GP
        # fitting and at least one observation has a meaningful yield.
        start = min(warmup, max(i - minpnts, 0)) if i > minpnts else 0
        # Initially retain 27 observations by moving the left edge. Once
        # warmup observations have been forgotten, the retained set grows;
        # warmup is not a cap on training size or classifier exposure rows.
        training_slice = slice(start, i)
        ready = i >= minpnts and np.any(dy_train[:i] > 0.5)
        if ready and dy_model == 2:
            successes, failures = exposure_counts(
                dy_train[training_slice], exp_train[training_slice]
            )
            if not np.any(successes) or not np.any(failures):
                # Stock binary GPC cannot fit a single class. Keep sampling
                # the initial design until the retained data contain both.
                warnings.warn(
                    "DY classifier needs both successes and failures; "
                    "taking the next initial-design measurement.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                ready = False
        if ready:
            # Slowly forget the initial warmup once the experiment is longer
            # than the minimum training set.
            x_observed = x_train[training_slice]
            dy_observed = dy_train[training_slice]

            # Select the same three DY model choices as the MATLAB example.
            dy_gp = _fit_yield_model(
                x_observed, dy_observed, exp_train[training_slice], dy_model
            )

            # Fit one GP per physical response. EUV retains its dose basis.
            euv_gp = _fit_gpr(
                x_observed,
                euv_train[training_slice],
                1e-2,
                dose_trend=True,
            )
            rx_gp = _fit_gpr(x_observed, rx_train[training_slice], 0.5 / 3.0)
            ry_gp = _fit_gpr(x_observed, ry_train[training_slice], 0.5 / 3.0)
            ts_gp = _fit_gpr(x_observed, ts_train[training_slice], 5.0 / 3.0)

            dy_pred_train = _predict_yield(dy_gp, x_observed)[0]
            # Compare predicted and actual yield as a simulation-only
            # diagnostic. In a real experiment actual yield is not available.
            measured_actual_dy = dy_actual_train[training_slice]
            r_squared, yield_rmse = _yield_scores(measured_actual_dy, dy_pred_train, r2_method)

            # Only measured points satisfying all target-shape constraints
            # can become the incumbent feasible target.
            shape_feasible = (
                (ts_train[training_slice] >= lbt)
                & (ts_train[training_slice] <= ubt)
                & (ry_train[training_slice] >= lbry)
                & (ry_train[training_slice] <= ubry)
                & (rx_train[training_slice] >= lbrx)
                & (rx_train[training_slice] <= ubrx)
            )
            feasible_indices = np.flatnonzero(shape_feasible)
            # "Best feasible" refers to shape only; the yield target enters
            # _cost rather than this feasibility mask.

            if feasible_indices.size:
                # Previously measured points inside the target-shape
                # constraints define the incumbent objective value.
                observed_euv = euv_train[training_slice][feasible_indices]
                observed_dy = dy_pred_train[feasible_indices]
                objective = _cost(observed_euv, observed_dy, cost_option, lbdy)
                best_index = feasible_indices[int(np.argmax(objective))]
                target = float(np.max(objective))
                best_dt = float(x_observed[best_index, -1])
                best_dy = float(dy_pred_train[best_index])
                best_x = x_observed[best_index].copy()

            # Generate a new Latin hypercube of candidate settings, sampling
            # dose around the best dose-yield product when possible.
            candidates = _latin_hypercube(n_test, num_variables, seed + i + 1)
            if np.isfinite(best_dt) and np.isfinite(best_dy) and 0.98 * best_dt * best_dy > lbr:
                # Sample around +/-3% of the best dose-target/yield product.
                lower_dt = 0.97 * best_dt * best_dy
                upper_dt = 1.03 * best_dt * best_dy
            else:
                lower_dt, upper_dt = lbr, ubr
            candidates[:, -1] = lower_dt + (upper_dt - lower_dt) * candidates[:, -1]

            euv_mean, euv_std = _predict(euv_gp, candidates)
            rx_mean, _ = _predict(rx_gp, candidates)
            ry_mean, _ = _predict(ry_gp, candidates)
            ts_mean, _ = _predict(ts_gp, candidates)
            dy_mean, dy_lower, dy_upper = _predict_yield(dy_gp, candidates)

            # Use two-standard-deviation bounds to represent the predicted
            # response uncertainty when calculating expected improvement.
            # Reject candidates whose predicted shape violates the target
            # size and angle constraints.
            candidate_feasible = (
                (ts_mean >= lbt)
                & (ts_mean <= ubt)
                & (ry_mean >= lbry)
                & (ry_mean <= ubry)
                & (rx_mean >= lbrx)
                & (rx_mean <= ubrx)
            )
            candidate_indices = np.flatnonzero(candidate_feasible)
            if candidate_indices.size and feasible_indices.size:
                # Improvement compares each candidate with the current
                # feasible target. The confidence bounds estimate uncertainty
                # in the objective; epsilon=0 gives standard expected
                # improvement.
                improvement = _cost(euv_mean, dy_mean, cost_option, lbdy) - target
                euv_lower = euv_mean - 2.0 * euv_std
                euv_upper = euv_mean + 2.0 * euv_std
                sigma = np.maximum(
                    np.abs(
                        _cost(euv_upper, dy_upper, cost_option, lbdy)
                        - _cost(euv_lower, dy_lower, cost_option, lbdy)
                    )
                    # This objective uncertainty is a bounds-based approximation,
                    # not exact propagation of the joint response distribution.
                    / 4.0,
                    tolerance,
                )
                gamma = improvement / sigma
                probability_of_improvement = norm.cdf(gamma)
                probability_density = norm.pdf(gamma)
                exploitation = improvement * probability_of_improvement
                exploration = sigma * probability_density
                expected_improvement = exploitation + exploration
                expected_improvement[~candidate_feasible] = -np.inf
                selected_index = int(np.argmax(expected_improvement))
                if np.isfinite(expected_improvement[selected_index]):
                    x_train[i] = candidates[selected_index]
                    selected_by_optimizer[i] = True

        x, y, z, e, d, p, k, b, dt = x_train[i]
        # Run the selected source-model measurement. As in the MATLAB example,
        # clear accumulated drift after each measurement.
        (
            state,
            euv,
            rx,
            ry,
            ts,
            dy,
            exposures,
            _de,
            euv_ol,
            dy_actual,
        ) = srcmdl(x, y, z, e, d, p, k, b, dt, state)
        state.drift.fill(0.0)  # Disable drift, as in the MATLAB optimizer.

        euv_train[i] = np.mean(euv)
        rx_train[i] = rx
        ry_train[i] = ry
        ts_train[i] = ts
        dy_train[i] = dy
        exp_train[i] = exposures
        euv_ol_train[i] = euv_ol
        dy_actual_train[i] = dy_actual
        # Actual yield and open-loop EUV are recorded for plots only; these
        # simulation-only values are not inputs to the response models.
        print(
            f"  Best DT: {best_dt:.4f} mJ | Best DY: {best_dy:.6f} "
            f"({100.0 * best_dy:.3f}%) | Training-window R^2: {r_squared:.3f}% "
            f"| DY RMSE: {100.0 * yield_rmse:.6f} percentage points"
        )

    # Predict yield for the complete experiment using the most recently
    # fitted model, as in the MATLAB final-prediction section.
    if dy_gp is not None:
        predicted_dy = _predict_yield(dy_gp, x_train)[0]
        # The final diagnostic must use the same predictions and observations
        # as the final actual-yield plot, not the previous training window.
        r_squared, yield_rmse = _yield_scores(dy_actual_train, predicted_dy, r2_method)
    else:
        predicted_dy = np.full(n, np.nan)

    # Solve for the dose target whose predicted yield meets the lower bound.
    # Only the final dose-target input varies; the other settings remain at
    # the best feasible measured point.
    if np.all(np.isfinite(best_x)) and dy_gp is not None:
        def yield_at_dose(dose: float) -> float:
            query = np.concatenate((best_x[:-1], [dose])).reshape(1, -1)
            return float(_predict_yield(dy_gp, query)[0][0])

        # Bounded scalar minimization finds DY(DT) closest to lbdy within the
        # allowed dose-target interval.
        dose_result = minimize_scalar(
            lambda dose: abs(yield_at_dose(dose) - lbdy),
            bounds=(lbr, 1.05 * best_x[-1]),
            method="bounded",
            options={"xatol": 1e-3},
        )
        dt_at_minimum_yield = float(dose_result.x)
        predicted_yield_at_dt = yield_at_dose(dt_at_minimum_yield)
        yield_residual = float(dose_result.fun)
    else:
        dt_at_minimum_yield = float("nan")
        predicted_yield_at_dt = float("nan")
        yield_residual = float("nan")

    # These are also displayed as percentages in the final MATLAB summary.
    # Return measurements, predictions, and final optimization diagnostics.
    return OptimizationResult(
        x_train=x_train,
        selected_by_optimizer=selected_by_optimizer,
        euv_train=euv_train,
        rx_train=rx_train,
        ry_train=ry_train,
        target_size_train=ts_train,
        die_yield_train=dy_train,
        exposures_train=exp_train,
        open_loop_euv_train=euv_ol_train,
        actual_die_yield_train=dy_actual_train,
        predicted_die_yield=predicted_dy,
        best_dt=best_dt,
        best_dy=best_dy,
        best_x=best_x,
        r_squared=r_squared,
        dt_at_minimum_yield=dt_at_minimum_yield,
        predicted_yield_at_dt=predicted_yield_at_dt,
        yield_residual=yield_residual,
        cost_option=cost_option,
        yield_rmse=yield_rmse,
        dy_model=dy_model,
        r2_method=r2_method,
    )


def plot_results(result: OptimizationResult, lbdy: float = 0.995) -> None:
    """Plot the optimizer's measurements and physical constraints."""
    import matplotlib.pyplot as plt

    n = len(result.euv_train)
    time = 10.0 * np.arange(1, n + 1)
    marker = "*"
    line_width = 1.2

    # Each point is one measurement; the time axis uses 10 seconds per
    # measurement. Reference lines show the constraints used by the search.
    def new_axis(name: str, title: str, ylabel: str):
        figure, axis = plt.subplots(num=name)
        axis.set_title(title)
        axis.set_xlabel("Elapsed time [s]")
        axis.set_ylabel(ylabel)
        axis.set_xlim(0, 10 * n)
        axis.grid(True)
        return figure, axis

    _, axis = new_axis("EUV response", "Closed-loop EUV response", "Average EUV [mJ]")
    axis.plot(time, result.euv_train, marker, label="Measured EUV")
    axis.legend(loc="best")

    _, axis = new_axis("Open-loop EUV response", "Open-loop EUV response", "Open-loop EUV [mJ]")
    axis.plot(time, result.open_loop_euv_train, marker, label="Open-loop EUV")
    axis.legend(loc="best")

    _, axis = new_axis("Target angles", "Target angles and feasible bounds", "Angle [deg]")
    axis.plot(time, result.rx_train, marker, label="Rx")
    axis.plot(time, result.ry_train, marker, label="Ry")
    for bound, style, label in (
        (-23.5, "--", "Rx lower bound"),
        (23.5, "--", "Rx upper bound"),
        (11.5, ":", "Ry lower bound"),
        (23.5, ":", "Ry upper bound"),
    ):
        axis.axhline(bound, linestyle=style, linewidth=line_width, label=label)
    axis.set_ylim(-25.5, 25.5)
    axis.legend(loc="best")

    _, axis = new_axis("Target size", "Target size and feasible bounds", "Target size [um]")
    axis.plot(time, result.target_size_train, marker, label="Target size")
    axis.axhline(0.1, linestyle="--", linewidth=line_width, label="Lower bound")
    axis.axhline(0.9, linestyle="--", linewidth=line_width, label="Upper bound")
    axis.set_ylim(0, 1)
    axis.legend(loc="best")

    _, axis = new_axis("Die yield", "Die yield and required minimum", "Die yield [%]")
    axis.plot(time, 100.0 * result.die_yield_train, marker, label="Observed DY")
    axis.plot(time, 100.0 * result.predicted_die_yield, marker, label="Predicted DY")
    axis.axhline(100.0 * lbdy, linestyle="--", linewidth=line_width, label="Required yield")
    axis.set_ylim(90, 100)
    axis.legend(loc="best")

    _, axis = new_axis("Actual die yield", "Actual die yield and required minimum", "Actual die yield [%]")
    axis.plot(time, 100.0 * result.actual_die_yield_train, marker, label="Actual DY")
    axis.plot(time, 100.0 * result.predicted_die_yield, marker, label="Predicted DY")
    axis.axhline(100.0 * lbdy, linestyle="--", linewidth=line_width, label="Required yield")
    axis.set_ylim(90, 100)
    axis.legend(loc="best")

    _, axis = new_axis("Dose target", "Dose target selected by the optimizer", "Dose target [mJ]")
    axis.plot(time, result.x_train[:, -1], marker, label="Dose target")
    axis.axhline(50, linestyle="--", linewidth=line_width, label="Lower bound")
    axis.axhline(60, linestyle="--", linewidth=line_width, label="Upper bound")
    axis.set_ylim(48, max(60, result.best_dt) + 2)
    axis.legend(loc="best")

    _, axis = new_axis("Optimization objective", "Optimization objective over time", "Objective value [EUV-equivalent units]")
    axis.plot(
        time,
        _cost(result.euv_train, result.predicted_die_yield, result.cost_option, lbdy),
        marker,
        label="Objective",
    )
    axis.legend(loc="best")
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=300, help="number of measurements")
    parser.add_argument("--warmup", type=int, default=90, help="maximum number of initial measurements to forget")
    parser.add_argument("--n-test", type=int, default=50_000, help="candidate settings per optimization step")
    parser.add_argument("--seed", type=int, default=0, help="random seed for Latin hypercube designs")
    parser.add_argument(
        "--cost-option", type=int, choices=(0, 1, 2), default=2,
        help="0: linear, 1: multiplicative, 2: logit yield penalty",
    )
    parser.add_argument(
        "--dy-model", type=int, choices=(0, 1, 2), default=1,
        help="0: binomial GLM, 1: logit GP regression, 2: exposure-expanded GP classifier (expensive)",
    )
    parser.add_argument("--no-plots", action="store_true", help="skip interactive plots")
    parser.add_argument(
        "--r2-method", type=int, choices=(0, 1, 2), default=1,
        help="0: SSE R^2, 1: deviance pseudo-R^2 (default), 2: McFadden log-likelihood pseudo-R^2",
    )
    args = parser.parse_args()

    result = run_optimization(
        args.n, warmup=args.warmup, n_test=args.n_test, seed=args.seed,
        cost_option=args.cost_option, dy_model=args.dy_model,
        r2_method=args.r2_method,
    )
    # Report the best feasible measured point and the dose target closest to
    # the required yield, if the optimizer found a feasible point.
    print("\nFinal Bayesian-optimization results:")
    print(f"  Cost option: {result.cost_option} | DY model: {result.dy_model} | R^2 method: {result.r2_method}")
    print(f"  Best feasible DT: {result.best_dt:.4f} mJ")
    print(f"  Best feasible DY: {result.best_dy:.6f} ({100.0 * result.best_dy:.3f}%)")
    print(f"  R^2 score: {result.r_squared:.3f}%")
    print(f"  DY probability RMSE: {100.0 * result.yield_rmse:.6f} percentage points")
    print("  R^2 uses clipped actual probabilities; SSE/deviance scores are undefined for constant DY.")
    if np.isfinite(result.dt_at_minimum_yield):
        print(f"\nDose target for DY = 0.995: {result.dt_at_minimum_yield:.4f} mJ")
        print(f"  Predicted DY: {result.predicted_yield_at_dt:.6f}")
        print(f"  |DY - 0.995| residual: {result.yield_residual:.3e}")
        if result.yield_residual > 1e-4:
            # The search interval may not contain a dose target at the
            # requested yield.
            print("  Note: the required yield was not reached within the search interval.")
    else:
        print("\nDose target solve unavailable (no feasible point found).")

    if not args.no_plots:
        plot_results(result)


if __name__ == "__main__":
    main()
