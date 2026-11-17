"""Sequential Bayesian optimization of the simulated EUV source.

The original loop is preserved, with an optional Kumo EUV prediction branch.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm

from srcmdl import SourceState, srcmdl
from .diagnostics import _yield_scores
from .models.gp import FittedGP, _fit_gpr, _predict
from .models.yield_models import (
    YieldClassifier, YieldGLM, exposure_counts, _fit_yield_model, _predict_yield,
)
from .objective import _cost
from .sampling import _latin_hypercube
from .types import CostOption, DYModelOption, OptimizationResult, R2Method


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
    euv_model: str = "gp",
    surrogate_scope: str = "euv",
    kumo_size: str = "small",
    kumo_device: str = "cpu",
    kumo_estimators: int = 4,
    kumo_batch_size: int = 256,
    kumo_checkpoint: str | Path | None = None,
) -> OptimizationResult:
    """Run sequential BO with GP responses and an optional Kumo EUV surrogate.

    cost_option: 0 = linear, 1 = multiplicative, 2 = logit yield penalty.
    dy_model: 0 = binomial polynomial GLM, 1 = logit GP regression,
    2 = GaussianProcessClassifier with exact per-exposure row replication.
    Mode 2 preserves success/failure counts but requires dense matrices over
    all exposures, not just observations; full runs can be very expensive.
    source_seed selects the simulated source independently of the design seed.
    r2_method: 0 = SSE, 1 = deviance (MATLAB default), 2 = McFadden
    log-likelihood. This diagnostic choice does not affect optimization.
    euv_model: 'gp' (default) or 'kumo'. surrogate_scope='all' replaces shape
    and logit yield too when Kumo is selected. warmup=0 retains all history.
    Kumo uses its median and a normal-equivalent scale from its 95% interval
    in the existing approximate expected-improvement calculation.
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    if warmup < 0:
        raise ValueError("warmup must be nonnegative (0 retains all history)")
    if n_test < 1:
        raise ValueError("n_test must be at least 1")
    if cost_option not in (0, 1, 2):
        raise ValueError("cost_option must be 0 (linear), 1 (multiplicative), or 2 (logit)")
    if dy_model not in (0, 1, 2):
        raise ValueError("dy_model must be 0 (binomial GLM), 1 (logit GP), or 2 (classifier GP)")
    if r2_method not in (0, 1, 2):
        raise ValueError("r2_method must be 0 (SSE), 1 (deviance), or 2 (log-likelihood)")
    if euv_model not in ("gp", "kumo"):
        raise ValueError("euv_model must be 'gp' or 'kumo'")
    if surrogate_scope not in ("euv", "all"):
        raise ValueError("surrogate_scope must be 'euv' or 'all'")
    if kumo_checkpoint is not None:
        if euv_model != "kumo":
            raise ValueError("kumo_checkpoint requires euv_model='kumo'")
        if not Path(kumo_checkpoint).is_file():
            raise FileNotFoundError(f"Kumo checkpoint not found: {kumo_checkpoint}")
    full_kumo = surrogate_scope == "all" and euv_model == "kumo"
    if full_kumo and dy_model != 1:
        raise ValueError("All-Kumo uses logit-yield regression; set dy_model=1")
    if euv_model == "kumo":
        if kumo_size not in ("small", "medium", "large") or kumo_device not in ("cpu", "cuda"):
            raise ValueError("Select a valid Kumo size and cpu/cuda device")
        if kumo_estimators < 1 or kumo_batch_size < 1:
            raise ValueError("Kumo estimator count and batch size must be positive")

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
    kumo_euv = None  # Load pretrained weights once, when model fitting begins.
    kumo_shape = None
    kumo_yield = None

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
            if full_kumo:
                if kumo_euv is None:
                    from .models.kumo import KumoEUV, KumoRegressor, KumoYield

                    options = dict(size=kumo_size, device=kumo_device,
                                   num_estimators=kumo_estimators, batch_size=kumo_batch_size)
                    print(f"Loading shared Kumo Tabular {kumo_size} on {kumo_device} for all five responses", flush=True)
                    kumo_euv = KumoEUV(**options, checkpoint=kumo_checkpoint)
                    shared = dict(**options, shared_model=kumo_euv.model)
                    kumo_shape = [KumoRegressor(target_name=name, **shared)
                                  for name in ("rx_deg", "ry_deg", "target_size")]
                    kumo_yield = KumoYield(**shared)
                # Yield is needed for the incumbent before candidate generation.
                kumo_yield.fit(x_observed, dy_observed, seed=seed + i)
                dy_gp = kumo_yield
            else:
                dy_gp = _fit_yield_model(
                    x_observed, dy_observed, exp_train[training_slice], dy_model
                )

            # Fit one GP per physical response. EUV retains its dose basis.
            # Full Kumo fits physical responses sequentially at prediction time.
            if euv_model == "gp":
                euv_gp = _fit_gpr(
                    x_observed,
                    euv_train[training_slice],
                    1e-2,
                    dose_trend=True,
                )
            elif not full_kumo:
                if kumo_euv is None:
                    from .models.kumo import KumoEUV

                    print(f"Loading Kumo Tabular {kumo_size} on {kumo_device} for EUV", flush=True)
                    kumo_euv = KumoEUV(
                        size=kumo_size, device=kumo_device,
                        num_estimators=kumo_estimators, batch_size=kumo_batch_size,
                        checkpoint=kumo_checkpoint,
                    )
                kumo_euv.fit(x_observed, euv_train[training_slice], seed=seed + i)
            if not full_kumo:
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

            if euv_model == "gp":
                euv_mean, euv_std = _predict(euv_gp, candidates)
            else:
                # For Kumo these names hold a median and normal-equivalent
                # scale, rather than a GP mean and standard deviation.
                if full_kumo:
                    kumo_euv.fit(x_observed, euv_train[training_slice], seed=seed + i)
                euv_mean, euv_std = kumo_euv.predict(candidates)
            if full_kumo:
                shape_predictions = []
                for model, observations in zip(kumo_shape, (rx_train, ry_train, ts_train)):
                    model.fit(x_observed, observations[training_slice], seed=seed + i)
                    shape_predictions.append(model.predict(candidates)[0])
                rx_mean, ry_mean, ts_mean = shape_predictions
                # One shared set of weights/cache: refit yield last so subsequent
                # diagnostics and the final dose solve keep the yield context.
                kumo_yield.fit(x_observed, dy_observed, seed=seed + i)
            else:
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
        euv_model=euv_model,
        surrogate_scope=surrogate_scope,
        kumo_used=kumo_euv is not None,
    )
