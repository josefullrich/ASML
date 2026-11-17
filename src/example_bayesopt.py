"""Bayesian optimization demonstration for the source model.

The optimizer selects nine process settings using target-shape constraints
and a selectable EUV/die-yield objective. Gaussian processes predict physical
responses; yield uses a binomial GLM, logit GP regression, or an
exposure-expanded GaussianProcessClassifier. Predicted uncertainty guides
expected improvement for each subsequent measurement.
"""

from __future__ import annotations

import argparse

import numpy as np

# Re-export the original names so existing imports continue to work.
from bayesopt.diagnostics import _yield_scores
from bayesopt.models.gp import FittedGP, _new_gpr, _fit_gpr, _predict
from bayesopt.models.yield_models import _fit_yield_model, _predict_yield
from bayesopt.objective import _cost
from bayesopt.optimization import run_optimization
from bayesopt.plotting import plot_results
from bayesopt.sampling import _latin_hypercube
from bayesopt.types import (
    FloatArray, CostOption, DYModelOption, R2Method, OptimizationResult,
)


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
    parser.add_argument("--euv-model", choices=("gp", "kumo"), default="gp",
                        help="EUV surrogate only; shape and yield models stay unchanged")
    parser.add_argument("--surrogate-scope", choices=("euv", "all"), default="euv")
    parser.add_argument("--kumo-size", choices=("small", "medium", "large"), default="small")
    parser.add_argument("--kumo-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--kumo-estimators", type=int, default=4)
    parser.add_argument("--kumo-batch-size", type=int, default=256)
    parser.add_argument("--kumo-checkpoint", help="local Kumo state_dict; omit for pretrained weights")
    parser.add_argument(
        "--r2-method", type=int, choices=(0, 1, 2), default=1,
        help="0: SSE R^2, 1: deviance pseudo-R^2 (default), 2: McFadden log-likelihood pseudo-R^2",
    )
    args = parser.parse_args()

    result = run_optimization(
        args.n, warmup=args.warmup, n_test=args.n_test, seed=args.seed,
        cost_option=args.cost_option, dy_model=args.dy_model,
        r2_method=args.r2_method,
        euv_model=args.euv_model, kumo_size=args.kumo_size,
        surrogate_scope=args.surrogate_scope,
        kumo_device=args.kumo_device, kumo_estimators=args.kumo_estimators,
        kumo_batch_size=args.kumo_batch_size, kumo_checkpoint=args.kumo_checkpoint,
    )
    # Report the best feasible measured point and the dose target closest to
    # the required yield, if the optimizer found a feasible point.
    print("\nFinal Bayesian-optimization results:")
    full_kumo = result.euv_model == "kumo" and result.surrogate_scope == "all"
    print(f"  EUV model: {result.euv_model} | Shape/yield: {'Kumo' if full_kumo else 'existing models'}")
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
