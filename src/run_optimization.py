"""Run a small, recorded optimization experiment with a noninteractive plot."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
from importlib.metadata import version
import hashlib
import json
from pathlib import Path
import platform
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from example_bayesopt import run_optimization


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--n-test", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=90)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--source-seed", type=int, default=0)
    parser.add_argument("--dy-model", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--cost-option", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    started = datetime.now(timezone.utc)
    output = args.output or Path(__file__).resolve().parents[1] / "results" / started.strftime("optimization_%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    print(f"Running {args.n} measurements; saving to {output}", flush=True)
    begin = time.perf_counter()
    with (output / "run.log").open("w") as log, redirect_stdout(log):
        result = run_optimization(
            args.n, warmup=args.warmup, n_test=args.n_test, seed=args.seed,
            source_seed=args.source_seed, dy_model=args.dy_model,
            cost_option=args.cost_option,
        )
    elapsed = time.perf_counter() - begin
    settings = ["x", "y", "z", "e", "d", "p", "k", "b", "dose_target_mJ"]
    frame = pd.DataFrame(result.x_train, columns=settings)
    frame.insert(0, "measurement", np.arange(1, args.n + 1))
    frame["selected_by_optimizer"] = result.selected_by_optimizer
    for name, values in {
        "euv_mJ": result.euv_train, "rx_deg": result.rx_train,
        "ry_deg": result.ry_train, "target_size_normalized": result.target_size_train,
        "observed_yield": result.die_yield_train, "exposures": result.exposures_train,
        "open_loop_euv_mJ": result.open_loop_euv_train,
        "actual_yield_simulation_only": result.actual_die_yield_train,
        "final_model_predicted_yield": result.predicted_die_yield,
    }.items():
        frame[name] = values
    shape = ((frame.target_size_normalized >= .1) & (frame.target_size_normalized <= .9)
             & frame.rx_deg.between(-23.5, 23.5) & frame.ry_deg.between(11.5, 23.5))
    frame["shape_feasible"] = shape
    frame["shape_and_observed_yield_feasible"] = shape & (frame.observed_yield >= .995)
    frame["shape_and_actual_yield_feasible"] = shape & (frame.actual_yield_simulation_only >= .995)
    raw = frame.drop(columns=["final_model_predicted_yield"]).select_dtypes(include="number")
    if not np.isfinite(raw.to_numpy()).all():
        raise ValueError("Non-finite measurement or setting")
    if not (frame.observed_yield.between(0, 1).all()
            and frame.actual_yield_simulation_only.between(0, 1).all()
            and (frame.exposures == 71).all()
            and ((result.x_train[:, :8] >= 0) & (result.x_train[:, :8] <= 1)).all()):
        raise ValueError("Measurement range check failed")
    if result.selected_by_optimizer.any() and not np.isfinite(result.predicted_die_yield).all():
        raise ValueError("Non-finite model predictions")
    frame.to_csv(output / "measurements.csv", index=False)

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True, layout="constrained")
    step = frame.measurement
    chosen = result.selected_by_optimizer
    for j, label in enumerate(settings[:8]):
        axes[0].plot(step, result.x_train[:, j], ".-", label=label, alpha=.7, lw=.8)
    axes[0].set(ylabel="Normalized setting", ylim=(-.05, 1.05))
    axes[0].legend(ncol=8, fontsize=9)
    axes[1].plot(step, frame.dose_target_mJ, ".-", color="tab:purple")
    axes[1].set(ylabel="Dose target [mJ]")
    axes[2].plot(step, frame.euv_mJ, ".-", color=".55", label="Measured EUV")
    axes[2].scatter(step[chosen], frame.euv_mJ[chosen], color="tab:blue", label="Optimizer-selected")
    axes[2].scatter(step[shape], frame.euv_mJ[shape], facecolors="none", edgecolors="tab:green", s=70, label="Measured shape feasible")
    axes[2].set(ylabel="Mean EUV [mJ]")
    axes[2].legend(fontsize=9, ncol=3)
    axes[3].plot(step, 100 * frame.observed_yield, ".-", label="Observed yield")
    axes[3].plot(step, 100 * frame.actual_yield_simulation_only, "--", label="Underlying simulated yield")
    axes[3].axhline(99.5, color="tab:red", ls=":", label="99.5% requirement")
    axes[3].set(ylabel="Die yield [%]", xlabel="Measurement (71 exposures each)", ylim=(-3, 103))
    axes[3].legend(fontsize=9, ncol=3)
    first = np.flatnonzero(chosen)
    for ax in axes:
        ax.grid(alpha=.2)
        if first.size:
            ax.axvline(first[0] + .5, color="k", ls=":", alpha=.5)
    axes[0].set_title("Dotted line: first optimizer-selected measurement", fontsize=10)
    fig.suptitle(f"EUV optimization smoke run | {args.n} measurements | logit GP" if args.dy_model == 1
                 else f"EUV optimization smoke run | {args.n} measurements | yield model {args.dy_model}")
    fig.savefig(output / "overview.png", dpi=160)
    plt.close(fig)

    def best_row(mask):
        subset = frame.loc[mask]
        if subset.empty:
            return None
        row = subset.loc[subset.euv_mJ.idxmax()]
        return {"measurement": int(row.measurement), "euv_mJ": float(row.euv_mJ),
                "observed_yield": float(row.observed_yield),
                "actual_yield_simulation_only": float(row.actual_yield_simulation_only)}

    summary = {
        "started_utc": started.isoformat(), "elapsed_seconds": elapsed,
        "configuration": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "source_drift_reset_after_each_measurement": True,
        "hardware": {"platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor()},
        "python": platform.python_version(),
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in Path(__file__).resolve().parent.glob("*.py")},
        "packages": {p: version(p) for p in ("numpy", "scipy", "scikit-learn", "matplotlib", "pandas", "statsmodels")},
        "optimizer_selected_measurements": int(chosen.sum()),
        "shape_feasible_measurements": int(shape.sum()),
        "shape_and_observed_yield_feasible_measurements": int(frame.shape_and_observed_yield_feasible.sum()),
        "shape_and_actual_yield_feasible_measurements": int(frame.shape_and_actual_yield_feasible.sum()),
        "highest_euv_shape_feasible": best_row(shape),
        "highest_euv_shape_and_observed_yield_feasible": best_row(frame.shape_and_observed_yield_feasible),
        "highest_euv_shape_and_actual_yield_feasible": best_row(frame.shape_and_actual_yield_feasible),
        "checks": "Finite measurements, normalized inputs, yield ranges, and 71 exposures checked.",
        "limitations": ["Small execution check, not evidence of superiority over random search.",
                        "Yield is a soft objective penalty in the supplied optimizer.",
                        "Final-model predictions are retrospective, not held-out accuracy.",
                        "MATLAB equivalence and foundation models have not been evaluated."],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "elapsed_seconds": elapsed,
                      "optimizer_selected_measurements": int(chosen.sum()),
                      "shape_feasible_measurements": int(shape.sum())}, indent=2))


if __name__ == "__main__":
    main()
