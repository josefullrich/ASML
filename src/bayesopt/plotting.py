"""Diagnostic plots for the standalone optimization example."""

import numpy as np

from .objective import _cost
from .types import OptimizationResult


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
