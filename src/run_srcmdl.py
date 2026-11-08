"""Repeated source-model measurements with optional die-yield feedback."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from srcmdl import SourceState, srcmdl


FloatArray = NDArray[np.float64]


@dataclass
class RunResult:
    """Measurements produced by :func:`run_experiment`."""

    time: FloatArray
    dose_target: FloatArray
    euv: FloatArray
    rx: FloatArray
    ry: FloatArray
    target_size: FloatArray
    die_yield: FloatArray
    open_loop_euv: FloatArray
    actual_die_yield: FloatArray


def run_experiment(
    n: int = 100_000,
    *,
    enable_dy_control: bool = True,
    integral_gain: float = 0.3,
    dy_target: float = 0.995,
    initial_dt: float = 62.5,
    state: SourceState | int = 0,
) -> RunResult:
    """Run repeated measurements and return their recorded responses."""
    if n < 1:
        raise ValueError("n must be at least 1")

    # Keep one fixed set of normalized process settings while the feedback
    # controller changes the dose target between measurements.
    x, y, z, e, d, p, k, b = 0.250, 0.700, 0.477, 0.798, 0.150, 0.275, 0.312, 0.657
    dt = initial_dt
    state_value: SourceState | int = state

    # One measurement represents about 10 seconds. Preallocate arrays so each
    # response has a fixed size rather than growing inside the loop.
    dt_train = np.empty(n)
    euv_train = np.empty(n)
    rx_train = np.empty(n)
    ry_train = np.empty(n)
    ts_train = np.empty(n)
    dy_train = np.empty(n)
    euv_ol_train = np.empty(n)
    dy_actual_train = np.empty(n)

    # Run one source-model measurement at a time. 'state' is returned by
    # srcmdl and passed into the next call to preserve source history.
    for i in range(n):
        (
            state_value,
            euv,
            rx,
            ry,
            ts,
            dy,
            _nr_exposures,
            _de,
            euv_ol,
            dy_actual,
        ) = srcmdl(x, y, z, e, d, p, k, b, dt, state_value)

        # EUV can contain multiple exposure values, so store its average for
        # the response-versus-time plots.
        euv_train[i] = np.mean(euv)
        rx_train[i] = rx
        ry_train[i] = ry
        ts_train[i] = ts
        dy_train[i] = dy
        euv_ol_train[i] = euv_ol
        dy_actual_train[i] = dy_actual

        # Save the dose target used for this measurement before updating it;
        # this keeps dose_target[i] aligned with measurement i.
        dt_train[i] = dt
        if enable_dy_control:
            # Integral-style feedback: adjust dose in proportion to the
            # measured difference from the desired die yield.
            dt += integral_gain * (dy - dy_target)

    return RunResult(
        time=10.0 * np.arange(1, n + 1),
        dose_target=dt_train,
        euv=euv_train,
        rx=rx_train,
        ry=ry_train,
        target_size=ts_train,
        die_yield=dy_train,
        open_loop_euv=euv_ol_train,
        actual_die_yield=dy_actual_train,
    )


def plot_results(result: RunResult) -> None:
    """Plot measurements against elapsed time, as in the MATLAB example."""
    # There are 10 seconds per measurement, as in the MATLAB example.
    # Average EUV is plotted alongside its open-loop reference. The remaining
    # panels show target shape, measured/actual yield, and the control signal.
    plots = (
        (result.euv, "Average EUV [mJ]"),
        (result.open_loop_euv, "Open-loop EUV [mJ]"),
        (result.rx, "Average Rx [deg]"),
        (result.ry, "Average Ry [deg]"),
        (result.target_size, "Average target size [um]"),
        (100.0 * result.die_yield, "Die yield [%]"),
        (100.0 * result.actual_die_yield, "Actual die yield [%]"),
        (result.dose_target, "Dose target [mJ]"),
    )
    for values, ylabel in plots:
        _, axis = plt.subplots()
        axis.plot(result.time, values)
        axis.grid(True)
        axis.set_xlabel("Elapsed time [s]")
        axis.set_ylabel(ylabel)
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100_000, help="number of measurements")
    parser.add_argument("--no-plots", action="store_true", help="skip interactive plots")
    args = parser.parse_args()

    result = run_experiment(args.n)
    if not args.no_plots:
        plot_results(result)


if __name__ == "__main__":
    main()
