"""Shared option types and the optimization result container."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray


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
    euv_model: str = "gp"
    surrogate_scope: str = "euv"
    kumo_used: bool = False
