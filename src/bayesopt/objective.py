"""The existing EUV and yield objective, maximized by the optimizer."""

import numpy as np

from .types import CostOption, FloatArray


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
