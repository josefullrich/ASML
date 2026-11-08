# pyobfus:generated format=1 edition=community
# Tool: pyobfus 0.12.0 - https://github.com/zhurong2020/pyobfus
# Source: srcmdl_full.py
# DO NOT EDIT - generated output
"""Source Model

Python conversion of MATLAB ``srcmdl_full.m``. A call simulates one source
measurement consisting of 71 exposures and returns, in MATLAB output order:

    (state, euv, rx, ry, ts, dy, nr_exposures, de, euvol, dy_actual)

Outputs
-------
state
    Persistent source state. Pass it into the next call to preserve its
    measurement count and accumulated drift. An integer state starts a
    repeatable sequence using that value as the seed; ``None`` chooses a
    random seed.
euv
    EUV for each exposure, in mJ. The MATLAB model's documented maximum is
    6.85 mJ at ``dt=100`` and 6.23 mJ at 99.5% die yield.
rx, ry
    Target angles Rx and Ry, in degrees. The documented ranges are
    23.5 deg <= Rx <= 23.5 deg and 11.5 deg <= Ry <= 23.5 deg.
ts
    Normalized target size (target size in um transformed from the model's
    300--480 um scale); the documented normalized range is 0.2--0.9.
dy
    Measured die yield as a fraction; the documented requirement is at least
    0.995.
nr_exposures
    Number of exposures in the measurement (71).
de
    Per-exposure die errors as fractions. Their documented maximum is 0.01.
    ``calculate_de=False`` follows the MATLAB fewer-than-eight-outputs path,
    returning ``None`` for ``de`` and drawing ``dy`` from a binomial model.
euvol
    Mean modeled open-loop EUV across the exposures, in mJ.
dy_actual
    Underlying logistic-model die yield before the measurement distribution.

Parameters
----------
x, y, z, e, d, p, k, b : float or None
    Normalized process inputs, normally in [0, 1]:
    ``x`` is l2dx (optimum 0.250), ``y`` is l2dy (0.700), ``z`` is l2dz
    (0.477), ``e`` is pp energy (0.798), ``d`` is mp2pp delay (0.150),
    ``p`` is drive-laser energy (0.275), ``k`` is drive-laser feed-forward
    gain (0.312), and ``b`` is beam size (0.657). ``None`` selects the
    documented default. Supplied settings are clipped to [0, 1] and, for a
    nonzero seed, transformed through a randomly selected beta CDF.
dt : float or None
    Dose target per burst, in mJ. The MATLAB default is 100 mJ open-loop;
    closed-loop operation is typically 60--65 mJ, with 62.5 mJ as the
    documented optimum.
state : SourceState, int, or None
    Initial or continuing source state.
calculate_de : bool, keyword-only
    Selects which MATLAB yield calculation to use. ``True`` returns
    per-exposure extreme-value dose errors and computes measured yield from
    them; ``False`` uses a binomial draw from ``dy_actual``.

Model details
-------------
The model maps l2dx/l2dy through the fitted quadratic ``poly22`` response,
applies exposure-dependent degradation, subtracts the six-dimensional
Hartmann response plus 3 for ``[e, d, z, p, k, b]``, then adds modified-Akima EUV
interpolation/noise. It also calculates target angles and target size,
converts dose to a logistic underlying yield, and samples the requested
measured-yield/dose-error distribution. MATLAB's ``evinv``/``evrnd`` correspond
to SciPy's minimum (left-skewed) Gumbel distribution. Drift is updated once
per call.

Random streams are repeatable for a given seed within this Python version,
but are not bit-for-bit compatible with MATLAB's random-number generator.
"""
from __future__ import annotations
from dataclasses import dataclass
from numbers import Integral
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import Akima1DInterpolator
from scipy.special import betainc, expit
from scipy.stats import binom, gumbel_l
I0 = NDArray[np.float64]

@dataclass
class SourceState:
    """Persistent source history returned by :func:`srcmdl`."""
    seed: int
    drift: I0
    k: int = 1

def I2(x: float, xp: I0, fp: I0) -> float:
    if x < xp[0]:
        I54 = (fp[1] - fp[0]) / (xp[1] - xp[0])
        return float(fp[0] + I54 * (x - xp[0]))
    if x > xp[-1]:
        I54 = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
        return float(fp[-1] + I54 * (x - xp[-1]))
    return float(np.interp(x, xp, fp))

def I1(xx: I0) -> float:
    I4 = np.array([1.0, 1.2, 3.0, 3.2])
    I3 = np.array([[10, 3, 17, 3.5, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14], [3, 3.5, 1.7, 10, 17, 8], [17, 8, 0.05, 10, 0.1, 14]], dtype=float)
    p = 0.0001 * np.array([[1312, 1696, 5569, 124, 8283, 5886], [2329, 4135, 8307, 3736, 1004, 9991], [2348, 1451, 3522, 2883, 3047, 6650], [4047, 8828, 8732, 5743, 1091, 381]], dtype=float)
    I35 = np.sum(I4 * np.exp(-np.sum(I3 * (xx - p) ** 2, axis=1)))
    return float(-(2.58 + I35) / 1.94)

def srcmdl(x: float | None=None, y: float | None=None, z: float | None=None, e: float | None=None, d: float | None=None, p: float | None=None, k: float | None=None, b: float | None=None, dt: float | None=None, state: SourceState | int | None=None, *, calculate_de: bool=True) -> tuple[SourceState, I0, float, float, float, float, int, I0 | None, float, float]:
    """Evaluate one 71-exposure source-model measurement.

    Inputs ``x`` through ``b`` use the normalized ranges and defaults in the
    MATLAB model. Pass the returned ``state`` into the next call to preserve
    source drift. An integer ``state`` starts a fresh sequence with that seed.
    ``calculate_de=False`` selects the MATLAB path used when fewer than eight
    outputs are requested; otherwise the returned ``de`` array and its
    corresponding measured yield are calculated.
    """
    I24 = 100
    I34 = 71
    I48 = 50000
    if state is None:
        I51 = np.random.default_rng()
        I55 = SourceState(seed=int(I51.integers(1, I24 + 1)), drift=np.zeros(6, dtype=float))
    elif isinstance(state, Integral):
        I55 = SourceState(seed=int(state), drift=np.zeros(6, dtype=float))
    elif isinstance(state, SourceState):
        I55 = state
        I55.k += 1
    else:
        raise TypeError('state must be None, an integer seed, or a SourceState')
    I45 = np.random.default_rng(I55.seed)

    def transform(value: float | None, default: float, scale: float=1.0, offset: float=0.0) -> float:
        if value is None:
            return default
        value = float(np.clip(value, 0.0, 1.0))
        if I55.seed:
            I52, I53 = I45.integers(1, 6, size=2)
            value = float(betainc(I52, I53, value))
        return scale * value + offset
    I58 = transform(x, -8.0, 16.0, -12.0)
    I59 = transform(y, 10.0, 20.0, -4.0)
    I10 = (0.20168952, 0.15001069, 0.47687398, 0.27533243, 0.31165162, 0.65730054)
    I60 = transform(z, I10[2])
    I17 = transform(None if e is None else 1.0 - float(np.clip(e, 0.0, 1.0)), I10[0])
    I6 = transform(d, I10[1])
    I44 = transform(p, I10[3])
    I26 = transform(k, I10[4])
    I5 = transform(b, I10[5])
    I14 = 100.0 if dt is None else float(dt)
    I46 = I14 * 0.997 / (I48 * 0.0002)
    I22 = np.arange(1, I34 + 1, dtype=float)
    I18 = I22 + I34 * (I55.k - 1)
    I27 = I58 - 1.0 + 0.999 ** I18
    I28 = I59 + 4.0 - 4.0 * 0.9993 ** I18
    I37, I41, I38, I43, I42, I39 = (6.142710350655271, -0.034842117259257, 0.071626594093166, -0.002053273246531, -0.0001843041278061222, -0.002595465689353)
    I20: I0 = I37 + I41 * I27 + I38 * I28 + I43 * I27 ** 2 + I42 * I27 * I28 + I39 * I28 ** 2
    I20 -= 1e-07 * (1.0 + 0.1 * I45.standard_normal()) * I55.k
    I13 = np.random.default_rng(I24 * I55.k + I55.seed + 1)
    I55.drift += 0.0001 * I13.standard_normal(6)
    I60 += I55.drift[0]
    I17 += I55.drift[1]
    I6 += I55.drift[2]
    I44 += I55.drift[3]
    I26 += I55.drift[4]
    I5 += I55.drift[5]
    I20 -= I1(np.array([I17, I6, I60, I44, I26, I5])) + 3.0
    I33 = np.random.default_rng(I24 * I55.k + I55.seed + 1)
    I32 = I33.standard_normal(I34)
    I12 = 0.09
    I31 = float(np.max(I20))
    if I46 < I31:
        I29 = np.array([0.0, (1.0 - I12) * I31, I31, 20.0])
        I30 = np.array([I46, I46, I31, I31])
        I25 = Akima1DInterpolator(I29, I30, method='makima', extrapolate=True)
        I19 = float(I25(I46)) + 0.025 * I32
    else:
        I19 = I20 + 0.25 * I32
    I49 = float(0.4 * (np.mean(I27) + 8.0) + (np.mean(I28) - 10.0) + 0.5 / 3.0 * I33.standard_normal())
    I50 = float(-1.4 * (np.mean(I27) - 5.0) - 0.5 * (np.mean(I28) - 10.0) + 0.5 / 3.0 * I33.standard_normal())
    I56 = I2(I17, np.array([0.0, 0.8, 1.0]), np.array([320.0, 400.0, 470.0]))
    I57 = max(300.0, I56 - abs(I58) / 5.0 - abs(I59) / 5.0 + 5.0 / 3.0 * I33.standard_normal())
    I57 = float((I57 - 300.0) / 180.0)
    I21 = float(np.mean(I20))
    I47 = 5.3
    I36 = I47 / I12
    I40 = -I47 / (I12 * I21)
    I16 = float(expit(I36 + I40 * I46))
    I7: I0 | None
    if calculate_de:
        I11 = 0.01
        I8 = 0.0111
        I9 = float(gumbel_l.ppf(1.0 - min(I16, 0.9999), loc=0.0, scale=0.005))
        I23 = gumbel_l.rvs(loc=0.0, scale=0.005, size=I34, random_state=I33)
        I7 = np.abs(I11 / (I8 - I9) * (I9 - I23) + I11)
        I15 = float(np.count_nonzero(I7 <= I11) / I34)
    else:
        I7 = None
        I15 = float(binom.rvs(I34, I16, random_state=I33) / I34)
    return (I55, np.asarray(I19, dtype=float), I49, I50, I57, I15, I34, I7, I21, I16)