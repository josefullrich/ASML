"""Latin hypercube designs for initial experiments and candidates."""

from scipy.stats import qmc

from .types import FloatArray


def _latin_hypercube(n: int, dimensions: int, seed: int) -> FloatArray:
    # SciPy's LHS has no correlation criterion; strength-1 randomized LHS is
    # used to preserve one sample per stratum in every dimension.
    sampler = qmc.LatinHypercube(d=dimensions, seed=seed)
    return sampler.random(n)
