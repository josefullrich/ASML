from .base import Pred, Backend, LEVELS, ACQ_IDX, point_pred


def make_backend(cfg, model: str) -> Backend:
    """Backend factory. Register a new model family here."""
    if model == "gp":
        from .gp import GPBackend
        return GPBackend(cfg.gp_maxiter)
    if cfg.standin:
        from .standin import StandinBackend
        return StandinBackend(model, seed=sum(map(ord, model)))
    if model == "kumo":
        from .tfm import KumoBackend
        return KumoBackend(cfg)
    if model == "causilo":
        from .tfm import CausiloBackend
        return CausiloBackend(cfg)
    raise ValueError(f"unknown model {model!r}")
