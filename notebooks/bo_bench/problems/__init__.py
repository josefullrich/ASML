from .base import Problem, OutputSpec, Evaluation, lhs, scale_to


def make_problem(cfg) -> Problem:
    """Problem factory. Add new black boxes here; nothing else needs to change."""
    if cfg.problem == "srcmdl":
        from .srcmdl_problem import SrcmdlProblem
        return SrcmdlProblem(cfg)
    if cfg.problem.startswith("bocode:"):
        from .bocode_problem import BocodeProblem
        return BocodeProblem(cfg, cfg.problem.split(":", 1)[1])
    raise ValueError(f"unknown problem {cfg.problem!r}")
