"""All benchmark settings in one place. Override from the CLI with --set key=value
or by passing a JSON file (--config my.json). Nothing else is hard-coded."""
from __future__ import annotations
import json, os
from dataclasses import dataclass, field, asdict
from typing import Any

# Yield arms. "own run" arms change what the acquisition sees, so each needs its own BO run.
YIELD_ARMS = {
    "reg_logit":    "Regressor on logit yield (baseline)",
    "clf_raw":      "Classifier, raw binomial (point)",
    "clf_bb":       "Classifier + beta-binomial",
    "clf_ess_bb":   "Classifier + sample-size weights + beta-binomial",
    "clf_phix":     "Classifier + beta-binomial, phi(x) from regressor",
    "clf_binned":   "Binned-count classifier",
}
ACQS = {"matlab": "MATLAB-style EI (mean + 95% bounds)", "spread": "Spread-aware EI (averages over the distribution)"}


@dataclass
class Config:
    # ---- what to run -------------------------------------------------
    problem: str = "srcmdl"                    # "srcmdl" or any BOCoDe name, e.g. "bocode:WeldedBeamSO"
    models: list = field(default_factory=lambda: ["kumo", "causilo"])   # TFM backends
    gp_baseline: bool = True                   # BoTorch GP control (no context)
    yield_arms: list = field(default_factory=lambda: list(YIELD_ARMS))
    acqs: list = field(default_factory=lambda: list(ACQS))
    contexts: list = field(default_factory=lambda: ["none", "machines"])
    seeds: list = field(default_factory=lambda: [1, 2])   # black-box seeds, shared by every run
    n_iter: int = 350                          # total black-box evaluations per run
    n_init: int | None = None                  # initial LHS points; None = 3 x number of inputs
    n_candidates: int = 4096                   # candidates scored per BO step (MATLAB uses 50,000)
    warmup_forget: int = 0                     # MATLAB forgets up to 90 initial points; 0 = keep all

    # ---- models ------------------------------------------------------
    kumo_size: str = "large"
    n_estimators: int = 1
    device: str = "auto"
    query_batch: int = 2048                    # candidates per forward pass
    weights_backend: str = "auto"              # auto | ivo | sample_weight | expand
    ivo_repo: str = "tfm_onnx_in_matlab"       # clone of iwoodsawyer/tfm_onnx_in_matlab (weighted graphs)
    expand_max_rows: int = 40000               # cap for the row-expansion fallback
    gp_maxiter: int = 50                       # optimiser steps per GP fit (warm-started from the previous step)
    logit_target: str = "halfcount"            # halfcount: log((k+.5)/(n-k+.5)); clip: MATLAB's 1e-4 clip

    # ---- beta-binomial / quiz ---------------------------------------
    phi_min_fit: int = 20                      # steps of raw binomial before a concurrent phi fit
    quiz_mode: str = "lodo"                    # lodo (leave one context dataset out) | fraction
    quiz_fraction: float = 0.2
    quiz_sweep: list = field(default_factory=lambda: [0.05, 0.1, 0.2, 0.4])   # prediction benchmark only
    quiz_repeats: int = 3
    pmf_floor: float = 1e-3                    # probability floor used in log scores (as in earlier reports)

    # ---- context -----------------------------------------------------
    data_dir: str = "data"                     # real optimisation logs (opt_results_<machine>_*.csv)
    context_per_machine: int = 2               # datasets taken from each machine
    context_input_scaling: str = "minmax"      # minmax | zscore | log | none   (per machine, per input)
    context_machine_column: bool = True        # add a machine-id feature
    context_max_trials: float = 2.0            # drop rows with NrOfExposures > this x the usual count

    # ---- srcmdl ------------------------------------------------------
    srcmdl_dir: str = "src"                    # folder that holds srcmdl.py
    cost_option: int = 2                       # 0 linear, 1 multiplicative, 2 logit penalty (MATLAB example)
    yield_target: float = 0.995
    extra_scatter_rho: float = 0.0             # >0 adds run-to-run yield scatter (beta-binomial) to the simulator
    euv_targets: list = field(default_factory=lambda: [0.95, 0.99])   # fractions of the best EUV at 99.5% yield

    # ---- BOCoDe ------------------------------------------------------
    bocode_noise: float = 0.0                  # relative Gaussian noise on outputs (0 = deterministic)
    bocode_n_iter: int = 300
    bocode_seeds: list = field(default_factory=lambda: [0, 1, 2])

    # ---- prediction benchmark (no acquisition) -----------------------
    pred_train_sizes: list = field(default_factory=lambda: [50, 100, 200, 350])
    pred_n_test: int = 500

    # ---- output normalisation (only used when combining outputs with different units) ----
    output_normalisation: dict = field(default_factory=dict)   # e.g. {"euv": "zscore", "ts": "minmax"}
    default_output_normalisation: str = "zscore"               # minmax | linear | log | zscore | none

    # ---- run control -------------------------------------------------
    results_dir: str = "results/bo_bench"
    budget_hours: float = 11.0                 # stop cleanly and save once this much wall-clock is used
    save_every: int = 1
    standin: bool = False                      # use the CPU stand-in instead of Kumo/Causilo (testing)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1)

    def update(self, **kw: Any) -> "Config":
        for k, v in kw.items():
            if not hasattr(self, k):
                raise KeyError(f"unknown config key: {k}")
            setattr(self, k, v)
        return self

    @staticmethod
    def load(path: str | None = None, overrides: list | None = None) -> "Config":
        cfg = Config()
        if path:
            with open(path) as f:
                cfg.update(**json.load(f))
        for item in overrides or []:
            k, v = item.split("=", 1)
            try:
                v = json.loads(v)
            except json.JSONDecodeError:
                pass
            cfg.update(**{k: v})
        return cfg


def problem_dir(cfg: Config) -> str:
    d = os.path.join(cfg.results_dir, cfg.problem.replace(":", "_"))
    os.makedirs(os.path.join(d, "runs"), exist_ok=True)
    return d
