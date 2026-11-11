"""Shared configuration for the TFM-vs-GP surrogate benchmark. Every tunable lives here."""
from __future__ import annotations

import glob
import hashlib
import os
import warnings

import numpy as np

# ----------------------------------------------------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------------------------------------------------
# Repo layout (all relative to this file, so it works from any clone):
#   <repo>/notebooks/tfm_bench/   this code
#   <repo>/data/                  the opt_results_YYYYMMDD_HHMMSS.csv logs (searched recursively; not in git)
#   <repo>/results/tfm_bench/     everything the runs produce (not in git)
# On Colab the notebooks link <repo>/data and <repo>/results to Google Drive, so the same relative paths hold.
# BENCH_DATA_DIR / BENCH_WORK_DIR override them (notebook 2 uses BENCH_WORK_DIR for a scratch rebuild).
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(os.path.dirname(CODE_DIR))
PROJECT_DIR = os.environ.get("BENCH_WORK_DIR", os.path.join(REPO_DIR, "results", "tfm_bench"))
DATA_DIR = os.environ.get("BENCH_DATA_DIR", os.path.join(REPO_DIR, "data"))
# ----------------------------------------------------------------------------------------------------------------------
# Target. "dy" = logit die yield (the original benchmark, results/). The regressor outputs each get their own results
# folder (results_EUV_mJ/, ...). Pick with the env var BENCH_TARGET or `--target` on run_gpr.py / run_tfm.py.
# ----------------------------------------------------------------------------------------------------------------------
TARGETS = {"dy": "die yield, logit((k+0.5)/(n+1))", "EUV_mJ": "EUV energy (mJ)", "Rx_deg": "target angle Rx (deg)",
           "Ry_deg": "target angle Ry (deg)", "TargetSize_um": "target size (um)"}
TARGET = os.environ.get("BENCH_TARGET", "dy")
assert TARGET in TARGETS, f"BENCH_TARGET={TARGET!r}; choose one of {list(TARGETS)}"
YIELD_ONLY_MODELS = {"gpr_laplace", "causilo_trials"}  # binomial models: die yield only


def results_dir_for(target: str) -> str:
    return os.path.join(PROJECT_DIR, "results" if target == "dy" else f"results_{target}")


RESULTS_DIR = results_dir_for(TARGET)
AVG_DIR = os.path.join(PROJECT_DIR, "avg_everything")
CONTEXT_SELECTION_CSV = os.path.join(PROJECT_DIR, "context_selection.csv")

# ----------------------------------------------------------------------------------------------------------------------
# Datasets (hard-coded; auto-discovery below warns on any mismatch)
# ----------------------------------------------------------------------------------------------------------------------
DATASETS_ALL = [
    "20261001_161410", "20261002_081740", "20261002_122830", "20261002_133805",
    "20261005_111638", "20261005_111644", "20261005_153712", "20261005_153716",
    "20261006_174410", "20261006_174913", "20261012_073415", "20261012_073422",
    "20261012_073431", "20261013_235925", "20261016_114304", "20261016_160240",
    "20261019_094809", "20261019_095858", "20261019_135845", "20261021_092803",
]

# Four pairs of files are byte-identical. With DEDUPLICATE=True the later copy of each pair is dropped, so a TFM can
# never receive the current day's exact data as "another day" in its context (ctxall would otherwise contain it).
DEDUPLICATE = True

# Each CSV is ~324 BO rows (NrOfExposures=70) followed in 14/20 files by a validation tail: 8-13 re-measurements with
# 420 (or 313) exposures and one open-loop row at DoseTarget=120 with 36 (or 27) exposures. With
# TRUNCATE_AT_BO_END=True each dataset stops before the first row whose NrOfExposures differs from the warm-up rows.
TRUNCATE_AT_BO_END = True

# ----------------------------------------------------------------------------------------------------------------------
# Columns
# ----------------------------------------------------------------------------------------------------------------------
ALL_COLUMNS = ["EUV_mJ", "NrOfExposures", "NrOfInSpecDose", "Rx_deg", "Ry_deg", "TargetSize_um", "LeakagePower_W",
               "RejectedPower_W", "PP2DX_um", "PP2DY_um", "PP2DZ_um", "MP2PPDelay_ns", "PPE_mJ", "MMPED_mJ", "FFWPED",
               "MPBS_um", "DoseTarget_mJ"]
INPUT_COLS = ["PP2DX_um", "PP2DY_um", "PP2DZ_um", "MP2PPDelay_ns", "PPE_mJ", "MMPED_mJ", "FFWPED", "MPBS_um",
              "DoseTarget_mJ"]
OUTPUT_COLS = ["EUV_mJ", "NrOfExposures", "NrOfInSpecDose", "Rx_deg", "Ry_deg", "TargetSize_um", "LeakagePower_W",
               "RejectedPower_W"]
K_COL, N_COL = "NrOfInSpecDose", "NrOfExposures"
assert not set(INPUT_COLS) & set(OUTPUT_COLS), "measured outputs leaked into the feature list"
assert len(INPUT_COLS) == 9

# ----------------------------------------------------------------------------------------------------------------------
# Protocol
# ----------------------------------------------------------------------------------------------------------------------
N_WARMUP = 27  # rows revealed before the first prediction (3 per input x 9 inputs, as in example_bayesopt.m)
PREDICT_REMAINING = False  # also predict all not-yet-revealed rows at each step (logged separately)
SEED = 0

# ----------------------------------------------------------------------------------------------------------------------
# Models and arms
# ----------------------------------------------------------------------------------------------------------------------
GP_MODELS = ["gpr_laplace", "gpr_botorch"]
# "kumo" is Kumo Tabular size="large" (folder name kept for compatibility with existing results);
# kumo_medium / kumo_small are the same model family at the smaller sizes. All six run with "--models all".
TFM_MODELS = ["causilo", "kumo", "kumo_medium", "kumo_small", "tabpfn2", "tabpfn35"]
# opt-in models that "--models all" skips; run them by name. causilo_trials = Causilo as a binomial classifier with
# frequency weights (needs the causilo PR #19 build).
EXTRA_TFM_MODELS: list[str] = ["causilo_trials"]
ALL_TFM_MODELS = TFM_MODELS + EXTRA_TFM_MODELS
ARMS = ["ctx0", "ctx5", "ctxall"]
BOTORCH_OBSERVATION_NOISE = True  # predictive for y includes the fitted Gaussian noise (it is a prediction of y)
LAPLACE_GRAD = "exact"  # "exact" analytic evidence gradient, or "fd" (finite differences, like MATLAB fminunc)
TFM_DEVICE = "cuda"  # set to "cpu" to debug without a GPU

# ----------------------------------------------------------------------------------------------------------------------
# Normalization used to combine metrics into the rankings (analyze.py; the HTML lets you switch live)
# ----------------------------------------------------------------------------------------------------------------------
# Each ranked metric is first turned into a "badness" (>= 0, lower = better: |coverage - 0.9|, 1 - tau, ...), then
# rescaled across configs WITHIN each dataset, then averaged over datasets -> category -> overall.
#   "robust_z"  (b - median) / (1.4826 * MAD)     one broken config can't stretch the scale        <- default
#   "z"         (b - mean) / std                  classic z-score; sensitive to a single outlier config
#   "skill"     log2(b / b_REF), clipped to +-3   0 = same as the reference GP, -1 = half its badness (2x better)
#   "rank"      (rank - 1) / (n - 1)              order only, ignores how big the gaps are (old behaviour)
#   "minmax"    (b - min) / (max - min)           0 = best, 1 = worst; most outlier-sensitive
# analyze.py (yield-only summary) default is "rank", the original Kaggle behaviour; targets_report.py (several outputs
# with different units, where rescaling is actually needed) defaults to TARGETS_NORMALIZATION.
NORMALIZATION = "rank"
TARGETS_NORMALIZATION = "robust_z"
NORMALIZATION_REF = os.environ.get("BENCH_NORM_REF", "gpr_botorch")  # reference config for "skill"


def model_folders():
    return GP_MODELS + [f"{m}_{a}" for m in ALL_TFM_MODELS for a in ARMS]


# ----------------------------------------------------------------------------------------------------------------------
# Context selection (TFMs)
# ----------------------------------------------------------------------------------------------------------------------
N_CONTEXT_DAYS = 5
CONTEXT_POOL = "all_other"  # or "previous_only" (only days dated before the current one)
CONTEXT_SELECTION = "most_different"  # "most_different" | "most_recent" | "random" (seeded)
# The summary vector of the current day is computed from its full BO run, as the spec defines it. Set to "warmup" to
# use only the current day's warm-up rows (strictly no look-ahead inside the current day).
CURRENT_DAY_SUMMARY_FROM = "full"

# ----------------------------------------------------------------------------------------------------------------------
# Output grid and metrics
# ----------------------------------------------------------------------------------------------------------------------
# BASE grid: the fixed 101 levels every CRPS / PIT / reliability number is computed on (unchanged, so all runs compare).
BASE_QUANTILE_LEVELS = np.unique(np.round(np.concatenate([np.arange(1, 100) / 100, [0.025, 0.975]]), 6))
# EXTRA tail levels, only for the 99% central interval. Runs made before these existed simply have no 99% point.
EXTRA_QUANTILE_LEVELS = np.array([0.005, 0.995])
QUANTILE_LEVELS = np.unique(np.round(np.concatenate([BASE_QUANTILE_LEVELS, EXTRA_QUANTILE_LEVELS]), 6))  # adapters output
# Central-interval levels for the coverage / width-quality analysis.
COVERAGE_LEVELS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99]
LAST_FRAC = 0.2  # "last 20%" = last 20% of the predicted iterations of a dataset
ROLLING_WINDOW = 10
BIG_ERROR_FACTOR = 2.0  # flag |error| > factor * median |error| (per model, per dataset)
PIT_LEVELS = np.round(np.arange(0.05, 0.951, 0.05), 6)


def qcol(level: float) -> str:
    return f"q_{level:.3f}"


QCOLS = [qcol(q) for q in QUANTILE_LEVELS]  # every column the adapters write
BASE_QCOLS = [qcol(q) for q in BASE_QUANTILE_LEVELS]  # the columns CRPS / PIT use


# ----------------------------------------------------------------------------------------------------------------------
# Dataset discovery / de-duplication
# ----------------------------------------------------------------------------------------------------------------------
_CSV_INDEX: dict[str, str] = {}
for _p in sorted(glob.glob(os.path.join(DATA_DIR, "**", "opt_results_*.csv"), recursive=True)):
    _CSV_INDEX.setdefault(os.path.basename(_p)[len("opt_results_"):-4], _p)  # first copy wins if a file repeats


def csv_path(ds_id: str) -> str:
    return _CSV_INDEX.get(ds_id, os.path.join(DATA_DIR, f"opt_results_{ds_id}.csv"))


def discover_datasets() -> list[str]:
    found = sorted(_CSV_INDEX)
    missing, extra = sorted(set(DATASETS_ALL) - set(found)), sorted(set(found) - set(DATASETS_ALL))
    if missing:
        warnings.warn(f"Datasets listed in config.DATASETS_ALL but not found in {DATA_DIR}: {missing}")
    if extra:
        warnings.warn(f"CSV files found in {DATA_DIR} but not listed in config.DATASETS_ALL: {extra}")
    return [d for d in DATASETS_ALL if d in found]


def duplicate_map(ids: list[str]) -> dict[str, str]:
    """{later_id: first_id} for files with identical bytes."""
    seen, dup = {}, {}
    for d in sorted(ids):
        with open(csv_path(d), "rb") as fh:
            h = hashlib.md5(fh.read()).hexdigest()
        if h in seen:
            dup[d] = seen[h]
        else:
            seen[h] = d
    return dup


def active_datasets() -> list[str]:
    ids = discover_datasets()
    if DEDUPLICATE:
        dup = duplicate_map(ids)
        if dup:
            warnings.warn(f"Dropping {len(dup)} duplicate dataset(s): " + ", ".join(f"{k}=={v}" for k, v in dup.items()))
        ids = [d for d in ids if d not in dup]
    return ids


DATASETS = active_datasets()
N_CONTEXT_ALL = len(DATASETS) - 1
