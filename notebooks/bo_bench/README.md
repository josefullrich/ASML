# bo_bench: closed-loop BO benchmark (TFM surrogates vs GP)

Drop this folder into `notebooks/` of the ASML repo. It reads `src/srcmdl.py` and `data/opt_results_*.csv`
and writes only under `results/bo_bench/` (already git-ignored).

## What it runs

| Factor | Levels |
|---|---|
| Model | Kumo large, Causilo, BoTorch GP (control, no context) |
| Context | none; an equal number of datasets from each real machine (3400 / 3600 / 3800) |
| Yield arm | logit regressor (baseline), classifier point, + beta-binomial, + sample-size weights, + phi(x) from the regressor, binned-count classifier |
| Acquisition | `matlab` (the example script's EI on mean + 95% bounds), `spread` (EI averaged over the distribution) |
| Seeds | the same black-box seeds for every run |

EUV, Rx, Ry and target size always use the same model's regressor, so the yield model is the only thing
that changes between yield arms.

Three stages, all resumable:

1. `pred`  : every arm fitted to the same black-box data and scored on the same held-out points (no acquisition).
2. `run`   : the BO runs. Saved after every evaluation.
3. `report`: one self-contained `report.html` per problem.

## Colab

Open `notebooks/bo_bench_srcmdl_colab.ipynb` (or `bo_bench_bocode_colab.ipynb`), fill in the settings cell, run all.
Results live on Google Drive, so after the 11 h budget or a disconnect you run the same cells again and it
continues. `pack` / `unpack` make a zip of the saved state if you would rather carry a file.

## Command line

    cd notebooks
    python -m bo_bench.run estimate                      # time one step per arm, project the total
    python -m bo_bench.run all --set budget_hours=11     # pred + run + report
    python -m bo_bench.run report
    python -m bo_bench.run all --suite --set bocode_noise=0.02    # the BOCoDe suite

Every setting is in `config.py` and can be overridden with `--set key=value` or `--config file.json`.

## Adding a black box

Write a `Problem` (see `problems/base.py`): inputs with bounds, a list of outputs (real or binomial, with
constraint bounds), `evaluate`, and `cost`. Register it in `problems/__init__.py`. The loop, both acquisition
functions, the metrics and the report work from that description alone.

## Weighted classifier

Frequency weights are not in either library's public API. `weights_backend = "auto"` tries, in order:
a `sample_weight` argument on the installed estimator (the Causilo PR build), the weighted PyTorch modules
from `iwoodsawyer/tfm_onnx_in_matlab` (clone it next to the repo; set `ivo_repo`), then plain row repetition
capped at `expand_max_rows`. The route used is printed once per model.

## Known deviations from the MATLAB example

* 4,096 candidates per step instead of 50,000 (`n_candidates`).
* No forgetting of the initial points by default (`warmup_forget=0`; set 90 to match).
* The dose window for candidates uses the previous step's incumbent, not the current one (saves a forward pass).
* Yield bounds use the 2.5% / 97.5% quantiles; EUV bounds use the same quantiles instead of mean ± 2 sd.
* GP baseline is BoTorch `SingleTaskGP` on the half-count logit, not the sklearn / fitcgp models.
