# ASML EUV optimization

This project explores ways to maximize EUV output while maintaining a die yield
of at least 99.5%. We use a simulated EUV source to test optimization methods,
trying to beat a Gaussian Process baseline with a Tabular Foundation Model approach. 

## Folder guide

```text
ASML/
├── src/               Python simulator and optimization scripts
├── matlab/            Original MATLAB code
├── notebooks/         Experiment notebooks, including colab_pipeline.ipynb
├── data/              Historical measurement logs
├── docs/              Project briefs and research reports
├── results/           Experiment outputs and plots
└── requirements.txt   Python dependencies
```

`data/`, `docs/`, and `results/` are kept locally and are not included in this
repository.

The Colab workflow is in [`notebooks/colab_pipeline.ipynb`](notebooks/colab_pipeline.ipynb).

## Python code guide

```text
src/
├── srcmdl.py                 Simulated EUV source (the black box)
├── run_srcmdl.py             Separate simulator experiment; not used by BO
├── run_optimization.py       Notebook/CLI launcher; saves results and plots
├── example_bayesopt.py       Original CLI and compatible optimizer imports
├── dy_models.py              Compatible imports for the moved yield models
└── bayesopt/
    ├── optimization.py      Main loop: fit, predict, select, measure, repeat
    ├── types.py             Result container and option types
    ├── sampling.py          Initial designs and candidate generation
    ├── objective.py         EUV/yield score
    ├── diagnostics.py       Yield prediction metrics
    ├── plotting.py          Standalone example's diagnostic plots
    └── models/
        ├── gp.py            GP construction, fitting, and prediction
        ├── kumo.py          Optional Kumo Tabular EUV surrogate
        └── yield_models.py  GLM, logit GP, and classifier yield models
```

The notebook launches `run_optimization.py`, which calls `run_optimization()`
in `bayesopt/optimization.py` through the existing `example_bayesopt` import.
That loop calls `srcmdl()` directly for each measurement. Start with
`bayesopt/optimization.py` to follow the algorithm; its model helpers live in
`bayesopt/models/`. The original functions, comments, and calculations were
moved together, including candidate selection inside the main loop.

For Colab, upload the complete `src/` folder, including `bayesopt/` and its
subfolders. Preserve the directory structure.

Existing commands still work, for example:

```sh
python src/run_optimization.py --n 40 --n-test 1000 --dy-model 1
python src/example_bayesopt.py --n 40 --n-test 1000 --dy-model 1 --no-plots
```

The default EUV model is still the GP. Select Kumo to use its pretrained
predictions for EUV while keeping the existing shape and yield models. Yield
remains a soft penalty in the objective.

## Run BO with Kumo EUV predictions

Install the optional dependency in your active environment, then run from the
project folder:

```sh
python -m pip install -r requirements-kumo.txt
python src/run_optimization.py --n 40 --n-test 1000 --euv-model kumo
```

Kumo Tabular Small runs on CPU by default. Its weights load once when model
fitting begins. Each iteration, `optimization.py` calls `KumoEUV.fit()` with
the same retained input settings and measured mean EUV used by the GP path.
This refreshes cached context without training the neural-network weights.
`KumoEUV.predict()` then scores candidates in batches. Both EUV backends retain
the original fixed dose trend (`0.1 * dose_target`), learning the residual and
adding the trend back to predictions. No simulation-only outputs are supplied
to Kumo.

Kumo returns quantiles. The adapter uses its median as the central prediction
and `(q975 - q025) / (2 * normal_ppf(0.975))` as an approximate standard
deviation for the existing expected-improvement calculation. This normal
approximation does not make Kumo uncertainty equivalent to a GP posterior or
establish calibration. Run summaries record this choice, backend configuration,
dependency versions, and whether Kumo actually ran. A run that never reaches
the model-fitting stage does not load Kumo.

Options: `--kumo-size small|medium|large`, `--kumo-device cpu|cuda`,
`--kumo-estimators 4`, and `--kumo-batch-size 256`. In the notebook's configuration
cell, set `EUV_MODEL = "kumo"`; select `KUMO_DEVICE = "cuda"` only in a CUDA-enabled
runtime. Colab installs the optional dependency when Kumo is selected. Upload
the complete `src/` folder after changing local code. Saved notebook outputs
describe the earlier run until its cells are executed again.

Use `--euv-model gp` (or omit the option) to run the original GP baseline. No
SDM/PyTorch import is required by that path. The standalone `example_bayesopt.py`
command accepts the same Kumo options.

Adapter checks (no weight downloads):

```sh
python -m unittest discover -s tests -v
```

## Compare all five Kumo surrogates against GP

The notebook now defaults to 200 measurements, 1,000 candidates, Kumo Large
on CUDA, and `SURROGATE_SCOPE = "all"`. `FORGET_INITIAL = 0` retains the complete
measured history. Set `EUV_MODEL = "gp"` and `KUMO_CHECKPOINT = None` for a matched GP baseline;
keep measurement budget, candidate count, seeds, and forgetting policy fixed.
The notebook's saved cell outputs have been cleared because these settings
have not yet been run on Colab.

Equivalent CLI runs (use distinct output directories):

```sh
python src/run_optimization.py --n 200 --n-test 1000 --warmup 0 --seed 0 --source-seed 0 --dy-model 1 --euv-model kumo --surrogate-scope all --kumo-size large --kumo-device cuda
python src/run_optimization.py --n 200 --n-test 1000 --warmup 0 --seed 0 --source-seed 0 --dy-model 1 --euv-model gp
```

All-Kumo models EUV, both rotations, target size, and clipped-logit yield.
Responses have separate contexts but reuse one pretrained model sequentially,
avoiding five copies of Large weights in GPU memory. Contexts are cleared and
refitted when switching response. Yield is fitted again after the physical
responses so the final diagnostics and dose solve use the yield context.
It uses observed yield fractions clipped to `[0.0001, 0.9999]`, matching the
logit GP target transform. Yield quantiles are transformed back to probabilities;
these are regression intervals, not calibrated binomial probability intervals.
Only EUV retains the fixed dose basis. Shape predictions use the Kumo median.
Expected improvement, shape thresholds, simulator, and soft yield objective
are unchanged. `summary.json` identifies the effective response backends,
history policy, uncertainty approximation, and whether Kumo actually ran.

Candidate count is a computational search budget, not extra measurements.
Start with 1,000; compare against 5,000 across matched seeds if runtime permits.
There is no established sufficient count for this nine-dimensional search.
A short GPU pilot verifies memory and timing, but should use the same total
measurement budget for GP and Kumo: changing `n` changes the initial Latin
hypercube design, even with the same seed. A 200-measurement run has at most
173 model-guided evaluations after the initial 27, and possibly fewer if the
yield readiness check or candidate feasibility checks delay selection.
Validate GPU memory/runtime on Colab; local adapter tests do not establish
Large-model capacity, calibration, or optimization performance.

## Run the fine-tuned Kumo Large checkpoint

The notebook defaults to pretrained Kumo Large with `KUMO_CHECKPOINT = None`. To run the fine-tuned model, set `KUMO_CHECKPOINT = "data/lr1e-05_T1000_tcont128_tquery512_vcont256_vquery82_checkpoint.pt"`. It preserves the comparison settings: 200 measurements, 1,000 candidates, both seeds 0, full history, all five Kumo responses, Large on CUDA, four estimators, and batch size 256. For Colab, re-upload the updated complete `src/` folder. For fine-tuned runs, upload `data/checkpoint_upload_parts/` to the same runtime; the preparation cell reassembles the 16 MB parts and verifies size and SHA-256. A checkpoint already present in `/content/data/` or `/content/ASML/data/` is also accepted. The checkpoint is not included in the repository.

Run notebook cells in order. A missing checkpoint produces an error. Set `KUMO_CHECKPOINT = None` in the configuration cell to compare against original pretrained Kumo Large; keep every other setting fixed. For a GP baseline also set `EUV_MODEL = "gp"`.

Both CLI launchers accept `--kumo-checkpoint PATH`. The checkpoint loads strictly into the selected architecture without downloading pretrained weights, and all response adapters share that loaded model. The recorded runner saves the checkpoint path, SHA-256, and whether Kumo actually ran in `summary.json`. Run folder names in the notebook distinguish fine-tuned, pretrained, and GP runs. Training provenance is unknown; existing EUV residual and logit-yield transforms remain in use.
