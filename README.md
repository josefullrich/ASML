# ASML EUV optimization project

This workspace is for understanding and running the supplied EUV source simulator,
reproducing its optimization example, and later comparing alternative models.

## Folder guide

```text
ASML/
├── README.md          Project guide and setup status
├── docs/              Project briefs, slides, and research reports
├── data/              Original 20 CSV measurement logs
├── src/               Python simulator, optimizer, and our future Python code
├── matlab/            MATLAB code, including evaluate_model.m
├── notebooks/         Guided notebooks, including future Colab setup
└── results/           Generated plots, tables, and experiment logs
```

Original downloads remain available in Downloads. The three project PDFs are
in `docs/`. Python files live together in `src/` so their existing imports work.
The MATLAB evaluation script reads CSVs from the sibling `data/` folder.

Put new experiment logic in `src/`; use notebooks to configure and explain runs.
Store outputs in a separate folder for each run under `results/`, recording the
settings, random seeds, package versions, hardware, and elapsed time alongside
the results. A first Colab notebook is available in
`notebooks/01_colab_smoke_test.ipynb`.

## What we have inspected

- `src/srcmdl.py`: source simulator, including state and drift.
- `src/run_srcmdl.py`: repeated-measurement demonstration.
- `src/example_bayesopt.py`: Gaussian-process Bayesian
  optimization example with selectable die-yield models.
- `src/dy_models.py`: supporting die-yield models.
- `matlab/evaluate_model.m`: offline surrogate evaluation on a CSV
  log. Its qualification set includes training and future rows, as documented in
  the script; its scores should not be described as held-out predictive accuracy.
- 20 CSV files: 322–338 measurements per file, with 17 columns each.

The supplied Python optimizer uses **nine settings**, including dose target.
Earlier project descriptions mention seven; use the actual code interface when
setting up runs, and reconcile this difference before defining benchmarks.

## Setup status

The project has an isolated Python 3.11 environment in `.venv/`, with NumPy,
SciPy, Matplotlib, scikit-learn, statsmodels, and pandas installed. Installed
versions are recorded in `requirements.txt`; the package consistency check passes.
All four Python modules import successfully, including the highlighted
`GLMResultsWrapper` import. A 10-measurement simulator run passed finite-output
and yield-range checks. These are basic execution checks, not MATLAB equivalence
or model-accuracy validation.
A reduced optimization run has now completed: 60 measurements, including 33
optimizer-selected measurements, using logit GP yield regression and 2,000
candidates per step. Finite-value, input-range, yield-range, and exposure-count
checks passed. This checks execution; full-run behavior and MATLAB equivalence
remain unvalidated.

For VS Code, open the ASML folder. If imports remain underlined, open the command
palette (Cmd+Shift+P), choose **Python: Select Interpreter**, and select
`ASML/.venv/bin/python`. A default interpreter setting is included in
`.vscode/settings.json`, but an existing editor selection may take precedence.

To recreate the environment from the ASML folder:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The measurement demo defaults to 100,000 measurements. Start with a small run
once dependencies are installed, rather than launching that default:

```bash
.venv/bin/python src/run_srcmdl.py --n 10 --no-plots
```

## Recorded optimization run

Run the small experiment and save its outputs without opening plot windows:

```bash
.venv/bin/python src/run_optimization.py
```

The default is 60 measurements (27 initial-design points followed by model-guided
selection when feasible), design seed 0, source seed 0, logit GP yield model,
and the supplied logit yield penalty. Source drift is reset after every
measurement, as in the supplied optimizer; other source history still advances.
Use `--help` for overrides. `--output` must name a new directory.

Each run saves `overview.png` (all nine settings, EUV, and yield),
`measurements.csv` (settings, responses, selection flags, and feasibility flags),
`summary.json` (configuration, seeds, versions, source hashes, hardware, runtime,
and checks), and `run.log` in a new subfolder of `results/`.
The vertical dotted line marks the first optimizer-selected measurement.

The first run is in `results/optimization_20261108T185437910534Z/`.
Of 60 measurements, 46 met the measured shape limits; 11 also met the observed
99.5% yield threshold, while 6 met the underlying simulated yield threshold.
The highest EUV among shape-feasible points with underlying yield at least
99.5% was 5.226 mJ (measurement 44, underlying yield 99.789%). The largest
shape-feasible EUV overall was 5.499 mJ, but its underlying yield was only 59.31%.

Yield is a soft objective penalty in the supplied optimizer, so its chosen
incumbent is not guaranteed to meet the yield requirement. With only 71
exposures, a measured 100% yield also does not establish an underlying yield
of at least 99.5%. Simulation-only actual yield is recorded for diagnosis and
does not enter model fitting or acquisition. Final-model predictions in the CSV
are retrospective and are not held-out accuracy estimates. This one short run
does not establish optimization superiority or convergence.

Next: implement the intended hard yield constraint, then define a repeated-seed
benchmark before comparing surrogates.

## First Colab test from VS Code

A notebook (`.ipynb`) contains explanatory text and individually runnable Python
cells. Here it configures and calls the existing scripts, then displays their
saved results. VS Code is the editor; the selected kernel determines whether
Python executes locally or on a Colab machine. The Python source stays in `src/`.

1. Install Google's official [Colab extension](https://marketplace.visualstudio.com/items?itemName=google.colab)
   in VS Code, plus its Jupyter dependency if prompted.
2. Open `notebooks/01_colab_smoke_test.ipynb`.
3. Choose **Select Kernel → Colab → Auto Connect** and sign in with your Google
   account. Use a free CPU runtime; the existing GP implementation uses CPU.
4. Right-click the local `src` folder in VS Code's Explorer and choose
   **Upload to Colab**. This puts the source files on the remote machine.
5. Run the notebook cells from top to bottom. The default test uses 40
   measurements and 1,000 candidates per step. It displays the plot and the
   highest-EUV tested setting satisfying the simulation diagnostic criteria,
   or reports that none qualifies.
6. In the final code cell, set `SAVE_TO_DRIVE = True` and run it to authorize
   Drive mounting and save a results ZIP under `My Drive/ASML/results/`.
   Download that ZIP to your Mac's `results/` folder when desired. Save the
   notebook itself with Cmd+S.

The remote runtime cannot read your Mac's `/Users/...` paths or `.venv`.
Connecting a notebook does not upload its sibling source files or synchronize
results. Re-upload `src/` after local source changes. Colab runs use their own
installed library versions, recorded in `summary.json`; the notebook installs
only the six required scientific libraries without forcing the local version
pins. Outputs can differ across package versions and platforms.

The notebook also includes instructions for using the Colab website instead.
No historical CSV files are required for this simulator test. The current
optimizer still uses a soft yield penalty; this notebook tests cloud execution
and does not change its objective.

Official references: [Colab VS Code guide](https://github.com/googlecolab/colab-vscode/wiki/User-Guide)
for connection, uploads, and Drive mounting; [Colab FAQ](https://research.google.com/colaboratory/faq.html)
for free resource limits and temporary runtime storage. A free-account Colab run has completed successfully through VS Code: 40
measurements, 13 optimizer-selected measurements, and a separate CUDA matrix
calculation on a Tesla T4. The baseline GP itself used CPU. Runtime access and
GPU availability still depend on the individual account.

## Version control

The project uses a `main` branch. Work in a branch for each change, push it, and
open a pull request for a teammate to review before merging. See
[CONTRIBUTING.md](CONTRIBUTING.md) for the local-copy workflow.

The `.gitignore` excludes datasets, project PDFs, generated results, local Python
environments, and secrets. These files remain on the original computer and will
not appear when teammates clone the repository. Obtain `data/` and `docs/`
separately through the team's agreed shared location; the simulator notebook
does not require either folder. `results/` is created by the experiment runner.
Notebook outputs are cleared before committing; keep executed copies and plots
under the ignored `results/` folder. The original executed Colab notebook has
been preserved locally under `results/notebook_snapshots/`.

After cloning, create your own `.venv` using the setup instructions above.
Cloning does not copy another teammate's Python environment. Keep this GitHub
repository private when sharing project code with invited teammates.
