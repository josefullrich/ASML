# TFM vs GP surrogates for EUV source BO

Can a tabular foundation model (Causilo, Kumo, TabPFN) replace the Gaussian-process surrogates in ASML's Bayesian
optimization of EUV source setpoints? This code replays the real BO runs and predicts each measured output one step
ahead with every model: die yield (the binomial one), and the regressor outputs EUV_mJ, Rx_deg, Ry_deg and
TargetSize_um. Then it scores the predictions.

> TabPFN ≥ 2.5 and Causilo weights are licensed for non-commercial use only. The data are real ASML BO runs: keep the
> repo private, and keep `data/` and `results/` out of git.

## Where things live (all paths relative, so any clone works)
```
ASML/
├── notebooks/
│   ├── tfm_01_run_benchmark.ipynb     run GPs + TFMs, build the pages (GPU)
│   ├── tfm_02_rebuild_reports.ipynb   rebuild the pages from any state zip (CPU)
│   ├── tfm_03_classifier_suite.ipynb  Causilo-classifier experiment (GPU)
│   └── tfm_bench/                     the code the notebooks call (this README)
├── data/                              opt_results_YYYYMMDD_HHMMSS.csv, searched recursively   (local, not in git)
└── results/tfm_bench/                 everything the runs produce                              (local, not in git)
    ├── benchmark_state_slim.zip       data + every die-yield prediction behind the current results
    ├── results/  results_EUV_mJ/ …    predictions per output / config / dataset
    ├── avg_everything/                the three pages + tables
    ├── clf_suite/                     classifier suite predictions + page
    └── tfm_bench_rebuild/<zip>/       (one level up) scratch rebuilds from notebook TFM 2
```
The notebooks run from `notebooks/` and use `../data`, `../results/tfm_bench`, and `tfm_bench/*.py`.

- **Locally** (Jupyter / VS Code): open the notebook in `notebooks/` and run it. Set keys as environment variables.
- **On Colab:** the notebook clones the repo to `/content/ASML` and links `data/` and `results/` to a Google Drive
  folder (`DRIVE_ROOT`, default `MyDrive/ASML`), so the same relative paths work and results survive disconnects.
  Copy your local `data/` and `results/tfm_bench/benchmark_state_slim.zip` to that Drive folder once. If `data/` has
  no CSVs, the first run restores them from the state zip.

## The three pages
| Page | Script | What it compares | Normalized? |
|---|---|---|---|
| `binomial_comparison.html` | `binomial_report.py` | Die yield as a distribution over 0…70 good dies: every GP and TFM, the raw Causilo classifier, and the fixes tried on it (beta-binomial φ, Platt logit a, b, both) | **No.** Every config predicts the same counts |
| `targets_comparison.html` | `targets_report.py` | Regressor outputs EUV, Rx, Ry, TargetSize (+ logit yield): GP BoTorch and the TFM regressors | Only to combine outputs; the per-output tables are raw |
| `summary.html` | `analyze.py` | The original die-yield summary: curves over the run, reliability by context arm, flagged errors | Overall rank uses per-dataset ranks |

All three are self-contained (Plotly is embedded from `vendor/`), so they open in any browser, offline too.

## Code (`notebooks/tfm_bench/`)
| File | Purpose |
|---|---|
| `config.py` | Every setting: paths, target (`BENCH_TARGET`), datasets, protocol, context selection, normalization |
| `common.py` | Data loading, the target, context selection, the one-step-ahead loop |
| `gp_laplace.py` | Python port of MATLAB `fitcgp`/`ClassificationGP` (binomial Laplace GP, ARD Matérn 5/2) |
| `models.py` | Adapters: `gpr_laplace`, `gpr_botorch`, `causilo`, `causilo_trials`, `kumo` / `kumo_medium` / `kumo_small`, `tabpfn2`, `tabpfn35`, `mock` (CPU stand-in) |
| `metrics.py`, `metrics_cont.py` | Die-yield metrics / continuous-output metrics |
| `run_gpr.py`, `run_tfm.py` | Run the GPs / TFMs: `--target --models --arms --datasets --skip-existing --max-steps` |
| `binomial_report.py`, `targets_report.py`, `analyze.py` (+ `*_template.html`) | Build the three pages |
| `classifier_suite.py` | The classifier experiment and its page |
| `htmlutil.py`, `vendor/` | Embed Plotly in the pages |
| `requirements-tfm.txt` | Model library versions |

Run a script by hand from `notebooks/`, e.g. `python tfm_bench/run_tfm.py --target EUV_mJ --models tabpfn35 --arms ctx0`.

## Protocol
At step t the model sees rows 1…t−1 of the current day and predicts row t. The first 27 rows are warm-up. Every
model refits from scratch at every step. For die yield the target is `y = logit((k+0.5)/(n+1))` with n = 70
exposures; the regressor outputs are modelled in their raw units. Each output has its own folder under
`results/tfm_bench/`: `results/` for die yield, `results_EUV_mJ/`, `results_Rx_deg/` and so on for the others. Pick one with
`--target` or the env var `BENCH_TARGET`. `gpr_laplace` and `causilo_trials` are binomial, so they're yield-only.
Context days are chosen from the yield data, so every output uses the same days.

Context arms (TFMs only; the GPs never see other days):
- `ctx0`: zero-shot
- `ctx5`: the 5 most different other days, prepended
- `ctxall`: all other days

Every model outputs 101 quantiles (on the target's scale) plus a mean.

Data handling: four CSVs are byte-identical copies, so they're dropped (`DEDUPLICATE`), leaving 16 days. The
validation tail after BO ends (n = 420 re-measurements and the open-loop row) is also cut (`TRUNCATE_AT_BO_END`).

## Metric glossary (die-yield `summary.html`, `metrics.py`)
"Per dataset" means a metric is computed on one day's ~297 predictions, then averaged over days. The **Last 20%**
tab repeats everything on the end of each run, where BO is near the optimum. ★ marks the metrics that enter the
default ranking.

### Uncertainty (is the predicted distribution honest?)
| Metric | Meaning | Better |
|---|---|---|
| ★ `crps` | Continuous ranked probability score: the distance between the predicted CDF and the truth, in logit units. Rewards being both right and sharp. Computed on the 101-quantile grid (2 × mean pinball loss). | lower |
| ★ `coverage90` | Fraction of truths inside the 5–95% interval. | 0.90 |
| ★ `width90` | Mean width of that interval. Only meaningful when coverage is right. | lower |
| ★ `calib_error` | Mean \|observed − nominal\| of the PIT over levels 5–95% (the reliability-diagram gap). | lower |
| `coverage_curve_error` | Mean \|coverage − level\| over central intervals at 10–99%. | lower |
| `width_ratio90`, `width_error` | Actual width ÷ the width this model *needed* to cover 90% of its own errors (1 = right). `width_error` = mean \|log₂ ratio\| over levels. | 1 / lower |
| `width_adaptivity` | Spearman correlation between interval width and \|error\|: does it widen where it's actually wrong? | higher |
| `crps_over_noise` | CRPS ÷ the CRPS a perfect predictive distribution would still get from binomial counting noise. 1 = at the physics floor. | lower |

### Ranking (does it order setpoints right? This is what BO actually uses.)
| Metric | Meaning | Better |
|---|---|---|
| ★ `kendall_tau` | Kendall's τ-b between predicted median and truth. τ-b because many rows tie at k = 0 or 70. | higher |
| ★ `top10_recall` | Share of the true top-10% rows that the model also puts in its top 10%. | higher |
| `tau_ceiling`, `top10_ceiling` | The best a *perfect* model could score, given counting noise and ties (simulated). Same for every model. | info |
| `tau_frac`, `top10_frac` | Score ÷ ceiling. | higher |

### Point (is the central prediction right?)
| Metric | Meaning | Better |
|---|---|---|
| ★ `binom_nll` | Binomial negative log-likelihood of the measured k of 70, given the predicted mean yield. | lower |
| ★ `rmse`, ★ `mae` | Error of the mean (RMSE) or median (MAE), logit units. | lower |
| ★ `rmse_mae_ratio` | RMSE ÷ MAE. About 1.25 for Gaussian errors; much higher means a few huge misses. | 1.25 |
| `nrmse` | RMSE ÷ SD of the truth (last 20% only). | lower |
| `noise_rmse` | RMSE that pure counting noise alone would produce. Same for every model. | info |
| `rmse_over_noise`, `mae_over_noise` | Error ÷ that noise floor. 1 = as good as the measurement allows. | lower |
| `rmse_noise_corrected` | √(RMSE² − noise²): estimated error against the *true* yield. | lower |
| `within_noise2` | Share of predictions within 2 noise SDs of the truth. | 0.95 |
| `nll_excess` | Mean binomial NLL − the **binomial entropy floor**, which is the NLL a model that knew the true yield would still average. 0 = at the floor. | lower |
| `overdispersion` (`_top`: rows with p > 0.95) | Mean deviance ÷ the deviance expected if the predicted p were exactly right. 1 = the misses are pure counting luck; > 1 = the model misses real structure or is overconfident. | 1 |

## Die yield as counts (`binomial_comparison.html`): no normalization
Every config is turned into a probability for each count k = 0…70, then scored on the same rows in the same units:
- Regressors on the logit target (Causilo, Kumo, TabPFN, GP BoTorch with observation noise): the quantile function is
  binned between the 71 possible target values.
- GP Laplace: its predictive is the *latent* yield, with no counting noise. It becomes the binomial mixture
  ∫ N(f) · Binomial(70, σ(f)) df (Gauss–Hermite). Binning it directly would make it look far too confident.
- Causilo classifier: p is recovered from the saved mean and used as Binomial(70, p). The fixes are applied to those
  same p's:
  - `_bb`: beta-binomial φ
  - `_calib`: Platt p' = σ(a·logit p + b)
  - `_calib_bb`: both

  a, b and φ are fitted by maximum likelihood on the config's **own earlier predictions that day**: at step t only
  steps < t are used, and the raw binomial is used for the first 20 steps. The page plots how a, b and φ evolve. (The
  classifier suite fits them differently, on held-out folds of the context, and also has the methods that need refits.)

| Metric | Meaning | Better |
|---|---|---|
| Log score | −log P(the count that happened) | lower |
| Log score − entropy | The log score minus the binomial entropy at that count (what a model that knew the true yield would still average). Same ranking as the log score; shows the room left | lower, 0 = floor |
| CRPS | Σₖ (F(k) − 1[k ≥ k_true])², in counts | lower |
| Coverage 90% (PIT) | The truth's PIT, spread over [F(k−1), F(k)] and computed exactly, falls in the central 90%. A calibrated count model scores exactly 0.90 | 0.90 |
| Coverage error | Mean \|PIT coverage − level\| over 10…99%, i.e. confidence at every level | lower |
| Coverage 90% (interval) | Truth inside the central 90% count interval, endpoints included. Reads high for every model on counts (the 0/70 and 70/70 spikes), so it's shown for reference | 0.90 |
| Overdispersion | Σ deviance / Σ expected deviance at the predicted *mean* yield. Judges the mean only, so the beta-binomial fix leaves it unchanged; ~13–36 here means the yield swings far beyond any binomial with that mean, which is why a plain binomial fails | 1 |

Results on the current state (whole run, mean over 6 days): the ctxall regressors lead with a log score of about 2.18
and PIT coverage of about 0.90. The raw classifier is badly overconfident (90% coverage 0.10–0.36). Platt alone barely
helps. The beta-binomial fixes most of the coverage (0.86–0.89) but still trails the regressors (log score 2.50 vs
2.18 at ctxall).

## Regressor outputs (`targets_comparison.html`): normalization only across outputs
Metrics (`metrics_cont.py`, all in the output's own units):
- `crps`, `coverage90`, `width90`
- `width_ratio90` and `width_error`: actual width ÷ the width needed to cover the config's own errors
- `calib_error` (reliability), `coverage_curve_error`, `width_adaptivity`
- `kendall_tau`, `top10_recall`
- `rmse`, `mae`, `nrmse`

Within one output these compare directly, so the page's raw table has no normalization. Ranking **across** outputs
works like this:
1. Each ticked metric becomes a badness b ≥ 0 (lower = better): \|coverage − 0.9\|, \|log₂ width ratio\|, 1 − τ, …
2. b is rescaled across configs per output and dataset, with the method picked in the page (or `--norm`):

   | Option | Formula | Note |
   |---|---|---|
   | `none` | — | No cross-output ranking; raw tables only |
   | `robust_z` (default) | (b − median)/(1.4826·MAD) | One broken config can't stretch the scale |
   | `z` | (b − mean)/SD | One very bad config inflates the SD |
   | `skill` | log₂(b / b_ref), clipped ±3 | 0 = same as the reference (GP BoTorch), −1 = 2× better |
   | `rank` | (rank − 1)/(n − 1) | Order only |
   | `minmax` | (b − min)/(max − min) | Most outlier-sensitive |

3. Average over datasets → metrics → categories to get the output's score. Overall = the mean over the chosen outputs,
   only for configs that have all of them.

The page also has PIT reliability, coverage and width-ratio curves per output. `targets_rankings_all_norms.csv` has
every normalization, so you can check a conclusion doesn't depend on the choice.

## Classifier suite (`tfm_03_classifier_suite.ipynb`, `notebooks/tfm_bench/classifier_suite.py`)
Is Causilo better as a binomial classifier than as a regressor on logit yield? Each method gives a full distribution
over 0…70 good dies for every row, one step ahead, over several seeds:

| Method | What it does |
|---|---|
| `trials` | Classifier p → Binomial(70, p) |
| `trials_bb` | Classifier p → beta-binomial with a φ learned on held-out folds |
| `calib` | Platt-sharpened p → binomial |
| `calib_bb` | Platt-sharpened p → beta-binomial |
| `no_mid` | Trained on 0/70 and 70/70 rows only |
| `multiclass_71`, `multiclass_4bin` | Causilo predicts the count, or a count bin, directly |
| `regressor` | CausiloRegressor quantiles → counts (the baseline to beat) |

**Scored only with metrics that check the whole distribution against what was measured, or against the data's own
noise:**
- **Log score**: −log P(the count that happened)
- **Log score − entropy floor**: the same minus the binomial entropy at that count. This gives the same ranking as
  the log score, because the floor is the same for every method on the same rows; it just shows how far from
  "physics-limited" each method is.
- **CRPS**: over counts
- **Coverage 90%**, and **coverage error**: mean \|coverage − level\| over 50–98%
- **Overdispersion**: as above

Interval width, width ratio, "needed width", Brier, MAE, P(extreme) and the bootstrap P(best) were removed. A
classifier outputs one p, so any spread comes from our conversion step (binomial / beta-binomial φ). Comparing widths
would grade the adapter, not Causilo. Coverage, CRPS and log score still penalize a made-up spread, because they're
scored against reality.

## Licences and versions
The saved results were made with causilo 1.0.3, structured-data-models (Kumo) 0.2.1, tabpfn 9.2.0, botorch 0.18.1 and
gpytorch 1.15.2 (`results/tfm_bench/results/versions_*.json`). Notebook TFM 1 pins these (`requirements-tfm.txt`). `causilo_trials` and the classifier suite need
the Causilo PR #19 build (commit `06b3fd0`).
