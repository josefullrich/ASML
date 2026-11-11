# ASML EUV optimization

This project explores ways to maximize EUV output while maintaining a die yield
of at least 99.5%. We use a simulated EUV source to test optimization methods,
starting with a Gaussian-process baseline and later comparing foundation models.
The current baseline uses a yield penalty; a hard yield constraint is still to be added.


## Folder guide

```text
ASML/
├── src/               Python simulator and optimization scripts
├── matlab/            Original MATLAB code
├── notebooks/         Experiment notebooks, including the Colab test
│   └── tfm_bench/     Code for the TFM vs GP surrogate benchmark
├── data/              Historical measurement logs
├── docs/              Project briefs and research reports
├── results/           Experiment outputs and plots
└── requirements.txt   Python dependencies
```

`data/`, `docs/`, and `results/` are kept locally and are not included in this
repository.


## Benchmarks

### TFM vs. GP surrogates (die yield)

We replayed real BO runs and scored one-step-ahead die-yield predictions from two GP
baselines (a Python port of ASML's MATLAB GP, and BoTorch), Causilo, Kumo (small /
medium / large) and TabPFN 2 / 3.5. Each TFM was tested with and without past days as
context. Models were judged on their full predicted distribution, not just the point
estimate.

- **Best models:** TabPFN 3.5, Kumo and Causilo, especially with all other days as
  context. Their 90% prediction intervals held the true value about 90% of the time.
- **Effect of context:** more past days generally improved accuracy, but made
  calibration slightly worse (the predicted spread drifts a little off).
- **vs. GPs:** with past days as context, every TFM beat both GP baselines on most
  metrics. Zero-shot, Causilo and Kumo still beat them; TabPFN 2 and 3.5 fell behind
  GP BoTorch.
- **Caveat:** this measures prediction quality only. It suggests TFMs could help BO,
  but doesn't show better BO performance; that needs a closed-loop test.

**Next steps:** run the same evaluation on the other outputs (EUV, Rx, Ry, TargetSize).
The code already supports them, including their metrics and the cross-output
normalization; only the runs are left to do.

### Causilo as a binomial classifier (die yield)

Die yield is really a count (good dies out of 70), so we also tried Causilo's
classifier, trained on pass/fail counts, instead of a regressor on yield.

- **Out of the box:** badly overconfident. Its 90% intervals held the true count
  only 10–36% of the time.
- **With a beta-binomial correction:** most of that is fixed (86–89%), but it still
  trails the regressors (log score ≈ 2.50 vs ≈ 2.18 for the best regressors, with all
  other days as context).
- **Platt scaling alone** (adding and re-fitting the classifier's logit) barely helped.
- **Takeaway:** the classifier approach can work, but it isn't better yet. Other
  ways of calibrating it are still being tested.

The classifier runs as one more model in the same replay benchmark
(`tfm_01_run_benchmark.ipynb`), and the corrections are compared on
`binomial_comparison.html`. The other classifier variants live in
`tfm_03_classifier_suite.ipynb`.

Notebooks: `notebooks/tfm_01_run_benchmark.ipynb` (run),
`tfm_02_rebuild_reports.ipynb` (rebuild the report pages from saved results, make sure to have a file named "benchmark_state_slim.zip" to use past data),
`tfm_03_classifier_suite.ipynb` (classifier experiments). Details and metric building
definitions: [`notebooks/tfm_bench/README.md`](notebooks/tfm_bench/README.md).
