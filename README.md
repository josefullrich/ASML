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
├── data/              Historical measurement logs
├── docs/              Project briefs and research reports
├── results/           Experiment outputs and plots
└── requirements.txt   Python dependencies
```

`data/`, `docs/`, and `results/` are kept locally and are not included in this
repository.
