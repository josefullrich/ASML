"""bo_bench: closed-loop Bayesian-optimisation benchmark for TFM vs GP surrogates.

Layout
------
config.py        every knob (arms, seeds, budgets, normalisation toggles)
problems/        black boxes behind one interface (srcmdl, BOCoDe, add your own)
surrogates/      model backends (BoTorch GP, Kumo, Causilo, CPU stand-in)
yield_arms.py    the yield-model arms (logit regressor, classifier and its fixes)
bb.py            beta-binomial maths and the context "quiz"
acquisition.py   the two acquisition functions
loop.py          resumable BO loop with a wall-clock budget
predbench.py     prediction-only benchmark on fixed data (no acquisition)
metrics.py       every metric shown in the report
report.py        builds the self-contained interactive HTML
run.py           command line entry point
"""
__version__ = "0.1.0"
