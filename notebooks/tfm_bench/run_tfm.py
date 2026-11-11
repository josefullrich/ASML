"""Run tabular foundation models with the one-step-ahead protocol.

Set MODEL / ARM below, or override on the command line. The output folder is chosen automatically
(results/<model>_<arm>/results_<dataset>/).

Examples
    python run_tfm.py                                       # uses MODEL and ARM below
    python run_tfm.py --models tabpfn35 --arms ctx0,ctx5
    python run_tfm.py --models all --arms all               # every model x arm combination (18 configs)
    python run_tfm.py --models kumo,kumo_medium,kumo_small --arms all   # Kumo size ablation
    python run_tfm.py --models mock --arms all --max-steps 20   # CPU stand-in, pipeline test only
    python run_tfm.py --models all --arms all --target Rx_deg   # a regressor output instead of die yield

Context days come from context_selection.csv (computed once, shared by every TFM).
"""
from __future__ import annotations

import os
import sys

if "--target" in sys.argv:  # must be set before config is imported
    os.environ["BENCH_TARGET"] = sys.argv[sys.argv.index("--target") + 1]

import argparse
import json

import numpy as np

import common
import config as C
import models

MODEL = "tabpfn35"  # causilo | kumo (= large) | kumo_medium | kumo_small | tabpfn2 | tabpfn35
ARM = "ctx0"        # ctx0 | ctx5 | ctxall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="all")
    ap.add_argument("--models", default=MODEL, help="comma list, or 'all' for " + ",".join(C.TFM_MODELS))
    ap.add_argument("--arms", default=ARM, help="comma list, or 'all' for " + ",".join(C.ARMS))
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--recompute-context", action="store_true", help="rebuild context_selection.csv")
    ap.add_argument("--target", default=C.TARGET, choices=list(C.TARGETS), help="what to predict (default dy)")
    args = ap.parse_args()

    datasets = common.parse_list(args.datasets, C.DATASETS)
    tfms = common.parse_list(args.models, C.TFM_MODELS)
    arms = common.parse_list(args.arms, C.ARMS)
    bad = set(tfms) - set(C.ALL_TFM_MODELS + ["mock"])
    if bad or set(arms) - set(C.ARMS):
        raise SystemExit(f"unknown model/arm: {bad or set(arms) - set(C.ARMS)}")
    if C.TARGET != "dy" and set(tfms) & C.YIELD_ONLY_MODELS:
        raise SystemExit(f"{set(tfms) & C.YIELD_ONLY_MODELS} only model die yield; drop it for --target {C.TARGET}")
    print(f"target: {C.TARGET} ({C.TARGETS[C.TARGET]}) -> {C.RESULTS_DIR}")
    np.random.seed(C.SEED)
    sel = common.load_context_selection(recompute=args.recompute_context)
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(C.RESULTS_DIR, "versions_tfm.json"), "w") as fh:
        json.dump(models.package_versions(), fh, indent=2)

    for model in tfms:
        predictor = models.FACTORIES[model]()  # weights load once per model
        for arm in arms:
            folder = f"{model}_{arm}"
            for ds in datasets:
                out = os.path.join(common.result_dir(folder, ds), "predictions.csv")
                if args.skip_existing and os.path.exists(out):
                    print(f"skip {folder} {ds}")
                    continue
                ctx = common.context_ids(ds, arm, sel)
                assert ds not in ctx
                print(f"== {folder} | {ds} | {len(ctx)} context days", flush=True)
                df = common.run_protocol(ds, folder, model, arm, predictor, ctx, max_steps=args.max_steps)
                common.write_metrics(df, folder, ds)


if __name__ == "__main__":
    main()
