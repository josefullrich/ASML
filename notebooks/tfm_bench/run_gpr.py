"""Run both GP surrogates (gpr_laplace, gpr_botorch) on every dataset with the one-step-ahead protocol.

Examples
    python run_gpr.py                                   # both GPs, all datasets
    python run_gpr.py --target EUV_mJ                   # gpr_botorch on EUV (gpr_laplace is yield-only)
    python run_gpr.py --models gpr_laplace --datasets 20261001_161410
    python run_gpr.py --max-steps 30                    # quick smoke test
GPs start from zero data every dataset and never see other days.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="all", help="comma-separated dataset IDs (default all)")
    ap.add_argument("--models", default="all", help="comma-separated subset of " + ",".join(C.GP_MODELS))
    ap.add_argument("--max-steps", type=int, default=None, help="limit predicted steps per dataset (smoke tests)")
    ap.add_argument("--skip-existing", action="store_true", help="skip datasets whose predictions.csv exists")
    ap.add_argument("--target", default=C.TARGET, choices=list(C.TARGETS), help="what to predict (default dy)")
    args = ap.parse_args()

    datasets = common.parse_list(args.datasets, C.DATASETS)
    default = C.GP_MODELS if C.TARGET == "dy" else [g for g in C.GP_MODELS if g not in C.YIELD_ONLY_MODELS]
    gp_models = common.parse_list(args.models, default)
    if C.TARGET != "dy" and set(gp_models) & C.YIELD_ONLY_MODELS:
        raise SystemExit(f"{set(gp_models) & C.YIELD_ONLY_MODELS} only models die yield; drop it for --target {C.TARGET}")
    print(f"target: {C.TARGET} ({C.TARGETS[C.TARGET]}) -> {C.RESULTS_DIR}")
    bad = set(gp_models) - set(C.GP_MODELS)
    if bad:
        raise SystemExit(f"unknown GP model(s): {bad}")
    np.random.seed(C.SEED)
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(C.RESULTS_DIR, "versions_gpr.json"), "w") as fh:
        json.dump(models.package_versions(), fh, indent=2)

    for model in gp_models:
        predictor = models.FACTORIES[model]()
        for ds in datasets:
            out = os.path.join(common.result_dir(model, ds), "predictions.csv")
            if args.skip_existing and os.path.exists(out):
                print(f"skip {model} {ds}")
                continue
            print(f"== {model} | {ds}", flush=True)
            df = common.run_protocol(ds, model, model, "none", predictor, [], max_steps=args.max_steps)
            common.write_metrics(df, model, ds)


if __name__ == "__main__":
    main()
