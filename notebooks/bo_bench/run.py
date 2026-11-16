"""Command line entry point.

    python -m bo_bench.run plan      --set problem=srcmdl
    python -m bo_bench.run estimate                         # time one step per arm, project the total
    python -m bo_bench.run run                              # BO runs; stops and saves at budget_hours
    python -m bo_bench.run pred                             # prediction-only benchmark on fixed data
    python -m bo_bench.run report                           # build results/.../report.html
    python -m bo_bench.run all                              # pred + run + report, within the budget
    python -m bo_bench.run pack   --zip state.zip           # zip the results folder to carry elsewhere
    python -m bo_bench.run unpack --zip state.zip

Any setting in config.py can be overridden:  --set n_iter=350 --set 'seeds=[1,2]'
BOCoDe: --set problem=bocode:WeldedBeamSO, or --suite to loop over the whole ASML-like suite.
"""
from __future__ import annotations
import argparse, os, sys, time, zipfile, copy


def _problems(cfg, suite):
    if not suite:
        return [cfg.problem]
    from .problems.bocode_problem import ASML_LIKE_SUITE
    return [f"bocode:{n}" for n in ASML_LIKE_SUITE]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["plan", "estimate", "run", "pred", "report", "all", "pack", "unpack"])
    ap.add_argument("--config", default=None, help="JSON file of settings")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--suite", action="store_true", help="loop over the BOCoDe ASML-like suite")
    ap.add_argument("--zip", default="bo_bench_state.zip")
    args = ap.parse_args(argv)

    from .config import Config, problem_dir
    cfg = Config.load(args.config, args.set)
    if args.command == "pack":
        with zipfile.ZipFile(args.zip, "w", zipfile.ZIP_DEFLATED) as z:
            for root, _d, files in os.walk(cfg.results_dir):
                for f in files:
                    if not f.endswith(".tmp.npz"):
                        p = os.path.join(root, f)
                        z.write(p, os.path.relpath(p, cfg.results_dir))
        print(f"wrote {args.zip} ({os.path.getsize(args.zip) / 1e6:.1f} MB)")
        return 0
    if args.command == "unpack":
        os.makedirs(cfg.results_dir, exist_ok=True)
        with zipfile.ZipFile(args.zip) as z:
            z.extractall(cfg.results_dir)
        print(f"restored into {cfg.results_dir}")
        return 0

    from .problems import make_problem
    from . import loop
    t_start = time.time()
    deadline = t_start + cfg.budget_hours * 3600
    finished = True
    for name in _problems(cfg, args.suite):
        c = copy.copy(cfg)
        c.problem = name
        c.budget_hours = max((deadline - time.time()) / 3600, 0.0)       # one budget shared by the whole session
        problem = make_problem(c)
        print(f"\n=== {name}: {problem.dim} inputs, {len(problem.outputs)} outputs -> {problem_dir(c)} ===", flush=True)
        if args.command == "plan":
            runs = loop.plan(c, problem)
            for r in runs:
                print("  ", loop.run_key(r))
            print(f"{len(runs)} runs x {loop.n_iter_for(c)} evaluations")
        elif args.command == "estimate":
            loop.run_all(c, problem, estimate=True)
        if args.command in ("pred", "all"):
            from .predbench import run_pred
            finished &= bool(run_pred(c, problem, deadline))
        if args.command in ("run", "all") and time.time() < deadline:
            finished &= bool(loop.run_all(c, problem))
        if args.command in ("report", "all", "run", "pred"):
            from .report import build
            try:
                out, _ = build(c, problem)
                print(f"report: {out}")
            except Exception as err:                                      # a report must never lose a run
                print(f"report not built ({type(err).__name__}: {err})")
        if time.time() > deadline:
            print("\nSession budget used. Re-run the same command to continue.")
            finished = False
            break
    return 0 if finished else 3


if __name__ == "__main__":
    sys.exit(main())
