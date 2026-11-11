"""Classifier suite: every Causilo-classifier method, on ALL rows of the chosen datasets, over several seeds,
split across every visible GPU (Colab: 1), with an interactive HTML report.

    python classifier_suite.py            # predict + build the report
    python classifier_suite.py --report   # only rebuild the report from saved predictions (no GPU)
Settings: edit SETTINGS below, or pass them as JSON in the env var CLF_SETTINGS (the Colab notebook does this).
Needs the PR #19 causilo build (commit 06b3fd0, `sample_weight`).

Scored ONLY with metrics that judge the whole predicted distribution against the measured count, or the data's own
noise: log score, CRPS, coverage (at 50-98%), NLL excess over the binomial entropy floor, and overdispersion.
Interval width / width ratio / "needed width" were removed on purpose: a classifier outputs one p, so any width is
built by our conversion step (binomial / beta-binomial phi), and comparing widths would grade the adapter, not
Causilo. Coverage, CRPS and log score still penalise a fake spread, because they are scored against what happened.

What changes per seed: Causilo's random_state (its ensemble shuffles) and the quiz's fold split.
The query rows do NOT change per seed: every row from MIN_CTX on is predicted (or every QUERY_STRIDE-th),
so seed-to-seed differences are model randomness only, never "different rows were sampled".
Each row is predicted one step ahead: the context is only the rows measured before it.

Methods (each yields a full predicted distribution over 0..70 good shots):
  trials          classifier p, Binomial(70, p)                          (the original causilo_trials)
  trials_bb       classifier p, beta-binomial with quiz-learned phi
  calib           quiz-sharpened p (a*logit p + b), binomial
  calib_bb        quiz-sharpened p, beta-binomial
  no_mid          classifier trained only on 0/70 and 70/70 context rows, binomial
  multiclass_71   classifier with label = count k (one row per setting): Causilo outputs P(0)..P(70) itself
  multiclass_4bin classifier with 4 labels: 0 | 1-34 | 35-69 | 70; inner bins spread evenly over their counts
  regressor       CausiloRegressor on logit yield, its 99 quantiles turned into the same 0..70 distribution
Every distribution gets the same tiny floor (EPS spread over all 71 counts) so an unseen count isn't log(0).
"""
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
import glob, json, subprocess, sys, time, warnings, zlib
import numpy as np
import pandas as pd
from scipy import optimize, special, stats

warnings.filterwarnings("ignore")

# ======================================================================== SETTINGS
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # <repo>/notebooks/tfm_bench/..
DATA_GLOB = os.path.join(_REPO, "data", "**", "opt_results_*.csv")
OUT_DIR = os.path.join(_REPO, "results", "tfm_bench", "clf_suite")
DATASETS = "auto:3"      # "auto:N" = the N datasets with the most 0%/middle/100% rows, "all", or a list of file names
SEEDS = [0, 1, 2, 3, 4]  # 5 seeds; add more for tighter seed spreads (cost grows linearly)
MIN_CTX = 40             # first predicted row; earlier rows are context only
QUERY_STRIDE = 1         # 1 = predict every row; 2 = every other row (half the cost)
METHODS = ["trials", "trials_bb", "calib", "calib_bb", "no_mid", "multiclass_71", "multiclass_4bin", "regressor"]
FOLDS = 4
EPS = 1e-3               # probability floor, spread evenly over the 71 counts, same for every method
TIME_BUDGET_H = 11.0     # each GPU worker stops cleanly before this; rerun to resume
N_EXP = 70
FEATURES = ["PP2DX_um", "PP2DY_um", "PP2DZ_um", "MP2PPDelay_ns", "PPE_mJ", "MMPED_mJ", "FFWPED", "MPBS_um",
            "DoseTarget_mJ"]
LEVELS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98]
REG_LEVELS = np.linspace(0.01, 0.99, 99)
BINS4 = [(0, 0), (1, 34), (35, 69), (70, 70)]
# ==================================================================================
for _k, _v in json.loads(os.environ.get("CLF_SETTINGS", "{}")).items():   # notebook overrides, e.g. {"SEEDS": [0, 1]}
    assert _k in globals() and _k.isupper(), f"unknown setting {_k}"
    globals()[_k] = _v

K_ALL = np.arange(N_EXP + 1)
T_ALL = np.log((K_ALL + 0.5) / (N_EXP - K_ALL + 0.5))
T_EDGES = (T_ALL[1:] + T_ALL[:-1]) / 2
PART_DIR = f"{OUT_DIR}/parts"


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def stratum(k):
    return 0 if k == 0 else (2 if k == N_EXP else 1)          # 0 = "0%", 1 = middle, 2 = "100%"


# ------------------------------------------------------------------ data
def dataset_paths():
    paths = {}
    for p in sorted(glob.glob(DATA_GLOB, recursive=True)):
        paths.setdefault(os.path.basename(p), p)               # same file in several inputs -> keep one
    return paths


def load_df(path):
    df = pd.read_csv(path)
    return df[df.NrOfExposures == N_EXP].head(324).reset_index(drop=True)


def choose_datasets():
    paths = dataset_paths()
    assert paths, f"no CSVs match {DATA_GLOB}"
    if isinstance(DATASETS, (list, tuple)):
        missing = [d for d in DATASETS if d not in paths]
        assert not missing, f"not found: {missing}. Available: {sorted(paths)}"
        return list(DATASETS), paths
    scored = []
    for name, p in paths.items():
        k = load_df(p).NrOfInSpecDose.to_numpy(int)[MIN_CTX:]
        if len(k):
            scored.append((min(np.sum(k == 0), np.sum((k > 0) & (k < N_EXP)), np.sum(k == N_EXP)), name))
    scored.sort(key=lambda s: (-s[0], s[1]))
    names = [n for _, n in scored]
    if DATASETS != "all":
        names = names[: int(str(DATASETS).split(":")[1])]
    return names, paths


# ------------------------------------------------------------------ distributions + scores
def floor(pmf):
    pmf = np.clip(np.asarray(pmf, float), 0, None)
    pmf = pmf / pmf.sum()
    return (1 - EPS) * pmf + EPS / len(pmf)


def pmf_binom(mu):
    return stats.binom.pmf(K_ALL, N_EXP, float(np.clip(mu, 1e-6, 1 - 1e-6)))


def pmf_bb(mu, phi):
    mu = float(np.clip(mu, 1e-6, 1 - 1e-6))
    return stats.betabinom.pmf(K_ALL, N_EXP, mu * phi, (1 - mu) * phi)


def score(pmf, k):
    pmf = floor(pmf)
    cdf = np.cumsum(pmf)
    med = int(np.argmax(cdf >= 0.5))
    cov = []
    for L in LEVELS:
        lo, hi = int(np.argmax(cdf >= (1 - L) / 2)), int(np.argmax(cdf >= (1 + L) / 2))
        cov.append(int(lo <= k <= hi))
    return dict(med=med, mean_k=float(pmf @ K_ALL), nll=float(-np.log(pmf[k])),
                crps=float(np.sum((cdf - (K_ALL >= k)) ** 2)),           # discrete CRPS over counts
                cov=cov)


# ------------------------------------------------------------------ noise-aware (computed in the report from k, mean_k)
_ENT = {}


def entropy_floor(k):
    """Entropy (nats) of Binomial(70, p_hat), p_hat = (k+0.5)/71: the log score a model that knew the true yield
    would average on this row. Same definition as metrics.py in the replay benchmark."""
    if k not in _ENT:
        w = stats.binom.pmf(K_ALL, N_EXP, (k + 0.5) / (N_EXP + 1))
        w = w[w > 0]
        _ENT[k] = float(-(w * np.log(w)).sum())
    return _ENT[k]


def deviance(k, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = np.where(k > 0, k * np.log(np.maximum(k, 1) / (N_EXP * p)), 0.0)
        t2 = np.where(N_EXP - k > 0, (N_EXP - k) * np.log(np.maximum(N_EXP - k, 1) / (N_EXP * (1 - p))), 0.0)
    return 2 * (t1 + t2)


def expected_deviance(p):
    """E[deviance] if K ~ Binomial(70, p): what the deviance averages when the predicted p is exactly right."""
    pmf = stats.binom.pmf(K_ALL[None, :], N_EXP, np.clip(p, 1e-12, 1 - 1e-12)[:, None])
    return (pmf * deviance(K_ALL[None, :], np.asarray(p)[:, None])).sum(1)


# ------------------------------------------------------------------ model pieces (worker side)
CLF = REG = None


def fit_p(X, k, n, Xq, seed):
    Xr = np.repeat(X, 2, axis=0)
    yr = np.tile([1, 0], len(X))
    wr = np.column_stack([k, n - k]).ravel()
    keep = wr > 0
    Xr, yr, wr = Xr[keep], yr[keep], wr[keep]
    if len(np.unique(yr)) < 2:
        return np.full(len(Xq), 1e-4 if yr[0] == 0 else 1 - 1e-4)
    CLF.random_state = seed
    CLF.fit(Xr, yr, sample_weight=wr)
    pr = CLF.predict_proba(Xq)
    return np.clip(pr[:, list(CLF.classes_).index(1)], 1e-6, 1 - 1e-6)


def fit_multiclass(X, labels, Xq, seed, n_classes):
    """Returns probabilities for labels 0..n_classes-1; labels never seen in context get 0 (floored later)."""
    out = np.zeros(n_classes)
    if len(np.unique(labels)) < 2:
        out[labels[0]] = 1.0
        return out
    CLF.random_state = seed
    CLF.fit(X, labels)
    out[np.asarray(CLF.classes_, int)] = CLF.predict_proba(Xq)[0]
    return out


def bb_loglik(z, k, n, a, b, logphi):
    mu = special.expit(a * z + b).clip(1e-6, 1 - 1e-6)
    if logphi is None:
        return np.sum(k * np.log(mu) + (n - k) * np.log1p(-mu))
    phi = np.exp(np.clip(logphi, -4, 9))
    return np.sum(special.betaln(k + mu * phi, n - k + (1 - mu) * phi) - special.betaln(mu * phi, (1 - mu) * phi))


def quiz(X, k, n, seed, rng):
    folds = rng.permutation(len(X)) % FOLDS
    z = np.full(len(X), np.nan)
    for f in range(FOLDS):
        tr, te = folds != f, folds == f
        z[te] = logit(fit_p(X[tr], k[tr], n[tr], X[te], seed=seed * 1000 + 100 + f))
    opts = dict(method="Nelder-Mead", options=dict(maxiter=3000, xatol=1e-4, fatol=1e-6))
    r1 = optimize.minimize(lambda t: -bb_loglik(z, k, n, 1.0, 0.0, t[0]), [np.log(10.0)], **opts)
    r2 = optimize.minimize(lambda t: -bb_loglik(z, k, n, t[0], t[1], None), [1.0, 0.0], **opts)
    r3 = optimize.minimize(lambda t: -bb_loglik(z, k, n, *t), [1.0, 0.0, np.log(10.0)], **opts)
    phi = lambda lp: float(np.exp(np.clip(lp, -4, 9)))
    return dict(trials_bb=(1.0, 0.0, phi(r1.x[0])), calib=(r2.x[0], r2.x[1], None),
                calib_bb=(r3.x[0], r3.x[1], phi(r3.x[2])))


def predict_row(X, k, n, r, seed, rng):
    Xc, kc, nc, Xq = X[:r], k[:r], n[:r], X[r:r + 1]
    res = {}
    p = fit_p(Xc, kc, nc, Xq, seed)[0]
    if "trials" in METHODS:
        res["trials"] = (pmf_binom(p), p, (1.0, 0.0, np.nan))
    if any(m in METHODS for m in ("trials_bb", "calib", "calib_bb")):
        cal = quiz(Xc, kc, nc, seed, rng)
        for m in ("trials_bb", "calib", "calib_bb"):
            if m in METHODS:
                a, b, phi = cal[m]
                mu = float(special.expit(a * logit(p) + b))
                res[m] = (pmf_binom(mu) if phi is None else pmf_bb(mu, phi), mu,
                          (a, b, np.nan if phi is None else phi))
    if "no_mid" in METHODS:
        ext = (kc == 0) | (kc == nc)
        if ext.sum() >= 5 and len(np.unique(kc[ext] > 0)) == 2:
            pn = fit_p(Xc[ext], kc[ext], nc[ext], Xq, seed)[0]
            res["no_mid"] = (pmf_binom(pn), pn, (1.0, 0.0, np.nan))
    if "multiclass_71" in METHODS:
        pm = fit_multiclass(Xc, kc, Xq, seed, N_EXP + 1)
        res["multiclass_71"] = (pm, float(pm @ K_ALL) / N_EXP, (np.nan,) * 3)
    if "multiclass_4bin" in METHODS:
        lab = np.digitize(kc, [1, 35, 70])                    # 0 | 1-34 | 35-69 | 70  ->  0,1,2,3
        pb = fit_multiclass(Xc, lab, Xq, seed, 4)
        pm = np.zeros(N_EXP + 1)
        for (lo, hi), pr in zip(BINS4, pb):
            pm[lo:hi + 1] = pr / (hi - lo + 1)
        res["multiclass_4bin"] = (pm, float(pm @ K_ALL) / N_EXP, (np.nan,) * 3)
    if "regressor" in METHODS:
        REG.random_state = seed
        REG.fit(Xc, T_ALL[kc].astype(np.float32))
        q = np.maximum.accumulate(np.asarray(
            REG.predict(Xq, output_type="quantiles", quantiles=list(REG_LEVELS))[0], float))
        cdf = np.interp(T_EDGES, q, REG_LEVELS, left=0.0, right=1.0)
        pm = np.diff(np.concatenate([[0.0], cdf, [1.0]]))
        res["regressor"] = (pm, float(special.expit(np.interp(0.5, REG_LEVELS, q))), (np.nan,) * 3)
    return res


def done_keys():
    keys = set()
    for f in glob.glob(f"{PART_DIR}/*.csv"):
        d = pd.read_csv(f, usecols=["dataset", "row", "seed"])
        keys |= set(zip(d.dataset, d.row, d.seed))
    return keys


def worker(gpu, jobs):
    """Runs in its own process with CUDA_VISIBLE_DEVICES set to one GPU."""
    global CLF, REG
    from causilo import CausiloClassifier, CausiloRegressor
    CLF, REG = CausiloClassifier(random_state=0), CausiloRegressor(random_state=0)
    os.makedirs(PART_DIR, exist_ok=True)
    out_path = f"{PART_DIR}/gpu{gpu}.csv"
    _, paths = choose_datasets()
    skip = done_keys()
    t0, buf, n_done = time.time(), [], 0
    total = sum(1 for name, seed, rows in jobs for r in rows if (name, r, seed) not in skip)
    print(f"[gpu{gpu}] {total} (row, seed) predictions to do ({len(skip)} already saved)", flush=True)

    def flush():
        if buf:
            pd.DataFrame(buf).to_csv(out_path, mode="a", header=not os.path.exists(out_path), index=False)
            buf.clear()

    for name, seed, rows in jobs:
        df = load_df(paths[name])
        X = df[FEATURES].to_numpy(np.float32)
        k = df.NrOfInSpecDose.to_numpy(int)
        n = df.NrOfExposures.to_numpy(int)
        for r in rows:
            if (name, r, seed) in skip:
                continue
            if time.time() - t0 > TIME_BUDGET_H * 3600:
                flush()
                print(f"[gpu{gpu}] time budget reached; rerun to resume", flush=True)
                return
            rng = np.random.default_rng([seed, r, zlib.crc32(name.encode())])   # same on every process/run
            for m, (pmf, p_used, (a, b, phi)) in predict_row(X, k, n, r, seed, rng).items():
                s = score(pmf, k[r])
                buf.append(dict(dataset=name, row=r, seed=seed, method=m, k=int(k[r]), stratum=stratum(k[r]),
                                p=p_used, a=a, b=b, phi=phi, **{x: s[x] for x in s if x != "cov"},
                                **{f"cov{i}": c for i, c in enumerate(s["cov"])}))
            n_done += 1
            if n_done % 20 == 0:
                el = time.time() - t0
                print(f"[gpu{gpu}] {n_done}/{total}  {el / 60:.1f} min, ETA {el / n_done * (total - n_done) / 60:.0f} min",
                      flush=True)
                flush()
    flush()
    print(f"[gpu{gpu}] finished", flush=True)


def launch():
    names, paths = choose_datasets()
    print("datasets:", names)
    jobs = []
    for name in names:
        n_rows = len(load_df(paths[name]))
        rows = list(range(MIN_CTX, n_rows, QUERY_STRIDE))
        jobs += [(name, s, rows) for s in SEEDS]
    try:
        n_gpu = len([l for l in subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
                     .splitlines() if l.startswith("GPU")])
    except FileNotFoundError:
        n_gpu = 0
    n_gpu = max(n_gpu, 1)
    print(f"{len(jobs)} (dataset, seed) jobs over {n_gpu} GPU(s); "
          f"{sum(len(r) for *_, r in jobs)} row-predictions x {len(METHODS)} methods")
    split = [jobs[g::n_gpu] for g in range(n_gpu)]
    os.makedirs(f"{OUT_DIR}/logs", exist_ok=True)
    procs, logs = [], []
    for g, js in enumerate(split):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), PYTHONWARNINGS="ignore")
        log = open(f"{OUT_DIR}/logs/gpu{g}.log", "w")
        logs.append(f"{OUT_DIR}/logs/gpu{g}.log")
        procs.append(subprocess.Popen([sys.executable, os.path.abspath(__file__), "--worker", str(g), json.dumps(js)],
                                      env=env, stdout=log, stderr=subprocess.STDOUT, text=True))
    seen = [0] * len(logs)
    while True:                                             # tail both logs; workers write to files, no pipes
        running = any(p.poll() is None for p in procs)
        for g, path in enumerate(logs):
            lines = open(path, errors="ignore").read().splitlines()
            for line in lines[seen[g]:]:
                if "[gpu" in line or "Error" in line or "Traceback" in line:
                    print(line, flush=True)
            seen[g] = len(lines)
        if not running:
            break
        time.sleep(5)
    bad = [g for g, p in enumerate(procs) if p.returncode]
    if bad:
        print(f"WARNING: worker(s) {bad} failed; full logs in {OUT_DIR}/logs/. Building the report from what exists.")


# ------------------------------------------------------------------ report
def build_report():
    parts = sorted(glob.glob(f"{PART_DIR}/*.csv"))
    assert parts, f"no predictions in {PART_DIR}"
    d = pd.concat([pd.read_csv(f) for f in parts], ignore_index=True)
    d = d.drop_duplicates(["dataset", "row", "seed", "method"]).reset_index(drop=True)
    # noise-aware per-row pieces (work on old saved predictions too: they only need k and mean_k)
    pm = (d["mean_k"].to_numpy(float) / N_EXP).clip(1e-6, 1 - 1e-6)
    d["H"] = d["k"].map(entropy_floor)
    d["dev"] = deviance(d["k"].to_numpy(float), pm)
    d["edev"] = expected_deviance(pm)
    d.to_csv(f"{OUT_DIR}/predictions_all.csv", index=False)
    methods = [m for m in METHODS if m in set(d.method)] + sorted(set(d.method) - set(METHODS))
    datasets = sorted(d.dataset.unique())
    d["mi"] = d.method.map({m: i for i, m in enumerate(methods)})
    d["di"] = d.dataset.map({x: i for i, x in enumerate(datasets)})
    cols = dict(di="di", r="row", s="seed", m="mi", k="k", st="stratum", phi="phi", nll="nll", crps="crps",
                H="H", dev="dev", edev="edev")
    payload = {short: np.round(d[c].to_numpy(float), 5).tolist() for short, c in cols.items()}
    payload = {key: [None if (isinstance(v, float) and not np.isfinite(v)) else v for v in vals]
               for key, vals in payload.items()}
    ncov = len([c for c in d.columns if c.startswith("cov") and c[3:].isdigit()])
    payload["cov"] = d[[f"cov{i}" for i in range(ncov)]].to_numpy(int).tolist()
    meta = dict(methods=methods, datasets=datasets, levels=LEVELS[:ncov], seeds=sorted(d.seed.unique().tolist()),
                n_exp=N_EXP, eps=EPS, min_ctx=MIN_CTX, stride=QUERY_STRIDE,
                generated=time.strftime("%Y-%m-%d %H:%M"))
    from htmlutil import inline_plotly
    html = inline_plotly(TEMPLATE.replace("/*__DATA__*/null", json.dumps(payload, separators=(",", ":")))
                         .replace("/*__META__*/null", json.dumps(meta)))
    out = f"{OUT_DIR}/classifier_suite.html"
    open(out, "w").write(html)
    print(f"report: {out}  ({len(d)} predictions, {os.path.getsize(out) / 1e6:.1f} MB)")
    i90 = LEVELS.index(0.9)
    g = d.groupby("method")
    tab = pd.DataFrame({"log_score": g.nll.mean(), "nll_excess": g.nll.mean() - g.H.mean(), "crps": g.crps.mean(),
                        "coverage90": g[f"cov{i90}"].mean(), "overdispersion": g.dev.mean() / g.edev.mean()})
    print(tab.reindex(methods).round(3).to_string())


TEMPLATE = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Causilo classifier suite</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root{--bg:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#8a8984;--line:#e4e3dd;--grid:#efeee9;--accent:#2a78d6}
@media (prefers-color-scheme:dark){:root{--bg:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#8f8e88;--line:#383835;--grid:#2a2a28;--accent:#3987e5}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1280px;margin:0 auto;padding:24px}
h1{font-size:22px;font-weight:500;margin:0 0 4px}
h2{font-size:17px;font-weight:500;margin:28px 0 4px}
.sub{color:var(--ink2);margin:0 0 12px;font-size:13px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:16px;margin:12px 0}
.controls{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:14px}
.controls fieldset{border:1px solid var(--line);border-radius:8px;padding:8px 10px;margin:0}
.controls legend{color:var(--ink2);font-size:12px;padding:0 4px}
.controls label{display:block;font-size:13px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.row{display:flex;gap:8px;flex-wrap:wrap;margin-top:6px}
button,select,input[type=number]{font:inherit;font-size:12px;background:var(--surface);color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:3px 8px}
button{cursor:pointer}
.swatch{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px}
table{border-collapse:collapse;width:100%;font-size:12.5px}
th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th{color:var(--ink2);font-weight:500;cursor:pointer;user-select:none}
td:first-child,th:first-child{text-align:left}
td.best{font-weight:600}
.pm{color:var(--muted);font-size:11px}
.note{color:var(--ink2);font-size:12.5px;margin:6px 0 0}
.plot{width:100%;height:380px}
.plot.tall{height:430px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:12px}
@media (max-width:900px){.grid2{grid-template-columns:1fr}}
.pill{display:inline-block;background:var(--grid);border-radius:10px;padding:1px 8px;font-size:12px;color:var(--ink2);margin-right:6px}
</style>
</head>
<body>
<main>
<h1>Causilo classifier suite</h1>
<p class="sub" id="meta"></p>

<div class="card">
  <div class="controls">
    <fieldset><legend>Datasets</legend><div id="ds"></div>
      <div class="row"><button data-all="ds">all</button><button data-none="ds">none</button></div></fieldset>
    <fieldset><legend>Methods</legend><div id="me"></div>
      <div class="row"><button data-all="me">all</button><button data-none="me">none</button></div></fieldset>
    <fieldset><legend>True outcome</legend><div id="st"></div></fieldset>
    <fieldset><legend>Rank on these metrics</legend><div id="mt"></div>
</fieldset>
  </div>
  <p class="note" id="sampling"></p>
</div>

<h2>Leaderboard</h2>
<p class="sub">Value = all chosen rows and seeds pooled; ± = spread across seeds (SD of the per-seed value). Overall = mean rank across the ticked metrics (1 = best). Click a header to sort.
Only metrics that score the whole predicted distribution against the measured count, or against the data's own counting noise, are shown. Interval width / width ratio are left out on purpose: a classifier gives one p, so its width comes from our conversion step, not from Causilo.</p>
<div class="card" style="overflow-x:auto"><table id="board"></table></div>

<h2>How many predictions are behind every point</h2>
<p class="sub">Every level in the sweep charts below uses all of these predictions: the levels are different interval widths applied to the same rows, not different samples.</p>
<div class="card"><div id="pSampling" class="plot" style="height:300px"></div></div>

<h2>Coverage vs asked-for level</h2>
<p class="sub">On the dashed diagonal = honest. Above = too cautious, below = overconfident. Error bars = 95% Wilson interval for that many predictions.</p>
<div class="card"><div id="pCov" class="plot tall"></div></div>

<h2>Seed-to-seed spread</h2>
<p class="sub">One dot per seed, bar = mean over seeds. Wide dot clouds mean the TFM's randomness alone moves the score that much.</p>
<div class="card"><div class="row" style="margin:0 0 6px">Metric <select id="seedMetric"></select></div><div id="pSeed" class="plot"></div></div>

</main>

<script>
const DATA = /*__DATA__*/null;
const META = /*__META__*/null;
const COLORS = ["#2a78d6","#eb6834","#1baf7a","#eda100","#e87ba4","#008300","#6250d6","#e34948"];
const SYMBOLS = ["circle","square","diamond","triangle-up","cross","x","triangle-down","star"];
const STRATA = ["0%","middle","100%"];
const L90 = META.levels.indexOf(0.9);
const METRICS = {
  nll:     {label:"Log score (−log P true count)",       dir:-1, fmt:3},
  excess:  {label:"Log score − entropy floor",           dir:-1, fmt:3},
  crps:    {label:"CRPS (counts)",                       dir:-1, fmt:3},
  cov90:   {label:"Coverage 90%",                        dir:0,  fmt:3, target:0.9},
  covErr:  {label:"Coverage error, mean |cov − level|",  dir:-1, fmt:3},
  od:      {label:"Overdispersion (1 = counting noise)", dir:0,  fmt:2, target:1, log:true},
};
const DEFAULT_RANK = ["nll","crps","covErr","od"];
const N = DATA.m.length;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const color = m => COLORS[m % COLORS.length];

/* ---------- controls ---------- */
function checks(id, items, checked, onLabel){
  document.getElementById(id).innerHTML = items.map((t,i)=>
    `<label><input type="checkbox" value="${i}" ${checked(i)?"checked":""}> ${onLabel?onLabel(i,t):t}</label>`).join("");
}
checks("ds", META.datasets, ()=>true);
checks("me", META.methods, ()=>true, (i,t)=>`<span class="swatch" style="background:${color(i)}"></span>${t}`);
document.getElementById("st").innerHTML = ["all","0%","middle","100%"].map((t,i)=>
  `<label><input type="radio" name="st" value="${i-1}" ${i===0?"checked":""}> ${t}</label>`).join("");
checks("mt", Object.keys(METRICS).map(k=>METRICS[k].label), i=>DEFAULT_RANK.includes(Object.keys(METRICS)[i]));
document.getElementById("seedMetric").innerHTML = Object.keys(METRICS).map(k=>`<option value="${k}">${METRICS[k].label}</option>`).join("");
document.querySelectorAll("[data-all]").forEach(b=>b.onclick=()=>{document.querySelectorAll(`#${b.dataset.all} input`).forEach(c=>c.checked=true);update();});
document.querySelectorAll("[data-none]").forEach(b=>b.onclick=()=>{document.querySelectorAll(`#${b.dataset.none} input`).forEach(c=>c.checked=false);update();});
document.querySelectorAll(".controls input, #seedMetric").forEach(el=>el.addEventListener("change", update));
const picked = id => [...document.querySelectorAll(`#${id} input:checked`)].map(c=>+c.value);
document.getElementById("meta").innerHTML =
  `${META.seeds.length} seeds (${META.seeds.join(", ")}) · rows from ${META.min_ctx}, every ${META.stride} · ` +
  `one-step-ahead · probability floor ${META.eps} · generated ${META.generated}`;

/* ---------- metric helpers ---------- */
function metric(idx, name){
  if(!idx.length) return NaN;
  const mean = f => idx.reduce((s,i)=>s+f(i),0)/idx.length;
  switch(name){
    case "nll": return mean(i=>DATA.nll[i]);
    case "excess": return mean(i=>DATA.nll[i]-DATA.H[i]);
    case "crps": return mean(i=>DATA.crps[i]);
    case "cov90": return mean(i=>DATA.cov[i][L90]);
    case "covErr": return META.levels.reduce((s,L,li)=>s+Math.abs(mean(i=>DATA.cov[i][li])-L),0)/META.levels.length;
    case "od": { const e = mean(i=>DATA.edev[i]); return e>0 ? mean(i=>DATA.dev[i])/e : NaN; }
  }
}
function badness(name, v){            // lower = better, for ranking
  const M = METRICS[name];
  if(!isFinite(v)) return Infinity;
  if(M.dir===0) return M.log ? Math.abs(Math.log(v/M.target)) : Math.abs(v-M.target);
  return -M.dir*v;
}
function wilson(x, n){ if(!n) return [NaN,NaN]; const z=1.96,p=x/n,d=1+z*z/n,c=(p+z*z/(2*n))/d,h=z*Math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d; return [c-h,c+h]; }
function sd(a){ const v=a.filter(isFinite); if(v.length<2) return NaN; const m=v.reduce((s,x)=>s+x,0)/v.length; return Math.sqrt(v.reduce((s,x)=>s+(x-m)**2,0)/(v.length-1)); }

/* ---------- selection ---------- */
function select(){
  const ds = new Set(picked("ds")), me = picked("me"), meSet = new Set(me);
  const st = +document.querySelector("input[name=st]:checked").value;
  const byM = new Map(me.map(m=>[m,[]]));
  for(let i=0;i<N;i++){
    if(!ds.has(DATA.di[i]) || !meSet.has(DATA.m[i]) || (st>=0 && DATA.st[i]!==st)) continue;
    byM.get(DATA.m[i]).push(i);
  }
  // rows (dataset,row) that every selected method predicted -> fair comparison set
  const key = i => DATA.di[i]*100000 + DATA.r[i];
  let common = null;
  for(const m of me){ const s=new Set(byM.get(m).map(key)); common = common===null ? s : new Set([...common].filter(x=>s.has(x))); }
  common = common || new Set();
  for(const m of me) byM.set(m, byM.get(m).filter(i=>common.has(key(i))));
  return {me, byM, common:[...common], key, st};
}

/* ---------- plots ---------- */
function baseLayout(extra){
  return Object.assign({paper_bgcolor:"rgba(0,0,0,0)", plot_bgcolor:"rgba(0,0,0,0)", font:{color:css("--ink2"), size:12},
    margin:{l:60,r:20,t:30,b:50}, legend:{orientation:"h", y:-0.22}, hovermode:"closest",
    xaxis:{gridcolor:css("--grid"), zeroline:false, linecolor:css("--line")},
    yaxis:{gridcolor:css("--grid"), zeroline:false, linecolor:css("--line")}}, extra||{});
}
const cfg = {displaylogo:false, responsive:true};

function plotSampling(S){
  const ds = picked("ds"), stOf = new Map(), common = new Set(S.common);
  for(const m of S.me) for(const i of S.byM.get(m)) stOf.set(S.key(i), DATA.st[i]);
  const traces = STRATA.map((nm,s)=>({type:"bar", name:nm, x: ds.map(d=>META.datasets[d]),
    y: ds.map(d=>[...common].filter(c=>Math.floor(c/100000)===d && stOf.get(c)===s).length),
    marker:{color:["#86b6ef","#2a78d6","#184f95"][s]}, hovertemplate:"%{x}<br>"+nm+": %{y} rows<extra></extra>"}));
  Plotly.react("pSampling", traces, baseLayout({barmode:"stack", yaxis:{title:"rows predicted (each × seeds)", gridcolor:css("--grid")},
    xaxis:{tickfont:{size:10}}}), cfg);
}

function plotSweeps(S){
  const lv = META.levels, cov=[];
  for(const m of S.me){
    const idx = S.byM.get(m); if(!idx.length) continue;
    const n = idx.length, hits = lv.map((_,li)=>idx.reduce((s,i)=>s+DATA.cov[i][li],0));
    const ci = hits.map(h=>wilson(h,n));
    cov.push({name:META.methods[m], mode:"lines+markers", line:{color:color(m), width:2},
      marker:{symbol:SYMBOLS[m%8], size:8, color:color(m)}, x:lv, y:hits.map(h=>h/n),
      error_y:{type:"data", symmetric:false, array:ci.map((c,j)=>c[1]-hits[j]/n), arrayminus:ci.map((c,j)=>hits[j]/n-c[0]), color:color(m), thickness:1, width:3},
      customdata: lv.map(()=>[n, new Set(idx.map(S.key)).size]),
      hovertemplate:"%{fullData.name}<br>asked %{x}: got %{y:.3f}<br>%{customdata[0]} predictions (%{customdata[1]} rows)<extra></extra>"});
  }
  cov.push({x:[lv[0],lv[lv.length-1]], y:[lv[0],lv[lv.length-1]], mode:"lines", line:{color:css("--ink2"), dash:"dash", width:1}, showlegend:false, hoverinfo:"skip"});
  Plotly.react("pCov", cov, baseLayout({xaxis:{title:"asked-for interval level", gridcolor:css("--grid")}, yaxis:{title:"actual coverage", range:[0,1.03], gridcolor:css("--grid")}}), cfg);
}

function plotSeeds(S){
  const name = document.getElementById("seedMetric").value, tr=[];
  S.me.forEach((m,j)=>{
    const idx = S.byM.get(m); if(!idx.length) return;
    const vals = META.seeds.map(s=>metric(idx.filter(i=>DATA.s[i]===s), name));
    const mu = vals.filter(isFinite).reduce((a,b)=>a+b,0)/vals.filter(isFinite).length;
    tr.push({type:"bar", x:[META.methods[m]], y:[mu], marker:{color:color(m), opacity:0.35}, showlegend:false, hovertemplate:`mean %{y:.4f}<extra></extra>`});
    tr.push({type:"scatter", mode:"markers", x:vals.map(()=>META.methods[m]), y:vals, text:META.seeds.map(s=>"seed "+s),
      marker:{color:color(m), size:9, symbol:SYMBOLS[m%8], line:{color:css("--surface"), width:1}}, showlegend:false, hovertemplate:"%{text}: %{y:.4f}<extra></extra>"});
  });
  const M = METRICS[name], shapes = M.dir===0 ? [{type:"line", xref:"paper", x0:0, x1:1, y0:M.target, y1:M.target, line:{dash:"dash", color:css("--ink2"), width:1}}] : [];
  Plotly.react("pSeed", tr, baseLayout({barmode:"overlay", shapes, yaxis:{title:M.label, type:M.log?"log":"linear", gridcolor:css("--grid")}}), cfg);
}

/* ---------- leaderboard ---------- */
let sortKey = "overall", sortDir = 1;
function leaderboard(S){
  const mkeys = Object.keys(METRICS), rankOn = picked("mt").map(i=>mkeys[i]);
  const rows = S.me.map(m=>{
    const idx = S.byM.get(m), o = {name:META.methods[m], m};
    for(const k of mkeys){ o[k]=metric(idx,k); o[k+"_sd"]=sd(META.seeds.map(s=>metric(idx.filter(i=>DATA.s[i]===s),k))); }
    return o;
  });
  for(const k of rankOn){ const order=[...rows].sort((x,y)=>badness(k,x[k])-badness(k,y[k])); order.forEach((r,i)=>r["rk_"+k]=i+1); }
  rows.forEach(r=>r.overall = rankOn.length ? rankOn.reduce((s,k)=>s+r["rk_"+k],0)/rankOn.length : NaN);
  rows.sort((x,y)=>{ const v=(r)=> sortKey==="overall" ? r.overall : sortKey==="name" ? r.name : badness(sortKey, r[sortKey]);
                     const a=v(x), b=v(y); return (a>b?1:a<b?-1:0)*sortDir; });
  const bestOf = k => Math.min(...rows.map(r=>badness(k,r[k])));
  const head = `<tr><th data-k="name">method</th><th data-k="overall">overall rank</th>` +
    mkeys.map(k=>`<th data-k="${k}" title="${METRICS[k].dir===0?'closest to '+METRICS[k].target:(METRICS[k].dir<0?'lower':'higher')+' is better'}">${METRICS[k].label}${rankOn.includes(k)?" ★":""}</th>`).join("") + `</tr>`;
  const body = rows.map(r=>`<tr><td><span class="swatch" style="background:${color(r.m)}"></span>${r.name}</td>
    <td>${isFinite(r.overall)?r.overall.toFixed(2):"–"}</td>` +
    mkeys.map(k=>`<td class="${badness(k,r[k])===bestOf(k)?"best":""}">${isFinite(r[k])?r[k].toFixed(METRICS[k].fmt):"–"}
      <span class="pm">${isFinite(r[k+"_sd"])?"±"+r[k+"_sd"].toFixed(METRICS[k].fmt):""}</span></td>`).join("") + `</tr>`).join("");
  const t = document.getElementById("board"); t.innerHTML = head + body;
  t.querySelectorAll("th[data-k]").forEach(th=>th.onclick=()=>{ const k=th.dataset.k; sortDir = sortKey===k ? -sortDir : 1; sortKey=k; leaderboard(S); });
}

function update(){
  const S = select(), nPred = S.me.reduce((s,m)=>s+S.byM.get(m).length,0);
  document.getElementById("sampling").innerHTML =
    `<span class="pill">${S.common.length} rows</span><span class="pill">${META.seeds.length} seeds</span>` +
    `<span class="pill">${nPred} predictions</span> Only rows that every ticked method predicted are used ` +
    `(no_mid skips rows whose context lacks both 0/70 and 70/70), so all methods are scored on the same rows.`;
  if(!S.me.length || !S.common.length){ document.getElementById("board").innerHTML="<tr><td>nothing selected</td></tr>"; return; }
  leaderboard(S); plotSampling(S); plotSweeps(S); plotSeeds(S);
}
update();
</script>
</body>
</html>
'''


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(int(sys.argv[2]), json.loads(sys.argv[3]))
    elif len(sys.argv) > 1 and sys.argv[1] == "--report":
        build_report()
    else:
        os.makedirs(OUT_DIR, exist_ok=True)
        launch()
        build_report()
