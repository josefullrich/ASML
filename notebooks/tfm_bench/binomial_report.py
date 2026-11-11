"""Die-yield models compared as what they really predict: a distribution over 0..70 good dies. No normalization.

    python binomial_report.py              # -> avg_everything/binomial_comparison.html (+ binomial_metrics.csv)

Every config's saved one-step-ahead prediction is turned into a probability for each count k = 0..70:
  * regressors on the observed logit target (Causilo, Kumo, TabPFN, GP BoTorch with observation noise): their
    quantile function is binned at the midpoints between the 71 possible target values;
  * GP Laplace (MATLAB port): its predictive is the LATENT logit yield f (no counting noise), so the count
    distribution is the binomial mixture  sum_f N(f; mean, sd) * Binomial(k; 70, sigmoid(f))  (Gauss-Hermite);
  * causilo_trials (Causilo classifier, frequency weights): the predicted p is recovered from the saved mean and
    used as Binomial(70, p). Three fixes we tried are then applied ON TOP of those same p's:
       _bb       beta-binomial: Binomial(70, p) widened by an overdispersion phi (small phi = U-shaped)
       _calib    Platt sharpening of the logit: p' = sigmoid(a * logit p + b), binomial
       _calib_bb both: sharpened p' and a beta-binomial phi
    a, b, phi are fitted by maximum likelihood on the config's OWN EARLIER predictions on that day (prequential:
    at step t only steps < t are used, so nothing is fitted on the row being scored). Before MIN_FIT past steps
    exist the raw binomial is used. The classifier suite (03 notebook) fits the same parameters a different way
    (held-out folds of the context) and also has the methods that need refits (no_mid, multiclass).
Every distribution gets the same tiny floor (EPS over the 71 counts) so an unseen count isn't log(0).

Metrics (per dataset, then mean +- SE over datasets; scope = whole run or last 20%):
  log score      -log P(the count that happened)                                         lower
  excess         log score - binomial entropy at that count (0 = as good as counting noise allows)   lower
  crps           sum_k (F(k) - 1[k >= k_true])^2, in counts                                lower
  cov90          PIT coverage at 90%: share of the truth's PIT inside the central 90% (exact for a calibrated
                 count model)                                                               0.90
  cov90_int      share of truths inside the central 90% COUNT interval, endpoints included. Biased high on counts
                 for every model (the 0/70 and 70/70 spikes), shown for reference                0.90
  cov_err        mean |PIT coverage - level| over 10..99%                                   lower
  overdisp       sum deviance / sum expected deviance at the predicted mean p (1 = misses are pure counting luck)  1
"""
from __future__ import annotations

import glob
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

import config as C

warnings.filterwarnings("ignore")

N = 70
K = np.arange(N + 1)
T = np.log((K + 0.5) / (N + 1 - K - 0.5))  # the 71 possible target values logit((k+.5)/(n+1))
T_EDGES = (T[1:] + T[:-1]) / 2
EPS = 1e-3
MIN_FIT = 20
LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]
L90 = LEVELS.index(0.9)
LATENT = {"gpr_laplace"}  # predictive is the latent logit yield, not the measurement
CLASSIFIER = "causilo_trials"
FIXES = {"bb": "beta-binomial phi", "calib": "Platt a, b", "calib_bb": "Platt a, b + beta-binomial phi"}
GH_X, GH_W = np.polynomial.hermite.hermgauss(40)

COLORS = {"gpr_laplace": "#2a78d6", "gpr_botorch": "#eb6834", "causilo": "#1baf7a", "kumo": "#3f2fa8",
          "kumo_medium": "#6f5fe0", "kumo_small": "#a499f0", "tabpfn2": "#e87ba4", "tabpfn35": "#008300",
          "causilo_trials": "#8a5a00", "causilo_trials_bb": "#d68a00", "causilo_trials_calib": "#c0392b",
          "causilo_trials_calib_bb": "#7d1f6b", "mock": "#8a8984"}
LABELS = {"gpr_laplace": "GP Laplace (latent → binomial mixture)", "gpr_botorch": "GP BoTorch",
          "causilo": "Causilo regressor", "kumo": "Kumo large", "kumo_medium": "Kumo medium", "kumo_small": "Kumo small",
          "tabpfn2": "TabPFN 2", "tabpfn35": "TabPFN 3.5", "causilo_trials": "Causilo classifier: binomial (raw)",
          "causilo_trials_bb": "Causilo classifier + beta-binomial", "causilo_trials_calib": "Causilo classifier + Platt",
          "causilo_trials_calib_bb": "Causilo classifier + Platt + beta-binomial", "mock": "mock"}


def split(folder):
    for m in sorted(C.ALL_TFM_MODELS + ["mock"], key=len, reverse=True):
        if folder.startswith(m + "_") and folder[len(m) + 1:] in C.ARMS:
            return m, folder[len(m) + 1:]
    return folder, "none"


# ---------------------------------------------------------------------------------------------------- distributions
def floor(P):
    P = np.clip(P, 0, None)
    P = P / P.sum(1, keepdims=True)
    return (1 - EPS) * P + EPS / P.shape[1]


def pmf_from_quantiles(df):
    lv = np.array([float(c[2:]) for c in df.columns if c.startswith("q_")])
    cols = [c for c in df.columns if c.startswith("q_")]
    o = np.argsort(lv)
    lv, Q = lv[o], np.maximum.accumulate(df[cols].to_numpy(float)[:, o], axis=1)
    cdf = np.stack([np.interp(T_EDGES, q, lv, left=0.0, right=1.0) for q in Q])
    return np.diff(np.concatenate([np.zeros((len(Q), 1)), cdf, np.ones((len(Q), 1))], 1), axis=1)


def pmf_latent(mean, sd):
    f = mean[:, None] + np.sqrt(2) * sd[:, None] * GH_X[None, :]             # (rows, nodes)
    p = special.expit(f)
    B = stats.binom.pmf(K[None, None, :], N, p[:, :, None])                  # (rows, nodes, 71)
    return (GH_W[None, :, None] * B).sum(1) / np.sqrt(np.pi)


_PG = special.expit(np.linspace(-14, 14, 4001))
_MG = stats.binom.pmf(K[None, :], N, _PG[:, None]) @ T                       # mean target for each p (monotone)


def p_from_mean(mean):
    return np.interp(mean, _MG, _PG)


def pmf_binom(p):
    return stats.binom.pmf(K[None, :], N, np.clip(p, 1e-9, 1 - 1e-9)[:, None])


def pmf_bb(p, phi):
    p = np.clip(p, 1e-6, 1 - 1e-6)[:, None]
    return stats.betabinom.pmf(K[None, :], N, p * phi, (1 - p) * phi)


def loglik(theta, z, k, kind):
    a, b = (theta[0], theta[1]) if kind != "bb" else (1.0, 0.0)
    mu = special.expit(a * z + b).clip(1e-6, 1 - 1e-6)
    if kind == "calib":
        return np.sum(k * np.log(mu) + (N - k) * np.log1p(-mu))
    phi = np.exp(np.clip(theta[-1], -4, 9))
    return np.sum(special.betaln(k + mu * phi, N - k + (1 - mu) * phi) - special.betaln(mu * phi, (1 - mu) * phi))


def prequential_fix(p, k, kind):
    """For each step t: fit (a, b, phi) on steps < t, return the pmf for step t and the parameters used."""
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    P = np.empty((len(p), N + 1))
    pars = np.full((len(p), 3), np.nan)
    theta = {"bb": np.array([np.log(10.0)]), "calib": np.array([1.0, 0.0]),
             "calib_bb": np.array([1.0, 0.0, np.log(10.0)])}[kind]
    for t in range(len(p)):
        if t < MIN_FIT:
            P[t] = pmf_binom(p[t:t + 1])[0]
            pars[t] = [1.0, 0.0, np.inf]
            continue
        r = optimize.minimize(lambda th: -loglik(th, z[:t], k[:t], kind), theta, method="Nelder-Mead",
                              options=dict(maxiter=600, xatol=1e-4, fatol=1e-6))
        if np.isfinite(r.fun):
            theta = r.x
        a, b = (theta[0], theta[1]) if kind != "bb" else (1.0, 0.0)
        phi = float(np.exp(np.clip(theta[-1], -4, 9))) if kind != "calib" else np.inf
        mu = special.expit(a * z[t] + b)
        P[t] = pmf_binom(np.array([mu]))[0] if kind == "calib" else pmf_bb(np.array([mu]), phi)[0]
        pars[t] = [a, b, phi]
    return P, pars


# ---------------------------------------------------------------------------------------------------- scores
_H = {}


def entropy_floor(k):
    if k not in _H:
        w = stats.binom.pmf(K, N, (k + 0.5) / (N + 1))
        w = w[w > 0]
        _H[k] = float(-(w * np.log(w)).sum())
    return _H[k]


def deviance(k, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = np.where(k > 0, k * np.log(np.maximum(k, 1) / (N * p)), 0.0)
        t2 = np.where(N - k > 0, (N - k) * np.log(np.maximum(N - k, 1) / (N * (1 - p))), 0.0)
    return 2 * (t1 + t2)


def row_scores(P, k):
    """Per-row pieces; every summary metric is a mean (or ratio of means) of these."""
    P = floor(P)
    cdf = np.cumsum(P, 1)
    kk = k.astype(int)
    nll = -np.log(P[np.arange(len(k)), kk])
    crps = ((cdf - (K[None, :] >= kk[:, None])) ** 2).sum(1)
    # two kinds of coverage at each level L:
    #  cov  (PIT): the truth's probability-integral-transform u ~ U(F(k-1), F(k)) falls in the central L of [0, 1].
    #       Exactly L for a calibrated count model; computed as the expected value, no random draw.
    #  covi (interval): k inside the central count interval [q_(1-L)/2, q_(1+L)/2], endpoints included. Biased
    #       ABOVE L for every model on counts (the endpoint bins carry extra mass, e.g. the spike at 70/70).
    Fhi = cdf[np.arange(len(k)), kk]
    Flo = Fhi - P[np.arange(len(k)), kk]
    cov = np.empty((len(k), len(LEVELS)))
    covi = np.empty((len(k), len(LEVELS)))
    for j, L in enumerate(LEVELS):
        a, b = (1 - L) / 2, (1 + L) / 2
        cov[:, j] = np.clip(np.minimum(Fhi, b) - np.maximum(Flo, a), 0, None) / np.maximum(Fhi - Flo, 1e-15)
        lo = np.argmax(cdf >= a, axis=1)
        hi = np.argmax(cdf >= b, axis=1)
        covi[:, j] = (lo <= kk) & (kk <= hi)
    pm = (P @ K) / N
    dev = deviance(k.astype(float), pm)
    edev = (stats.binom.pmf(K[None, :], N, np.clip(pm, 1e-12, 1 - 1e-12)[:, None]) *
            deviance(K[None, :].astype(float), pm[:, None])).sum(1)
    H = np.array([entropy_floor(int(x)) for x in kk])
    return dict(nll=nll, H=H, crps=crps, cov=cov, covi=covi, dev=dev, edev=edev)


def summarize(rs, sel):
    m = lambda x: float(np.mean(x[sel]))
    cov, covi = rs["cov"][sel].mean(0), rs["covi"][sel].mean(0)
    return dict(nll=m(rs["nll"]), excess=m(rs["nll"]) - m(rs["H"]), crps=m(rs["crps"]), cov90=float(cov[L90]),
                cov_err=float(np.mean(np.abs(cov - np.array(LEVELS)))), cov90_int=float(covi[L90]),
                overdisp=float(rs["dev"][sel].sum() / rs["edev"][sel].sum()), cov_curve=cov.tolist(),
                covi_curve=covi.tolist(), n=int(sel.sum()))


# ---------------------------------------------------------------------------------------------------- main
def configs_from(df_pred, model, arm):
    """-> list of (config_id, pmf, params or None)."""
    k = df_pred["k"].to_numpy(float)
    if model in LATENT:
        return [(model, pmf_latent(df_pred["mean"].to_numpy(float), df_pred["sd"].to_numpy(float)), None)]
    if model == CLASSIFIER:
        p = p_from_mean(df_pred["mean"].to_numpy(float))
        out = [(f"{CLASSIFIER}_{arm}", pmf_binom(p), None)]
        for kind in FIXES:
            P, pars = prequential_fix(p, k, kind)
            out.append((f"{CLASSIFIER}_{kind}_{arm}", P, pars))
        return out
    cid = model if arm == "none" else f"{model}_{arm}"
    return [(cid, pmf_from_quantiles(df_pred), None)]


def main():
    t0 = time.time()
    rows, params, cfg_meta = [], {}, {}
    folders = sorted(os.path.basename(p) for p in glob.glob(os.path.join(C.RESULTS_DIR, "*")) if os.path.isdir(p))
    for folder in folders:
        model, arm = split(folder)
        if model == "mock":
            continue
        for path in sorted(glob.glob(os.path.join(C.RESULTS_DIR, folder, "results_*", "predictions.csv"))):
            d = pd.read_csv(path).sort_values("iter").reset_index(drop=True)
            if not (d["n"] == N).all():
                d = d[d["n"] == N].reset_index(drop=True)
            ds = d["dataset"].iloc[0]
            k = d["k"].to_numpy(float)
            last = np.zeros(len(d), bool)
            last[len(d) - int(np.ceil(C.LAST_FRAC * len(d))):] = True
            for cid, P, pars in configs_from(d, model, arm):
                base = cid[: -len(arm) - 1] if arm != "none" else cid
                cfg_meta[cid] = dict(id=cid, model=base, arm=arm, label=LABELS.get(base, base),
                                     color=COLORS.get(base, "#52514e"))
                rs = row_scores(P, k)
                for scope, sel in [("all", np.ones(len(d), bool)), ("last20", last)]:
                    rows.append(dict(config=cid, dataset=ds, scope=scope, **summarize(rs, sel)))
                if pars is not None:
                    params.setdefault(cid, {})[ds] = dict(iter=d["iter"].astype(int).tolist(),
                                                         a=np.round(pars[:, 0], 4).tolist(),
                                                         b=np.round(pars[:, 1], 4).tolist(),
                                                         phi=[None if not np.isfinite(v) else round(float(v), 3)
                                                              for v in pars[:, 2]])
        print(f"  {folder}  ({time.time() - t0:.0f}s)", flush=True)
    R = pd.DataFrame(rows)
    os.makedirs(C.AVG_DIR, exist_ok=True)
    R.drop(columns=["cov_curve", "covi_curve"]).to_csv(os.path.join(C.AVG_DIR, "binomial_metrics.csv"), index=False)
    datasets = sorted(R.dataset.unique())
    order = ["gpr_laplace", "gpr_botorch", "causilo", "kumo", "kumo_medium", "kumo_small", "tabpfn2", "tabpfn35",
             CLASSIFIER] + [f"{CLASSIFIER}_{k}" for k in FIXES]
    arms_order = ["none"] + C.ARMS
    cfgs = sorted(cfg_meta.values(), key=lambda c: (order.index(c["model"]) if c["model"] in order else 99,
                                                     arms_order.index(c["arm"])))
    per = {}
    for scope in ["all", "last20"]:
        S = R[R.scope == scope].set_index(["config", "dataset"])
        per[scope] = {}
        for c in cfgs:
            per[scope][c["id"]] = {ds: ({k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                                         for k, v in S.loc[(c["id"], ds)].items() if k != "n"}
                                        if (c["id"], ds) in S.index else None) for ds in datasets}
    data = dict(title="Die yield as a count distribution: every model and every fix",
                generated=time.strftime("%Y-%m-%d %H:%M"), levels=LEVELS, datasets=datasets, configs=cfgs,
                arms=[a for a in C.ARMS if any(c["arm"] == a for c in cfgs)], per=per, params=params,
                minFit=MIN_FIT, eps=EPS)
    from htmlutil import inline_plotly
    html = inline_plotly(TEMPLATE.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")).replace("</", "<\\/")))
    out = os.path.join(C.AVG_DIR, "binomial_comparison.html")
    open(out, "w", encoding="utf-8").write(html)
    print(f"done in {time.time() - t0:.0f}s -> {out} ({os.path.getsize(out) / 1e6:.1f} MB)")
    tab = R[R.scope == "all"].groupby("config")[["nll", "excess", "crps", "cov90", "cov90_int", "cov_err",
                                                  "overdisp"]].mean()
    print(tab.reindex([c["id"] for c in cfgs]).round(3).to_string())


TEMPLATE = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "binomial_template.html"),
                encoding="utf-8").read() if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                         "binomial_template.html")) else ""

if __name__ == "__main__":
    main()
