"""Data loading, target, context selection, and the shared one-step-ahead protocol loop."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

import config as C


# ----------------------------------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------------------------------
def logit_dy(k, n):
    """Modelling/scoring scale: logit((k + 0.5) / (n + 1))."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = (k + 0.5) / (n + 1.0)
    return np.log(p) - np.log1p(-p)


@dataclass
class Dataset:
    ds_id: str
    X: np.ndarray  # (T, 9) raw inputs in CSV row order
    k: np.ndarray
    n: np.ndarray
    y: np.ndarray  # the modelled target: logit yield for TARGET == "dy", else the raw column (EUV_mJ, Rx_deg, ...)
    n_rows_file: int
    ydy: np.ndarray | None = None  # logit yield, always (context selection uses it so every target sees the same days)


def load_dataset(ds_id: str) -> Dataset:
    df = pd.read_csv(C.csv_path(ds_id))
    if list(df.columns) != C.ALL_COLUMNS:
        raise ValueError(f"{ds_id}: unexpected columns {list(df.columns)}")
    n_file = len(df)
    if C.TRUNCATE_AT_BO_END:
        n0 = df[C.N_COL].iloc[: C.N_WARMUP].mode().iloc[0]
        changed = np.flatnonzero(df[C.N_COL].values != n0)
        if len(changed):
            df = df.iloc[: changed[0]]
    X = df[C.INPUT_COLS].to_numpy(float)
    assert X.shape[1] == len(C.INPUT_COLS) and not set(C.INPUT_COLS) & set(C.OUTPUT_COLS)
    k = df[C.K_COL].to_numpy(float)
    n = df[C.N_COL].to_numpy(float)
    ydy = logit_dy(k, n)
    y = ydy if C.TARGET == "dy" else df[C.TARGET].to_numpy(float)
    return Dataset(ds_id, X, k, n, y, n_file, ydy)


_CACHE: dict[str, Dataset] = {}


def get_dataset(ds_id: str) -> Dataset:
    if ds_id not in _CACHE:
        _CACHE[ds_id] = load_dataset(ds_id)
    return _CACHE[ds_id]


# ----------------------------------------------------------------------------------------------------------------------
# Context selection (computed once, saved to context_selection.csv, read by every TFM)
# ----------------------------------------------------------------------------------------------------------------------
def _summary_vectors(ids: list[str]) -> pd.DataFrame:
    """Per dataset: mean and std of each input (z-scored with pooled stats) plus mean and std of logit DY."""
    allX = np.vstack([get_dataset(d).X for d in ids])
    mu, sd = allX.mean(0), allX.std(0)
    sd[sd == 0] = 1
    rows = {}
    for d in ids:
        ds = get_dataset(d)
        Z = (ds.X - mu) / sd
        rows[d] = np.concatenate([Z.mean(0), Z.std(0), [ds.ydy.mean(), ds.ydy.std()]])
    return pd.DataFrame.from_dict(rows, orient="index")


def _summary_current_warmup(d: str, ids: list[str]) -> np.ndarray:
    allX = np.vstack([get_dataset(i).X for i in ids])
    mu, sd = allX.mean(0), allX.std(0)
    sd[sd == 0] = 1
    ds = get_dataset(d)
    Z = (ds.X[: C.N_WARMUP] - mu) / sd
    y = ds.ydy[: C.N_WARMUP]
    return np.concatenate([Z.mean(0), Z.std(0), [y.mean(), y.std()]])


def compute_context_selection(ids: list[str] | None = None, n_days: int | None = None) -> pd.DataFrame:
    ids = list(C.DATASETS if ids is None else ids)
    n_days = C.N_CONTEXT_DAYS if n_days is None else n_days
    S = _summary_vectors(ids)
    # Each summary feature is put on a common scale so no single input dominates the distance.
    Sz = (S - S.mean()) / S.std(ddof=0).replace(0, 1)
    rng = np.random.default_rng(C.SEED)
    out = []
    for d in ids:
        pool = [o for o in ids if o != d]
        if C.CONTEXT_POOL == "previous_only":
            pool = [o for o in pool if o < d]
        if C.CONTEXT_SELECTION == "most_different":
            if C.CURRENT_DAY_SUMMARY_FROM == "warmup":
                v = (_summary_current_warmup(d, ids) - S.mean().values) / S.std(ddof=0).replace(0, 1).values
            else:
                v = Sz.loc[d].values
            dist = {o: float(np.linalg.norm(Sz.loc[o].values - v)) for o in pool}
            chosen = sorted(pool, key=lambda o: (-dist[o], o))[:n_days]
        elif C.CONTEXT_SELECTION == "most_recent":
            chosen = sorted(pool, key=lambda o: abs(int(o.replace("_", "")) - int(d.replace("_", ""))))[:n_days]
        elif C.CONTEXT_SELECTION == "random":
            chosen = list(rng.choice(sorted(pool), size=min(n_days, len(pool)), replace=False))
        else:
            raise ValueError(C.CONTEXT_SELECTION)
        out.append(dict(dataset=d, ctx5=";".join(chosen), ctxall=";".join(sorted(pool)),
                        selection=C.CONTEXT_SELECTION, pool=C.CONTEXT_POOL))
    return pd.DataFrame(out)


def load_context_selection(recompute: bool = False) -> pd.DataFrame:
    if recompute or not os.path.exists(C.CONTEXT_SELECTION_CSV):
        df = compute_context_selection()
        df.to_csv(C.CONTEXT_SELECTION_CSV, index=False)
        return df
    df = pd.read_csv(C.CONTEXT_SELECTION_CSV, dtype=str)
    if sorted(df.dataset) != sorted(C.DATASETS):
        raise RuntimeError("context_selection.csv does not match config.DATASETS; rerun with --recompute-context")
    return df


def context_ids(ds_id: str, arm: str, sel: pd.DataFrame) -> list[str]:
    if arm == "ctx0":
        return []
    s = sel.set_index("dataset").loc[ds_id, arm]
    return [] if pd.isna(s) or s == "" else s.split(";")


def context_arrays(ids: list[str]):
    if not ids:
        return np.empty((0, len(C.INPUT_COLS))), np.empty(0)
    dss = [get_dataset(i) for i in ids]
    return np.vstack([d.X for d in dss]), np.concatenate([d.y for d in dss])


# ----------------------------------------------------------------------------------------------------------------------
# Adapter output contract
# ----------------------------------------------------------------------------------------------------------------------
def finalize_output(mean, quantiles, sd=None):
    """Sort quantiles per row (fixes crossings) and derive the median. Returns dict of arrays."""
    q = np.sort(np.asarray(quantiles, float), axis=1)
    if q.shape[1] != len(C.QUANTILE_LEVELS):
        raise ValueError(f"expected {len(C.QUANTILE_LEVELS)} quantile levels, got {q.shape[1]}")
    mean = np.asarray(mean, float).reshape(-1)
    if not (np.isfinite(q).all() and np.isfinite(mean).all()):
        raise ValueError("non-finite prediction")
    med = q[:, int(np.flatnonzero(np.isclose(C.QUANTILE_LEVELS, 0.5))[0])]
    out = dict(mean=mean, median=med, quantiles=q)
    out["sd"] = np.full(len(mean), np.nan) if sd is None else np.asarray(sd, float).reshape(-1)
    return out


def interp_quantiles(native_levels, native_q, levels=None):
    """Linear interpolation of a fixed quantile set (rows x native) onto the shared grid."""
    levels = C.QUANTILE_LEVELS if levels is None else levels
    native_q = np.sort(np.asarray(native_q, float), axis=1)
    return np.stack([np.interp(levels, native_levels, row) for row in native_q])


# ----------------------------------------------------------------------------------------------------------------------
# Protocol loop
# ----------------------------------------------------------------------------------------------------------------------
def result_dir(folder: str, ds_id: str) -> str:
    return os.path.join(C.RESULTS_DIR, folder, f"results_{ds_id}")


def run_protocol(ds_id: str, folder: str, model: str, arm: str, predictor, ctx_ids: list[str],
                 max_steps: int | None = None, verbose: bool = True) -> pd.DataFrame:
    """predictor(X_cur, y_cur, k_cur, n_cur, X_ctx, y_ctx, X_query) -> finalize_output(...) dict.

    At step t (1-based CSV row t = N_WARMUP+1 ... T) the model sees rows 1..t-1 of the current dataset (plus the full
    context days for ctx5/ctxall) and predicts row t only. Everything is refit from scratch at every step.
    Partial progress is checkpointed to predictions.partial.csv so an interrupted run resumes where it stopped.
    """
    ds = get_dataset(ds_id)
    Xc, yc = context_arrays(ctx_ids)
    out_dir = result_dir(folder, ds_id)
    os.makedirs(out_dir, exist_ok=True)
    partial = os.path.join(out_dir, "predictions.partial.csv")
    partial_rem = os.path.join(out_dir, "predictions_remaining.partial.csv")
    rows, rem_rows = [], []
    done = set()
    if os.path.exists(partial):
        prev = pd.read_csv(partial)
        rows = prev.to_dict("records")
        done = set(prev["iter"].astype(int))
        if os.path.exists(partial_rem):
            rem_rows = pd.read_csv(partial_rem).to_dict("records")
    T = len(ds.y)
    steps = list(range(C.N_WARMUP, T))  # 0-based index of the row being predicted
    if max_steps is not None:
        steps = steps[:max_steps]
    t0 = time.time()
    for j, t in enumerate(steps):
        it = t + 1
        if it in done:
            continue
        Xq = ds.X[t:] if C.PREDICT_REMAINING else ds.X[t:t + 1]
        tic = time.time()
        pred = predictor(ds.X[:t], ds.y[:t], ds.k[:t], ds.n[:t], Xc, yc, Xq)
        fit_s = time.time() - tic
        rec = dict(dataset=ds_id, model=model, arm=arm, iter=it, y_true=ds.y[t], k=ds.k[t], n=ds.n[t],
                   mean=pred["mean"][0], median=pred["median"][0], sd=pred["sd"][0], fit_seconds=fit_s,
                   n_context_rows=len(yc), n_train_rows=t)
        rec.update({c: v for c, v in zip(C.QCOLS, pred["quantiles"][0])})
        rows.append(rec)
        if C.PREDICT_REMAINING:
            for r in range(1, len(Xq)):
                rr = dict(dataset=ds_id, model=model, arm=arm, step_iter=it, target_iter=t + r + 1,
                          y_true=ds.y[t + r], k=ds.k[t + r], n=ds.n[t + r], mean=pred["mean"][r],
                          median=pred["median"][r], sd=pred["sd"][r])
                rr.update({c: v for c, v in zip(C.QCOLS, pred["quantiles"][r])})
                rem_rows.append(rr)
        if (j + 1) % 25 == 0 or j == len(steps) - 1:
            pd.DataFrame(rows).to_csv(partial, index=False)
            if rem_rows:
                pd.DataFrame(rem_rows).to_csv(partial_rem, index=False)
            if verbose:
                print(f"  [{folder}] {ds_id} step {it}/{T}  {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows).sort_values("iter").reset_index(drop=True)
    df.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)
    if rem_rows:
        pd.DataFrame(rem_rows).to_csv(os.path.join(out_dir, "predictions_remaining.csv"), index=False)
    for p in (partial, partial_rem):
        if os.path.exists(p):
            os.remove(p)
    return df


def write_metrics(df: pd.DataFrame, folder: str, ds_id: str):
    if C.TARGET != "dy":  # binomial metrics make no sense for EUV / Rx / Ry / TargetSize
        import metrics_cont as M
    else:
        import metrics as M
    out_dir = result_dir(folder, ds_id)
    M.step_metrics(df).to_csv(os.path.join(out_dir, "step_metrics.csv"), index=False)
    M.final_metrics(df).to_csv(os.path.join(out_dir, "final_metrics.csv"), index=False)


def parse_list(s: str | None, default: list[str]) -> list[str]:
    if s is None or s in ("all", ""):
        return list(default)
    return [x.strip() for x in s.split(",") if x.strip()]
