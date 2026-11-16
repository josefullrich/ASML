"""Kumo Tabular and Causilo backends.

Regression and unweighted classification use each library's own public API.
Weighted classification (frequency weights: a row with weight m behaves like m copies)
is not in either public API, so three routes are supported, tried in this order when
weights_backend = "auto":

  sample_weight  the installed estimator's fit() accepts sample_weight (e.g. the Causilo PR build)
  ivo            the weighted PyTorch modules from iwoodsawyer/tfm_onnx_in_matlab
                 (convert_kumo_to_onnx.WeightedKumoExport / convert_causilo_to_onnx.WeightedCausiloExport)
  expand         repeat rows (integer weights), thinning proportionally above expand_max_rows
"""
from __future__ import annotations
import inspect, os, sys
import numpy as np
from .base import Backend, LEVELS, interp_levels

RAW_LEVELS = np.arange(1, 1000) / 1000.0


def _device(cfg):
    import torch
    if cfg.device != "auto":
        return cfg.device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def _import_ivo(cfg, module):
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, "..", "..", ".."))
    for cand in [cfg.ivo_repo, os.path.join(repo, cfg.ivo_repo), os.path.join(os.getcwd(), cfg.ivo_repo),
                 os.path.join("/content", "tfm_onnx_in_matlab")]:
        if os.path.isfile(os.path.join(cand, module + ".py")):
            if cand not in sys.path:
                sys.path.insert(0, cand)
            return __import__(module)
    raise FileNotFoundError(f"{module}.py not found; clone iwoodsawyer/tfm_onnx_in_matlab and set ivo_repo")


def expand_rows(X, y, w, max_rows, rng):
    """Integer row repetition; above max_rows the weights are thinned proportionally (noted once)."""
    w = np.asarray(w, float)
    total = w.sum()
    if total > max_rows:
        w = w * (max_rows / total)
    reps = np.floor(w).astype(int)
    reps += rng.random(len(w)) < (w - reps)            # unbiased rounding of the fractional part
    idx = np.repeat(np.arange(len(w)), reps)
    return X[idx], y[idx]


def weighted_stats(X, w):
    """Mean and sample standard deviation of the frequency-expanded rows."""
    w = np.asarray(w, float)[:, None]
    n = w.sum()
    mu = (w * X).sum(0) / n
    var = (w * (X - mu) ** 2).sum(0) / max(n - 1.0, 1.0)
    sd = np.sqrt(var)
    sd[~(sd > 0)] = 1.0
    return mu, sd


class _TFM(Backend):
    def __init__(self, cfg):
        self.cfg, self.dev = cfg, _device(cfg)
        self._weighted, self._route = None, None
        self.rng = np.random.default_rng(0)
        self._warned = False

    def _route_for_weights(self, fit_fn):
        if self._route:
            return self._route
        want = self.cfg.weights_backend
        has_sw = "sample_weight" in inspect.signature(fit_fn).parameters
        if want in ("auto", "sample_weight") and has_sw:
            self._route = "sample_weight"
        elif want in ("auto", "ivo"):
            try:
                self._load_weighted()
                self._route = "ivo"
            except Exception as err:
                if want == "ivo":
                    raise
                print(f"  [{self.name}] weighted graph unavailable ({err}); falling back to row expansion")
                self._route = "expand"
        else:
            self._route = "expand"
        print(f"  [{self.name}] weighted classifier route: {self._route}")
        return self._route


class KumoBackend(_TFM):
    name = "kumo"

    def __init__(self, cfg, pretrained=True):
        super().__init__(cfg)
        self.pretrained, self._m = pretrained, {}

    def _model(self, task):
        if task not in self._m:
            from sdm.models import KumoTabular
            self._m[task] = KumoTabular(task=task, size=self.cfg.kumo_size, pretrained=self.pretrained, device=self.dev).eval()
        return self._m[task]

    def _fit_predict(self, task, Xc, yc, Xq):
        import torch
        m = self._model(task)
        g = torch.Generator().manual_seed(0)
        x = torch.as_tensor(np.asarray(Xc), dtype=torch.float32)
        y = torch.as_tensor(np.asarray(yc), dtype=torch.float32 if task == "regression" else torch.long).unsqueeze(-1)
        outs = []
        with torch.no_grad():
            m.fit(x, y, num_estimators=self.cfg.n_estimators, generator=g)
            for i in range(0, len(Xq), self.cfg.query_batch):
                outs.append(m.predict(torch.as_tensor(np.asarray(Xq[i:i + self.cfg.query_batch]), dtype=torch.float32)).to_pandas())
        import pandas as pd
        return pd.concat(outs, ignore_index=True)

    def regress(self, Xc, yc, Xq, tag=None):
        df = self._fit_predict("regression", Xc, yc, Xq)
        raw = df[sorted(df.columns)].to_numpy(float)          # q001..q999
        Q = interp_levels(raw, RAW_LEVELS)
        return raw.mean(1), Q, Q

    def _load_weighted(self):
        if self._weighted is None:
            mod = _import_ivo(self.cfg, "convert_kumo_to_onnx")
            (inner,) = self._model("classification").models.values()
            self._weighted = mod.WeightedKumoExport(inner.float().eval(), True).eval()
        return self._weighted

    def classify(self, Xc, yc, Xq, n_classes, weights=None):
        Xc, yc, Xq = np.asarray(Xc, float), np.asarray(yc).astype(int), np.asarray(Xq, float)
        if weights is not None:
            keep = np.asarray(weights) > 0
            Xc, yc, weights = Xc[keep], yc[keep], np.asarray(weights, float)[keep]
            route = self._route_for_weights(self._model("classification").fit)
            if route == "ivo":
                return self._classify_weighted(Xc, yc, Xq, n_classes, weights)
            Xc, yc = expand_rows(Xc, yc, weights, self.cfg.expand_max_rows, self.rng)
        df = self._fit_predict("classification", Xc, yc, Xq)
        P = np.zeros((len(Xq), n_classes))
        for col in df.columns:                                # columns are class labels, not in sorted order
            P[:, int(float(col))] = df[col].to_numpy(float)
        return P

    def _classify_weighted(self, Xc, yc, Xq, n_classes, w):
        import torch
        net = self._load_weighted()
        mu, sd = weighted_stats(Xc, w)
        xc = torch.as_tensor((Xc - mu) / sd, dtype=torch.float32, device=self.dev)
        y = torch.as_tensor(yc, dtype=torch.int32, device=self.dev)
        wt = torch.as_tensor(w, dtype=torch.float32, device=self.dev)
        mask = torch.zeros(Xc.shape[1], dtype=torch.bool, device=self.dev)
        out = []
        with torch.no_grad():
            for i in range(0, len(Xq), self.cfg.query_batch):
                xq = torch.as_tensor((Xq[i:i + self.cfg.query_batch] - mu) / sd, dtype=torch.float32, device=self.dev)
                out.append(net.run(torch.cat((xc, xq)), y, mask, wt).float().cpu().numpy())
        return _softmax(np.concatenate(out)[:, :n_classes])


class CausiloBackend(_TFM):
    name = "causilo"

    def _est(self, cls):
        return cls(n_estimators=self.cfg.n_estimators, device=self.dev)

    def regress(self, Xc, yc, Xq, tag=None):
        from causilo import CausiloRegressor
        est = self._est(CausiloRegressor).fit(np.asarray(Xc, np.float32), np.asarray(yc, np.float32))
        raws = [est.predict(np.asarray(Xq[i:i + self.cfg.query_batch], np.float32), output_type="raw")
                for i in range(0, len(Xq), self.cfg.query_batch)]
        raw = np.concatenate(raws).astype(float)
        Q = interp_levels(raw, RAW_LEVELS)
        return raw.mean(1), Q, Q

    def _load_weighted(self):
        if self._weighted is None:
            mod = _import_ivo(self.cfg, "convert_causilo_to_onnx")
            from causilo.checkpoints import load_pretrained_model
            model = load_pretrained_model("classification").float().eval()
            self._weighted = mod.WeightedCausiloExport(model).eval().to(self.dev)
        return self._weighted

    def classify(self, Xc, yc, Xq, n_classes, weights=None):
        from causilo import CausiloClassifier
        Xc, yc, Xq = np.asarray(Xc, np.float32), np.asarray(yc).astype(int), np.asarray(Xq, np.float32)
        kw = {}
        if weights is not None:
            keep = np.asarray(weights) > 0
            Xc, yc, weights = Xc[keep], yc[keep], np.asarray(weights, float)[keep]
            route = self._route_for_weights(CausiloClassifier.fit)
            if route == "ivo":
                return self._classify_weighted(Xc, yc, Xq, n_classes, weights)
            if route == "sample_weight":
                kw = {"sample_weight": weights}
            else:
                Xc, yc = expand_rows(Xc, yc, weights, self.cfg.expand_max_rows, self.rng)
        est = self._est(CausiloClassifier).fit(Xc, yc, **kw)
        P = np.zeros((len(Xq), n_classes))
        for i in range(0, len(Xq), self.cfg.query_batch):
            P[i:i + self.cfg.query_batch][:, est.classes_.astype(int)] = est.predict_proba(Xq[i:i + self.cfg.query_batch])
        return P

    def _classify_weighted(self, Xc, yc, Xq, n_classes, w):
        """Causilo's first-ensemble-member preprocessing (weighted z-score, arcsinh beyond four sigma),
        then the weighted network. One estimator, as configured."""
        import torch
        net = self._load_weighted()
        wn = np.asarray(w, float)[:, None]
        mu = (wn * Xc).sum(0) / wn.sum()
        sd = np.sqrt((wn * (Xc - mu) ** 2).sum(0) / wn.sum())
        sd[~(sd > 0)] = 1.0

        def norm(X):
            Z = (X - mu) / sd
            hi, lo = Z > 4, Z < -4
            Z = np.where(hi, 4 + np.arcsinh(Z - 4), Z)
            return np.where(lo, -4 - np.arcsinh(-4 - Z), Z)
        t = lambda a, dt: torch.as_tensor(a, dtype=dt, device=self.dev).unsqueeze(0)
        cx, cy, cw = t(norm(Xc), torch.float32), t(yc, torch.int64), t(w, torch.float32)
        out = []
        with torch.no_grad():
            for i in range(0, len(Xq), self.cfg.query_batch):
                out.append(net.run(cx, cy, t(norm(Xq[i:i + self.cfg.query_batch]), torch.float32), cw)[0].float().cpu().numpy())
        return _softmax(np.concatenate(out)[:, :n_classes])
