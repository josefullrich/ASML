"""Python port of the MATLAB ClassificationGP / fitcgp used in example_bayesopt.m (dy_model = 2).

Faithful to the MATLAB call
    fitcgp(X, [k n], 'Standardize', 1, 'Basis', 'none', 'KernelFunction', 'ardmatern52')
with all other options at their MATLAB defaults:

* Inputs are z-scored with the training mean and std (std with ddof=1, zero std -> 1).
* Latent GP f with zero prior mean (Basis 'none').
* ARD Matern 5/2 kernel. Initial kernel parameters are all ones (length scales = 1, signal sd = 1),
  as in ClassificationGP.defaultKernelParameters (NOT the [std(X); std(Y)/sqrt(2)] from the docstring).
* Binomial likelihood fed [k, n]: y = k/n, weight w = n, logit link,
  log p(k | f) = n*(y*log(p) + (1-y)*log(1-p)) with p clipped to [eps, 1-eps].
* Laplace approximation: Newton iterations with backtracking line search, MaxIter = 50, Tol = 1e-6,
  W = max(w*p*(1-p), eps), cholSafe jitter ladder 1e-10 ... 1e-6.
* Approximate log evidence logZ = sum log p(k|f_hat) - 0.5*alpha'f_hat - sum(log(diag(L))), without the binomial
  coefficient (a constant that does not change the optimum).
* Kernel hyperparameters optimised in log space by a quasi-Newton (BFGS) method, starting from ones at every
  refit. Lambda = 0 (ConstantLambda = true). If the optimum is non-finite the initial parameters are kept.

Differences from MATLAB, all deliberate:

* Gradient: MATLAB's fminunc uses finite differences (no gradient is supplied). Here the exact gradient of the Laplace
  evidence is used (explicit + implicit term, Rasmussen & Williams Alg. 5.1). Same objective, same local optimum,
  about 10x fewer evidence evaluations. Set `grad="fd"` to reproduce the finite-difference behaviour.
* The two known bugs are not ported: `loss()` with a function handle referenced an undefined `yPred`, and the binomial
  log-coefficient constant used gammaln(n+1)+gammaln(k+1)+gammaln(n-k+1) instead of
  gammaln(n+1)-gammaln(k+1)-gammaln(n-k+1). Neither is used for fitting; the correct constant is in
  `binomial_log_coef`.

Prediction returns the latent mean and variance of f at new inputs. f lives on the logit scale, so the predictive
distribution of logit(DY) used by the benchmark is Gaussian(mean_f, var_f).
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.linalg import cho_solve, solve_triangular
from scipy.optimize import minimize
from scipy.special import expit, gammaln

EPS = np.finfo(float).eps
MIN_JITTER, MAX_JITTER = 1e-10, 1e-6
MAX_ITER, TOL = 50, 1e-6
SQRT5 = np.sqrt(5.0)


def binomial_log_coef(k, n):
    """log C(n, k) with the correct signs (the MATLAB constant had the wrong sign on two terms)."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def chol_safe(B):
    """Lower Cholesky with the same jitter ladder as ClassificationGP.cholSafe. Returns None on failure."""
    if not np.isfinite(B).all():
        return None
    try:
        return np.linalg.cholesky(B)
    except np.linalg.LinAlgError:
        pass
    n = B.shape[0]
    B = 0.5 * (B + B.T) + 1e-12 * np.eye(n)
    try:
        return np.linalg.cholesky(B)
    except np.linalg.LinAlgError:
        pass
    jitter = MIN_JITTER
    while jitter <= MAX_JITTER:
        try:
            return np.linalg.cholesky(B + jitter * np.eye(n))
        except np.linalg.LinAlgError:
            jitter = 10 * max(jitter, 1e-10)
    return None


def _sq_dist_per_dim(X1, X2):
    """(n1, n2, d) array of squared coordinate differences."""
    return (X1[:, None, :] - X2[None, :, :]) ** 2


def matern52(X1, X2, log_theta, D2=None):
    """ARD Matern 5/2. log_theta = [log ell_1..log ell_d, log sf]. Returns K and (optionally) the cached D2."""
    d = X1.shape[1]
    ell = np.exp(log_theta[:d])
    sf2 = np.exp(2 * log_theta[d])
    if D2 is None:
        D2 = _sq_dist_per_dim(X1, X2)
    S = D2 / ell**2  # scaled squared differences, (n1, n2, d)
    r = np.sqrt(np.maximum(S.sum(-1), 0.0))
    a = SQRT5 * r
    K = sf2 * (1 + a + a**2 / 3) * np.exp(-a)
    return K, S, a


def matern52_grads(K, S, a, log_theta):
    """dK/dlog(ell_i) for each i and dK/dlog(sf). Returns list of (n, n) arrays, length d+1."""
    d = S.shape[-1]
    sf2 = np.exp(2 * log_theta[d])
    common = sf2 * (5.0 / 3.0) * (1 + a) * np.exp(-a)  # dk/dlog ell_i = common * S_i
    grads = [common * S[:, :, i] for i in range(d)]
    grads.append(2 * K)
    return grads


def likelihood_moments(f, y, w):
    """Logit-link binomial log-likelihood per row, gradient, W, p (MATLAB likelihoodMoments, logit case)."""
    p = np.clip(expit(f), EPS, 1 - EPS)
    logp = w * (y * np.log(p) + (1 - y) * np.log(1 - p))
    grad = w * (y - p)
    W = np.maximum(w * p * (1 - p), EPS)
    return logp, grad, W, p


def infer_laplace(K, y, w):
    """Newton iterations with backtracking (MATLAB inferLaplace, zero mean). Returns dict or None on failure."""
    n = K.shape[0]
    f = np.zeros(n)
    alpha = np.zeros(n)
    obj_old = -np.inf
    it = 0
    for it in range(1, MAX_ITER + 1):
        logL, grad, W, _ = likelihood_moments(f, y, w)
        obj = logL.sum() - 0.5 * alpha @ f
        if abs(obj - obj_old) < TOL * max(1.0, abs(obj)):
            break
        obj_old = obj
        sW = np.sqrt(W)
        L = chol_safe(sW[:, None] * K * sW[None, :] + np.eye(n))
        if L is None:
            return None
        b = W * f + grad
        a_dir = b - sW * cho_solve((L, True), sW * (K @ b))
        d_alpha = a_dir - alpha
        d_f = K @ d_alpha
        step = 1.0
        while step > EPS:
            alpha_new = alpha + step * d_alpha
            f_new = f + step * d_f
            logL_new = likelihood_moments(f_new, y, w)[0]
            if logL_new.sum() - 0.5 * alpha_new @ f_new > obj:
                f, alpha = f_new, alpha_new
                break
            step *= 0.5
    if it == MAX_ITER:
        warnings.warn("ClassificationGP:MaxIterReached", RuntimeWarning)
    logL, grad, W, p = likelihood_moments(f, y, w)
    sW = np.sqrt(W)
    L = chol_safe(sW[:, None] * K * sW[None, :] + np.eye(n))
    if L is None:
        return None
    logZ = logL.sum() - 0.5 * alpha @ f - np.log(np.diag(L)).sum()
    return dict(f=f, alpha=alpha, W=W, sW=sW, L=L, p=p, grad=grad, logZ=logZ)


def neg_log_evidence_and_grad(log_theta, X, y, w, D2, want_grad=True):
    fail = (1e10, np.zeros_like(log_theta)) if want_grad else 1e10
    if not np.isfinite(log_theta).all() or np.abs(log_theta).max() > 30:  # exp(30): far outside any sane scale
        return fail
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        K, S, a = matern52(X, X, log_theta, D2)
    if not np.isfinite(K).all():
        return fail
    try:
        post = infer_laplace(K, y, w)
    except (ValueError, np.linalg.LinAlgError):
        post = None
    if post is None or not np.isfinite(post["logZ"]):
        return fail
    nlz = -post["logZ"]
    if not want_grad:
        return nlz
    # Exact gradient of the Laplace evidence (R&W Alg. 5.1, logit binomial likelihood).
    L, sW, alpha, p = post["L"], post["sW"], post["alpha"], post["p"]
    n = len(y)
    Linv_sW = solve_triangular(L, np.diag(sW), lower=True)  # L^{-1} diag(sW)
    R = Linv_sW.T @ Linv_sW  # sW B^{-1} sW
    C = solve_triangular(L, sW[:, None] * K, lower=True)
    d3 = -w * p * (1 - p) * (1 - 2 * p)  # third derivative of log-lik w.r.t. f
    # d/df_i of -0.5*log|B| = -0.5*Sigma_ii*dW_ii/df_i and dW_ii/df_i = -d3_i, so the sign is + (as in GPML's
    # infLaplace `dfhat = g.*d3lp`; the book's Alg. 5.1 prints it with a minus).
    s2 = 0.5 * (np.diag(K) - (C**2).sum(0)) * d3
    g = np.empty_like(log_theta)
    for j, dK in enumerate(matern52_grads(K, S, a, log_theta)):
        explicit = 0.5 * alpha @ dK @ alpha - 0.5 * np.sum(R * dK)
        b = dK @ post["grad"]
        implicit = s2 @ (b - K @ (R @ b))
        g[j] = -(explicit + implicit)
    del n
    if not np.isfinite(g).all():
        return fail
    return nlz, g


class LaplaceGPC:
    """Binomial GP classifier with Laplace inference. fit(X, k, n) then predict_latent(Xq) -> (mean_f, var_f)."""

    def __init__(self, grad: str = "exact", max_opt_iter: int = 400):
        self.grad = grad
        self.max_opt_iter = max_opt_iter

    def fit(self, X, k, n):
        X = np.asarray(X, float)
        k, n = np.asarray(k, float), np.asarray(n, float)
        self.mu_ = X.mean(0)
        sig = X.std(0, ddof=1)
        sig[sig == 0] = 1.0
        self.sig_ = sig
        Xs = (X - self.mu_) / self.sig_
        y, w = k / n, n
        D2 = _sq_dist_per_dim(Xs, Xs)
        theta0 = np.zeros(X.shape[1] + 1)  # log(ones)
        if self.grad == "exact":
            res = minimize(neg_log_evidence_and_grad, theta0, args=(Xs, y, w, D2, True), jac=True,
                           method="BFGS", options=dict(maxiter=self.max_opt_iter, gtol=1e-6))
        else:  # finite differences, like MATLAB fminunc without a supplied gradient
            res = minimize(neg_log_evidence_and_grad, theta0, args=(Xs, y, w, D2, False), jac=None,
                           method="BFGS", options=dict(maxiter=self.max_opt_iter, gtol=1e-6))
        self.opt_result_ = res
        theta = res.x if np.isfinite(res.fun) and res.fun < 1e10 else theta0
        self.log_theta_ = theta
        K, _, _ = matern52(Xs, Xs, theta, D2)
        post = infer_laplace(K, y, w)
        if post is None:  # fall back to the initial parameters, as MATLAB keeps argsIn on failure
            self.log_theta_ = theta0
            K, _, _ = matern52(Xs, Xs, theta0, D2)
            post = infer_laplace(K, y, w)
        self.post_ = post
        self.Xs_ = Xs
        return self

    def predict_latent(self, Xq):
        Xq = (np.atleast_2d(np.asarray(Xq, float)) - self.mu_) / self.sig_
        Kxs, _, _ = matern52(self.Xs_, Xq, self.log_theta_)
        mean = Kxs.T @ self.post_["alpha"]
        v = solve_triangular(self.post_["L"], self.post_["sW"][:, None] * Kxs, lower=True)
        sf2 = np.exp(2 * self.log_theta_[-1])
        var = np.maximum(sf2 - (v**2).sum(0), 0.0)
        return mean, var

    def predict_proba(self, Xq):
        """MATLAB predict score(:,2) with ProbitScaling 'none': sigmoid(mean_f)."""
        return expit(self.predict_latent(Xq)[0])
