"""LEAR, point and quantiles (handover Section 8, row "LEAR and naive implementation (step 2)", (b)-(d)).

(b) VST as Stan: med and MAD / Phi^-1(0.75) of the window target (MAD < 1e-6 -> 1), asinh((x - med) / MAD)
    on the target and the 5 price features. Then the 11 continuous regressors z-scored with the window mean
    and population std (ddof 0, std < 1e-12 -> 1), same transform on the test rows; dummies 0 / 1.
(c) Point: LassoCV(alphas=100, eps=1e-4), lambda.min, refit on the window, max_iter 1e5, tol 1e-7,
    10 folds of whole days (one default_rng(2027) per fit, permutation of the window's dates, fold =
    position mod 10, PredefinedSplit); "row" folds for R0. Smearing back-transform with the in-sample
    residuals at the selected alpha.
(d) Quantiles: QuantileRegressor(quantile=tau, alpha=lambda/(2n), solver="highs") for lambda in
    10^seq(-1, 3, length 20). BIC as Stan: log(sum rho_tau) + m log(n) / (2 n log(p)),
    m = #{j >= 1: |beta_j| > 1e-8}, strict minimum (smallest lambda wins ties), plain inverse VST,
    point forecast as fallback if no lambda solves, the 9 levels sorted per row.
"""
import time
import warnings

import numpy as np
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LassoCV, QuantileRegressor
from sklearn.model_selection import PredefinedSplit

from ..data import CONTINUOUS, DOW, PRICE, PRICE_FEATURES, QCOLS, QDUM, TAUS

Z075 = norm.ppf(0.75)
LAMBDA_GRID = 10.0 ** np.linspace(-1, 3, 20)
SEED = 2027
N_FOLDS = 10


def vst(x, med, mad):
    return np.arcsinh((x - med) / mad)


def design(fit, qdum=True, standardize=True):
    """X_train, y_train, X_test and the VST parameters for one hour-fit."""
    tr, te = fit.train, fit.test
    price = tr[PRICE].to_numpy(dtype=float)
    med = float(np.median(price))
    mad = float(np.median(np.abs(price - med))) / Z075
    if mad < 1e-6:
        mad = 1.0
    y = vst(price, med, mad)

    def cont(df):
        cols = []
        for c in CONTINUOUS:
            v = df[c].to_numpy(dtype=float)
            cols.append(vst(v, med, mad) if c in PRICE_FEATURES else v)
        return np.column_stack(cols)

    Ctr, Cte = cont(tr), cont(te)
    if standardize:
        mu = Ctr.mean(axis=0)
        sd = Ctr.std(axis=0, ddof=0)
        sd = np.where(sd < 1e-12, 1.0, sd)
        Ctr, Cte = (Ctr - mu) / sd, (Cte - mu) / sd
    dums = DOW + (QDUM if (qdum and fit.is15) else [])        # hourly: no quarter dummies, p = 17
    Xtr = np.column_stack([Ctr, tr[dums].to_numpy(dtype=float)])
    Xte = np.column_stack([Cte, te[dums].to_numpy(dtype=float)])
    return Xtr, y, Xte, med, mad


def fold_ids(fit, folds="day", seed=SEED):
    rng = np.random.default_rng(seed)
    if folds == "day":
        perm = rng.permutation(fit.window_days)
        pos = {int(d): k % N_FOLDS for k, d in enumerate(perm)}
        return np.array([pos[int(d)] for d in fit.train["day"].to_numpy()])
    if folds == "row":                     # glmnet: sample(rep(seq(nfolds), length = N))
        return rng.permutation(np.arange(len(fit.train)) % N_FOLDS)
    raise ValueError(folds)


def pinball_sum(resid, tau):
    return float(np.sum(np.where(resid >= 0, tau * resid, (tau - 1) * resid)))


def fit_lqr(X, y, tau, lam):
    """One LQR program. Returns (model or None, linprog status)."""
    n = X.shape[0]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning)
        try:
            m = QuantileRegressor(quantile=tau, alpha=lam / (2 * n), fit_intercept=True, solver="highs").fit(X, y)
        except Exception:                  # no solution returned (linprog x is None)
            return None, -1
    for wi in w:
        if issubclass(wi.category, ConvergenceWarning) and "did not succeed" in str(wi.message):
            msg = str(wi.message)
            status = int(msg.split("Status is ")[1].split(":")[0]) if "Status is " in msg else -1
            return None, status            # R: tryCatch(error) -> next
    return m, 0


def select_bic(X, y, tau, grid=LAMBDA_GRID, Xte=None, med=None, mad=None):
    """Stan's BIC selection over the grid (rule (i), in-fit). With Xte, also the lambda path: per lambda the
    back-transformed test-row quantile (NaN if the program failed), the window check loss, m and the status."""
    n, p = X.shape
    best = (np.inf, None, None, None)      # bic, lambda, model, m
    n_failed = 0
    path = None if Xte is None else {"q": np.full((len(Xte), len(grid)), np.nan), "loss": np.full(len(grid), np.nan),
                                      "m": np.full(len(grid), -1, dtype=int), "status": np.zeros(len(grid), dtype=int)}
    for k, lam in enumerate(grid):
        m_, status = fit_lqr(X, y, tau, lam)
        if path is not None:
            path["status"][k] = status
        if m_ is None:
            n_failed += 1
            continue
        loss = pinball_sum(y - m_.predict(X), tau)
        m = int(np.sum(np.abs(m_.coef_) > 1e-8))
        if path is not None:
            path["q"][:, k] = med + mad * np.sinh(m_.predict(Xte))
            path["loss"][k], path["m"][k] = loss, m
        bic = np.log(loss) + m * np.log(n) / (2 * n * np.log(p))
        if bic < best[0]:
            best = (bic, lam, m_, m)
    return (best, n_failed) if path is None else (best, n_failed, path)


def fit_predict(fit, qdum=True, standardize=True, folds="day", seed=SEED, path=False):
    """point (n_test,), Q (n_test, 9) sorted, log dict; with path=True also a dict with the lambda path:
    q (n_test, 9, 20) unsorted back-transformed quantiles, loss / m / status (9, 20)."""
    t0 = time.perf_counter()
    Xtr, y, Xte, med, mad = design(fit, qdum=qdum, standardize=standardize)
    n, p = Xtr.shape

    # (c) point
    cv = PredefinedSplit(fold_ids(fit, folds, seed))
    las = LassoCV(alphas=100, eps=1e-4, cv=cv, max_iter=100000, tol=1e-7, fit_intercept=True).fit(Xtr, y)
    resid = y - las.predict(Xtr)
    yhat = las.predict(Xte)
    point = med + mad * np.mean(np.sinh(yhat[:, None] + resid[None, :]), axis=1)
    t1 = time.perf_counter()

    # (d) quantiles
    Q = np.empty((len(Xte), len(TAUS)))
    log = {"n_train": n, "n_test": len(Xte), "p": p, "med": med, "mad": mad,
           "lasso_alpha": float(las.alpha_), "lasso_nnz": int(np.sum(las.coef_ != 0)),
           "lasso_alpha_max": float(las.alphas_.max()), "lasso_at_grid_min": bool(las.alpha_ == las.alphas_.min())}
    G = len(LAMBDA_GRID)
    lp = {"q": np.full((len(Xte), len(TAUS), G), np.nan), "loss": np.full((len(TAUS), G), np.nan),
          "m": np.full((len(TAUS), G), -1, dtype=int), "status": np.zeros((len(TAUS), G), dtype=int)} if path else None
    for k, tau in enumerate(TAUS):
        if path:
            (bic, lam, model, m), n_failed, pk = select_bic(Xtr, y, tau, Xte=Xte, med=med, mad=mad)
            lp["q"][:, k, :] = pk["q"]
            for key in ("loss", "m", "status"):
                lp[key][k] = pk[key]
        else:
            (bic, lam, model, m), n_failed = select_bic(Xtr, y, tau)
        c = QCOLS[k]
        if model is None:
            Q[:, k] = point
            log.update({f"lambda_{c}": np.nan, f"m_{c}": -1, f"bic_{c}": np.nan, f"fallback_{c}": True})
        else:
            Q[:, k] = med + mad * np.sinh(model.predict(Xte))
            log.update({f"lambda_{c}": float(lam), f"m_{c}": m, f"bic_{c}": float(bic), f"fallback_{c}": False})
        log[f"nfail_{c}"] = n_failed
        log[f"status_{c}"] = 0 if model is not None else -1
    Q = np.sort(Q, axis=1)
    t2 = time.perf_counter()
    log.update({"t_lasso_s": t1 - t0, "t_lqr_s": t2 - t1, "wall_s": t2 - t0})
    return (point, Q, log, lp) if path else (point, Q, log)
