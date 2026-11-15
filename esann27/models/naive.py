"""Naive benchmark as Stan coded it (handover Section 8, step 2 row (e), model_naive_*.R).

Point = price_lag_d7 of the test row. Quantiles = point + empirical quantile (R type 7, numpy "linear") of
price - price_lag_d7 over the window rows of the same clock hour (the 4 quarters pooled), sorted per row.
"""
import time

import numpy as np

from ..data import PRICE, TAUS


def fit_predict(fit, **_):
    t0 = time.perf_counter()
    err = (fit.train[PRICE] - fit.train["price_lag_d7"]).to_numpy(dtype=float)
    err = err[~np.isnan(err)]
    point = fit.test["price_lag_d7"].to_numpy(dtype=float)
    if len(err):
        offs = np.quantile(err, TAUS, method="linear")
        Q = np.sort(point[:, None] + offs[None, :], axis=1)
    else:
        Q = np.repeat(point[:, None], len(TAUS), axis=1)
    log = {"n_train": len(fit.train), "n_test": len(fit.test), "wall_s": time.perf_counter() - t0}
    return point, Q, log
