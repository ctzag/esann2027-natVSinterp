"""Point and probabilistic metrics against the native target (handover Section 8, "Primary metrics").

CRPS = 2 x mean pinball over the 9 levels 0.1 ... 0.9 (quantile score at nine levels). PICP and mean width
of the central 80% [q10, q90] and 60% [q20, q80] intervals, bounds inclusive as in Stan's scripts.
"""
import numpy as np
import pandas as pd

from .data import QCOLS, TAUS


def pinball(y, q, tau):
    d = y - q
    return np.where(d >= 0, tau * d, (tau - 1) * d)


def pinball_by_level(df):
    y = df["actual"].to_numpy(float)
    return {c: float(np.mean(pinball(y, df[c].to_numpy(float), t))) for c, t in zip(QCOLS, TAUS)}


def summary(df):
    y, p = df["actual"].to_numpy(float), df["point"].to_numpy(float)
    pl = pinball_by_level(df)
    out = {"n": len(df), "MAE": float(np.mean(np.abs(y - p))), "RMSE": float(np.sqrt(np.mean((y - p) ** 2))),
           "CRPS": 2.0 * float(np.mean(list(pl.values())))}
    for lev, lo, hi in ((80, "q10", "q90"), (60, "q20", "q80")):
        a, b = df[lo].to_numpy(float), df[hi].to_numpy(float)
        out[f"PICP{lev}"] = 100.0 * float(np.mean((y >= a) & (y <= b)))
        out[f"width{lev}"] = float(np.mean(b - a))
    out.update({f"pinball_{c}": v for c, v in pl.items()})
    return out


def period(df, start, end):
    d = pd.to_datetime(df["date"])
    return df[(d >= pd.Timestamp(start)) & (d < pd.Timestamp(end))]
