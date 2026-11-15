"""Evaluation of the raw and conformalized forecasts (handover Section 8, row "Evaluation implementation
(step 5.2)", items (a)-(l); Section 8.2 for tables, figures, numbers.tex and the stylized model).

  python -m esann27.evaluate --model lear [--n-boot 1000] [--n-jobs 8]

Order of work (item (i) first): the Proposition A prediction is computed from the burn-in only and written to
results/pred_doseresponse.csv and to numbers.tex before any evaluation-period metric is computed.

Inputs (a): forecasts/<setting>_<model>.parquet (raw), conformal/<setting>_<model>_<regime>.parquet, the naive
forecasts for rMAE, evaluation period 29 Oct 2025 - 30 Sep 2026 (32,348 quarters), target = `actual`.

Metrics: MAE, RMSE, rMAE (b). CRPS = 2 x mean over the 9 levels of the mean pinball (c). CORP per level (d):
isotonic tau-quantile regression of y on the issued quantile x by PAV with the pooled lower tau-quantile
(np.quantile(method="inverted_cdf")) as pooling functional, ties in x pooled first;
MCB_tau = S - S(recal), UNC_tau = S(unconditional lower tau-quantile of y), DSC_tau = UNC_tau - S(recal);
MCB, DSC, UNC = 2 x mean over levels, CRPS = MCB - DSC + UNC (asserted). PICP and mean width at 80 and 60,
bounds inclusive; Kupiec and Christoffersen per quarter-hour slot across days, Holm over the 96 slots per family
and level (e, redefined 7 Nov).
DM on the daily CRPS difference native - interp, Newey-West lag 7 (f). Circular block bootstrap over days (g).
pi-regression of the daily gap interp - native on the native share pi_D, Newey-West lag 7 (h).

Newey-West convention (f), (h): Bartlett weights 1 - l/(L + 1), l = 1..7, no small-sample correction, normal
reference distribution, two-sided p (= statsmodels OLS cov_type="HAC", maxlags=7, use_correction=False).

Outputs (j), in results/: table1.tex, table2[_lear].tex, fig1.pdf, fig2[_lear].pdf, numbers.tex, monthly.csv,
bootstrap.csv, regression.csv, kupiec.csv, pred_doseresponse.csv, summary_<model>.md
(and summary.md = the latest run), metrics_<model>.csv (per file
metrics, read back when Table 1 is assembled from several models). CSVs carry a `model` column and the rows of
the model being run replace its earlier rows, so LEAR and XGB accumulate in the same files. Until XGB exists
Table 2 and Figure 2 are produced on interp LEAR and the file names carry `_lear`.

Step 5c (handover 8.1 row ACI-block, step 5c specification B3): when the three
conformal/<setting>_<model>_aciblock files exist, block_stage() adds ACI-block to the CORP metrics, PICP, width,
shares, DM, bootstrap (the same draws as raw and ACI-CQR) and the per-slot coverage tests, appends its rows to
metrics_<model>.csv, bootstrap.csv and kupiec.csv (existing rows unchanged, checked), writes 12 macros per model
and Section 11 of the summary. It is not in REGIMES / LAYERS, so no existing table, figure, CSV row or macro sees it.

Step 5d (handover Section 8 row "XGB monthly re-tuning", step 5d specification B4): in the XGB run, when
--rdir (default forecasts_retune/) holds interp_xgbm.parquet and native_xgbm.parquet, retune_stage() evaluates xgbm
raw, ACI-CQR and ACI-block (conformal/<setting>_xgbm_{aci,aciblock}.parquet) in interp and native: CORP metrics,
PICP, width, DM on the daily CRPS native - interp, the block bootstrap on the same draws as every other regime. It
writes results/retune_xgb_{metrics,bootstrap,monthly,hyperparameters}.csv, the 14 step 5d macros and Section 12 of
summary_xgb.md, and touches no existing table, figure, CSV row or macro.

Step 5e (handover 8.1 rows Sym-block, Loc-block, Loc-slot, Section 8 row "Paper decisions of 10 Nov and step 5e
scope" with its implementation decisions; step 5e specification B3, B5): when the conformal files of
symblock, locblock and locslot exist (and those of aciblock), diag_stage() adds the three diagnostic layers to the
CORP metrics, PICP, width, shares, DM, the bootstrap (the same draws as every other regime) and the per-slot
coverage tests, appends their rows to metrics_<model>.csv, bootstrap.csv and kupiec.csv (existing rows unchanged,
checked), writes results/profile_<model>.csv (raw residual y - q50 by the 24 block groups), the 16 step 5e macros
and the last section of the summary (12 in summary_lear.md, 13 in summary_xgb.md) with the shares W and L of both
models and the reading line. The XGB run also writes results/fig2a.pdf (Figure 2 panel (a) alone). Nothing enters
Table 1, Table 2, fig1.pdf, fig2.pdf, monthly.csv, the pi-regression, the ablation or an existing macro.
"""
import argparse
import re
import subprocess
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.special import xlogy
from scipy.stats import chi2, norm

from . import conformal as cf
from . import data, metrics
from .models import xgb as xgbmod

ROOT = Path(__file__).resolve().parent.parent
SETTINGS = ("hourly", "interp", "native")
REGIMES = tuple(cf.REGIMES)                               # split rolling acifrozen aci acipooled
LAYERS = ("raw",) + REGIMES
TABLE2_REGIMES = ("split", "rolling", "acifrozen", "aci")
EVAL_START, EVAL_END = cf.EVAL_START, cf.EVAL_END         # inclusive
BURN_START, BURN_END = cf.BURN_START, cf.BURN_END         # inclusive
GO_LIVE = "2025-10-01"
WINDOW = data.WINDOW
SEED = 2027
N_BOOT = 1000
BLOCK = 7
NW_LAG = 7
N_PI = 101
LEVELS = {80: ("q10", "q90", 0.2), 60: ("q20", "q80", 0.4)}

LAYER_LABEL = {"raw": "Raw", "split": "Split-CQR", "rolling": "Rolling-CQR", "acifrozen": "ACI-frozen",
               "aci": "ACI-CQR", "acipooled": "Pooled ACI-CQR", "aciblock": "ACI-block", "symblock": "Sym-block",
               "locblock": "Loc-block", "locslot": "Loc-slot"}
REG_BLOCK = "aciblock"                                    # step 5c, not in REGIMES / LAYERS (block_stage)
DIAG = tuple(cf.DIAG_REGIMES)                             # step 5e: symblock locblock locslot (diag_stage)
SET_LABEL = {"hourly": "Hourly", "interp": "Interpolated", "native": "Native"}
MODEL_LABEL = {"lear": "LEAR", "xgb": "XGB"}
MODELS_TABLE1 = ("lear", "xgb")

# ---- interpretation switches (spec points left open, see the step 5.2 report) ----------------------------------
# Figure 1 dashed curve and \predCrpsGapHalf: "interp_minus_native" = CRPS(0) - CRPS(pi), the predicted gap
# interp - native at share pi (the interp model is the pi = 0 model). "item_i" = gap(pi) = CRPS(pi) - CRPS(1).
# Both columns are always written to pred_doseresponse.csv.
PRED_GAP = "interp_minus_native"
# Figure 2(b) predicted raw PICP-80 by position: interp at pi = 0, native = mean over the evaluation days of the
# prediction at pi_D, s_H pooled over hours (item (i)).
# Macros named for XGB in main.tex are written only by an XGB run. LEAR writes LEAR-suffixed stand-ins.
XGB_MACRO_STANDINS = False


# ================================================================================ quantiles and CORP (item (d))
def _frac(tau):
    f = Fraction(float(tau)).limit_denominator(1000)
    return f.numerator, f.denominator


def lq_index(m, tau):
    """0-based index of the lower tau-quantile (inverted_cdf) in a sorted sample of size m: ceil(tau m) - 1,
    in exact integer arithmetic (tau a decimal fraction)."""
    p, q = _frac(tau)
    return (p * np.asarray(m, dtype=np.int64) + q - 1) // q - 1


def lower_quantile(y, tau):
    y = np.sort(np.asarray(y, float))
    return float(y[int(lq_index(len(y), tau))])


def pav_quantile(x, y, tau):
    """Isotonic (nondecreasing in x) tau-quantile regression of y on x by pool-adjacent-violators.

    Ties in x are pooled first. Pooling functional: the lower tau-quantile of the pooled y. Blocks are merged
    while the previous block's value exceeds the current one. Returns the fitted value of every observation, in
    the input order."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    order = np.lexsort((y, x))                      # by x, then y within ties
    xs, ys = x[order], y[order]
    brk = np.flatnonzero(xs[1:] != xs[:-1]) + 1
    st = np.concatenate(([0], brk))
    en = np.concatenate((brk, [len(xs)]))
    v0 = ys[st + lq_index(en - st, tau)]            # ys sorted within each tie group
    p, q = _frac(tau)
    bs, bv = [], []
    for s, e, v in zip(st.tolist(), en.tolist(), v0.tolist()):
        while bv and bv[-1] > v:
            bv.pop()
            s = bs.pop()
            k = (p * (e - s) + q - 1) // q - 1
            v = float(np.partition(ys[s:e], k)[k])
        bs.append(s)
        bv.append(v)
    bs.append(len(xs))
    fs = np.repeat(np.asarray(bv), np.diff(bs))
    out = np.empty_like(fs)
    out[order] = fs
    return out


def corp(y, Q):
    """CORP decomposition of the mean pinball score per level and aggregated on the CRPS scale.
    y (n,), Q (n, 9) in the order of data.TAUS."""
    y = np.asarray(y, float)
    per = []
    for j, tau in enumerate(data.TAUS):
        x = Q[:, j]
        s = float(np.mean(metrics.pinball(y, x, tau)))
        s_rc = float(np.mean(metrics.pinball(y, pav_quantile(x, y, tau), tau)))
        unc = float(np.mean(metrics.pinball(y, lower_quantile(y, tau), tau)))
        per.append((s, s - s_rc, unc - s_rc, unc))
    per = np.asarray(per)
    crps, mcb, dsc, unc = (2.0 * per.mean(axis=0)).tolist()
    if abs(crps - (mcb - dsc + unc)) > 1e-9 * max(1.0, abs(crps)):
        raise AssertionError(f"CORP identity fails: {crps} vs {mcb - dsc + unc}")
    return {"CRPS": crps, "MCB": mcb, "DSC": dsc, "UNC": unc,
            "per_level": pd.DataFrame(per, columns=["S", "MCB", "DSC", "UNC"], index=data.QCOLS)}


# ================================================================================ HAC, DM, regression (f, h)
def ols_nw(y, X, lag=NW_LAG):
    """OLS with Newey-West (Bartlett, lag L, no small-sample correction) covariance, normal two-sided p."""
    y = np.asarray(y, float)
    X = np.asarray(X, float).reshape(len(y), -1)
    XtXi = np.linalg.inv(X.T @ X)
    beta = XtXi @ X.T @ y
    u = y - X @ beta
    Z = X * u[:, None]
    S = Z.T @ Z
    for lg in range(1, lag + 1):
        G = Z[lg:].T @ Z[:-lg]
        S += (1.0 - lg / (lag + 1.0)) * (G + G.T)
    cov = XtXi @ S @ XtXi
    se = np.sqrt(np.diag(cov))
    z = beta / se
    return beta, se, z, 2.0 * norm.sf(np.abs(z))


def dm_test(d, lag=NW_LAG):
    """Diebold-Mariano on a daily loss difference d: mean, HAC se, statistic, two-sided p."""
    b, se, z, p = ols_nw(d, np.ones(len(d)), lag)
    return {"mean": float(b[0]), "se": float(se[0]), "stat": float(z[0]), "p": float(p[0]), "T": len(d)}


def native_share(dates):
    """pi_D = number of window days D-364..D-1 on or after 1 Oct 2025, divided by 364."""
    dn = pd.to_datetime(pd.Series(dates)).values.astype("datetime64[D]").astype(np.int64)
    return np.clip(dn - data.day_number(GO_LIVE), 0, WINDOW) / WINDOW


# ================================================================================ coverage tests (e)
def kupiec_lr(x, n, p):
    ph = x / n
    ll0 = xlogy(n - x, 1 - p) + xlogy(x, p)
    ll1 = xlogy(n - x, 1 - ph) + xlogy(x, ph)
    return float(-2.0 * (ll0 - ll1))


def christoffersen_ind_lr(hits):
    h = np.asarray(hits, int)
    a, b = h[:-1], h[1:]
    n00, n01 = int(np.sum((a == 0) & (b == 0))), int(np.sum((a == 0) & (b == 1)))
    n10, n11 = int(np.sum((a == 1) & (b == 0))), int(np.sum((a == 1) & (b == 1)))
    p01 = n01 / (n00 + n01) if n00 + n01 else 0.0
    p11 = n11 / (n10 + n11) if n10 + n11 else 0.0
    p1 = (n01 + n11) / (n00 + n01 + n10 + n11)
    ll1 = xlogy(n00, 1 - p01) + xlogy(n01, p01) + xlogy(n10, 1 - p11) + xlogy(n11, p11)
    ll0 = xlogy(n00 + n10, 1 - p1) + xlogy(n01 + n11, p1)
    return float(-2.0 * (ll0 - ll1))


def holm(p):
    p = np.asarray(p, float)
    m = len(p)
    o = np.argsort(p, kind="stable")
    adj = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * p[o]))
    out = np.empty(m)
    out[o] = adj
    return out


N_SLOTS = 96


def coverage_tests(df):
    """Kupiec (unconditional coverage, chi2(1)) and Christoffersen (conditional coverage LR_uc + LR_ind,
    chi2(2)) per quarter-hour slot (slot = 4 hour + quarter, 96 slots) and level, across days (redefined 7 Nov,
    Section 8 row "Evaluation implementation" (e)): the hit sequence of a slot is its days in chronological order,
    hit = actual outside [lo, hi] (bounds inclusive). Holm over the 96 tests of each family and level."""
    df = df.sort_values("datetime_local")
    y = df["actual"].to_numpy(float)
    slot = (4 * df["hour"] + df["quarter"]).to_numpy()
    rows = []
    for lev, (lo, hi, a) in LEVELS.items():
        hit_all = ((y < df[lo].to_numpy(float)) | (y > df[hi].to_numpy(float))).astype(int)
        for s in range(N_SLOTS):
            hit = hit_all[slot == s]
            n, x = len(hit), int(hit.sum())
            if n == 0:
                raise RuntimeError(f"slot {s} has no observations")
            luc = kupiec_lr(x, n, a)
            lind = christoffersen_ind_lr(hit)
            rows.append({"slot": s, "hour": s // 4, "quarter": s % 4, "level": lev, "n": n, "misses": x,
                         "miss_rate": x / n, "nominal": a, "LR_uc": luc, "p_uc": float(chi2.sf(luc, 1)),
                         "LR_ind": lind, "p_ind": float(chi2.sf(lind, 1)), "LR_cc": luc + lind,
                         "p_cc": float(chi2.sf(luc + lind, 2))})
    out = pd.DataFrame(rows)
    for lev in LEVELS:
        sel = (out["level"] == lev).to_numpy()
        out.loc[sel, "p_uc_holm"] = holm(out.loc[sel, "p_uc"])
        out.loc[sel, "p_cc_holm"] = holm(out.loc[sel, "p_cc"])
    return out


def coverage_rejections(kp, alpha=0.05):
    """Per file and level: number of the 96 slots rejected after Holm (Kupiec, Christoffersen cc)."""
    keys = [c for c in ("model", "setting", "layer") if c in kp.columns] + ["level"]
    g = kp.groupby(keys, sort=False)
    return pd.DataFrame({"slots": g.size(), "kupiec_rej": g["p_uc_holm"].apply(lambda p: int((p < alpha).sum())),
                         "cc_rej": g["p_cc_holm"].apply(lambda p: int((p < alpha).sum()))}).reset_index()


# ================================================================================ Proposition A (item (i))
def burnin_inputs(model, fdir):
    """sigma_q, kappa_q (native and interpolated within-hour deviation std by position, burn-in, ddof 1), s_H and
    the standardized residuals Z of the hourly setting's point forecast against the hourly mean native price."""
    def burn(df):
        d = df["date"].astype(str).str[:10]
        return df[(d >= BURN_START) & (d <= BURN_END)].copy()

    tgt = burn(data.load_target())
    itp = burn(data.load_setting("interp"))
    if len(tgt) != 2688 or len(itp) != 2688:
        raise RuntimeError("burn-in must have 2,688 quarters")

    def dev_sd(df, col):
        key = [df["date"].astype(str), df["hour"]]
        d = df[col] - df.groupby(key)[col].transform("mean")
        return np.array([d[df["quarter"] == q].std(ddof=1) for q in range(4)])

    sig = dev_sd(tgt, "actual")
    sig_i = dev_sd(itp, data.PRICE)
    hf = burn(pd.read_parquet(Path(fdir) / f"hourly_{model}.parquet"))
    g = hf.groupby(["date", "hour"])
    if (g["point"].nunique() != 1).any():
        raise RuntimeError("hourly point forecast not constant within the hour")
    res = (g["actual"].mean() - g["point"].first()).to_numpy(float)      # y - yhat, 672 hours
    s_h = float(np.std(res, ddof=1))
    by_hour = pd.Series(res, index=g["point"].first().index.get_level_values("hour")).groupby(level=0).std(ddof=1)
    return {"sigma": sig, "sigma_interp": sig_i, "kappa": sig_i / sig, "s_H": s_h, "Z": np.sort(res / s_h),
            "n_resid": len(res), "resid_mean": float(res.mean()), "s_H_by_hour": by_hour}


def _ecdf(Zs, v):
    return np.searchsorted(Zs, v, side="right") / len(Zs)


def mixture_quantile(Zs, s_i, s_n, pi, tau, tol=1e-12):
    """Smallest x with (1 - pi) F(x / s_i) + pi F(x / s_n) >= tau (F = ECDF of Zs), by bisection."""
    G = lambda x: (1.0 - pi) * _ecdf(Zs, x / s_i) + pi * _ecdf(Zs, x / s_n)
    big = (np.abs(Zs).max() + 1.0) * max(s_i, s_n)
    lo, hi = -big, big
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if G(mid) >= tau - 1e-12:
            hi = mid
        else:
            lo = mid
        if hi - lo <= tol * (1.0 + abs(hi)):
            break
    return hi


def prop_a_point(inp, q, pi):
    """Predicted quantiles, PICP-80 (percent) and CRPS of the native setting at position q and share pi."""
    Zs, sH, sg, kp = inp["Z"], inp["s_H"], inp["sigma"][q], inp["kappa"][q]
    s_i = np.sqrt(sH ** 2 + kp ** 2 * sg ** 2)
    s_n = np.sqrt(sH ** 2 + sg ** 2)
    qt = np.array([mixture_quantile(Zs, s_i, s_n, pi, t) for t in data.TAUS])
    cov = _ecdf(Zs, qt[-1] / s_n) - _ecdf(Zs, qt[0] / s_n)
    crps = 2.0 * np.mean([s_n * np.mean(metrics.pinball(Zs, c / s_n, t)) for c, t in zip(qt, data.TAUS)])
    return qt, 100.0 * float(cov), float(crps), s_i, s_n


def prop_a(inp, pis=None):
    """Grid of 101 shares x 4 positions + pooled (mean over positions)."""
    pis = np.linspace(0.0, 1.0, N_PI) if pis is None else pis
    rows = []
    for pi in pis:
        for q in range(4):
            qt, cov, crps, s_i, s_n = prop_a_point(inp, q, pi)
            rows.append({"pi": round(float(pi), 10), "position": str(q), "s_I": s_i, "s_N": s_n,
                         "picp80": cov, "crps": crps, **{f"q{int(round(100 * t))}": v for t, v in zip(data.TAUS, qt)}})
    df = pd.DataFrame(rows)
    pooled = df.groupby("pi", as_index=False)[["picp80", "crps"]].mean()
    pooled["position"] = "pooled"
    df = pd.concat([df, pooled], ignore_index=True)
    for pos, sub in df.groupby("position"):
        c0 = sub.loc[sub["pi"] == 0.0, "crps"].iloc[0]
        c1 = sub.loc[sub["pi"] == 1.0, "crps"].iloc[0]
        df.loc[sub.index, "gap"] = sub["crps"] - c1                       # item (i): CRPS(pi) - CRPS(1)
        df.loc[sub.index, "gap_interp_minus_native"] = c0 - sub["crps"]   # CRPS(0) - CRPS(pi)
    df.insert(0, "kind", "grid")
    return df


def prop_a_eval_levels(inp, pis):
    """Predicted raw PICP-80 by position over the evaluation days: interp (pi = 0) and native (mean over the
    days of the prediction at pi_D). For Figure 2(b)."""
    upi = np.unique(np.round(pis, 12))
    cnt = pd.Series(np.round(pis, 12)).value_counts()
    rows = []
    for q in range(4):
        nat = {p: prop_a_point(inp, q, p)[1:3] for p in upi}
        w = np.array([cnt[p] for p in upi], float)
        rows.append({"kind": "eval_native", "pi": float(np.mean(pis)), "position": str(q),
                     "picp80": float(np.average([nat[p][0] for p in upi], weights=w)),
                     "crps": float(np.average([nat[p][1] for p in upi], weights=w))})
        _, c0, r0, _, _ = prop_a_point(inp, q, 0.0)
        rows.append({"kind": "eval_interp", "pi": 0.0, "position": str(q), "picp80": c0, "crps": r0})
    return pd.DataFrame(rows)


# ================================================================================ loading
def _eval_rows(df):
    d = df["date"].astype(str)
    df = df[(d >= EVAL_START) & (d <= EVAL_END)].sort_values("datetime_local").reset_index(drop=True)
    if len(df) != cf.N_EVAL_ROWS:
        raise RuntimeError(f"{len(df)} evaluation rows, expected {cf.N_EVAL_ROWS}")
    return df


def load_layer(model, setting, layer, fdir="forecasts", cdir="conformal"):
    p = Path(fdir) / f"{setting}_{model}.parquet" if layer == "raw" else Path(cdir) / f"{setting}_{model}_{layer}.parquet"
    return _eval_rows(pd.read_parquet(p))


def load_all(model, fdir="forecasts", cdir="conformal"):
    frames = {("naive", s, "raw"): load_layer("naive", s, "raw", fdir, cdir) for s in SETTINGS}
    for s in SETTINGS:
        for lay in LAYERS:
            frames[model, s, lay] = load_layer(model, s, lay, fdir, cdir)
    ref = frames[model, "native", "raw"]
    for k, f in frames.items():
        if not (f["datetime_local"].equals(ref["datetime_local"]) and f["actual"].equals(ref["actual"])):
            raise RuntimeError(f"{k}: rows or actuals differ from the native raw file")
    return frames


def row_crps(df):
    y = df["actual"].to_numpy(float)
    return 2.0 * np.mean([metrics.pinball(y, df[c].to_numpy(float), t) for c, t in zip(data.QCOLS, data.TAUS)], axis=0)


def daily_crps(df):
    return pd.Series(row_crps(df)).groupby(df["date"].astype(str).to_numpy()).mean()


def frame_metrics(df, naive_mae=None, with_corp=True):
    s = metrics.summary(df)
    out = {k: s[k] for k in ("n", "MAE", "RMSE", "CRPS", "PICP80", "width80", "PICP60", "width60")}
    out["rMAE"] = out["MAE"] / naive_mae if naive_mae else np.nan
    if with_corp:
        c = corp(df["actual"].to_numpy(float), df[data.QCOLS].to_numpy(float))
        if abs(c["CRPS"] - s["CRPS"]) > 1e-9 * s["CRPS"]:
            raise AssertionError("CRPS of CORP and of metrics.summary differ")
        out.update({k: c[k] for k in ("MCB", "DSC", "UNC")})
    return out


# ================================================================================ bootstrap (g)
def boot_days(n_days, n_draws=N_BOOT, block=BLOCK, seed=SEED):
    """Circular block bootstrap: ceil(n_days / block) blocks of `block` consecutive days per draw (49 for 337),
    start days uniform, wrapped at the end, truncated to n_days. One default_rng(seed), draws in order."""
    rng = np.random.default_rng(seed)
    nb = -(-n_days // block)
    out = np.empty((n_draws, n_days), dtype=np.int64)
    for b in range(n_draws):
        st = rng.integers(0, n_days, nb)
        out[b] = ((st[:, None] + np.arange(block)) % n_days).ravel()[:n_days]
    return out


def _boot_chunk(draws, day_start, day_len, y, Qs):
    res = []
    for di in draws:
        rows = np.concatenate([np.arange(day_start[d], day_start[d] + day_len[d]) for d in di])
        yy = y[rows]
        res.append({k: corp(yy, Q[rows]) for k, Q in Qs.items()})
    return [{k: (c["CRPS"], c["MCB"], c["DSC"]) for k, c in r.items()} for r in res]


def bootstrap(frames, model, n_draws=N_BOOT, n_jobs=8, regimes=("raw", "aci"), D=None):
    """Per draw: CRPS, MCB, DSC of interp and native for each regime on the same resampled days. Returns the
    long frame of draws (draw, regime, quantity, native, interp, diff = native - interp). D = the day-index draws
    (boot_days), passed in when several calls must use the same draws (step 5c)."""
    ref = frames[model, "native", "raw"]
    dates = ref["date"].astype(str).to_numpy()
    _, day_start, day_len = np.unique(dates, return_index=True, return_counts=True)
    y = ref["actual"].to_numpy(float)
    Qs = {(r, s): frames[model, s, r][data.QCOLS].to_numpy(float) for r in regimes for s in ("interp", "native")}
    D = boot_days(len(day_start), n_draws) if D is None else np.asarray(D)
    if D.shape != (n_draws, len(day_start)):
        raise ValueError(f"bootstrap draws of shape {D.shape}, expected {(n_draws, len(day_start))}")
    chunks = np.array_split(np.arange(n_draws), max(1, min(n_draws, 8 * n_jobs)))
    parts = Parallel(n_jobs=n_jobs)(delayed(_boot_chunk)(D[c], day_start, day_len, y, Qs) for c in chunks if len(c))
    rows = []
    for b, r in enumerate([x for p in parts for x in p]):
        for reg in regimes:
            for j, qn in enumerate(("CRPS", "MCB", "DSC")):
                nat, itp = r[reg, "native"][j], r[reg, "interp"][j]
                rows.append((b, reg, qn, nat, itp, nat - itp))
    return pd.DataFrame(rows, columns=["draw", "regime", "quantity", "native", "interp", "diff"])


def boot_summary(bt, met, model):
    """bootstrap.csv rows per (regime, quantity): point values from the metrics frame, 90% interval, mean and SD of
    the draws."""
    brows = []
    for (reg, qn), g in bt.groupby(["regime", "quantity"], sort=False):
        mrow = lambda s: met[(met["model"] == model) & (met["setting"] == s) & (met["layer"] == reg)].iloc[0][qn]
        est = mrow("native") - mrow("interp")
        brows.append({"model": model, "regime": reg, "quantity": qn, "native": mrow("native"),
                      "interp": mrow("interp"), "diff_native_minus_interp": est,
                      "lo5": float(np.percentile(g["diff"], 5)), "hi95": float(np.percentile(g["diff"], 95)),
                      "boot_mean": float(g["diff"].mean()), "boot_sd": float(g["diff"].std(ddof=1)),
                      "n_draws": len(g), "block": BLOCK, "seed": SEED})
    return pd.DataFrame(brows)


# ================================================================================ formatting and files
def fmt(v, nd=2):
    s = f"{v:.{nd}f}"
    if s.startswith("-"):
        return s[1:] if float(s) == 0 else "\\ensuremath{-" + s[1:] + "}"
    return s


def fmt_p(p):
    return "\\ensuremath{<}0.001" if p < 0.001 else f"{p:.3f}"


def fmt_p_txt(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


MACRO_RE = re.compile(r"^\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}\s*$")


def read_macros(path):
    out = {}
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            m = MACRO_RE.match(line)
            if m:
                out[m.group(1)] = m.group(2)
    return out


def write_macros(path, new):
    """Merge `new` into numbers.tex (existing macros of other models are kept)."""
    cur = read_macros(path)
    cur.update(new)
    lines = ["% results/numbers.tex, written by python -m esann27.evaluate (handover Section 8.2). Do not edit."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(cur.items())]
    Path(path).write_text("\n".join(lines) + "\n")


def upsert_csv(path, df, model):
    path = Path(path)
    if path.exists():
        old = pd.read_csv(path, float_precision="round_trip", dtype={"position": str})
        old = old[old["model"] != model]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(path, index=False)
    return df


def md_table(df, floatfmt=None):
    floatfmt = floatfmt or {}
    cols = list(df.columns)

    def cell(c, v):
        if isinstance(v, (float, np.floating)):
            if np.isnan(v):
                return ""
            f = floatfmt.get(c, ".4f")
            return f"{v:{f}}"
        return str(v)

    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(cell(c, v) for c, v in zip(cols, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


# ================================================================================ tables (8.2)
def table1_tex(metrics_all, dm):
    """Body of Table 1. metrics_all: rows model, setting, layer, MAE, CRPS, MCB, DSC, PICP80, width80.
    dm: {(model, regime): p}. Blocks model x regime (raw, aci), rows hourly, interp, native; DM p-value
    native vs interpolated on the native row of each block."""
    lines = ["\\begin{tabular}{|l|l|c|c|c|c|c|c|c|}", "\\hline",
             "Setting & Model & MAE & CRPS & MCB & DSC & PICP & Width & DM \\\\", "\\hline"]
    for m in MODELS_TABLE1:
        for reg in ("raw", "aci"):
            sub = metrics_all[(metrics_all["model"] == m) & (metrics_all["layer"] == reg)]
            if sub.empty:
                continue
            for s in SETTINGS:
                r = sub[sub["setting"] == s].iloc[0]
                name = MODEL_LABEL[m] + ("" if reg == "raw" else "+ACI")
                dmc = fmt_p(dm[m, reg]) if s == "native" else "--"
                lines.append(f"{SET_LABEL[s]} & {name} & {fmt(r['MAE'])} & {fmt(r['CRPS'])} & {fmt(r['MCB'])} & "
                             f"{fmt(r['DSC'])} & {fmt(r['PICP80'], 1)} & {fmt(r['width80'])} & {dmc} \\\\")
            lines.append("\\hline")
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"


def table2_rows(metrics_all, unbounded, model, setting="interp"):
    rows = []
    for reg in TABLE2_REGIMES:
        r = metrics_all[(metrics_all["model"] == model) & (metrics_all["setting"] == setting)
                        & (metrics_all["layer"] == reg)].iloc[0]
        rows.append({"regime": reg, "CRPS": r["CRPS"], "PICP80": r["PICP80"], "width80": r["width80"],
                     "unbounded": 100.0 * unbounded[setting, reg]})
    return pd.DataFrame(rows)


def table2_tex(t2):
    lines = ["\\begin{tabular}{|l|c|c|c|c|}", "\\hline",
             "Regime & CRPS & PICP-80 & Width-80 & Unbounded \\\\", "\\hline"]
    for r in t2.itertuples(index=False):
        lines.append(f"{LAYER_LABEL[r.regime]} & {fmt(r.CRPS)} & {fmt(r.PICP80, 1)} & {fmt(r.width80)} & "
                     f"{fmt(r.unbounded, 1)} \\\\")
    lines += ["\\hline", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


# ================================================================================ figures (8.2)
CM = 1 / 2.54
FIG_W, FIG_H = 12.2, 4.6         # cm, Figures 1 (two panels) and 2 (handover rev 2.8.14); 5.0 for Figure 1 (a) only
LEG_ROW_CM = 0.4125              # height reserved per legend row below the panels (0.075 x 5.5 cm of rev 2.8.12)
C_BLUE, C_ORANGE, C_AQUA, C_YELLOW, C_VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600, "font.size": 8,
                         "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "legend.fontsize": 8, "font.family": "DejaVu Sans", "axes.linewidth": 0.6,
                         "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.labelcolor": INK, "text.color": INK, "xtick.major.width": 0.6,
                         "ytick.major.width": 0.6, "lines.linewidth": 1.2, "legend.frameon": False})
    return plt


def _share_to_num(pi):
    import matplotlib.dates as mdates
    return mdates.date2num(pd.Timestamp(GO_LIVE)) + np.asarray(pi) * WINDOW


def _num_to_share(x):
    import matplotlib.dates as mdates
    return (np.asarray(x) - mdates.date2num(pd.Timestamp(GO_LIVE))) / WINDOW


def figure1(monthly, pred, path, abl=None, abl_pred=None):
    """Panel (a): monthly CRPS gap interp - native, raw and ACI-CQR, per model, one dashed Proposition A curve per
    model (s_H of that model), native share on the top axis. Panel (b), when the step 5b files exist: observed
    CRPS(pi) - CRPS(regular) at the four ablation levels and the regular run at pi_bar (black), LEAR predicted
    curve dashed, PICP-80 on the right axis. 12.2 x 5.0 cm with panel (a) only, 12.2 x 4.6 cm with both
    (Section 8.2, step row 5b, rev 2.8.14)."""
    import matplotlib.dates as mdates
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    plt = _mpl()
    two = abl is not None
    if two:
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(FIG_W * CM, FIG_H * CM), gridspec_kw={"width_ratios": [1.3, 1]})
    else:
        fig, ax = plt.subplots(figsize=(12.2 * CM, 5.0 * CM))
    ax.axhline(0, color=GRID, lw=0.6, zorder=0)
    style = {"raw": (C_BLUE, "o"), "aci": (C_ORANGE, "s")}
    mstyle = {"lear": "-", "xgb": ":"}
    pcol = {"lear": INK2, "xgb": C_VIOLET}
    present = []
    for m in MODELS_TABLE1:
        sub = monthly[monthly["model"] == m]
        if sub.empty:
            continue
        present.append(m)
        for reg in ("raw", "aci"):
            w = sub[sub["layer"] == reg].pivot_table(index=["month", "mid"], columns="setting", values="CRPS")
            w = w.reset_index().sort_values("month")
            col, mk = style[reg]
            ax.plot(pd.to_datetime(w["mid"]), w["interp"] - w["native"], ls=mstyle[m], color=col, marker=mk,
                    ms=3.2, mfc=col, mec="white", mew=0.5, label=f"{MODEL_LABEL[m]} {LAYER_LABEL[reg]}")
        p = pred[(pred["model"] == m) & (pred["kind"] == "grid") & (pred["position"] == "pooled")]
        p = p[p["pi"] >= native_share([EVAL_START])[0] - 1e-9]
        ycol = "gap_interp_minus_native" if PRED_GAP == "interp_minus_native" else "gap"
        ax.plot(mdates.num2date(_share_to_num(p["pi"].to_numpy())), p[ycol], ls="--", lw=1.0, color=pcol[m],
                label=f"{MODEL_LABEL[m]} predicted (raw)")
    ax.set_ylabel("CRPS gap interp. $-$ native" if not two else "CRPS gap\ninterp. $-$ native", labelpad=2)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(11, 1, 3, 5, 7, 9) if not two else (11, 3, 7)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax.set_xlim(pd.Timestamp("2025-10-25"), pd.Timestamp("2026-10-04"))
    top = ax.secondary_xaxis("top", functions=(_num_to_share, _share_to_num))
    top.set_xticks([0.25, 0.5, 0.75, 1.0])
    top.tick_params(colors=INK2, pad=1)
    ax.grid(axis="y", color=GRID, lw=0.4)
    ax.spines["right"].set_visible(False)
    if not two:
        top.set_xlabel("native share $\\pi$", labelpad=2)
        h, lab = ax.get_legend_handles_labels()
        fig.legend(h, lab, loc="lower center", ncol=3, handlelength=1.8, columnspacing=1.0, borderaxespad=0.1)
        fig.tight_layout(pad=0.3, rect=(0, 0.1 * (1 + (len(h) > 3)), 1, 1))
        fig.savefig(path)
        plt.close(fig)
        return
    top.set_xticklabels(["$\\pi$ = .25", ".50", ".75", "1"])
    ax.set_title("(a)", loc="left", pad=2)
    _figure1_ablation(bx, abl, abl_pred)
    h = [Line2D([], [], color=C_BLUE, marker="o", ms=3.2, mec="white", mew=0.5, lw=1.0),
         Line2D([], [], color=C_ORANGE, marker="s", ms=3.2, mec="white", mew=0.5, lw=1.0)]
    lab = ["raw", "ACI-CQR"]
    for m in present:
        h.append(Line2D([], [], color=INK, ls=mstyle[m], lw=1.0))
        lab.append(MODEL_LABEL[m])
    for m in present:
        h.append(Line2D([], [], color=pcol[m], ls="--", lw=1.0))
        lab.append(f"Prop. A (scale), {MODEL_LABEL[m]}")
    h += [Line2D([], [], ls="none", marker="o", ms=3.8, mfc=INK, mec="white"),
          Line2D([], [], ls="none", marker="s", ms=3.2, mfc="white", mec=C_ORANGE, mew=0.9)]
    lab += ["regular run (b)", "PICP-80 (b, right)"]
    fig.legend(h, lab, loc="lower center", ncol=4, handlelength=1.8, columnspacing=1.0, borderaxespad=0.1)
    nrow = -(-len(h) // 4)
    fig.tight_layout(pad=0.3, w_pad=0.4, rect=(0, nrow * LEG_ROW_CM / FIG_H, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def _figure1_ablation(bx, abl, abl_pred):
    """Panel (b) of Figure 1 from results/ablation.csv and ablation_pred.csv (step 5b): LEAR, native setting."""
    from matplotlib.ticker import MaxNLocator
    lev = abl[abl["level"] != "regular"].sort_values("pi")
    reg = abl[abl["level"] == "regular"].iloc[0]
    curve = abl_pred[abl_pred["kind"] == "grid"].sort_values("pi")
    bx.axhline(0, color=GRID, lw=0.6, zorder=0)
    bx.plot(curve["pi"], curve["pred_gap"], ls="--", lw=1.0, color=INK2)
    x = np.append(lev["pi"].to_numpy(), reg["pi"])
    bx.plot(x, np.append(lev["gap_crps"].to_numpy(), 0.0), ls="-", lw=1.0, color=C_BLUE, marker="o", ms=3.2,
            mfc=C_BLUE, mec="white", mew=0.5)
    bx.plot([reg["pi"]], [0.0], ls="none", marker="o", ms=3.8, mfc=INK, mec="white", mew=0.5, zorder=5)
    bx.set_xlim(-0.05, 1.05)
    bx.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    bx.set_xticklabels(["0", ".25", ".50", ".75", "1"])
    bx.set_xlabel("native share $\\pi$ of the window", labelpad=1)
    bx.set_ylabel("CRPS($\\pi$) $-$ regular", labelpad=2)
    bx.yaxis.set_major_locator(MaxNLocator(4))
    bx.grid(axis="y", color=GRID, lw=0.4)
    bx.set_title("(b)", loc="left", pad=2)
    rx = bx.twinx()
    rx.plot(x, np.append(lev["PICP80"].to_numpy(), reg["PICP80"]), ls="none", marker="s", ms=3.2, mfc="white",
            mec=C_ORANGE, mew=0.9)
    rx.set_ylabel("PICP-80 (%)", color=C_ORANGE, labelpad=2)
    rx.yaxis.set_major_locator(MaxNLocator(3))
    rx.tick_params(axis="y", colors=C_ORANGE, pad=1)
    rx.spines["right"].set_color(C_ORANGE)
    rx.spines["top"].set_visible(False)
    bx.spines["top"].set_visible(False)


def rolling_picp(df, k=cf.K_WIN):
    y = df["actual"].to_numpy(float)
    cov = ((y >= df["q10"].to_numpy(float)) & (y <= df["q90"].to_numpy(float))).astype(float)
    g = pd.DataFrame({"date": df["date"].astype(str).to_numpy(), "c": cov}).groupby("date")["c"].agg(["sum", "count"])
    r = g.rolling(k).sum().dropna()
    return pd.Series((100.0 * r["sum"] / r["count"]).to_numpy(), index=pd.to_datetime(r.index))


def picp_by_position(df):
    y = df["actual"].to_numpy(float)
    cov = (y >= df["q10"].to_numpy(float)) & (y <= df["q90"].to_numpy(float))
    return pd.Series(cov).groupby(df["quarter"].to_numpy()).mean() * 100.0


def figure2(frames, model, pred_eval, path):
    """(a) 28-day rolling PICP-80 of the four variants on interp; (b) PICP-80 by quarter position, raw, pooled
    ACI-CQR and Mondrian ACI-CQR on interp and native, predicted raw levels as black dashes. 12.2 x 4.6 cm, each
    legend in two rows below its panel, outside the data (rev 2.8.14: the in-panel legend of (b) covered the raw
    markers at :00)."""
    import matplotlib.dates as mdates
    from matplotlib.lines import Line2D
    plt = _mpl()
    fig, (a, b) = plt.subplots(1, 2, figsize=(FIG_W * CM, FIG_H * CM), gridspec_kw={"width_ratios": [1.15, 1]})
    cols = {"split": (C_BLUE, "-"), "rolling": (C_ORANGE, "--"), "acifrozen": (C_AQUA, "-."), "aci": (C_VIOLET, "-")}
    for reg in TABLE2_REGIMES:
        r = rolling_picp(frames[model, "interp", reg])
        a.plot(r.index, r.values, color=cols[reg][0], ls=cols[reg][1], lw=1.0, label=LAYER_LABEL[reg])
    a.axhline(80, color=INK, lw=0.6, ls=":")
    a.set_ylabel("PICP-80 (%), 28-day")
    a.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7)))
    a.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    a.set_title("(a) interpolated, four variants", loc="left")
    a.grid(axis="y", color=GRID, lw=0.4)

    lay = {"raw": C_BLUE, "acipooled": C_ORANGE, "aci": C_AQUA}
    off = {("interp", "raw"): -0.30, ("interp", "acipooled"): -0.20, ("interp", "aci"): -0.10,
           ("native", "raw"): 0.10, ("native", "acipooled"): 0.20, ("native", "aci"): 0.30}
    for (s, reg), dx in off.items():
        v = picp_by_position(frames[model, s, reg])
        filled = s == "native"
        b.plot(np.arange(4) + dx, v.values, ls="none", marker="o" if filled else "D", ms=3.6,
               mfc=lay[reg] if filled else "white", mec=lay[reg], mew=0.9)
    for s, kind in (("interp", "eval_interp"), ("native", "eval_native")):
        p = pred_eval[(pred_eval["kind"] == kind)].sort_values("position")
        b.plot(np.arange(4) + off[s, "raw"], p["picp80"].to_numpy(), ls="none", marker="_", ms=9, mew=1.2,
               color=INK)
    b.axhline(80, color=INK, lw=0.6, ls=":")
    b.set_xticks(range(4))
    b.set_xticklabels([":00", ":15", ":30", ":45"])
    b.set_xlim(-0.5, 3.5)
    b.set_title("(b) by quarter position", loc="left")
    b.grid(axis="y", color=GRID, lw=0.4)
    col = [Line2D([], [], ls="none", marker="s", ms=4, color=lay[r]) for r in lay]
    shp = [Line2D([], [], ls="none", marker="D", ms=3.6, mfc="white", mec=INK2),
           Line2D([], [], ls="none", marker="o", ms=3.6, color=INK2),
           Line2D([], [], ls="none", marker="_", ms=8, mew=1.2, color=INK)]
    hb = [x for pair in zip(col, shp) for x in pair]            # column-major fill: colours row 1, shapes row 2
    lb = [x for pair in zip(["raw", "pooled ACI", "Mondrian ACI"], ["interp.", "native", "predicted raw"])
          for x in pair]
    for ax in (a, b):
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
    fig.tight_layout(pad=0.3, w_pad=0.8, rect=(0, 2 * LEG_ROW_CM / FIG_H, 1, 1))
    ha, la = a.get_legend_handles_labels()
    # 8 pt legends of 4.6 + 5.9 cm on a 12.2 cm figure: (a) aligned left on its axes, (b) right on its axes
    pa, pb = a.get_position(), b.get_position()
    fig.legend(ha, la, loc="lower left", bbox_to_anchor=(pa.x0, 0), ncol=2, handlelength=1.4, handletextpad=0.4,
               columnspacing=0.8, borderaxespad=0.1)
    fig.legend(hb, lb, loc="lower right", bbox_to_anchor=(pb.x1, 0), ncol=3, handlelength=0.8, handletextpad=0.3,
               columnspacing=0.5, borderaxespad=0.1)
    fig.savefig(path)
    plt.close(fig)


FIG2A_H = 4.0                    # cm, Figure 2 panel (a) alone (step 5e spec B5)


def figure2a(frames, model, path):
    """Panel (a) of Figure 2 alone (step 5e spec B5, handover 8.2 "Paper decisions of 10 Nov"): 28-day rolling
    PICP-80 of the four variants on interp, nominal line, legend in two rows below the panel, aligned left on the
    axes as in fig2.pdf. 12.2 x 4.0 cm, written next to fig2.pdf, which is unchanged. The panel spans the full width,
    so the month ticks follow Figure 1 (a) alone and the title drops the "(a)"."""
    import matplotlib.dates as mdates
    plt = _mpl()
    fig, a = plt.subplots(figsize=(FIG_W * CM, FIG2A_H * CM))
    cols = {"split": (C_BLUE, "-"), "rolling": (C_ORANGE, "--"), "acifrozen": (C_AQUA, "-."), "aci": (C_VIOLET, "-")}
    for reg in TABLE2_REGIMES:
        r = rolling_picp(frames[model, "interp", reg])
        a.plot(r.index, r.values, color=cols[reg][0], ls=cols[reg][1], lw=1.0, label=LAYER_LABEL[reg])
    a.axhline(80, color=INK, lw=0.6, ls=":")
    a.set_ylabel("PICP-80 (%), 28-day")
    a.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(11, 1, 3, 5, 7, 9)))
    a.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    a.set_title("interpolated, four variants", loc="left")
    a.grid(axis="y", color=GRID, lw=0.4)
    a.spines["right"].set_visible(False)
    a.spines["top"].set_visible(False)
    fig.tight_layout(pad=0.3, rect=(0, 2 * LEG_ROW_CM / FIG2A_H, 1, 1))
    ha, la = a.get_legend_handles_labels()
    pa = a.get_position()
    fig.legend(ha, la, loc="lower left", bbox_to_anchor=(pa.x0, 0), ncol=2, handlelength=1.4, handletextpad=0.4,
               columnspacing=0.8, borderaxespad=0.1)
    fig.savefig(path)
    plt.close(fig)


# ================================================================================ macros (k)
REG_NAME = {"split": "Split", "rolling": "Rolling", "acifrozen": "AciFrozen", "aci": "Aci"}
REGIME_MACROS = (("Crps", "CRPS", 2, ("split", "rolling", "acifrozen", "aci")),
                 ("Picp", "PICP80", 1, ("rolling", "acifrozen")),
                 ("Width", "width80", 2, ("acifrozen", "aci")),
                 ("Unbounded", "unbounded", 1, ("aci",)))


def regime_macros(t2):
    """The nine regime macros of handover rev 2.8.14 (Table 2 folded into the text): interp XGB, evaluation
    period, from the Table 2 rows, formatted as the table cells (unbounded in percent of quarter-days with a
    formal unbounded side at the 80% pair)."""
    t = t2.set_index("regime")
    return {f"reg{name}{REG_NAME[r]}": fmt(t.loc[r, col], nd) for name, col, nd, regs in REGIME_MACROS for r in regs}


def model_macros(model, met, t2, boot, reg_df, naive_mae):
    """numbers.tex macros of one model run. Gaps, slopes and DM p-values carry the model suffix. The headline
    macros named for XGB in main.tex (\\hlGapRemovedXGB, \\hlPicpSplit, \\hlPicpAci on interp XGB) are written
    by the XGB run only; a LEAR run writes them with the LEAR suffix (decided 7 Nov)."""
    M = MODEL_LABEL[model]
    macros = {f"naiveMAE{s}": fmt(naive_mae[s]) for s in SETTINGS}
    qname = {"CRPS": "Crps", "MCB": "Mcb", "DSC": "Dsc"}
    for r in boot.itertuples():
        base = f"gap{qname[r.quantity]}{'Raw' if r.regime == 'raw' else 'Aci'}{M}"
        macros[base] = fmt(r.diff_native_minus_interp)
        macros[base + "Lo"] = fmt(r.lo5)
        macros[base + "Hi"] = fmt(r.hi95)
    for r in reg_df.itertuples():
        R = "Raw" if r.regime == "raw" else "Aci"
        macros[f"slope{R}{M}"] = fmt(r.slope)
        macros[f"slope{R}{M}Se"] = fmt(r.slope_se)
        macros[f"slope{R}{M}P"] = fmt_p(r.slope_p)
        macros[f"dmP{R}{M}"] = fmt_p(r.dm_p)
    crps = lambda s, lay: met[(met["model"] == model) & (met["setting"] == s) & (met["layer"] == lay)].iloc[0]["CRPS"]
    g_raw = crps("interp", "raw") - crps("native", "raw")
    g_aci = crps("interp", "aci") - crps("native", "aci")
    hl = {"hlGapRemoved": fmt(100.0 * (g_raw - g_aci) / g_raw, 1),
          "hlPicpSplit": fmt(t2.set_index("regime").loc["split", "PICP80"], 1),
          "hlPicpAci": fmt(t2.set_index("regime").loc["aci", "PICP80"], 1)}
    if model == "xgb":
        macros.update({"hlGapRemovedXGB": hl["hlGapRemoved"], "hlPicpSplit": hl["hlPicpSplit"],
                       "hlPicpAci": hl["hlPicpAci"]})
        macros.update(regime_macros(t2))
    else:
        macros.update({k + M: v for k, v in hl.items()})
        if XGB_MACRO_STANDINS:
            macros.update({"hlGapRemovedXGB": hl["hlGapRemoved"], "hlPicpSplit": hl["hlPicpSplit"],
                           "hlPicpAci": hl["hlPicpAci"]})
    return macros


def load_ablation(out):
    """results/ablation.csv and ablation_pred.csv of step 5b, or (None, None)."""
    a, p = Path(out) / "ablation.csv", Path(out) / "ablation_pred.csv"
    if not (a.exists() and p.exists()):
        return None, None
    return (pd.read_csv(a, float_precision="round_trip", dtype={"level": str}),
            pd.read_csv(p, float_precision="round_trip"))


# ================================================================================ step 5c: ACI-block (spec B3)
def block_files_exist(model, cdir):
    return all((Path(cdir) / f"{s}_{model}_{REG_BLOCK}{x}.parquet").exists() for s in SETTINGS for x in ("", "_alpha"))


def append_csv(path, df, model, what=REG_BLOCK):
    """upsert_csv with the guarantee of spec B3.4: the file written before this call is a prefix of the file
    written by it, i.e. every existing row stays present and unchanged and the new rows are appended."""
    path = Path(path)
    prev = path.read_text() if path.exists() else ""
    upsert_csv(path, df, model)
    if not path.read_text().startswith(prev):
        raise RuntimeError(f"{path}: appending the {what} rows changed existing rows")


def block_macros(model, met, boot_b, dm_b):
    """The 12 step 5c macros of one model (handover 8.2, "Step 5c macros"): gaps native - interp under ACI-block
    with the 90% bootstrap bounds, the DM p-value, and CRPS(ACI-block) - CRPS(ACI-CQR) in interp and native."""
    M = MODEL_LABEL[model]
    qname = {"CRPS": "Crps", "MCB": "Mcb", "DSC": "Dsc"}
    out = {}
    for r in boot_b.itertuples():
        if r.regime != REG_BLOCK:
            raise ValueError(f"block_macros got a {r.regime} row")
        base = f"gap{qname[r.quantity]}Block{M}"
        out[base] = fmt(r.diff_native_minus_interp)
        out[base + "Lo"] = fmt(r.lo5)
        out[base + "Hi"] = fmt(r.hi95)
    out[f"dmPBlock{M}"] = fmt_p(dm_b["p"])
    crps = lambda s, lay: met[(met["model"] == model) & (met["setting"] == s) & (met["layer"] == lay)].iloc[0]["CRPS"]
    out[f"dCrpsBlockInterp{M}"] = fmt(crps("interp", REG_BLOCK) - crps("interp", "aci"))
    out[f"dCrpsBlockNative{M}"] = fmt(crps("native", REG_BLOCK) - crps("native", "aci"))
    if len(out) != 12:
        raise RuntimeError(f"expected 12 step 5c macros, got {len(out)}")
    return out


def block_rule(boot_b, dm_b):
    """Decision rule of handover 8.1, row ACI-block: DM p < 0.05 and the 90% interval of the CRPS gap excludes 0."""
    cr = boot_b[boot_b["quantity"] == "CRPS"].iloc[0]
    excludes = bool(cr["lo5"] > 0 or cr["hi95"] < 0)
    holds = bool(dm_b["p"] < 0.05 and excludes)
    return {"p": dm_b["p"], "lo5": float(cr["lo5"]), "hi95": float(cr["hi95"]), "excludes": excludes, "holds": holds,
            "line": f"rule: gap holds = DM p < 0.05 and lo5..hi95 excludes 0 -> {'yes' if holds else 'no'}"}


def block_stage(model, frames, met, boot, kp, reg_df, D, n_jobs, naive_mae, cdir, out):
    """Step 5c (spec B3): ACI-block in the CORP metrics, PICP and width, the unbounded / empty / clipped shares,
    the DM test, the block bootstrap on the same draws D as raw and ACI-CQR, and Kupiec and Christoffersen per
    slot. Appends its rows to metrics_<model>.csv, bootstrap.csv and kupiec.csv (existing rows unchanged) and
    returns what Section 11 and the 12 macros need. Nothing here enters Table 1, Table 2, the figures,
    monthly.csv, the pi-regression, regression.csv, the ablation, Proposition A or an existing macro (B3.2)."""
    out = Path(out)
    bw = {}

    t0 = time.perf_counter()
    ref = frames[model, "native", "raw"]
    fb = {}
    for s in SETTINGS:
        f = load_layer(model, s, REG_BLOCK, cdir=cdir)
        if not (f["datetime_local"].equals(ref["datetime_local"]) and f["actual"].equals(ref["actual"])):
            raise RuntimeError(f"{s} {REG_BLOCK}: rows or actuals differ from the native raw file")
        fb[s] = f
    met_b = pd.DataFrame([{"model": model, "setting": s, "layer": REG_BLOCK,
                           **frame_metrics(fb[s], naive_mae[s])} for s in SETTINGS])
    met_full = pd.concat([met, met_b], ignore_index=True)
    if not met_full.to_csv(index=False).startswith(met.to_csv(index=False)):
        raise RuntimeError("appending the aciblock rows changed existing rows of the metrics file")
    met_full.to_csv(out / f"metrics_{model}.csv", index=False)
    bw["load + metrics + CORP"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    d = (daily_crps(fb["native"]) - daily_crps(fb["interp"])).to_numpy()
    if len(d) != cf.N_EVAL_DAYS:
        raise RuntimeError("expected 337 evaluation days")
    dm_b = dm_test(d)
    ra = reg_df[reg_df["regime"] == "aci"].iloc[0]
    dm_rows = pd.DataFrame([{"regime": "aci", "mean": ra["dm_mean_native_minus_interp"], "se": ra["dm_se"],
                             "stat": ra["dm_stat"], "p": ra["dm_p"], "T": int(ra["T"])},
                            {"regime": REG_BLOCK, "mean": dm_b["mean"], "se": dm_b["se"], "stat": dm_b["stat"],
                             "p": dm_b["p"], "T": dm_b["T"]}])
    bw["DM"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    fr = {**frames, **{(model, s, REG_BLOCK): fb[s] for s in SETTINGS}}
    bt_b = bootstrap(fr, model, len(D), n_jobs, regimes=(REG_BLOCK,), D=D)
    boot_b = boot_summary(bt_b, met_full, model)
    append_csv(out / "bootstrap.csv", pd.concat([boot, boot_b], ignore_index=True), model)
    bw["bootstrap"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    kb = []
    for s in SETTINGS:
        k = coverage_tests(fb[s])
        k.insert(0, "layer", REG_BLOCK)
        k.insert(0, "setting", s)
        kb.append(k)
    kb = pd.concat(kb, ignore_index=True)
    kb.insert(0, "model", model)
    append_csv(out / "kupiec.csv", pd.concat([kp, kb], ignore_index=True), model)
    rej = coverage_rejections(pd.concat([kp[kp["layer"] == "aci"], kb], ignore_index=True))
    rej = rej[rej["setting"].isin(["interp", "native"])]
    kup = rej.pivot_table(index=["setting", "level"], columns="layer", values=["kupiec_rej", "cc_rej"], aggfunc="first")
    kup.columns = [f"{a} {b}" for a, b in kup.columns]
    kup = kup.reset_index()[["setting", "level", "kupiec_rej aci", f"kupiec_rej {REG_BLOCK}", "cc_rej aci",
                             f"cc_rej {REG_BLOCK}"]]
    kup = kup.sort_values(["setting", "level"], ascending=[True, False]).reset_index(drop=True)
    kup[kup.columns[2:]] = kup[kup.columns[2:]].astype(int)
    bw["Kupiec + Christoffersen"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    mrow = lambda s, lay: met_full[(met_full["model"] == model) & (met_full["setting"] == s)
                                   & (met_full["layer"] == lay)].iloc[0]
    per_set, grp, shr = [], [], []
    dev = {k: {"k": k, "target": cf.PAIRS[k][2] / 2, "bound": cf.aci_bound(k)} for k in cf.PAIRS}
    for s in SETTINGS:
        b, a = mrow(s, REG_BLOCK), mrow(s, "aci")
        per_set.append({"setting": s, "CRPS": b["CRPS"], "MCB": b["MCB"], "DSC": b["DSC"], "UNC": b["UNC"],
                        "PICP80": b["PICP80"], "PICP60": b["PICP60"], "width80": b["width80"],
                        "CRPS aci": a["CRPS"], "MCB aci": a["MCB"], "CRPS - aci": b["CRPS"] - a["CRPS"],
                        "MCB - aci": b["MCB"] - a["MCB"]})
        f = fb[s]
        y = f["actual"].to_numpy(float)
        cov = (y >= f["q10"].to_numpy(float)) & (y <= f["q90"].to_numpy(float))
        pg = pd.Series(cov).groupby(cf.group_key(f, cf.ALL_REGIMES[REG_BLOCK][2])).mean() * 100.0
        if len(pg) != 24:
            raise RuntimeError(f"{s}: {len(pg)} groups, expected 24")
        grp.append({"setting": s, "groups": len(pg), "PICP80 min": pg.min(), "group min": int(pg.idxmin()),
                    "PICP80 max": pg.max(), "group max": int(pg.idxmax())})
        al = pd.read_parquet(Path(cdir) / f"{s}_{model}_{REG_BLOCK}_alpha.parquet")
        s1, sa = cf.shares(al, 1), cf.shares(al)
        shr.append({"setting": s, **{f"{x} 80 (%)": 100.0 * s1[x] for x in s1},
                    **{f"{x} all pairs (%)": 100.0 * sa[x] for x in sa}})
        e = al.groupby(["group", "k", "side"])["err"].agg(["mean", "size"]).reset_index()
        if not (e["size"] == cf.N_EVAL_DAYS).all() or len(e) != 24 * 4 * 2:
            raise RuntimeError(f"{s}: alpha log not 337 days per (group, k, side)")
        e["dev"] = (e["mean"] - e["k"].map(lambda k: cf.PAIRS[k][2] / 2)).abs()
        for k in cf.PAIRS:
            dev[k][s] = float(e.loc[e["k"] == k, "dev"].max())
    devt = pd.DataFrame(list(dev.values()))
    devt["within bound"] = (devt[list(SETTINGS)].max(axis=1) <= devt["bound"]).map({True: "yes", False: "no"})
    order = {"aci": 0, REG_BLOCK: 1}
    bshow = pd.concat([boot[boot["regime"] == "aci"], boot_b], ignore_index=True)
    bshow = bshow.assign(_q=bshow["quantity"].map({"CRPS": 0, "MCB": 1, "DSC": 2}), _r=bshow["regime"].map(order))
    bshow = bshow.sort_values(["_q", "_r"]).drop(columns=["_q", "_r", "model", "block", "seed"]).reset_index(drop=True)
    macros = block_macros(model, met_full, boot_b, dm_b)
    rule = block_rule(boot_b, dm_b)
    bw["shares + groups + macros"] = time.perf_counter() - t0
    return {"met": met_b, "per_setting": pd.DataFrame(per_set), "groups": pd.DataFrame(grp), "shares": pd.DataFrame(shr),
            "dev": devt, "dm": dm_rows, "dm_block": dm_b, "boot": boot_b, "boot_show": bshow, "kupiec": kup,
            "macros": macros, "rule": rule, "walls": bw, "kb": kb}


def section11_lines(model, b):
    """Section 11 of summary_<model>.md (spec B3.6)."""
    M = MODEL_LABEL[model]
    r = b["rule"]
    L = [f"## 11. Step 5c: ACI-CQR with 24 groups ({REG_BLOCK})", "",
         f"ACI-CQR with 24 Mondrian groups g = 4 (h // 4) + p (h = clock hour, p = quarter position), handover 8.1 "
         f"row ACI-block, step 5c specification. Files conformal/<setting>_{model}_{REG_BLOCK}.parquet, "
         f"evaluation period as above. `aci` = ACI-CQR with the four position groups; `- aci` = {REG_BLOCK} minus "
         f"ACI-CQR in the same setting. {M}.", "",
         "Metrics per setting (CRPS, MCB, DSC, UNC on the CRPS scale; PICP and width at the issued bounds):", "",
         md_table(b["per_setting"], {"PICP80": ".2f", "PICP60": ".2f"}), "",
         "PICP-80 (%) by group, minimum and maximum over the 24 groups:", "",
         md_table(b["groups"], {"PICP80 min": ".2f", "PICP80 max": ".2f"}), "",
         "Shares (%) of quarter-days with a formal unbounded side, a formal empty side, a clipped index: at the 80% "
         "pair (k = 1) and over all four pairs (share of quarter-day x pair combinations):", "",
         md_table(b["shares"], {c: ".3f" for c in b["shares"].columns if c != "setting"}), "",
         "Prop. 4.1 per side: max over the 24 groups and both sides of |mean over the 337 days of err - 0.1k| "
         "(err of the formal side), per setting, next to the bound (max(0.1k, 1 - 0.1k) + gamma) / (gamma 337):", "",
         md_table(b["dev"], {"target": ".1f", "bound": ".5f", "hourly": ".5f", "interp": ".5f", "native": ".5f"}), "",
         "Diebold-Mariano on the daily CRPS difference native - interp (Newey-West, Bartlett lag 7, normal p), "
         "ACI-CQR beside ACI-block:", "",
         md_table(b["dm"], {"p": ".4g"}), "",
         "Bootstrap 90% intervals of native - interp (the same 1,000 draws as raw and ACI-CQR, circular blocks of "
         "7 days, default_rng(2027)), ACI-CQR beside ACI-block:", "",
         md_table(b["boot_show"]), "",
         "Kupiec and Christoffersen (cc) per quarter-hour slot, rejections at 5% after Holm over the 96 slots, of 96 "
         "(descriptive only):", "",
         md_table(b["kupiec"]), "",
         f"Decision rule of 8.1 (row ACI-block), inputs: DM p = {fmt_p_txt(r['p'])} ({r['p']:.4g}), CRPS gap "
         f"lo5..hi95 = {r['lo5']:.4f} .. {r['hi95']:.4f}.", "",
         r["line"], "",
         "Macros written to results/numbers.tex by this run for step 5c (not listed in Section 9):", "",
         md_table(pd.DataFrame([{"macro": "\\" + a, "value": v} for a, v in sorted(b["macros"].items())])), "",
         "Wall time of the step 5c work (not included in the stage rows of Section 10):", "",
         md_table(pd.DataFrame([{"stage": a, "seconds": v} for a, v in b["walls"].items()]), {"seconds": ".1f"}), ""]
    return L


# ================================================================================ step 5d: xgbm (spec B4)
RETUNE = "xgbm"
RETUNE_SETTINGS = xgbmod.RETUNE_SETTINGS                    # interp, native
RETUNE_LAYERS = ("raw", "aci", REG_BLOCK)
RETUNE_NAME = {"raw": "Raw", "aci": "Aci", REG_BLOCK: "Block"}
RETUNE_MACROS = tuple([f"gapCrps{RETUNE_NAME[l]}RetuneXGB{x}" for l in RETUNE_LAYERS for x in ("", "Lo", "Hi")]
                      + [f"dmP{RETUNE_NAME[l]}RetuneXGB" for l in RETUNE_LAYERS]
                      + ["dCrpsRetuneInterpXGB", "dCrpsRetuneNativeXGB"])
RETUNE_FILES = ("retune_xgb_metrics.csv", "retune_xgb_bootstrap.csv", "retune_xgb_monthly.csv",
                "retune_xgb_hyperparameters.csv")
RETUNE_QTY = ("CRPS", "MCB", "DSC", "UNC", "PICP80", "PICP60", "width80")
SEC12 = "## 12. Step 5d: XGB with monthly re-tuning (xgbm)"


def retune_files_exist(rdir):
    return all((Path(rdir) / f"{s}_{RETUNE}.parquet").exists() for s in RETUNE_SETTINGS)


def retune_macros(met_r, met_x, boot_r, dm_r):
    """The 14 step 5d macros (handover 8.2, "Step 5d macros"): the CRPS gap native - interp under xgbm, raw, ACI-CQR
    and ACI-block, with the 90% bootstrap bounds, the DM p-values, and raw CRPS of xgbm minus raw CRPS of xgb per
    setting. met_r: metrics of xgbm, met_x: metrics of xgb, boot_r: boot_summary rows of xgbm, dm_r: {layer: dm}."""
    out = {}
    for lay in RETUNE_LAYERS:
        b = boot_r[(boot_r["regime"] == lay) & (boot_r["quantity"] == "CRPS")]
        if len(b) != 1:
            raise ValueError(f"retune_macros: {len(b)} CRPS bootstrap rows for {lay}")
        r = b.iloc[0]
        base = f"gapCrps{RETUNE_NAME[lay]}RetuneXGB"
        out[base] = fmt(r["diff_native_minus_interp"])
        out[base + "Lo"] = fmt(r["lo5"])
        out[base + "Hi"] = fmt(r["hi95"])
        out[f"dmP{RETUNE_NAME[lay]}RetuneXGB"] = fmt_p(dm_r[lay]["p"])
    raw = lambda m, model, s: m[(m["model"] == model) & (m["setting"] == s) & (m["layer"] == "raw")]["CRPS"].iloc[0]
    for s, S in (("interp", "Interp"), ("native", "Native")):
        out[f"dCrpsRetune{S}XGB"] = fmt(raw(met_r, RETUNE, s) - raw(met_x, "xgb", s))
    if sorted(out) != sorted(RETUNE_MACROS):
        raise RuntimeError(f"expected the 14 step 5d macros, got {sorted(out)}")
    return out


def retune_rule(boot_r, dm_r):
    """Decision rule of the Section 8 row "XGB monthly re-tuning": robust if the CRPS gap native - interp under xgbm
    has DM p < 0.05 and a 90% interval that excludes 0, raw and under ACI-CQR."""
    parts = {}
    for lay in ("raw", "aci"):
        cr = boot_r[(boot_r["regime"] == lay) & (boot_r["quantity"] == "CRPS")].iloc[0]
        exc = bool(cr["lo5"] > 0 or cr["hi95"] < 0)
        p = float(dm_r[lay]["p"])
        parts[lay] = {"p": p, "lo5": float(cr["lo5"]), "hi95": float(cr["hi95"]), "excludes": exc,
                      "holds": bool(p < 0.05 and exc)}
    holds = all(v["holds"] for v in parts.values())
    return {"parts": parts, "holds": holds,
            "line": "rule: robust = DM p < 0.05 and lo5..hi95 excludes 0, raw and ACI-CQR -> " + ("yes" if holds else "no")}


def retune_hp_tables(log, frozen):
    """Per setting and re-tune date: the selected point configuration, the quantile eta counts and rounds range, and
    whether the set differs from the frozen one (any of the 10 models in depth, eta or rounds; the point model).
    Second table: per re-tune date, whether interp and native selected the same 10 configurations."""
    models = ["point"] + data.QCOLS
    d0 = int(frozen["point"]["max_depth"])
    fz = {"point": (d0, float(frozen["point"]["eta"]), int(frozen["point"]["rounds"]))}
    fz.update({c: (d0, float(frozen["quantiles"][c]["eta"]), int(frozen["quantiles"][c]["rounds"])) for c in data.QCOLS})
    rows, cfgs = [], {}
    for (s, r), g in log.groupby(["setting", "retune_date"], sort=True):
        g = g.set_index("model")
        if list(g.index) != models:
            raise RuntimeError(f"retune_log.csv: models of {s} {r} are {list(g.index)}")
        cfg = {m: (int(g.loc[m, "max_depth"]), float(g.loc[m, "eta"]), int(g.loc[m, "rounds"])) for m in models}
        cfgs[s, r] = cfg
        q = g.loc[data.QCOLS]
        nd = sum(cfg[m] != fz[m] for m in models)
        rows.append({"setting": s, "retune_date": r, "point depth": cfg["point"][0], "point eta": cfg["point"][1],
                     "point rounds": cfg["point"][2], "q eta 0.05": int((q["eta"] == 0.05).sum()),
                     "q eta 0.1": int((q["eta"] == 0.1).sum()), "q rounds min": int(q["rounds"].min()),
                     "q rounds max": int(q["rounds"].max()), "models differing": nd, "differs": "yes" if nd else "no",
                     "point differs": "yes" if cfg["point"] != fz["point"] else "no"})
    same = []
    for r in xgbmod.RETUNE_DATES:
        if ("interp", r) in cfgs and ("native", r) in cfgs:
            a, b = cfgs["interp", r], cfgs["native", r]
            nd = sum(a[m] != b[m] for m in models)
            same.append({"retune_date": r, "interp = native": "yes" if nd == 0 else "no", "models differing": nd})
    qr = [int(frozen["quantiles"][c]["rounds"]) for c in data.QCOLS]
    frozen_txt = (f"point depth {fz['point'][0]}, eta {fz['point'][1]}, rounds {fz['point'][2]}; quantile eta 0.05 / 0.1: "
                  f"{sum(frozen['quantiles'][c]['eta'] == 0.05 for c in data.QCOLS)} / "
                  f"{sum(frozen['quantiles'][c]['eta'] == 0.1 for c in data.QCOLS)}, rounds {min(qr)}-{max(qr)}")
    return pd.DataFrame(rows), pd.DataFrame(same), frozen_txt


def retune_stage(frames, met, boot, reg_df, block, D, n_jobs, naive_mae, rdir, cdir, out, mon_all):
    """Step 5d (spec B4): xgbm raw, ACI-CQR and ACI-block in interp and native: CORP metrics, PICP, width, DM on the
    daily CRPS native - interp, the block bootstrap on the same draws D as every other regime, the monthly raw CRPS
    of xgb and xgbm, the re-tuned hyperparameters. Writes the four retune_xgb_*.csv files and returns what Section 12
    and the 14 macros need. Nothing here enters Table 1, Table 2, the figures, monthly.csv, the regression, the
    ablation, Proposition A, Kupiec, an existing CSV row or an existing macro (B4.2)."""
    out, rdir, cdir = Path(out), Path(rdir), Path(cdir)
    if block is None:
        raise RuntimeError("step 5d needs the xgb aciblock files of step 5c (ACI-block of xgb beside xgbm)")
    bw = {}

    t0 = time.perf_counter()
    ref = frames["xgb", "native", "raw"]
    fr = {}
    for s in RETUNE_SETTINGS:
        for lay in RETUNE_LAYERS:
            p = rdir / f"{s}_{RETUNE}.parquet" if lay == "raw" else cdir / f"{s}_{RETUNE}_{lay}.parquet"
            if not p.exists():
                raise RuntimeError(f"step 5d: {p} missing ({rdir} holds the xgbm forecasts)")
            f = load_layer(RETUNE, s, lay, fdir=rdir, cdir=cdir)
            if not (f["datetime_local"].equals(ref["datetime_local"]) and f["actual"].equals(ref["actual"])):
                raise RuntimeError(f"{s} {RETUNE} {lay}: rows or actuals differ from the native raw xgb file")
            fr[RETUNE, s, lay] = f
    met_r = pd.DataFrame([{"model": RETUNE, "setting": s, "layer": lay,
                           **frame_metrics(fr[RETUNE, s, lay], naive_mae[s])}
                          for s in RETUNE_SETTINGS for lay in RETUNE_LAYERS])
    met_r.to_csv(out / "retune_xgb_metrics.csv", index=False)
    bw["load + metrics + CORP"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    dm_r = {}
    for lay in RETUNE_LAYERS:
        d = (daily_crps(fr[RETUNE, "native", lay]) - daily_crps(fr[RETUNE, "interp", lay])).to_numpy()
        if len(d) != cf.N_EVAL_DAYS:
            raise RuntimeError("expected 337 evaluation days")
        dm_r[lay] = dm_test(d)
    bw["DM"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    bt = bootstrap(fr, RETUNE, len(D), n_jobs, regimes=RETUNE_LAYERS, D=D)
    boot_r = boot_summary(bt, met_r, RETUNE)
    boot_r.to_csv(out / "retune_xgb_bootstrap.csv", index=False)
    bw["bootstrap"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    mrows = []
    for m, src in (("xgb", frames), (RETUNE, fr)):
        for s in RETUNE_SETTINGS:
            f = src[m, s, "raw"]
            dates = f["date"].astype(str)
            rc = pd.Series(row_crps(f)).groupby(dates.str[:7].to_numpy()).mean()     # as monthly.csv
            ud = pd.Series(dates.unique())
            for mth in rc.index:
                dd = ud[ud.str[:7] == mth]
                mrows.append({"model": m, "setting": s, "layer": "raw", "month": mth, "n_days": len(dd),
                              "mean_pi": float(native_share(dd).mean()), "CRPS": float(rc[mth])})
    mon_r = pd.DataFrame(mrows)
    mon_r.to_csv(out / "retune_xgb_monthly.csv", index=False)
    log = pd.read_csv(rdir / "retune_log.csv", float_precision="round_trip", dtype={"retune_date": str})
    log.drop(columns=["wall_s"]).to_csv(out / "retune_xgb_hyperparameters.csv", index=False)
    hp, same, frozen_txt = retune_hp_tables(log, xgbmod.load_hp("15min"))
    bw["monthly + hyperparameters"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    met_x = pd.concat([met[met["model"] == "xgb"], block["met"]], ignore_index=True)
    rows = []
    for s in RETUNE_SETTINGS:
        for lay in RETUNE_LAYERS:
            a = met_r[(met_r["setting"] == s) & (met_r["layer"] == lay)].iloc[0]
            b = met_x[(met_x["setting"] == s) & (met_x["layer"] == lay)].iloc[0]
            for name in (RETUNE, "xgb", "xgbm - xgb"):
                v = {q: a[q] if name == RETUNE else (b[q] if name == "xgb" else a[q] - b[q]) for q in RETUNE_QTY}
                rows.append({"setting": s, "layer": lay, "model": name, **v})
    met_show = pd.DataFrame(rows)
    dm_rows = []
    for lay in RETUNE_LAYERS:
        if lay == REG_BLOCK:
            x = block["dm_block"]
            xd = {k: x[k] for k in ("mean", "se", "stat", "p", "T")}
        else:
            r = reg_df[reg_df["regime"] == lay].iloc[0]
            xd = {"mean": r["dm_mean_native_minus_interp"], "se": r["dm_se"], "stat": r["dm_stat"], "p": r["dm_p"],
                  "T": int(r["T"])}
        dm_rows.append({"layer": lay, "model": "xgb", **xd})
        dm_rows.append({"layer": lay, "model": RETUNE, **{k: dm_r[lay][k] for k in ("mean", "se", "stat", "p", "T")}})
    bs = pd.concat([boot, block["boot"], boot_r], ignore_index=True)
    bs = bs.assign(_q=bs["quantity"].map({"CRPS": 0, "MCB": 1, "DSC": 2}),
                   _l=bs["regime"].map({l: i for i, l in enumerate(RETUNE_LAYERS)}),
                   _m=bs["model"].map({"xgb": 0, RETUNE: 1}))
    bs = bs.sort_values(["_q", "_l", "_m"]).reset_index(drop=True)
    bs = bs[["regime", "quantity", "model", "native", "interp", "diff_native_minus_interp", "lo5", "hi95", "boot_mean",
             "boot_sd", "n_draws"]]
    lr = mon_all[(mon_all["model"] == "lear") & (mon_all["layer"] == "raw")]
    mon_show = mon_r[(mon_r["model"] == "xgb") & (mon_r["setting"] == "interp")][["month", "n_days", "mean_pi"]]
    mon_show = mon_show.reset_index(drop=True)
    for s in RETUNE_SETTINGS:
        get = lambda df, m: df[(df["model"] == m) & (df["setting"] == s)].set_index("month")["CRPS"]
        x, y = get(mon_r, "xgb"), get(mon_r, RETUNE)
        lv = lr[lr["setting"] == s].set_index("month")["CRPS"]
        mon_show[f"{s} LEAR"] = mon_show["month"].map(lv)
        mon_show[f"{s} xgb"] = mon_show["month"].map(x)
        mon_show[f"{s} xgbm"] = mon_show["month"].map(y)
        mon_show[f"{s} xgbm - xgb"] = mon_show[f"{s} xgbm"] - mon_show[f"{s} xgb"]
    macros = retune_macros(met_r, met_x, boot_r, dm_r)
    rule = retune_rule(boot_r, dm_r)
    bw["tables + macros"] = time.perf_counter() - t0

    walls = []
    for s in RETUNE_SETTINGS:
        g = log[log["setting"] == s].groupby("retune_date")["wall_s"].first()
        walls.append({"stage": f"re-tuning {s} ({len(g)} dates)", "seconds": float(g.sum()),
                      "per item (s)": float(g.mean())})
    for s in RETUNE_SETTINGS:
        dw = pd.read_parquet(rdir / f"{s}_{RETUNE}_daywall.parquet")
        walls.append({"stage": f"forecasts {s} ({len(dw)} days)", "seconds": float(dw["wall_s"].sum()),
                      "per item (s)": float(dw["wall_s"].mean())})
    cw = rdir / "conformal_wall.csv"
    if cw.exists():
        for r in pd.read_csv(cw, float_precision="round_trip").itertuples():
            walls.append({"stage": f"conformal {r.setting} {r.regime}", "seconds": float(r.wall_s),
                          "per item (s)": float(r.wall_s) / cf.N_EVAL_DAYS})
    else:
        walls.append({"stage": f"conformal (not recorded, {cw} missing)", "seconds": np.nan, "per item (s)": np.nan})
    walls += [{"stage": f"evaluation: {k}", "seconds": v, "per item (s)": np.nan} for k, v in bw.items()]
    return {"met": met_r, "boot": boot_r, "dm": dm_r, "monthly": mon_r, "hp": hp, "same": same,
            "frozen_txt": frozen_txt, "met_show": met_show, "dm_show": pd.DataFrame(dm_rows), "boot_show": bs,
            "mon_show": mon_show, "macros": macros, "rule": rule, "walls": bw, "walls_table": pd.DataFrame(walls)}


def section12_lines(b):
    """Section 12 of summary_xgb.md (spec B4.5)."""
    hp, same, r = b["hp"], b["same"], b["rule"]
    nd, npd = int((hp["differs"] == "yes").sum()), int((hp["point differs"] == "yes").sum())
    pr, pa = r["parts"]["raw"], r["parts"]["aci"]
    nov = same[same["retune_date"] == "2025-11-01"]
    nov_txt = ("not available" if nov.empty else
               ("identical" if nov.iloc[0]["interp = native"] == "yes" else
                f"different ({int(nov.iloc[0]['models differing'])} of 10 models)"))
    L = [SEC12, "",
         "XGB with its hyperparameters re-tuned on the first day of each month (handover Section 8 row \"XGB monthly "
         "re-tuning\", step 5d specification): model `xgbm`, settings interp and native, forecasts in "
         "forecasts_retune/, conformal/<setting>_xgbm_{aci,aciblock}.parquet. `xgb` = the frozen design (main "
         "analysis). Evaluation period as above.", "",
         "Hyperparameters per re-tune date and setting (forecasts_retune/retune_log.csv, "
         f"results/retune_xgb_hyperparameters.csv). Frozen set (hyperparameters_15min.json): {b['frozen_txt']}. "
         "differs = any of the 10 models (point, q10..q90) differs from the frozen set in depth, eta or rounds; "
         "point differs = the point model does.", "",
         md_table(hp, {"point eta": ".2f"}), "",
         f"Setting-dates differing from the frozen set: {nd} of {len(hp)} (point model: {npd} of {len(hp)}).", "",
         "Selected configurations, interp against native per re-tune date (10 models). The 1 Nov 2025 re-tune "
         f"(validation 6 Sep - 31 Oct 2025, 31 native days in the native file) is {nov_txt} across the settings.", "",
         md_table(same), "",
         "Metrics per setting and layer (CRPS, MCB, DSC, UNC on the CRPS scale; PICP and width at the issued bounds), "
         "xgbm next to xgb and the difference:", "",
         md_table(b["met_show"], {"PICP80": ".2f", "PICP60": ".2f"}), "",
         "Diebold-Mariano on the daily CRPS difference native - interp (Newey-West, Bartlett lag 7, normal p), xgb "
         "beside xgbm:", "",
         md_table(b["dm_show"], {"p": ".4g"}), "",
         "Bootstrap 90% intervals of native - interp (the same 1,000 draws as every other regime, circular blocks of "
         "7 days, default_rng(2027)), xgb beside xgbm:", "",
         md_table(b["boot_show"]), "",
         "Monthly raw CRPS (results/retune_xgb_monthly.csv, LEAR from results/monthly.csv):", "",
         md_table(b["mon_show"], {"mean_pi": ".3f"}), "",
         f"Decision rule of Section 8 (row \"XGB monthly re-tuning\"), inputs: raw DM p = {fmt_p_txt(pr['p'])} "
         f"({pr['p']:.4g}), CRPS gap lo5..hi95 = {pr['lo5']:.4f} .. {pr['hi95']:.4f}; ACI-CQR DM p = "
         f"{fmt_p_txt(pa['p'])} ({pa['p']:.4g}), CRPS gap lo5..hi95 = {pa['lo5']:.4f} .. {pa['hi95']:.4f}.", "",
         r["line"], "",
         "Macros written to results/numbers.tex by this run for step 5d (not listed in Section 9):", "",
         md_table(pd.DataFrame([{"macro": "\\" + a, "value": v} for a, v in sorted(b["macros"].items())])), "",
         "Wall time of step 5d: re-tuning and forecasts from forecasts_retune/ (per item = per re-tune, per day), "
         "conformal from forecasts_retune/conformal_wall.csv (per item = per evaluation day), evaluation = this run "
         "(not included in the stage rows of Section 10):", "",
         md_table(b["walls_table"], {"seconds": ".1f", "per item (s)": ".3f"}), ""]
    return L


# ================================================================================ step 5e: diagnostic layers (spec B3)
DIAG_NAME = {"symblock": "SymBlock", "locblock": "LocBlock"}          # the two layers quoted through macros
PROFILE_MACROS = (("profMornQzeroInterp", 8), ("profMornQthreeInterp", 11), ("profEveQzeroInterp", 16),
                  ("profEveQthreeInterp", 19))                       # groups 08-11 h and 16-19 h at :00 and :45
SEC5E = "Step 5e: diagnostic layers (symblock, locblock, locslot), defined after the 24-group result"
SHOW5E = ("aci", REG_BLOCK) + DIAG                                    # regimes side by side in the summary


def diag_macro_names(model):
    """The 16 step 5e macros of one model (handover 8.2 "Step 5e macros")."""
    M = MODEL_LABEL[model]
    names = []
    for R in DIAG_NAME.values():
        names += [f"gapCrps{R}{M}", f"gapCrps{R}{M}Lo", f"gapCrps{R}{M}Hi", f"dmP{R}{M}"]
    names += [f"gapDsc{R}{M}" for R in DIAG_NAME.values()]
    names += [f"widthLocBlockInterp{M}", f"widthLocBlockNative{M}"] + [f"{p}{M}" for p, _ in PROFILE_MACROS]
    return names


DIAG_MACROS = tuple(n for m in MODELS_TABLE1 for n in diag_macro_names(m))


def diag_files_exist(model, cdir):
    return all((Path(cdir) / f"{s}_{model}_{r}{x}.parquet").exists()
               for s in SETTINGS for r in DIAG for x in cf.regime_files(r))


def fmt_signed(v, nd=1):
    """One decimal with an explicit sign: +9.3, \\ensuremath{-}10.5 (handover 8.2, profile macros)."""
    s = f"{abs(v):.{nd}f}"
    if float(s) == 0:
        return s
    return ("+" if v > 0 else "\\ensuremath{-}") + s


def residual_profile(frames, model):
    """results/profile_<model>.csv (spec B3.4): per setting (interp, native) and block group 0..23 (g = 4 (h // 4)
    + p) the mean of the raw residual y - q50 of forecasts/<setting>_<model>.parquet over the evaluation period,
    the unscaled median absolute deviation around the group median, n."""
    rows = []
    for s in ("interp", "native"):
        f = frames[model, s, "raw"]
        g = cf.group_key(f, "block")
        r = pd.Series(f["actual"].to_numpy(float) - f["q50"].to_numpy(float))
        mean = r.groupby(g).mean()
        for gg, v in r.groupby(g):
            v = v.to_numpy()
            b = 4 * (int(gg) // 4)
            rows.append({"model": model, "setting": s, "group": int(gg), "block": f"{b:02d}-{b + 3:02d}",
                         "position": int(gg) % 4, "mean_resid": float(mean.loc[gg]),
                         "mad_resid": float(np.median(np.abs(v - np.median(v)))), "n": len(v)})
    out = pd.DataFrame(rows)
    if len(out) != 48:
        raise RuntimeError(f"profile: {len(out)} rows, expected 48")
    return out


def wl_shares(met, model):
    """Shares of the reading rule (handover Section 8 row "Paper decisions of 10 Nov and step 5e scope"): with G =
    |CRPS gap native - interp| under aci (G4), aciblock (G24), symblock (GS), locblock (GL), width share W = (G4 -
    GS) / (G4 - G24), location share L = (G4 - GL) / (G4 - G24). None when a layer is missing for the model."""
    sub = met[met["model"] == model]
    G = {}
    for lay in ("aci", REG_BLOCK, "symblock", "locblock"):
        c = {}
        for s in ("interp", "native"):
            r = sub[(sub["setting"] == s) & (sub["layer"] == lay)]
            if len(r) != 1:
                return None
            c[s] = float(r.iloc[0]["CRPS"])
        G[lay] = abs(c["native"] - c["interp"])
    den = G["aci"] - G[REG_BLOCK]
    return {"model": MODEL_LABEL[model], "G4": G["aci"], "G24": G[REG_BLOCK], "GS": G["symblock"],
            "GL": G["locblock"], "W": (G["aci"] - G["symblock"]) / den, "L": (G["aci"] - G["locblock"]) / den}


def reading_label(sh):
    """recentering if W <= 1/3 and L >= 2/3 in both models, width if W >= 2/3 and L <= 1/3 in both, mixed otherwise;
    pending while a model has no step 5e rows."""
    v = [sh[m] for m in MODELS_TABLE1]
    if any(x is None for x in v):
        return "pending"
    if all(x["W"] <= 1 / 3 and x["L"] >= 2 / 3 for x in v):
        return "recentering"
    if all(x["W"] >= 2 / 3 and x["L"] <= 1 / 3 for x in v):
        return "width"
    return "mixed"


def reading_line(sh):
    f = lambda m, q: "pending" if sh[m] is None else f"{sh[m][q]:.3f}"
    return (f"reading: W = {f('lear', 'W')} / {f('xgb', 'W')}, L = {f('lear', 'L')} / {f('xgb', 'L')} (LEAR / XGB) "
            f"-> {reading_label(sh)}")


def diag_macros(model, met, boot_e, dm_e, prof):
    """The 16 step 5e macros of one model, formatted as the existing gap macros (profile: one decimal, signed)."""
    M = MODEL_LABEL[model]
    out = {}
    for reg, R in DIAG_NAME.items():
        b = boot_e[boot_e["regime"] == reg].set_index("quantity")
        out[f"gapCrps{R}{M}"] = fmt(b.loc["CRPS", "diff_native_minus_interp"])
        out[f"gapCrps{R}{M}Lo"] = fmt(b.loc["CRPS", "lo5"])
        out[f"gapCrps{R}{M}Hi"] = fmt(b.loc["CRPS", "hi95"])
        out[f"dmP{R}{M}"] = fmt_p(dm_e[reg]["p"])
        out[f"gapDsc{R}{M}"] = fmt(b.loc["DSC", "diff_native_minus_interp"])
    w = lambda s: met[(met["model"] == model) & (met["setting"] == s) & (met["layer"] == "locblock")].iloc[0]["width80"]
    out[f"widthLocBlockInterp{M}"] = fmt(w("interp"))
    out[f"widthLocBlockNative{M}"] = fmt(w("native"))
    pi = prof[prof["setting"] == "interp"].set_index("group")
    for name, g in PROFILE_MACROS:
        out[f"{name}{M}"] = fmt_signed(pi.loc[g, "mean_resid"])
    if sorted(out) != sorted(diag_macro_names(model)):
        raise RuntimeError(f"step 5e macros: {sorted(out)}")
    return out


def diag_stage(model, frames, met, boot, kp, reg_df, block, D, n_jobs, naive_mae, cdir, out):
    """Step 5e (spec B3): symblock, locblock, locslot in the CORP metrics, PICP and width, the unbounded / empty /
    clipped shares, the DM test, the block bootstrap on the same draws D as every other regime, Kupiec and
    Christoffersen per slot; the residual profile; W, L and the reading line; the 16 macros. Appends rows to
    metrics_<model>.csv, bootstrap.csv and kupiec.csv (the files written before are prefixes of the new ones),
    writes results/profile_<model>.csv. Nothing here enters Table 1, Table 2, fig1.pdf, fig2.pdf, monthly.csv,
    the pi-regression, the ablation, Proposition A or an existing macro (B3.2)."""
    out = Path(out)
    ew = {}

    t0 = time.perf_counter()
    ref = frames[model, "native", "raw"]
    fe = {}
    for r in DIAG:
        for s in SETTINGS:
            f = load_layer(model, s, r, cdir=cdir)
            if not (f["datetime_local"].equals(ref["datetime_local"]) and f["actual"].equals(ref["actual"])):
                raise RuntimeError(f"{s} {r}: rows or actuals differ from the native raw file")
            fe[s, r] = f
    met_e = pd.DataFrame([{"model": model, "setting": s, "layer": r, **frame_metrics(fe[s, r], naive_mae[s])}
                          for r in DIAG for s in SETTINGS])
    met_prev = pd.concat([met, block["met"]], ignore_index=True)            # the file written by block_stage
    met_full = pd.concat([met_prev, met_e], ignore_index=True)
    mpath = out / f"metrics_{model}.csv"
    prev = mpath.read_text()
    if prev != met_prev.to_csv(index=False) or not met_full.to_csv(index=False).startswith(prev):
        raise RuntimeError("appending the step 5e rows would change existing rows of the metrics file")
    met_full.to_csv(mpath, index=False)
    ew["load + metrics + CORP"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    dm_e = {}
    for r in DIAG:
        d = (daily_crps(fe["native", r]) - daily_crps(fe["interp", r])).to_numpy()
        if len(d) != cf.N_EVAL_DAYS:
            raise RuntimeError("expected 337 evaluation days")
        dm_e[r] = dm_test(d)
    ra = reg_df[reg_df["regime"] == "aci"].iloc[0]
    dmr = [{"regime": "aci", "mean": ra["dm_mean_native_minus_interp"], "se": ra["dm_se"], "stat": ra["dm_stat"],
            "p": ra["dm_p"], "T": int(ra["T"])}]
    dmr += [{"regime": rg, **{k: v[k] for k in ("mean", "se", "stat", "p", "T")}}
            for rg, v in [(REG_BLOCK, block["dm_block"])] + [(r, dm_e[r]) for r in DIAG]]
    ew["DM"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    fr = {**frames, **{(model, s, r): fe[s, r] for r in DIAG for s in SETTINGS}}
    bt_e = bootstrap(fr, model, len(D), n_jobs, regimes=DIAG, D=D)
    boot_e = boot_summary(bt_e, met_full, model)
    append_csv(out / "bootstrap.csv", pd.concat([boot, block["boot"], boot_e], ignore_index=True), model, "step 5e")
    ew["bootstrap"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    ke = []
    for r in DIAG:
        for s in SETTINGS:
            k = coverage_tests(fe[s, r])
            k.insert(0, "layer", r)
            k.insert(0, "setting", s)
            ke.append(k)
    ke = pd.concat(ke, ignore_index=True)
    ke.insert(0, "model", model)
    append_csv(out / "kupiec.csv", pd.concat([kp, block["kb"], ke], ignore_index=True), model, "step 5e")
    rej = coverage_rejections(pd.concat([kp[kp["layer"] == "aci"], block["kb"], ke], ignore_index=True))
    kup = {}
    for col in ("kupiec_rej", "cc_rej"):
        t = rej.pivot_table(index=["setting", "level"], columns="layer", values=col, aggfunc="first")
        t = t[list(SHOW5E)].reset_index()
        t["_s"] = t["setting"].map({s: i for i, s in enumerate(SETTINGS)})
        t = t.sort_values(["_s", "level"], ascending=[True, False]).drop(columns="_s").reset_index(drop=True)
        t[list(SHOW5E)] = t[list(SHOW5E)].astype(int)
        t.columns.name = None
        kup[col] = t
    ew["Kupiec + Christoffersen"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    mrow = lambda s, lay: met_full[(met_full["model"] == model) & (met_full["setting"] == s)
                                   & (met_full["layer"] == lay)].iloc[0]
    per = [{"setting": s, "regime": lay, **{q: mrow(s, lay)[q] for q in ("CRPS", "MCB", "DSC", "UNC", "PICP80",
                                                                        "PICP60", "width80")}}
           for s in SETTINGS for lay in SHOW5E]
    shr, dev = [], {k: {"k": k, "target": cf.PAIRS[k][2], "bound": cf.sym_bound(k)} for k in cf.PAIRS}
    for r in DIAG:
        for s in SETTINGS:
            al = pd.read_parquet(Path(cdir) / f"{s}_{model}_{r}_alpha.parquet")
            s1, sa = cf.shares(al, 1), cf.shares(al)
            shr.append({"regime": r, "setting": s, **{f"{x} 80 (%)": 100.0 * s1[x] for x in s1},
                        **{f"{x} all pairs (%)": 100.0 * sa[x] for x in sa}})
            if r == "symblock":
                e = al.groupby(["group", "k"])["err"].agg(["mean", "size"]).reset_index()
                if not (e["size"] == cf.N_EVAL_DAYS).all() or len(e) != 24 * 4 or set(al["side"]) != {cf.SYM}:
                    raise RuntimeError(f"{s} symblock: alpha log not 337 days per (group, k)")
                e["dev"] = (e["mean"] - e["k"].map(lambda k: cf.PAIRS[k][2])).abs()
                for k in cf.PAIRS:
                    dev[k][s] = float(e.loc[e["k"] == k, "dev"].max())
    devt = pd.DataFrame(list(dev.values()))
    devt["within bound"] = (devt[list(SETTINGS)].max(axis=1) <= devt["bound"]).map({True: "yes", False: "no"})
    order = {r: i for i, r in enumerate(SHOW5E)}
    bshow = pd.concat([boot[boot["regime"] == "aci"], block["boot"], boot_e], ignore_index=True)
    bshow = bshow.assign(_q=bshow["quantity"].map({"CRPS": 0, "MCB": 1, "DSC": 2}), _r=bshow["regime"].map(order))
    bshow = bshow.sort_values(["_q", "_r"]).drop(columns=["_q", "_r", "model", "block", "seed"]).reset_index(drop=True)

    prof = residual_profile(frames, model)
    prof.to_csv(out / f"profile_{model}.csv", index=False)
    pw = prof.pivot_table(index=["group", "block", "position"], columns="setting", values=["mean_resid", "mad_resid"],
                          aggfunc="first")
    pw.columns = [f"{a.replace('_resid', '')} {b}" for a, b in pw.columns]
    pw = pw.reset_index()[["group", "block", "position", "mean interp", "mad interp", "mean native", "mad native"]]
    pw["n"] = prof[prof["setting"] == "interp"].set_index("group").loc[pw["group"], "n"].to_numpy()

    sh = {model: wl_shares(met_full, model)}
    for m in MODELS_TABLE1:
        if m != model:
            p = out / f"metrics_{m}.csv"
            sh[m] = wl_shares(pd.read_csv(p, float_precision="round_trip"), m) if p.exists() else None
    wl = pd.DataFrame([dict(sh[m]) if sh[m] is not None else
                       {"model": MODEL_LABEL[m], **{q: "pending" for q in ("G4", "G24", "GS", "GL", "W", "L")}}
                       for m in MODELS_TABLE1])
    macros = diag_macros(model, met_full, boot_e, dm_e, prof)
    ew["shares + profile + macros"] = time.perf_counter() - t0
    return {"met": met_e, "per_setting": pd.DataFrame(per), "shares": pd.DataFrame(shr), "dev": devt,
            "dm": pd.DataFrame(dmr), "boot": boot_e, "boot_show": bshow, "kupiec": kup, "profile": pw,
            "wl": wl, "reading": reading_line(sh), "label": reading_label(sh), "macros": macros, "walls": ew}


def section5e_lines(model, e, number):
    """Last section of summary_<model>.md (spec B3.6): 12 in summary_lear.md, 13 in summary_xgb.md."""
    M = MODEL_LABEL[model]
    L = [f"## {number}. {SEC5E}", "",
         "Diagnostic layers, defined after the 24-group result of step 5c and run with its workspace values known "
         "(handover 8.1 rows Sym-block, Loc-block, Loc-slot, step 5e specification). Diagnostics, not "
         f"tests. Files conformal/<setting>_{model}_{{symblock,locblock,locslot}}.parquet, evaluation period as above. "
         "`symblock` = ACI-CQR with the 24 groups of aciblock and a symmetric score (width only); `locblock` / "
         "`locslot` = recentering by the rolling 28-day median of y - q50 per block group (24) / quarter-hour slot "
         f"(96), then the four-group ACI-CQR (location only). `aci` and `aciblock` repeated beside them. {M}.", "",
         "Metrics per setting and regime (CRPS, MCB, DSC, UNC on the CRPS scale; PICP and width at the issued "
         "bounds):", "",
         md_table(e["per_setting"], {"PICP80": ".2f", "PICP60": ".2f"}), "",
         "Diebold-Mariano on the daily CRPS difference native - interp (Newey-West, Bartlett lag 7, normal p):", "",
         md_table(e["dm"], {"p": ".4g"}), "",
         "Bootstrap 90% intervals of native - interp (the same 1,000 draws as every other regime, circular blocks of "
         "7 days, default_rng(2027)):", "",
         md_table(e["boot_show"]), "",
         "Shares W and L (handover Section 8 row \"Paper decisions of 10 Nov and step 5e scope\"): G = |CRPS gap "
         "native - interp| under aci (G4), aciblock (G24), symblock (GS), locblock (GL); W = (G4 - GS) / (G4 - G24), "
         "L = (G4 - GL) / (G4 - G24); recentering if W <= 1/3 and L >= 2/3 in both models, width if W >= 2/3 and "
         "L <= 1/3 in both, mixed otherwise. The other model is read from its metrics file (pending until its step "
         "5e rows exist).", "",
         md_table(e["wl"], {"G4": ".4f", "G24": ".4f", "GS": ".4f", "GL": ".4f", "W": ".3f", "L": ".3f"}), "",
         e["reading"], "",
         f"Residual profile (results/profile_{model}.csv): mean and unscaled MAD of the raw residual y - q50 per "
         "block group g = 4 (h // 4) + p over the evaluation period, interp and native:", "",
         md_table(e["profile"], {"mean interp": ".2f", "mad interp": ".2f", "mean native": ".2f",
                                 "mad native": ".2f"}), "",
         "Sym-block, two-sided Prop. 4.1: max over the 24 groups of |mean over the 337 days of err - 0.2k| (err of "
         "the formal two-sided set), per setting, next to the bound (max(0.2k, 1 - 0.2k) + gamma) / (gamma 337):", "",
         md_table(e["dev"], {"target": ".1f", "bound": ".5f", "hourly": ".5f", "interp": ".5f", "native": ".5f"}), "",
         "Shares (%) of quarter-days with a formal unbounded side, a formal empty side, a clipped index: at the 80% "
         "pair (k = 1) and over all four pairs:", "",
         md_table(e["shares"], {c: ".3f" for c in e["shares"].columns if c not in ("regime", "setting")}), "",
         "Kupiec per quarter-hour slot, rejections at 5% after Holm over the 96 slots, of 96 (descriptive only):", "",
         md_table(e["kupiec"]["kupiec_rej"]), "",
         "Christoffersen (cc) per quarter-hour slot, rejections at 5% after Holm over the 96 slots, of 96:", "",
         md_table(e["kupiec"]["cc_rej"]), "",
         "Macros written to results/numbers.tex by this run for step 5e (not listed in Section 9):", "",
         md_table(pd.DataFrame([{"macro": "\\" + a, "value": v} for a, v in sorted(e["macros"].items())])), "",
         "Wall time of the step 5e work (not included in the stage rows of Section 10):", "",
         md_table(pd.DataFrame([{"stage": a, "seconds": v} for a, v in e["walls"].items()]), {"seconds": ".1f"}), ""]
    return L


# ================================================================================ main
def git_head():
    """Short HEAD hash, with -dirty when tracked files under esann27/ differ from HEAD."""
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                           check=True).stdout.strip()
        d = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "--", "esann27"], cwd=ROOT,
                           capture_output=True, text=True, check=True).stdout.strip()
        return h + ("-dirty" if d else "")
    except Exception:
        return "unknown"


def run(model, n_boot=N_BOOT, n_jobs=8, fdir="forecasts", cdir="conformal", out="results", rdir="forecasts_retune"):
    t_all = time.perf_counter()
    walls = {}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    M = MODEL_LABEL[model]
    macros = {}

    # ---- (i) Proposition A, burn-in only, written before anything else --------------------------------------
    t0 = time.perf_counter()
    inp = burnin_inputs(model, fdir)
    grid = prop_a(inp)
    ev_dates = [data.day_str(d) for d in cf.eval_days()]
    pis_eval = native_share(ev_dates)
    pred_eval = prop_a_eval_levels(inp, pis_eval)
    pred = pd.concat([grid, pred_eval], ignore_index=True)
    pred.insert(0, "model", model)
    pred_all = upsert_csv(out / "pred_doseresponse.csv", pred, model)
    pooled = grid[grid["position"] == "pooled"].set_index("pi")
    gap_col = "gap_interp_minus_native" if PRED_GAP == "interp_minus_native" else "gap"
    pred_macros = {"predPicpZero": fmt(pooled.loc[0.0, "picp80"], 1),
                   "predPicpHalf": fmt(pooled.loc[0.5, "picp80"], 1),
                   "predCrpsGapHalf": fmt(pooled.loc[0.5, gap_col])}
    if model == "lear":                    # the stylized-model macros are calibrated on the LEAR hourly residuals
        macros.update(pred_macros)
        write_macros(out / "numbers.tex", pred_macros)
    walls["Proposition A"] = time.perf_counter() - t0
    print(f"Proposition A written ({walls['Proposition A']:.1f} s): s_H = {inp['s_H']:.4f}, "
          f"sigma = {np.round(inp['sigma'], 2)}, kappa = {np.round(inp['kappa'], 4)}, {pred_macros}", flush=True)

    # ---- (a)-(e) metrics per file -----------------------------------------------------------------------------
    t0 = time.perf_counter()
    frames = load_all(model, fdir, cdir)
    naive_mae = {s: metrics.summary(frames["naive", s, "raw"])["MAE"] for s in SETTINGS}
    mrows = []
    for (m, s, lay), f in frames.items():
        mrows.append({"model": m, "setting": s, "layer": lay,
                      **frame_metrics(f, naive_mae[s], with_corp=(m != "naive"))})
    met = pd.DataFrame(mrows)
    met.to_csv(out / f"metrics_{model}.csv", index=False)
    walls["metrics + CORP"] = time.perf_counter() - t0

    # ---- unbounded shares (Table 2) ------------------------------------------------------------------------
    unbounded = {}
    for s in SETTINGS:
        for reg in REGIMES:
            al = pd.read_parquet(Path(cdir) / f"{s}_{model}_{reg}_alpha.parquet")
            unbounded[s, reg] = cf.shares(al, 1)["unbounded"]

    # ---- (f) DM, (h) pi-regression, monthly ----------------------------------------------------------------
    t0 = time.perf_counter()
    daily = {(s, lay): daily_crps(frames[model, s, lay]) for s in SETTINGS for lay in LAYERS}
    days = daily["native", "raw"].index.to_numpy()
    if len(days) != cf.N_EVAL_DAYS:
        raise RuntimeError("expected 337 evaluation days")
    pi_d = native_share(days)
    dm, regs = {}, []
    for reg in ("raw", "aci"):
        d = (daily["native", reg] - daily["interp", reg]).to_numpy()
        r = dm_test(d)
        dm[model, reg] = r["p"]
        gap = -d                                                    # interp - native
        b, se, z, p = ols_nw(gap, np.column_stack([np.ones(len(gap)), pi_d]))
        regs.append({"model": model, "regime": reg, "dm_mean_native_minus_interp": r["mean"], "dm_se": r["se"],
                     "dm_stat": r["stat"], "dm_p": r["p"], "intercept": b[0], "intercept_se": se[0],
                     "slope": b[1], "slope_se": se[1], "slope_z": z[1], "slope_p": p[1], "T": len(gap)})
    reg_df = pd.DataFrame(regs)
    upsert_csv(out / "regression.csv", reg_df, model)

    month = pd.Series(days).str[:7].to_numpy()
    mon_rows = []
    for (s, lay), dc in daily.items():
        f = frames[model, s, lay]
        rc = pd.Series(row_crps(f)).groupby(f["date"].astype(str).str[:7].to_numpy()).mean()
        for mth in rc.index:
            sel = month == mth
            mon_rows.append({"model": model, "setting": s, "layer": lay, "month": mth, "n_days": int(sel.sum()),
                             "mean_pi": float(pi_d[sel].mean()),
                             "mid": str(pd.to_datetime(days[sel]).mean().date()), "CRPS": float(rc[mth])})
    mon = pd.DataFrame(mon_rows)
    mon_all = upsert_csv(out / "monthly.csv", mon, model)
    walls["DM + regression + monthly"] = time.perf_counter() - t0

    # ---- (g) bootstrap -------------------------------------------------------------------------------------
    t0 = time.perf_counter()
    D = boot_days(len(days), n_boot)                 # generated once, the step 5c draws reuse them
    bt = bootstrap(frames, model, n_boot, n_jobs, D=D)
    boot = boot_summary(bt, met, model)
    upsert_csv(out / "bootstrap.csv", boot, model)
    walls["bootstrap"] = time.perf_counter() - t0

    # ---- (e) Kupiec / Christoffersen ----------------------------------------------------------------------
    t0 = time.perf_counter()
    kp = []
    for s in SETTINGS:
        for lay in LAYERS:
            k = coverage_tests(frames[model, s, lay])
            k.insert(0, "layer", lay)
            k.insert(0, "setting", s)
            kp.append(k)
    kp = pd.concat(kp, ignore_index=True)
    kp.insert(0, "model", model)
    upsert_csv(out / "kupiec.csv", kp, model)
    walls["Kupiec + Christoffersen"] = time.perf_counter() - t0

    # ---- tables, macros, figures --------------------------------------------------------------------------
    t0 = time.perf_counter()
    met_all = pd.concat([pd.read_csv(out / f"metrics_{m}.csv", float_precision="round_trip")
                         for m in MODELS_TABLE1 if (out / f"metrics_{m}.csv").exists()], ignore_index=True)
    met_all = met_all[met_all["model"].isin(MODELS_TABLE1)]
    reg_all = pd.read_csv(out / "regression.csv", float_precision="round_trip")
    dm_all = {(r.model, r.regime): r.dm_p for r in reg_all.itertuples()}
    (out / "table1.tex").write_text(table1_tex(met_all, dm_all))
    t2 = table2_rows(met, unbounded, model)
    sfx = "" if model == "xgb" else f"_{model}"
    (out / f"table2{sfx}.tex").write_text(table2_tex(t2))

    macros.update(model_macros(model, met, t2, boot, reg_df, naive_mae))
    write_macros(out / "numbers.tex", macros)

    abl, abl_pred = load_ablation(out)
    figure1(mon_all, pred_all, out / "fig1.pdf", abl, abl_pred)
    figure2(frames, model, pred_eval, out / f"fig2{sfx}.pdf")
    if model == "xgb":                               # step 5e spec B5: panel (a) alone, fig2.pdf unchanged
        figure2a(frames, model, out / "fig2a.pdf")
    walls["tables + figures"] = time.perf_counter() - t0

    # ---- step 5c, ACI-block (timed on its own, Section 11) ----------------------------------------------------
    block = None
    if block_files_exist(model, cdir):
        block = block_stage(model, frames, met, boot, kp, reg_df, D, n_jobs, naive_mae, cdir, out)
        write_macros(out / "numbers.tex", block["macros"])
    else:
        print(f"step 5c: conformal files of {REG_BLOCK} for {model} not all present, Section 11 not written", flush=True)

    # ---- step 5d, xgbm (XGB run only, timed on its own, Section 12) ----------------------------------------------
    rt5d = None
    if model == "xgb":
        if retune_files_exist(rdir):
            rt5d = retune_stage(frames, met, boot, reg_df, block, D, n_jobs, naive_mae, rdir, cdir, out, mon_all)
            write_macros(out / "numbers.tex", rt5d["macros"])
        else:
            print(f"step 5d: {rdir}/<setting>_{RETUNE}.parquet not both present, Section 12 not written", flush=True)

    # ---- step 5e, diagnostic layers (timed on its own, last section) -------------------------------------------
    diag = None
    if block is not None and diag_files_exist(model, cdir):
        diag = diag_stage(model, frames, met, boot, kp, reg_df, block, D, n_jobs, naive_mae, cdir, out)
        write_macros(out / "numbers.tex", diag["macros"])
    else:
        print(f"step 5e: conformal files of {', '.join(DIAG)} (or of {REG_BLOCK}) for {model} not all present, "
              f"step 5e section not written", flush=True)
    walls["total"] = time.perf_counter() - t_all

    write_summary(out, model, inp, grid, pred_eval, met, reg_df, boot, mon, kp, t2, unbounded, macros,
                  (out / "table1.tex").read_text(), (out / f"table2{sfx}.tex").read_text(), walls, n_boot, sfx,
                  block, rt5d, diag)
    for k, v in walls.items():
        print(f"  {k:28s} {v:8.1f} s")
    if block is not None:
        for k, v in block["walls"].items():
            print(f"  {REG_BLOCK}: {k:28s} {v:8.1f} s")
        print(f"  {block['rule']['line']}", flush=True)
    if rt5d is not None:
        for k, v in rt5d["walls"].items():
            print(f"  {RETUNE}: {k:28s} {v:8.1f} s")
        print(f"  {rt5d['rule']['line']}", flush=True)
    if diag is not None:
        for k, v in diag["walls"].items():
            print(f"  step 5e: {k:28s} {v:8.1f} s")
        print(f"  {diag['reading']}", flush=True)
    return {"metrics": met, "regression": reg_df, "bootstrap": boot, "macros": macros, "walls": walls,
            "block": block, "retune": rt5d, "diag": diag}


def write_summary(out, model, inp, grid, pred_eval, met, reg_df, boot, mon, kp, t2, unbounded, macros,
                  t1tex, t2tex, walls, n_boot, sfx, block=None, retune=None, diag=None):
    M = MODEL_LABEL[model]
    L = [f"# Step 5.2 evaluation summary: {M}", "",
         f"`python -m esann27.evaluate --model {model}`, git {git_head()}, run {pd.Timestamp.now():%Y-%m-%d %H:%M}. "
         f"Evaluation period {EVAL_START} .. {EVAL_END} (337 days, 32,348 quarters), target = native price. "
         f"CRPS = 2 x mean pinball over 9 levels. MCB / DSC / UNC on the CRPS scale (2 x mean over levels). "
         f"Newey-West: Bartlett, lag 7, no small-sample correction, normal p. Bootstrap: circular blocks of 7 days, "
         f"49 blocks truncated to 337 days, {n_boot} draws, default_rng({SEED}), the same draws for both settings "
         f"and both regimes.", ""]

    L += ["## 1. Proposition A (burn-in 1-28 Oct 2025 only, computed and written before any evaluation metric)", ""]
    ip = pd.DataFrame({"position": [":00", ":15", ":30", ":45"], "sigma_q native": inp["sigma"],
                       "interp sd": inp["sigma_interp"], "kappa_q": inp["kappa"],
                       "s_I": np.sqrt(inp["s_H"] ** 2 + inp["kappa"] ** 2 * inp["sigma"] ** 2),
                       "s_N": np.sqrt(inp["s_H"] ** 2 + inp["sigma"] ** 2)})
    L += [md_table(ip), "",
          f"s_H = {inp['s_H']:.4f} EUR/MWh (std, ddof 1, of the {inp['n_resid']} hourly residuals y - yhat of the "
          f"hourly {M} point forecast against the hourly mean native price; residual mean {inp['resid_mean']:.4f}). "
          f"F = ECDF of the residuals / s_H (not centered). Per-hour s_H (information only, not used): min "
          f"{inp['s_H_by_hour'].min():.2f} (h{int(inp['s_H_by_hour'].idxmin())}), max {inp['s_H_by_hour'].max():.2f} "
          f"(h{int(inp['s_H_by_hour'].idxmax())}).", ""]
    sel = grid[grid["pi"].isin([0.0, 0.5, 1.0])][["pi", "position", "picp80", "crps", "gap", "gap_interp_minus_native"]]
    L += ["Predicted raw PICP-80 (%) and CRPS of the native setting at pi = 0, 0.5, 1. gap = CRPS(pi) - CRPS(1) "
          "(item (i)), gap_interp_minus_native = CRPS(0) - CRPS(pi) (used for Fig. 1 and \\predCrpsGapHalf).", "",
          md_table(sel.sort_values(["position", "pi"])), "",
          "Predicted raw PICP-80 by position over the evaluation days (Fig. 2(b) dashes): interp at pi = 0, native "
          "= mean over the 337 days of the prediction at pi_D.", "",
          md_table(pred_eval[["kind", "position", "pi", "picp80", "crps"]]), ""]

    L += ["## 2. Metrics per file (evaluation period)", ""]
    show = met.copy()
    show["setting"] = show["setting"].map(SET_LABEL)
    L += [md_table(show[["model", "setting", "layer", "MAE", "RMSE", "rMAE", "CRPS", "MCB", "DSC", "UNC", "PICP80",
                         "width80", "PICP60", "width60"]], {"PICP80": ".2f", "PICP60": ".2f"}), "",
          "Check CRPS = MCB - DSC + UNC: max abs residual "
          f"{np.nanmax(np.abs(met['CRPS'] - (met['MCB'] - met['DSC'] + met['UNC']))):.2e}.", ""]

    L += ["## 3. Table 1 (results/table1.tex)", "", "```latex", t1tex.rstrip(), "```", "",
          f"## 4. Table 2 (results/table2{sfx}.tex), interpolated {M}", "", md_table(t2), "",
          "```latex", t2tex.rstrip(), "```", "",
          "Unbounded = share (%) of quarter-days with a formal unbounded side at the 80% pair. All regimes:", "",
          md_table(pd.DataFrame([{"setting": s, "regime": r, "unbounded_pct": 100 * unbounded[s, r]}
                                 for s in SETTINGS for r in REGIMES]), {"unbounded_pct": ".3f"}), ""]

    L += ["## 5. Diebold-Mariano (daily CRPS native - interp) and pi-regression (daily gap interp - native on pi_D)",
          "", md_table(reg_df.drop(columns=["model"]), {"dm_p": ".4g", "slope_p": ".4g"}), ""]
    L += ["## 6. Bootstrap 90% intervals of native - interp", "",
          md_table(boot.drop(columns=["model", "block", "seed"])), ""]

    mw = mon.pivot_table(index=["month", "n_days", "mean_pi"], columns=["setting", "layer"], values="CRPS")
    mw.columns = [f"{s}_{lay}" for s, lay in mw.columns]
    mw = mw.reset_index()
    for lay in ("raw", "aci"):
        mw[f"gap_{lay} (interp-native)"] = mw[f"interp_{lay}"] - mw[f"native_{lay}"]
    L += ["## 7. Monthly CRPS (results/monthly.csv)", "", md_table(mw, {"mean_pi": ".3f"}), ""]

    k = kp.copy()
    rej = coverage_rejections(k).drop(columns=["model"])
    L += ["## 8. Kupiec and Christoffersen per quarter-hour slot across days (supplement), Holm over the 96 slots "
          "per family, level and file", "",
          f"Rejections at 5% after Holm, all files: Kupiec {int((k['p_uc_holm'] < 0.05).sum())} of {len(k)}, "
          f"Christoffersen (cc) {int((k['p_cc_holm'] < 0.05).sum())} of {len(k)}. Per file and level (of 96 slots; "
          "slot detail in results/kupiec.csv):", "", md_table(rej), ""]

    abl, _ = load_ablation(out)
    if abl is not None:
        L += ["## 8b. Share ablation (step 5b, results/ablation.csv, written by esann27.ablation)", "",
              "Native LEAR, rule (iii), on the ablation days; gap_crps = CRPS(level) - CRPS(regular), DM of the level "
              "against the regular run on the daily CRPS (Newey-West lag 7), pred_gap = Proposition A (LEAR burn-in "
              "inputs, pooled) CRPS_A(pi) - CRPS_A(pi_bar).", "",
              md_table(abl[["level", "pi", "m", "n_quarters", "CRPS", "MCB", "DSC", "UNC", "PICP80", "width80",
                            "gap_crps", "gap_dsc_regular_minus", "dm_p", "pred_gap"]],
                       {"PICP80": ".2f", "dm_p": ".4g"}), ""]

    L += ["## 9. Macros written to results/numbers.tex by this run", "",
          md_table(pd.DataFrame([{"macro": "\\" + a, "value": b} for a, b in sorted(macros.items())])), ""]
    L += ["## 10. Wall time", "", md_table(pd.DataFrame([{"stage": a, "seconds": b} for a, b in walls.items()]),
                                           {"seconds": ".1f"}), ""]
    if block is not None:
        L += section11_lines(model, block)
    if retune is not None:
        L += section12_lines(retune)
    if diag is not None:
        L += section5e_lines(model, diag, 12 + (retune is not None))
    text = "\n".join(L) + "\n"
    (out / f"summary_{model}.md").write_text(text)      # kept per model
    (out / "summary.md").write_text(text)               # latest run


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m esann27.evaluate")
    p.add_argument("--model", required=True, choices=list(MODEL_LABEL))
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    p.add_argument("--n-jobs", type=int, default=8)
    p.add_argument("--fdir", default="forecasts")
    p.add_argument("--cdir", default="conformal")
    p.add_argument("--out", default="results")
    p.add_argument("--rdir", default="forecasts_retune", help="step 5d: folder of the xgbm forecasts (XGB run only)")
    a = p.parse_args(argv)
    run(a.model, a.n_boot, a.n_jobs, a.fdir, a.cdir, a.out, a.rdir)


if __name__ == "__main__":
    main()
