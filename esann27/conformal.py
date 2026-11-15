"""Conformal layer on the base quantiles (handover Section 8.1, step 5.1 of Section 9).

  python -m esann27.conformal --setting native --model lear [--regime aci | all]

Input: forecasts/<setting>_<model>.parquet (standard schema, rows sorted per row, test year; LEAR = rule (iii)).
Target y = the `actual` column (native price of D in every setting).

Scores (asymmetric CQR, Romano et al. 2019 Thm 2), pair k = 1..4 with levels (0.1k, 1 - 0.1k), alpha_k = 0.2k:
  E_lo = q_{0.1k} - y,  E_hi = y - q_{1-0.1k}, from the uncalibrated quantiles, one score per quarter row.
Groups (group_key): quarter position 0..3 ("position", Mondrian, 24 scores per group-day, 23 on 29 Mar 2026),
one group ("pooled", label -1), or 24 groups ("block", step 5c, handover 8.1 row ACI-block): g = 4 (h // 4) + p,
h = clock hour of datetime_local, p = minute // 15, 4 scores per group-day (3 for g = 0..3 on 29 Mar 2026).
Calibration set for target day D and group g: scores of D-28..D-1 (rolling) or of the burn-in
1-28 Oct 2025 (frozen); n = number of scores in it.
Per side with level a: i = ceil((1 - a)(n + 1) - 1e-9) on the ascending scores (1-based).
  formal side: unbounded if i > n, empty if i < 1, otherwise bounded with offset = S_(i).
  issued side: offset = S_(clip(i, 1, n)), used for every output quantile.
Lower bound = q_lo - offset_lo, upper bound = q_hi + offset_hi, q50 unchanged, then the 9 levels sorted per row.
err of side s, pair k, group g, day t = share of the group's quarters of day t with y < formal lower (lo) or
y > formal upper (hi), unsorted formal bounds; unbounded side -> 0, empty side -> 1. The issued offset never
enters err or the level update.
ACI (Gibbs and Candes 2021) per (group, pair, side): a_{t+1} = a_t + gamma (alpha_k/2 - err_t), gamma = 0.05,
a = alpha_k/2 on the first evaluation day 29 Oct 2025, so the level for day D includes the err of D-1.
Fixed regimes keep a = alpha_k/2.

Regimes: split (frozen, fixed), rolling (rolling, fixed), acifrozen (frozen, ACI), aci (rolling, ACI),
acipooled (rolling, ACI, one group): REGIMES, the five of step 5.1, run by --regime all.
aciblock (rolling, ACI, 24 block groups, step 5c): EXTRA_REGIMES, run only by name (--regime aciblock), so that
--regime all, the 5.1 tests and every existing evaluate path keep the five regimes (decided 9 Nov).

Outputs, evaluation days 29 Oct 2025 - 30 Sep 2026 only (337 days):
  conformal/<setting>_<model>_<regime>.parquet        standard schema, 32,348 rows
  conformal/<setting>_<model>_<regime>_alpha.parquet  one row per (date, group, k, side): a, n, i, formal
      (bounded / unbounded / empty), clipped, offset (issued), err (formal), plus nq (quarters of the group on
      that day) and win_first / win_last (first and last calibration day) for the window checks.

Step 5e (handover 8.1 rows Sym-block, Loc-block, Loc-slot; step 5e specification B1): DIAG_REGIMES,
diagnostic layers defined after the 24-group result, run only by name like aciblock.
  symblock: symmetric score S_k = max(q_{0.1k} - y, y - q_{1-0.1k}) per quarter, 24 block groups, rolling K = 28,
      one level a per (group, pair) starting at alpha_k = 0.2k, i = ceil((1 - a)(n + 1) - 1e-9), formal set
      unbounded if i > n (err 0), empty if i < 1 (err 1), issued offset S_(clip(i, 1, n)) subtracted from the lower
      and added to the upper quantile of the pair, err = share of the group's quarters of the day outside the
      formal [lower, upper] (two-sided), a <- a + gamma (alpha_k - err); median passes through, rows sorted.
      Alpha log as above with side = "sym".
  locblock, locslot: recentering, then the aci path above unchanged. shift_{D,g} = numpy.median of the base
      residual y - q50 (unshifted q50 of the forecast file) over the rows of group g with date in D-28..D-1, 0 where
      that window is empty (1 Oct 2025); all nine quantiles of every row of D in g shifted by it, point and actual
      unchanged. Key "block" (24 groups) or "slot" (96 groups, label 4h + p). The aci regime (position key) then
      runs on the recentered frame, its scores formed from the recentered quantiles. Extra output
      conformal/<setting>_<model>_<regime>_shift.parquet, one row per test-year quarter (35,036): datetime_local,
      date, group, n_shift (residual rows in the window), shift.
"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import data, metrics

K_WIN = 28
GAMMA = 0.05
EPS = 1e-9
BURN_START, BURN_END = "2025-10-01", "2025-10-28"          # inclusive
EVAL_START, EVAL_END = "2025-10-29", "2026-09-30"          # inclusive, 337 days
N_EVAL_DAYS = 337
N_EVAL_ROWS = 32348
POOLED = -1                                               # group label of the one-group regime

# pair k: (lower column, upper column, alpha_k)
PAIRS = {k: (f"q{10 * k}", f"q{100 - 10 * k}", round(0.2 * k, 1)) for k in (1, 2, 3, 4)}
SIDES = ("lo", "hi")
REGIMES = {                                               # window, adaptive level, group scheme
    "split": ("frozen", False, "position"),
    "rolling": ("rolling", False, "position"),
    "acifrozen": ("frozen", True, "position"),
    "aci": ("rolling", True, "position"),
    "acipooled": ("rolling", True, "pooled"),
}
EXTRA_REGIMES = {"aciblock": ("rolling", True, "block")}   # step 5c, run by name only
ALL_REGIMES = {**REGIMES, **EXTRA_REGIMES}
DIAG_REGIMES = {                                          # step 5e, run by name only: (path, group key)
    "symblock": ("sym", "block"),                         # symmetric score, 24 groups, adaptive level
    "locblock": ("loc", "block"),                         # recentering by 24 groups, then aci
    "locslot": ("loc", "slot"),                           # recentering by 96 slots, then aci
}
N_GROUPS = {"position": 4, "pooled": 1, "block": 24, "slot": 96}
STD_COLS = ["datetime_local", "date", "hour", "quarter", "actual", "point"] + data.QCOLS
ALPHA_COLS = ["date", "group", "k", "side", "a", "n", "i", "formal", "clipped", "offset", "err", "nq",
              "win_first", "win_last"]
SHIFT_COLS = ["datetime_local", "date", "group", "n_shift", "shift"]
SYM = "sym"                                               # side label of the symmetric path in the alpha log


def order_index(a, n):
    """1-based order-statistic index ceil((1 - a)(n + 1) - 1e-9)."""
    return int(math.ceil((1.0 - a) * (n + 1) - EPS))


def formal_state(i, n):
    return "unbounded" if i > n else ("empty" if i < 1 else "bounded")


def clip_index(i, n):
    return min(max(i, 1), n)


def calib_days(dn, window):
    """Day numbers of the calibration set for target day dn."""
    if window == "frozen":
        return list(range(data.day_number(BURN_START), data.day_number(BURN_END) + 1))
    if window == "rolling":
        return list(range(dn - K_WIN, dn))
    raise ValueError(window)


def eval_days():
    return list(range(data.day_number(EVAL_START), data.day_number(EVAL_END) + 1))


def _check_input(fc):
    miss = [c for c in STD_COLS if c not in fc.columns]
    if miss:
        raise ValueError(f"forecast frame lacks {miss}")
    q = fc[data.QCOLS].to_numpy(float)
    if not np.isfinite(q).all() or not np.isfinite(fc["actual"].to_numpy(float)).all():
        raise ValueError("non-finite quantiles or actuals")
    per_day = fc.groupby("date").size()
    burn = [data.day_str(d) for d in calib_days(None, "frozen")]
    ev = [data.day_str(d) for d in eval_days()]
    if not set(burn + ev) <= set(per_day.index):
        raise ValueError("forecast frame does not cover the burn-in and the evaluation period")
    if (per_day.loc[burn] != 96).any():
        raise ValueError("burn-in days must have 96 quarters each")


def group_key(df, scheme):
    """Group label of every row of df: "position" = the quarter column (0..3), "pooled" = -1, "block" =
    4 (h // 4) + p with h = datetime_local hour (0..23) and p = datetime_local minute // 15 (0..3, 0 = HH:00),
    labels 0..23 (handover 8.1 row ACI-block), "slot" = 4h + p, labels 0..95 (8.1 row Loc-slot, step 5e), with
    the same row-by-row assertions as "block"."""
    if scheme == "position":
        return df["quarter"].to_numpy(int)
    if scheme == "pooled":
        return np.full(len(df), POOLED)
    if scheme in ("block", "slot"):
        t = pd.to_datetime(df["datetime_local"])
        h = t.dt.hour.to_numpy(int)
        p = (t.dt.minute // 15).to_numpy(int)
        if not np.array_equal(h, df["hour"].to_numpy(int)):
            raise AssertionError("hour column differs from the clock hour of datetime_local")
        if not (t.dt.minute % 15 == 0).all():
            raise AssertionError("datetime_local not on a quarter-hour")
        q = df["quarter"].to_numpy(int)
        if not (np.array_equal(p, q) or np.array_equal(p, q - 1)):
            raise AssertionError("quarter position of datetime_local does not map one to one onto the quarter column")
        if scheme == "block":
            g = 4 * (h // 4) + p
            if g.min() < 0 or g.max() > 23:
                raise AssertionError("block group label outside 0..23")
        else:
            g = 4 * h + p
            if g.min() < 0 or g.max() > 95:
                raise AssertionError("slot label outside 0..95")
        return g
    raise ValueError(f"unknown group scheme {scheme!r}")


def calibrate(fc, regime, gamma=GAMMA, issued_index=clip_index, scheme=None):
    """Run one regime on a forecast frame in the standard schema.

    issued_index(i, n) -> index used for the issued offset (default: clip to [1, n]); exposed so that a test can
    show that it never enters the level update. scheme overrides the regime's group scheme (the reduction test:
    aciblock with "position" is aci). Returns (issued forecasts of the evaluation days, alpha frame)."""
    window, adaptive, grouping = ALL_REGIMES[regime]
    grouping = grouping if scheme is None else scheme
    _check_input(fc)
    fc = fc.sort_values("datetime_local").reset_index(drop=True)
    dn = pd.to_datetime(fc["date"]).values.astype("datetime64[D]").astype(np.int64)
    y = fc["actual"].to_numpy(float)
    Q = fc[data.QCOLS].to_numpy(float)
    col = {c: j for j, c in enumerate(data.QCOLS)}
    g_all = group_key(fc, grouping)
    groups = sorted(set(g_all.tolist()))

    # scores per row and pair (columns k-1)
    QLO = np.column_stack([Q[:, col[PAIRS[k][0]]] for k in PAIRS])
    QHI = np.column_stack([Q[:, col[PAIRS[k][1]]] for k in PAIRS])
    SC = {"lo": QLO - y[:, None], "hi": y[:, None] - QHI}

    key = pd.Series(np.arange(len(fc))).groupby([dn, g_all]).indices          # (day, group) -> row indices
    rows_of = lambda d, g: key.get((d, g), np.empty(0, dtype=np.int64))

    lower = np.full((len(fc), len(PAIRS)), np.nan)
    upper = np.full((len(fc), len(PAIRS)), np.nan)
    recs = []
    for g in groups:
        a = {(k, s): PAIRS[k][2] / 2 for k in PAIRS for s in SIDES}
        frozen_sorted = None
        for d in eval_days():
            cal = calib_days(d, window)
            if window == "frozen" and frozen_sorted is not None:
                srt, n = frozen_sorted
            else:
                idx = np.concatenate([rows_of(c, g) for c in cal])
                srt = {s: np.sort(SC[s][idx], axis=0) for s in SIDES}
                n = len(idx)
                if window == "frozen":
                    frozen_sorted = (srt, n)
            r = rows_of(d, g)
            yr = y[r]
            for k in PAIRS:
                j = k - 1
                target = PAIRS[k][2] / 2
                for s in SIDES:
                    at = a[k, s]
                    i = order_index(at, n)
                    formal = formal_state(i, n)
                    # formal side -> err (the issued offset is not used here)
                    if formal == "unbounded":
                        err = 0.0
                    elif formal == "empty":
                        err = 1.0
                    elif s == "lo":
                        err = float(np.mean(yr < QLO[r, j] - srt["lo"][i - 1, j]))
                    else:
                        err = float(np.mean(yr > QHI[r, j] + srt["hi"][i - 1, j]))
                    # issued side -> output quantiles
                    off = float(srt[s][issued_index(i, n) - 1, j])
                    if s == "lo":
                        lower[r, j] = QLO[r, j] - off
                    else:
                        upper[r, j] = QHI[r, j] + off
                    recs.append((data.day_str(d), g, k, s, at, n, i, formal, formal != "bounded", off, err,
                                 len(r), data.day_str(cal[0]), data.day_str(cal[-1])))
                    if adaptive:
                        a[k, s] = at + gamma * (target - err)

    ev = (dn >= data.day_number(EVAL_START)) & (dn <= data.day_number(EVAL_END))
    out_q = np.empty((int(ev.sum()), len(data.QCOLS)))
    for k in PAIRS:
        out_q[:, col[PAIRS[k][0]]] = lower[ev, k - 1]
        out_q[:, col[PAIRS[k][1]]] = upper[ev, k - 1]
    out_q[:, col["q50"]] = Q[ev, col["q50"]]
    if not np.isfinite(out_q).all():
        raise RuntimeError("evaluation quarters without an issued bound")
    out = fc.loc[ev, STD_COLS[:6]].reset_index(drop=True)
    out[data.QCOLS] = np.sort(out_q, axis=1)
    alpha = pd.DataFrame.from_records(recs, columns=ALPHA_COLS)
    alpha = alpha.sort_values(["date", "group", "k", "side"], kind="stable").reset_index(drop=True)
    return out[STD_COLS], alpha


# ================================================================================ step 5e: diagnostic layers
def pair_scores(fc):
    """Per row and pair k (columns k - 1): the lower and upper base quantiles of the pair, the one-sided scores
    E_lo = q_lo - y, E_hi = y - q_hi and the symmetric score S = max(E_lo, E_hi)."""
    y = fc["actual"].to_numpy(float)
    QLO = np.column_stack([fc[PAIRS[k][0]].to_numpy(float) for k in PAIRS])
    QHI = np.column_stack([fc[PAIRS[k][1]].to_numpy(float) for k in PAIRS])
    e_lo, e_hi = QLO - y[:, None], y[:, None] - QHI
    return QLO, QHI, e_lo, e_hi, np.maximum(e_lo, e_hi)


def calibrate_sym(fc, gamma=GAMMA, issued_index=clip_index, scheme="block", with_bounds=False):
    """Regime symblock (handover 8.1 row Sym-block): rolling K = 28, adaptive level, symmetric score, one level and
    one offset per (group, pair, day) moving both bounds of the pair alike. Returns (issued forecasts of the
    evaluation days, alpha log) and, with with_bounds, the unsorted issued bounds of the evaluation rows
    {"lower", "upper"} (columns k - 1) in the row order of the output."""
    _check_input(fc)
    fc = fc.sort_values("datetime_local").reset_index(drop=True)
    dn = pd.to_datetime(fc["date"]).values.astype("datetime64[D]").astype(np.int64)
    y = fc["actual"].to_numpy(float)
    Q = fc[data.QCOLS].to_numpy(float)
    col = {c: j for j, c in enumerate(data.QCOLS)}
    g_all = group_key(fc, scheme)
    groups = sorted(set(g_all.tolist()))
    QLO, QHI, _, _, SC = pair_scores(fc)

    key = pd.Series(np.arange(len(fc))).groupby([dn, g_all]).indices
    rows_of = lambda d, g: key.get((d, g), np.empty(0, dtype=np.int64))

    lower = np.full((len(fc), len(PAIRS)), np.nan)
    upper = np.full((len(fc), len(PAIRS)), np.nan)
    recs = []
    for g in groups:
        a = {k: PAIRS[k][2] for k in PAIRS}
        for d in eval_days():
            cal = calib_days(d, "rolling")
            idx = np.concatenate([rows_of(c, g) for c in cal])
            srt = np.sort(SC[idx], axis=0)
            n = len(idx)
            r = rows_of(d, g)
            yr = y[r]
            for k in PAIRS:
                j = k - 1
                target = PAIRS[k][2]
                at = a[k]
                i = order_index(at, n)
                formal = formal_state(i, n)
                # formal set -> err (two-sided; the issued offset is not used here)
                if formal == "unbounded":
                    err = 0.0
                elif formal == "empty":
                    err = 1.0
                else:
                    o = srt[i - 1, j]
                    err = float(np.mean((yr < QLO[r, j] - o) | (yr > QHI[r, j] + o)))
                # issued offset -> both bounds of the pair
                off = float(srt[issued_index(i, n) - 1, j])
                lower[r, j] = QLO[r, j] - off
                upper[r, j] = QHI[r, j] + off
                recs.append((data.day_str(d), g, k, SYM, at, n, i, formal, formal != "bounded", off, err,
                             len(r), data.day_str(cal[0]), data.day_str(cal[-1])))
                a[k] = at + gamma * (target - err)

    ev = (dn >= data.day_number(EVAL_START)) & (dn <= data.day_number(EVAL_END))
    out_q = np.empty((int(ev.sum()), len(data.QCOLS)))
    for k in PAIRS:
        out_q[:, col[PAIRS[k][0]]] = lower[ev, k - 1]
        out_q[:, col[PAIRS[k][1]]] = upper[ev, k - 1]
    out_q[:, col["q50"]] = Q[ev, col["q50"]]
    if not np.isfinite(out_q).all():
        raise RuntimeError("evaluation quarters without an issued bound")
    out = fc.loc[ev, STD_COLS[:6]].reset_index(drop=True)
    out[data.QCOLS] = np.sort(out_q, axis=1)
    alpha = pd.DataFrame.from_records(recs, columns=ALPHA_COLS)
    alpha = alpha.sort_values(["date", "group", "k", "side"], kind="stable").reset_index(drop=True)
    if with_bounds:
        return out[STD_COLS], alpha, {"lower": lower[ev], "upper": upper[ev]}
    return out[STD_COLS], alpha


def recenter(fc, scheme, zero=False):
    """Recentering step of Loc-block / Loc-slot (handover 8.1): returns (the frame with all nine quantile columns
    shifted per row by shift_{D,g}, sorted by datetime_local; the shift frame, SHIFT_COLS). shift_{D,g} =
    numpy.median of y - q50 (base q50) over the rows of group g with date in D-28..D-1, 0 where the window is
    empty. zero=True forces the shift to 0 on the same code path (reduction test)."""
    _check_input(fc)
    fc = fc.sort_values("datetime_local").reset_index(drop=True)
    dn = pd.to_datetime(fc["date"]).values.astype("datetime64[D]").astype(np.int64)
    g_all = group_key(fc, scheme)
    resid = fc["actual"].to_numpy(float) - fc["q50"].to_numpy(float)
    key = pd.Series(np.arange(len(fc))).groupby([dn, g_all]).indices
    shift = np.zeros(len(fc))
    n_shift = np.zeros(len(fc), dtype=np.int64)
    for (d, g), rows in key.items():
        win = [key[(c, g)] for c in range(d - K_WIN, d) if (c, g) in key]
        if win:
            w = np.concatenate(win)
            n_shift[rows] = len(w)
            shift[rows] = np.median(resid[w])
    if zero:
        shift[:] = 0.0
    out = fc.copy()
    out[data.QCOLS] = fc[data.QCOLS].to_numpy(float) + shift[:, None]
    sf = pd.DataFrame({"datetime_local": fc["datetime_local"], "date": fc["date"], "group": g_all,
                       "n_shift": n_shift, "shift": shift})
    return out, sf


def run_diag(fc, regime, zero_shift=False):
    """One step 5e regime: (issued forecasts of the evaluation days, alpha log, shift frame or None)."""
    path, scheme = DIAG_REGIMES[regime]
    if path == "sym":
        out, alpha = calibrate_sym(fc, scheme=scheme)
        return out, alpha, None
    shifted, sf = recenter(fc, scheme, zero=zero_shift)
    out, alpha = calibrate(shifted, "aci")
    return out, alpha, sf


def regime_files(regime):
    """File suffixes written per (setting, model) for a regime: output, alpha, and the shift frame for loc*."""
    return ["", "_alpha"] + (["_shift"] if DIAG_REGIMES.get(regime, ("",))[0] == "loc" else [])


def sym_bound(k, T=N_EVAL_DAYS, gamma=GAMMA):
    """Two-sided bound of Gibbs and Candes 2021, Prop. 4.1, with target alpha_k = 0.2k (Sym-block)."""
    t = PAIRS[k][2]
    return (max(t, 1 - t) + gamma) / (T * gamma)


def sym_table(alpha, out, k=1, scheme="block"):
    """Per group of pair k (Sym-block): realized formal two-sided miscoverage (mean daily err, what the two-sided
    Prop. 4.1 bounds) and the issued two-sided miscoverage of the sorted output."""
    lo, hi, _ = PAIRS[k]
    gk = group_key(out, scheme)
    rows = []
    for g, ag in alpha[alpha["k"] == k].groupby("group"):
        sel = out[gk == g]
        y = sel["actual"].to_numpy(float)
        rows.append({"group": str(g), "err_sym": float(ag["err"].mean()),
                     "iss_sym": float(np.mean((y < sel[lo].to_numpy(float)) | (y > sel[hi].to_numpy(float))))})
    return pd.DataFrame(rows)


def aci_bound(k, T=N_EVAL_DAYS, gamma=GAMMA):
    """Per-side bound of Gibbs and Candes 2021, Prop. 4.1, with target alpha_k / 2."""
    t = PAIRS[k][2] / 2
    return (max(t, 1 - t) + gamma) / (T * gamma)


def side_table(alpha, out, k=1, scheme="position"):
    """Per group and side of pair k: realized formal miscoverage (mean daily err, what Prop. 4.1 bounds) and the
    issued miscoverage of the sorted output (share of quarters below q_lo / above q_hi). scheme = the group
    scheme of the regime that produced alpha."""
    lo, hi, _ = PAIRS[k]
    gk = group_key(out, scheme) if scheme != "pooled" else None
    rows = []
    for g, ag in alpha[alpha["k"] == k].groupby("group"):
        sel = out if g == POOLED else out[gk == g]
        y = sel["actual"].to_numpy(float)
        rec = {"group": "all" if g == POOLED else str(g)}
        for s in SIDES:
            rec[f"err_{s}"] = float(ag.loc[ag["side"] == s, "err"].mean())
        rec["iss_lo"] = float(np.mean(y < sel[lo].to_numpy(float)))
        rec["iss_hi"] = float(np.mean(y > sel[hi].to_numpy(float)))
        rows.append(rec)
    return pd.DataFrame(rows)


def shares(alpha, k=None):
    """Shares of quarter-days (weighted by nq) with a formal unbounded side, a formal empty side, a clipped index.
    k=None: over all pairs, i.e. the share of (quarter-day, pair) combinations."""
    a = alpha if k is None else alpha[alpha["k"] == k]
    w = a.pivot_table(index=["date", "group", "k"], columns="side", values="formal", aggfunc="first")
    nq = a.groupby(["date", "group", "k"])["nq"].first().reindex(w.index).to_numpy(float)
    f = lambda m: float(np.sum(nq * m) / np.sum(nq))
    unb = (w == "unbounded").any(axis=1).to_numpy()
    emp = (w == "empty").any(axis=1).to_numpy()
    clp = (w != "bounded").any(axis=1).to_numpy()
    return {"unbounded": f(unb), "empty": f(emp), "clipped": f(clp)}


def eval_summary(fc):
    s = metrics.summary(metrics.period(fc, EVAL_START, data.TEST_END))
    return {k: s[k] for k in ("n", "PICP80", "PICP60", "width80", "CRPS")}


def compare_files(new, old):
    """Regeneration check of step 5c (spec B2, B4.4): "bytes" if the two parquet files are byte-identical, "frame"
    if they are not but the frames are equal with the same columns, dtypes and float bit patterns (allowed by the
    spec only because a parquet writer may embed something run-specific), "" otherwise."""
    if Path(new).read_bytes() == Path(old).read_bytes():
        return "bytes"
    a, b = pd.read_parquet(new), pd.read_parquet(old)
    if list(a.columns) != list(b.columns) or len(a) != len(b) or not a.dtypes.equals(b.dtypes):
        return ""
    for c in a.columns:
        x, y = a[c].to_numpy(), b[c].to_numpy()
        if x.dtype.kind == "f":
            if not np.array_equal(x.view(np.int64), y.view(np.int64)):
                return ""
        elif not a[c].equals(b[c]):
            return ""
    return "frame"


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m esann27.conformal")
    p.add_argument("--setting", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--regime", default="all", choices=list(ALL_REGIMES) + list(DIAG_REGIMES) + ["all"],
                   help="all = the five regimes of step 5.1; aciblock (step 5c) and symblock, locblock, locslot "
                        "(step 5e) by name only")
    p.add_argument("--src", default="forecasts")
    p.add_argument("--out", default="conformal")
    a = p.parse_args(argv)
    fc = pd.read_parquet(Path(a.src) / f"{a.setting}_{a.model}.parquet")
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    regimes = list(REGIMES) if a.regime == "all" else [a.regime]
    rows = [{"regime": "raw", **eval_summary(fc)}]
    detail, detail_sym, shift_info = [], [], []
    for r in regimes:
        t0 = time.perf_counter()
        sf = None
        if r in DIAG_REGIMES:
            out, alpha, sf = run_diag(fc, r)
        else:
            out, alpha = calibrate(fc, r)
        wall = time.perf_counter() - t0
        if len(out) != N_EVAL_ROWS:
            raise RuntimeError(f"{r}: {len(out)} rows, expected {N_EVAL_ROWS}")
        stem = f"{a.setting}_{a.model}_{r}"
        out.to_parquet(out_dir / f"{stem}.parquet", index=False)
        alpha.to_parquet(out_dir / f"{stem}_alpha.parquet", index=False)
        if sf is not None:
            if len(sf) != len(fc):
                raise RuntimeError(f"{r}: shift frame of {len(sf)} rows, expected {len(fc)}")
            sf.to_parquet(out_dir / f"{stem}_shift.parquet", index=False)
            evr = (sf["date"] >= EVAL_START).to_numpy()
            shift_info.append({"regime": r, "mean_abs_shift_eval": float(np.abs(sf["shift"].to_numpy()[evr]).mean()),
                               "n_shift_min_eval": int(sf["n_shift"].to_numpy()[evr].min()),
                               "n_shift_max_eval": int(sf["n_shift"].to_numpy()[evr].max())})
        s1, sa = shares(alpha, 1), shares(alpha)
        rows.append({"regime": r, **eval_summary(out),
                     **{f"{x}_k1": s1[x] for x in s1}, **{f"{x}_all": sa[x] for x in sa},
                     "wall_s": wall, "ms_per_day": 1000 * wall / N_EVAL_DAYS})
        if r in DIAG_REGIMES and DIAG_REGIMES[r][0] == "sym":
            st = sym_table(alpha, out, 1, DIAG_REGIMES[r][1])
            st.insert(0, "regime", r)
            st["bound"] = sym_bound(1)
            detail_sym.append(st)
        elif r in DIAG_REGIMES or ALL_REGIMES[r][1]:
            st = side_table(alpha, out, 1, "position" if r in DIAG_REGIMES else ALL_REGIMES[r][2])
            st.insert(0, "regime", r)
            st["bound"] = aci_bound(1)
            detail.append(st)
        extra = " and _shift" if sf is not None else ""
        print(f"wrote {out_dir / stem}.parquet ({len(out)} rows), _alpha ({len(alpha)} rows){extra}, "
              f"{wall:.2f} s ({1000 * wall / N_EVAL_DAYS:.1f} ms per day)", flush=True)
    with pd.option_context("display.width", 250, "display.float_format", "{:.4f}".format,
                           "display.max_columns", 30):
        print(f"\n{a.setting} {a.model}, evaluation period {EVAL_START} .. {EVAL_END}")
        print(pd.DataFrame(rows).to_string(index=False))
        if detail:
            print("\npair k = 1, per-side miscoverage: formal (mean daily err) and issued (sorted output)")
            print(pd.concat(detail, ignore_index=True).to_string(index=False))
        if detail_sym:
            print("\npair k = 1, two-sided miscoverage (symmetric score): formal (mean daily err) and issued "
                  "(sorted output), two-sided bound")
            print(pd.concat(detail_sym, ignore_index=True).to_string(index=False))
        if shift_info:
            print("\nrecentering shift, evaluation rows")
            print(pd.DataFrame(shift_info).to_string(index=False))


if __name__ == "__main__":
    main()
