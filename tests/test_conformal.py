"""Conformal layer tests (handover Section 8.1 "Tests" row, step 5.1 of Section 9)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from esann27 import conformal as cf
from esann27 import data

ROOT = Path(__file__).resolve().parent.parent
SETTINGS = ("hourly", "interp", "native")
ADAPTIVE = [r for r, (_, ad, _) in cf.REGIMES.items() if ad]
FROZEN = [r for r, (w, _, _) in cf.REGIMES.items() if w == "frozen"]
ROLLING = [r for r, (w, _, _) in cf.REGIMES.items() if w == "rolling"]
DST_SPRING = "2026-03-29"


def _path(s):
    return ROOT / "forecasts" / f"{s}_lear.parquet"


@pytest.fixture(scope="session")
def lear():
    missing = [s for s in SETTINGS if not _path(s).exists()]
    if missing:
        pytest.skip(f"LEAR forecasts missing for {missing}")
    return {s: pd.read_parquet(_path(s)) for s in SETTINGS}


@pytest.fixture(scope="session")
def runs(lear):
    return {(s, r): cf.calibrate(lear[s], r) for s in SETTINGS for r in cf.REGIMES}


def _dn(datestr):
    return pd.to_datetime(pd.Series(datestr)).values.astype("datetime64[D]").astype(np.int64)


# ---------------------------------------------------------------- order statistic and formal / issued sides
def test_order_index_and_formal_state():
    n = 672
    assert cf.order_index(0.1, n) == 606 and cf.order_index(0.1, 671) == 605
    assert cf.order_index(0.0, n) == n + 1 and cf.formal_state(n + 1, n) == "unbounded"
    assert cf.formal_state(cf.order_index(-0.03, n), n) == "unbounded"
    # refinement of 6 Nov: 0 < a < 1/(n+1) is unbounded too
    assert cf.formal_state(cf.order_index(0.5 / (n + 1), n), n) == "unbounded"
    assert cf.formal_state(cf.order_index(1.5 / (n + 1), n), n) == "bounded"
    assert cf.order_index(1.0, n) == 0 and cf.formal_state(0, n) == "empty"
    assert cf.formal_state(cf.order_index(1.04, n), n) == "empty"
    assert cf.order_index(1 - 0.5 / (n + 1), n) == 1
    assert cf.clip_index(n + 1, n) == n and cf.clip_index(0, n) == 1 and cf.clip_index(5, n) == 5
    # exact integer products are not pushed up by float noise
    assert cf.order_index(0.5, 3) == 2 and cf.order_index(0.25, 3) == 3


def test_aci_bound_values():
    assert [round(cf.aci_bound(k), 3) for k in cf.PAIRS] == [0.056, 0.050, 0.045, 0.039]


# ---------------------------------------------------------------- schema
def test_output_schema(lear, runs):
    ev = [data.day_str(d) for d in cf.eval_days()]
    assert len(ev) == 337 and ev[0] == "2025-10-29" and ev[-1] == "2026-09-30"
    for (s, r), (out, alpha) in runs.items():
        assert list(out.columns) == cf.STD_COLS and len(out) == cf.N_EVAL_ROWS
        assert out["date"].unique().tolist() == ev
        raw = lear[s][lear[s]["date"] >= cf.EVAL_START].reset_index(drop=True)
        for c in ("datetime_local", "date", "hour", "quarter", "actual", "point"):
            assert out[c].equals(raw[c]), (s, r, c)
        ng = 1 if cf.REGIMES[r][2] == "pooled" else 4
        assert len(alpha) == 337 * ng * 4 * 2
        assert not alpha.duplicated(["date", "group", "k", "side"]).any()
        assert np.isfinite(out[data.QCOLS].to_numpy()).all()


# ---------------------------------------------------------------- ACI bound and recursion
@pytest.mark.parametrize("regime", ADAPTIVE)
def test_aci_per_side_bound(runs, regime):
    for s in SETTINGS:
        alpha = runs[s, regime][1]
        for (g, k, side), a in alpha.groupby(["group", "k", "side"]):
            assert len(a) == 337
            dev = abs(a["err"].mean() - cf.PAIRS[k][2] / 2)
            bound = (max(cf.PAIRS[k][2] / 2, 1 - cf.PAIRS[k][2] / 2) + 0.05) / (337 * 0.05)
            assert dev <= bound, (s, regime, g, k, side, dev, bound)


def test_alpha_paths_consistent(runs):
    """a follows the update on the formal err exactly, i / formal / clipped follow from (a, n), err is 0 on
    unbounded and 1 on empty sides; fixed regimes keep alpha_k / 2."""
    for (s, r), (_, alpha) in runs.items():
        for (g, k, side), a in alpha.groupby(["group", "k", "side"]):
            a = a.sort_values("date")
            t = cf.PAIRS[k][2] / 2
            av, ev = a["a"].to_numpy(), a["err"].to_numpy()
            assert av[0] == t
            if cf.REGIMES[r][1]:
                assert np.array_equal(av[1:], av[:-1] + cf.GAMMA * (t - ev[:-1])), (s, r, g, k, side)
            else:
                assert (av == t).all()
            i = np.array([cf.order_index(x, n) for x, n in zip(av, a["n"])])
            assert np.array_equal(i, a["i"].to_numpy())
            fs = [cf.formal_state(x, n) for x, n in zip(i, a["n"])]
            assert fs == a["formal"].tolist()
            assert np.array_equal(a["clipped"].to_numpy(), a["formal"].to_numpy() != "bounded")
            assert (ev[a["formal"].to_numpy() == "unbounded"] == 0).all()
            assert (ev[a["formal"].to_numpy() == "empty"] == 1).all()
            assert ((ev >= 0) & (ev <= 1)).all()


def test_issued_offset_never_enters_update(lear):
    """Swap the issued rule (clip -> always the smallest / largest score): a, i, formal and err are unchanged,
    issued offsets change. gamma = 2 drives the levels outside (0, 1), so both kinds of clipping occur."""
    fc = lear["interp"]
    for regime in ("aci", "acifrozen", "acipooled"):
        out0, a0 = cf.calibrate(fc, regime, gamma=2.0)
        assert a0["clipped"].any() and (a0["formal"] == "unbounded").any() and (a0["formal"] == "empty").any()
        for rule in (lambda i, n: 1, lambda i, n: n):
            out1, a1 = cf.calibrate(fc, regime, gamma=2.0, issued_index=rule)
            for c in ("a", "n", "i", "formal", "err"):
                assert a0[c].equals(a1[c]), (regime, c)
            assert not a0["offset"].equals(a1["offset"])
        # non-vacuous: on unbounded sides the issued (clipped) bound misses some quarters while err is 0
        fe = fc[fc["date"] >= cf.EVAL_START].reset_index(drop=True)
        key = pd.DataFrame({"date": fe["date"].to_numpy(),
                            "group": cf.POOLED if regime == "acipooled" else fe["quarter"].to_numpy()})
        hit = False
        for k in cf.PAIRS:
            sel = a0[(a0["k"] == k) & (a0["side"] == "hi")][["date", "group", "offset", "formal"]]
            m = key.merge(sel, on=["date", "group"], how="left", validate="many_to_one")
            unb = (m["formal"] == "unbounded").to_numpy()
            issued_upper = fe[cf.PAIRS[k][1]].to_numpy() + m["offset"].to_numpy()
            hit |= bool((fe["actual"].to_numpy()[unb] > issued_upper[unb]).any())
        assert hit


# ---------------------------------------------------------------- sorting and reconstruction
def _rebuild_unsorted(raw_eval, alpha, pooled):
    """Unsorted issued quantiles rebuilt from the base quantiles and the issued offsets of the alpha file."""
    g = np.full(len(raw_eval), cf.POOLED) if pooled else raw_eval["quarter"].to_numpy()
    Q = raw_eval[data.QCOLS].to_numpy(float).copy()
    col = {c: j for j, c in enumerate(data.QCOLS)}
    key = pd.DataFrame({"date": raw_eval["date"].to_numpy(), "group": g})
    for k, (lo, hi, _) in cf.PAIRS.items():
        for side, c, sign in (("lo", lo, -1.0), ("hi", hi, 1.0)):
            off = alpha[(alpha["k"] == k) & (alpha["side"] == side)][["date", "group", "offset"]]
            o = key.merge(off, on=["date", "group"], how="left", validate="many_to_one")["offset"].to_numpy()
            Q[:, col[c]] = raw_eval[c].to_numpy(float) + sign * o
    return Q


def test_sorting_idempotent_and_rebuild(lear, runs):
    for (s, r), (out, alpha) in runs.items():
        raw = lear[s][lear[s]["date"] >= cf.EVAL_START].reset_index(drop=True)
        Q = out[data.QCOLS].to_numpy()
        assert (np.diff(Q, axis=1) >= 0).all()
        assert np.array_equal(np.sort(Q, axis=1), Q)
        U = _rebuild_unsorted(raw, alpha, cf.REGIMES[r][2] == "pooled")
        assert np.array_equal(U[:, 4], raw["q50"].to_numpy())             # median passes through
        assert np.array_equal(np.sort(U, axis=1), Q), (s, r)
        assert np.array_equal(np.sort(np.sort(U, axis=1), axis=1), Q)


# ---------------------------------------------------------------- windows
def test_calib_days():
    D = data.day_number("2025-10-29")
    burn = cf.calib_days(D, "frozen")
    assert [data.day_str(burn[0]), data.day_str(burn[-1])] == ["2025-10-01", "2025-10-28"] and len(burn) == 28
    assert cf.calib_days(D, "rolling") == burn
    for d in cf.eval_days():
        r = cf.calib_days(d, "rolling")
        assert r == list(range(d - 28, d))
        assert cf.calib_days(d, "frozen") == burn


def test_window_dates_exact(runs):
    first = "2025-10-29"
    for (s, r), (_, alpha) in runs.items():
        f = alpha[alpha["date"] == first]
        assert (f["win_first"] == "2025-10-01").all() and (f["win_last"] == "2025-10-28").all()
        D, w0, w1 = _dn(alpha["date"]), _dn(alpha["win_first"]), _dn(alpha["win_last"])
        if r in ROLLING:
            assert (w0 == D - 28).all() and (w1 == D - 1).all(), (s, r)
        else:
            assert (alpha["win_first"] == "2025-10-01").all() and (alpha["win_last"] == "2025-10-28").all()
            assert (w1 <= data.day_number("2025-10-28")).all()


def _scores(df, k, side):
    lo, hi, _ = cf.PAIRS[k]
    return (df[lo] - df["actual"]) if side == "lo" else (df["actual"] - df[hi])


@pytest.mark.parametrize("setting", SETTINGS)
def test_offsets_independent_recompute(lear, runs, setting):
    """Offsets recomputed with plain pandas filters on the date strings of the forecast file."""
    fc = lear[setting]
    days = ["2025-10-29", "2025-11-26", "2026-03-29", "2026-03-30", "2026-04-26", "2026-04-27", "2026-09-30"]
    for r in cf.REGIMES:
        alpha = runs[setting, r][1].set_index(["date", "group", "k", "side"])
        frozen = cf.REGIMES[r][0] == "frozen"
        for D in days:
            Dt = pd.Timestamp(D)
            a0 = "2025-10-01" if frozen else (Dt - pd.Timedelta(days=28)).strftime("%Y-%m-%d")
            a1 = "2025-10-28" if frozen else (Dt - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            win = fc[(fc["date"] >= a0) & (fc["date"] <= a1)]
            for g in ([cf.POOLED] if cf.REGIMES[r][2] == "pooled" else [0, 1, 2, 3]):
                wg = win if g == cf.POOLED else win[win["quarter"] == g]
                for k in cf.PAIRS:
                    for side in cf.SIDES:
                        rec = alpha.loc[(D, g, k, side)]
                        srt = np.sort(_scores(wg, k, side).to_numpy())
                        assert rec["n"] == len(srt)
                        i = cf.order_index(rec["a"], len(srt))
                        assert rec["offset"] == srt[min(max(i, 1), len(srt)) - 1], (setting, r, D, g, k, side)


def test_frozen_windows_never_after_burnin(lear, runs):
    """split and acifrozen: every issued offset is a burn-in score of its group / pair / side, and the split
    output does not move when every actual after 28 Oct 2025 is changed (fixed level, frozen window)."""
    for s in SETTINGS:
        fc = lear[s]
        burn = fc[(fc["date"] >= cf.BURN_START) & (fc["date"] <= cf.BURN_END)]
        assert len(burn) == 28 * 96
        for r in FROZEN:
            alpha = runs[s, r][1]
            for (g, k, side), a in alpha.groupby(["group", "k", "side"]):
                pool = set(_scores(burn[burn["quarter"] == g], k, side).tolist())
                assert set(a["offset"].tolist()) <= pool, (s, r, g, k, side)
        pert = fc.copy()
        after = (pert["date"] > cf.BURN_END).to_numpy()
        pert.loc[after, "actual"] = pert.loc[after, "actual"] * 3.1 + 250.0
        out0, a0 = runs[s, "split"]
        out1, a1 = cf.calibrate(pert, "split")
        assert np.array_equal(out0[data.QCOLS].to_numpy(), out1[data.QCOLS].to_numpy())
        assert a0["offset"].equals(a1["offset"])


# ---------------------------------------------------------------- no look-ahead
@pytest.mark.parametrize("cut", ["2025-10-29", "2026-03-02", "2026-03-29"])
def test_no_lookahead(lear, runs, cut):
    """Actuals changed from day D on leave the issued quantiles of every day up to D unchanged."""
    for s in SETTINGS:
        fc = lear[s]
        pert = fc.copy()
        on = (pert["date"] >= cut).to_numpy()
        rng = np.random.default_rng(2027)
        pert.loc[on, "actual"] = pert.loc[on, "actual"] * 1.9 - 60.0 + 80.0 * rng.random(on.sum())
        for r in cf.REGIMES:
            out0, a0 = runs[s, r]
            out1, a1 = cf.calibrate(pert, r)
            upto = (out0["date"] <= cut).to_numpy()
            assert np.array_equal(out0.loc[upto, data.QCOLS].to_numpy(), out1.loc[upto, data.QCOLS].to_numpy()), \
                (s, r, cut)
            au = (a0["date"] <= cut).to_numpy()
            for c in ("a", "n", "i", "offset"):
                assert a0.loc[au, c].equals(a1.loc[au, c]), (s, r, cut, c)
            if r != "split":                    # sanity: the change reaches the following days
                assert not np.array_equal(out0.loc[~upto, data.QCOLS].to_numpy(),
                                          out1.loc[~upto, data.QCOLS].to_numpy())


# ---------------------------------------------------------------- window sizes around DST
def test_window_sizes_dst(runs):
    dst = data.day_number(DST_SPRING)
    for (s, r), (_, alpha) in runs.items():
        pooled = cf.REGIMES[r][2] == "pooled"
        full, short = (2688, 2684) if pooled else (672, 671)
        D = _dn(alpha["date"])
        if r in ROLLING:
            contains = (D - 28 <= dst) & (dst <= D - 1)
            assert contains.sum() > 0
            assert (alpha["n"].to_numpy()[contains] == short).all(), (s, r)
            assert (alpha["n"].to_numpy()[~contains] == full).all(), (s, r)
        else:
            assert (alpha["n"] == full).all()
        nq = alpha["nq"].to_numpy()
        is_dst = D == dst
        assert (nq[is_dst] == (92 if pooled else 23)).all() and (nq[~is_dst] == (96 if pooled else 24)).all()


# ---------------------------------------------------------------- coverage under exchangeability
def test_split_cqr_iid_core():
    """Order-statistic rule on i.i.d. scores: mean two-sided coverage >= 1 - alpha_k (Monte Carlo, 20,000
    calibration sets of n = 50, 20 test points each), and the exact per-side rank probability
    i / (n + 1) >= 1 - alpha_k / 2 for the window sizes of the design."""
    rng = np.random.default_rng(2027)
    R, n, m = 20000, 50, 20
    for k, (lo_c, hi_c, alpha_k) in cf.PAIRS.items():
        tau_lo, tau_hi = int(lo_c[1:]) / 100, int(hi_c[1:]) / 100
        q_lo, q_hi = 0.6 * norm.ppf(tau_lo), 0.6 * norm.ppf(tau_hi)          # under-dispersed base quantiles
        yc, yt = rng.standard_normal((R, n)), rng.standard_normal((R, m))
        Slo, Shi = np.sort(q_lo - yc, axis=1), np.sort(yc - q_hi, axis=1)
        i = cf.clip_index(cf.order_index(alpha_k / 2, n), n)
        lower, upper = q_lo - Slo[:, i - 1], q_hi + Shi[:, i - 1]
        cov = np.mean((yt >= lower[:, None]) & (yt <= upper[:, None]))
        assert cov >= 1 - alpha_k, (k, cov)
        for nn in (23, 24, 50, 671, 672, 2684, 2688):
            ii = cf.order_index(alpha_k / 2, nn)
            assert 1 <= ii <= nn and ii / (nn + 1) >= 1 - alpha_k / 2


def test_split_cqr_iid_engine(lear):
    """The split regime of the engine on synthetic i.i.d. targets with fixed base quantiles (scores
    exchangeable): mean coverage over 30 seeds >= 1 - alpha_k up to Monte Carlo error (4 sd, about 0.6 points
    at k = 1; per-side expectation 606 / 673 at n = 672)."""
    base = lear["native"][cf.STD_COLS].copy()
    for j, c in enumerate(data.QCOLS):
        base[c] = 0.6 * norm.ppf(data.TAUS[j])
    base["point"] = 0.0
    covs = {k: [] for k in cf.PAIRS}
    rng = np.random.default_rng(2027)
    for _ in range(30):
        fc = base.assign(actual=rng.standard_normal(len(base)))
        out, _ = cf.calibrate(fc, "split")
        y = out["actual"].to_numpy()
        for k, (lo, hi, _) in cf.PAIRS.items():
            covs[k].append(np.mean((y >= out[lo].to_numpy()) & (y <= out[hi].to_numpy())))
    for k, (_, _, alpha_k) in cf.PAIRS.items():
        c = np.array(covs[k])
        se = c.std(ddof=1) / np.sqrt(len(c))
        assert c.mean() >= 1 - alpha_k - 4 * se, (k, c.mean(), se)
        assert c.mean() <= 1 - alpha_k + 2 / 673 * 2 + 4 * se


# ---------------------------------------------------------------- files on disk
@pytest.mark.parametrize("setting", SETTINGS)
def test_files_on_disk_match(runs, setting):
    for r in cf.REGIMES:
        f = ROOT / "conformal" / f"{setting}_lear_{r}.parquet"
        fa = ROOT / "conformal" / f"{setting}_lear_{r}_alpha.parquet"
        if not (f.exists() and fa.exists()):
            pytest.skip("conformal outputs not written yet")
        out, alpha = runs[setting, r]
        assert pd.read_parquet(f).equals(out), (setting, r)
        assert pd.read_parquet(fa).equals(alpha), (setting, r)
