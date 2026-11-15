"""Evaluation tests (handover Section 8, row "Evaluation implementation (step 5.2)", item (l))."""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy.optimize import linprog
from statsmodels.stats.multitest import multipletests

from esann27 import conformal as cf
from esann27 import data, metrics
from esann27 import evaluate as ev

from .conftest import reference_hashes, require, require_data

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FDIR, CDIR = ROOT / "forecasts", ROOT / "conformal"


def _have_lear():
    return all((FDIR / f"{s}_lear.parquet").exists() and (FDIR / f"{s}_naive.parquet").exists()
               and all((CDIR / f"{s}_lear_{r}.parquet").exists() for r in ev.REGIMES) for s in ev.SETTINGS)


@pytest.fixture(scope="session")
def frames():
    if not _have_lear():
        pytest.skip("LEAR forecasts or conformal files missing")
    return ev.load_all("lear", FDIR, CDIR)


# ---------------------------------------------------------------- lower quantile = np.quantile(inverted_cdf)
def test_lq_index_matches_numpy():
    rng = np.random.default_rng(2027)
    sizes = list(range(1, 400)) + rng.integers(400, 40000, 200).tolist()
    for m in sizes:
        y = np.arange(m, dtype=float)
        for tau in data.TAUS:
            assert y[ev.lq_index(m, tau)] == np.quantile(y, tau, method="inverted_cdf"), (m, tau)


# ---------------------------------------------------------------- PAV against a linear program (brute force)
def _lp_isotonic_quantile(x, y, tau):
    """min sum rho_tau(y_i - f(x_i)) over nondecreasing f, as a linear program (f constant on ties of x)."""
    ux, g = np.unique(x, return_inverse=True)
    G, n = len(ux), len(y)
    c = np.concatenate([np.zeros(G), np.full(n, tau), np.full(n, 1 - tau)])
    A_eq = np.zeros((n, G + 2 * n))
    A_eq[np.arange(n), g] = 1.0
    A_eq[np.arange(n), G + np.arange(n)] = 1.0
    A_eq[np.arange(n), G + n + np.arange(n)] = -1.0
    A_ub = np.zeros((G - 1, G + 2 * n))
    A_ub[np.arange(G - 1), np.arange(G - 1)] = 1.0
    A_ub[np.arange(G - 1), np.arange(1, G)] = -1.0
    bounds = [(None, None)] * G + [(0, None)] * (2 * n)
    r = linprog(c, A_ub=A_ub, b_ub=np.zeros(G - 1), A_eq=A_eq, b_eq=y, bounds=bounds, method="highs")
    assert r.status == 0
    return r.fun / n


@pytest.mark.parametrize("seed", range(6))
def test_pav_matches_linprog(seed):
    rng = np.random.default_rng(2027 + seed)
    n = 30
    x = rng.normal(size=n) if seed % 2 == 0 else rng.integers(0, 8, n).astype(float)    # odd seeds: ties in x
    y = 2 * x + rng.standard_t(3, size=n)
    if seed == 5:
        y = np.round(y)                                                                   # ties in y as well
    for tau in data.TAUS:
        f = ev.pav_quantile(x, y, tau)
        o = np.argsort(x, kind="stable")
        assert np.all(np.diff(f[o]) >= 0)
        assert all(np.ptp(f[x == v]) == 0 for v in np.unique(x))                         # a function of x
        s_pav = np.mean(metrics.pinball(y, f, tau))
        s_lp = _lp_isotonic_quantile(x, y, tau)
        assert s_pav == pytest.approx(s_lp, abs=1e-9), (seed, tau, s_pav, s_lp)


def test_pav_block_values_are_lower_quantiles():
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=200), rng.normal(size=200)
    for tau in (0.1, 0.5, 0.9):
        f = ev.pav_quantile(x, y, tau)
        for v in np.unique(f):
            assert v == np.quantile(y[f == v], tau, method="inverted_cdf")


# ---------------------------------------------------------------- CORP on the LEAR files
def test_corp_identity_and_pav_on_files(frames):
    for key, f in frames.items():
        if key[0] == "naive":
            continue
        y = f["actual"].to_numpy(float)
        Q = f[data.QCOLS].to_numpy(float)
        c = ev.corp(y, Q)
        assert c["CRPS"] == pytest.approx(c["MCB"] - c["DSC"] + c["UNC"], abs=1e-9), key
        assert c["CRPS"] == pytest.approx(metrics.summary(f)["CRPS"], rel=1e-12), key
        pl = c["per_level"]
        assert (pl["MCB"] >= -1e-12).all() and (pl["DSC"] >= -1e-12).all(), key
    # PAV nondecreasing and never worse than the raw quantile, every level, on two files
    for key in (("lear", "native", "raw"), ("lear", "hourly", "aci")):
        f = frames[key]
        y = f["actual"].to_numpy(float)
        for c_, tau in zip(data.QCOLS, data.TAUS):
            x = f[c_].to_numpy(float)
            fit = ev.pav_quantile(x, y, tau)
            o = np.argsort(x, kind="stable")
            assert np.all(np.diff(fit[o]) >= 0)
            assert np.mean(metrics.pinball(y, fit, tau)) <= np.mean(metrics.pinball(y, x, tau)) + 1e-12


def test_unc_is_the_same_for_every_file(frames):
    y = frames["lear", "native", "raw"]["actual"].to_numpy(float)
    unc = [2 * np.mean([np.mean(metrics.pinball(y, ev.lower_quantile(y, t), t)) for t in data.TAUS])]
    assert ev.corp(y, frames["lear", "interp", "split"][data.QCOLS].to_numpy(float))["UNC"] == pytest.approx(unc[0])


# ---------------------------------------------------------------- HAC, DM, pi-regression
def test_ols_nw_equals_statsmodels():
    rng = np.random.default_rng(2027)
    T = 337
    x = np.linspace(0.077, 1, T)
    e = np.convolve(rng.normal(size=T + 5), np.ones(6) / 6, mode="valid")[:T]
    y = 0.1 + 0.3 * x + e
    X = np.column_stack([np.ones(T), x])
    b, se, z, p = ev.ols_nw(y, X, 7)
    r = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 7, "use_correction": False})
    assert np.allclose(b, r.params) and np.allclose(se, r.bse, rtol=1e-10) and np.allclose(p, r.pvalues, rtol=1e-8)
    d = ev.dm_test(e)
    r0 = sm.OLS(e, np.ones(T)).fit(cov_type="HAC", cov_kwds={"maxlags": 7, "use_correction": False})
    assert d["p"] == pytest.approx(r0.pvalues[0], rel=1e-8)
    assert ev.ols_nw(y, X, 7)[1].tolist() == se.tolist()                               # reproducible


def test_native_share():
    pi = ev.native_share(["2025-10-01", "2025-10-29", "2026-03-31", "2026-09-30"])
    assert pi[0] == 0 and pi[1] == pytest.approx(28 / 364) and round(pi[1], 3) == 0.077 and pi[3] == 1.0
    assert pi[2] == pytest.approx(181 / 364)


# ---------------------------------------------------------------- bootstrap reproducibility
def test_boot_days_reproducible_and_circular():
    a, b = ev.boot_days(337, 20), ev.boot_days(337, 20)
    assert np.array_equal(a, b) and a.shape == (20, 337)
    assert a.min() >= 0 and a.max() <= 336
    d = np.diff(a[0].reshape(-1)[: 7 * 48].reshape(48, 7), axis=1) % 337
    assert (d == 1).all()                                                               # blocks of 7 consecutive days
    assert not np.array_equal(ev.boot_days(337, 1, seed=1), ev.boot_days(337, 1, seed=2027))


def test_bootstrap_reproducible(frames):
    b1 = ev.bootstrap(frames, "lear", n_draws=3, n_jobs=1)
    b2 = ev.bootstrap(frames, "lear", n_draws=3, n_jobs=2)
    pd.testing.assert_frame_equal(b1, b2)
    # draw 0 recomputed directly
    D = ev.boot_days(337, 3)
    f = frames["lear", "native", "raw"]
    dates = f["date"].astype(str).to_numpy()
    _, st, ln = np.unique(dates, return_index=True, return_counts=True)
    rows = np.concatenate([np.arange(st[d], st[d] + ln[d]) for d in D[0]])
    c = ev.corp(f["actual"].to_numpy(float)[rows], f[data.QCOLS].to_numpy(float)[rows])
    got = b1[(b1["draw"] == 0) & (b1["regime"] == "raw") & (b1["quantity"] == "CRPS")]["native"].iloc[0]
    assert got == c["CRPS"]


# ---------------------------------------------------------------- coverage tests
def test_kupiec_christoffersen_basics():
    assert ev.kupiec_lr(20, 100, 0.2) == pytest.approx(0.0, abs=1e-12)
    assert ev.kupiec_lr(30, 100, 0.2) > 0
    assert ev.christoffersen_ind_lr(np.tile([0, 1], 50)) > 50                            # alternating hits
    p = np.array([0.01, 0.04, 0.03, 0.2, 0.5, 0.001, 0.02, 0.9])
    assert np.allclose(ev.holm(p), multipletests(p, method="holm")[1])


# ---------------------------------------------------------------- Proposition A
@pytest.fixture(scope="session")
def prop_inputs():
    require_data()
    if not (FDIR / "hourly_lear.parquet").exists():
        pytest.skip("hourly LEAR forecasts missing")
    return ev.burnin_inputs("lear", FDIR)


@reference_hashes
def test_prop_a_inputs_equal_burnin_table(prop_inputs):
    # data checks of 6 Nov 2026, burn-in column
    assert np.round(prop_inputs["sigma"], 2).tolist() == [22.07, 9.11, 8.90, 21.98]
    assert np.round(prop_inputs["sigma_interp"], 2).tolist() == [7.27, 2.83, 2.85, 7.26]
    # the table's ratio row is the ratio of the rounded stds (the burn-in statistics script); kappa itself is unrounded
    sr, ir = np.round(prop_inputs["sigma"], 2), np.round(prop_inputs["sigma_interp"], 2)
    assert np.round(sr / ir, 2).tolist() == [3.04, 3.22, 3.12, 3.03]
    assert np.allclose(prop_inputs["kappa"], prop_inputs["sigma_interp"] / prop_inputs["sigma"])
    assert prop_inputs["n_resid"] == 28 * 24
    assert np.std(prop_inputs["Z"], ddof=1) == pytest.approx(1.0)


def test_prop_a_limits(prop_inputs):
    inp = prop_inputs
    Zs = inp["Z"]
    for q in range(4):
        qt1, _, c1, _, s_n = ev.prop_a_point(inp, q, 1.0)
        qt0, _, c0, s_i, _ = ev.prop_a_point(inp, q, 0.0)
        for t, a1, a0 in zip(data.TAUS, qt1, qt0):
            assert a1 == pytest.approx(s_n * ev.lower_quantile(Zs, t), rel=1e-9, abs=1e-9)
            assert a0 == pytest.approx(s_i * ev.lower_quantile(Zs, t), rel=1e-9, abs=1e-9)
        for pi in (0.0, 0.25, 0.5, 0.75):
            assert ev.prop_a_point(inp, q, pi)[2] >= c1 - 1e-12        # pi = 1 is pinball-optimal on native


def test_pred_file_written_first_and_consistent():
    p = RESULTS / "pred_doseresponse.csv"
    if not p.exists():
        pytest.skip("results not produced yet")
    pr = pd.read_csv(p, dtype={"position": str})
    pr = pr[(pr["model"] == "lear") & (pr["kind"] == "grid")]
    assert len(pr) == 101 * 5
    for pos, s in pr.groupby("position"):
        s = s.sort_values("pi")
        assert s["gap"].iloc[-1] == 0 and s["gap_interp_minus_native"].iloc[0] == 0
        assert np.allclose(s["gap"] + s["gap_interp_minus_native"], s["gap"].iloc[0])
    pooled = pr[pr["position"] == "pooled"]
    pos = pr[pr["position"] != "pooled"]
    assert np.allclose(pooled.sort_values("pi")["crps"].to_numpy(),
                       pos.groupby("pi")["crps"].mean().sort_index().to_numpy())


# ---------------------------------------------------------------- macros equal the table cells they duplicate
def _macros():
    p = RESULTS / "numbers.tex"
    if not p.exists():
        pytest.skip("results not produced yet")
    return ev.read_macros(p)


def _rows(path):
    body = Path(path).read_text()
    out = []
    for line in body.splitlines():
        if line.endswith("\\\\"):
            out.append([c.strip() for c in line[:-2].split("&")])
    return out


def test_macros_equal_table_cells():
    require(*[RESULTS / f for f in ("table1.tex", "table2_lear.tex", "metrics_lear.csv", "bootstrap.csv",
                                    "pred_doseresponse.csv")])
    m = _macros()
    t1 = _rows(RESULTS / "table1.tex")
    assert t1[0] == ["Setting", "Model", "MAE", "CRPS", "MCB", "DSC", "PICP", "Width", "DM"]
    nat = {r[1]: r for r in t1[1:] if r[0] == "Native"}
    assert nat["LEAR"][8] == m["dmPRawLEAR"] and nat["LEAR+ACI"][8] == m["dmPAciLEAR"]
    t2 = {r[0]: r for r in _rows(RESULTS / "table2_lear.tex")[1:]}
    assert t2["Split-CQR"][2] == m["hlPicpSplitLEAR"] and t2["ACI-CQR"][2] == m["hlPicpAciLEAR"]
    met = pd.read_csv(RESULTS / "metrics_lear.csv", float_precision="round_trip")
    for s in ev.SETTINGS:
        r = met[(met["model"] == "naive") & (met["setting"] == s)].iloc[0]
        assert m[f"naiveMAE{s}"] == ev.fmt(r["MAE"])
        for lay, name in (("raw", "LEAR"), ("aci", "LEAR+ACI")):
            r = met[(met["model"] == "lear") & (met["setting"] == s) & (met["layer"] == lay)].iloc[0]
            row = [x for x in t1[1:] if x[0] == ev.SET_LABEL[s] and x[1] == name][0]
            assert row[2:8] == [ev.fmt(r["MAE"]), ev.fmt(r["CRPS"]), ev.fmt(r["MCB"]), ev.fmt(r["DSC"]),
                                ev.fmt(r["PICP80"], 1), ev.fmt(r["width80"])]
    bt = pd.read_csv(RESULTS / "bootstrap.csv", float_precision="round_trip")
    rname = {"raw": "Raw", "aci": "Aci", "aciblock": "Block"}        # aciblock rows of step 5c (handover 8.2)
    # the step 5e rows are checked by tests/test_diag5e.py (decision (1) of 10 Nov, handover Section 8 row "Paper
    # decisions of 10 Nov and step 5e scope")
    for r in bt[(bt["model"] == "lear") & bt["regime"].isin(list(rname))].itertuples():
        base = f"gap{r.quantity.capitalize()}{rname[r.regime]}LEAR"
        assert m[base] == ev.fmt(r.diff_native_minus_interp)
        assert m[base + "Lo"] == ev.fmt(r.lo5) and m[base + "Hi"] == ev.fmt(r.hi95)
        assert r.lo5 <= r.hi95
    pr = pd.read_csv(RESULTS / "pred_doseresponse.csv", dtype={"position": str})
    pooled = pr[(pr["model"] == "lear") & (pr["kind"] == "grid") & (pr["position"] == "pooled")].set_index("pi")
    assert m["predPicpZero"] == ev.fmt(pooled.loc[0.0, "picp80"], 1)
    assert m["predPicpHalf"] == ev.fmt(pooled.loc[0.5, "picp80"], 1)
    gcol = "gap_interp_minus_native" if ev.PRED_GAP == "interp_minus_native" else "gap"
    assert m["predCrpsGapHalf"] == ev.fmt(pooled.loc[0.5, gcol])


def test_regime_macros_equal_table2_cells():
    """The nine regime macros of rev 2.8.14 duplicate the cells of results/table2.tex (interp XGB)."""
    require(RESULTS / "table2.tex")
    m = _macros()
    if "regCrpsAci" not in m:
        pytest.skip("XGB regime macros not produced yet")
    t2 = {r[0]: r for r in _rows(RESULTS / "table2.tex")[1:]}
    col = {"Crps": 1, "Picp": 2, "Width": 3, "Unbounded": 4}
    lab = {"Split": "Split-CQR", "Rolling": "Rolling-CQR", "AciFrozen": "ACI-frozen", "Aci": "ACI-CQR"}
    names = [k for k in m if k.startswith("reg")]
    assert len(names) == 9
    for k in names:
        q, r = re.fullmatch(r"reg(Crps|Picp|Width|Unbounded)(Split|Rolling|AciFrozen|Aci)", k).groups()
        assert m[k] == t2[lab[r]][col[q]]


def test_number_formatting():
    assert ev.fmt(-0.3849) == "\\ensuremath{-0.38}" and ev.fmt(-0.001) == "0.00" and ev.fmt(80.06, 1) == "80.1"
    assert ev.fmt_p(0.0004) == "\\ensuremath{<}0.001" and ev.fmt_p(0.1234) == "0.123"
    assert re.fullmatch(r"[A-Za-z]+", "gapCrpsRawLEARLo")
