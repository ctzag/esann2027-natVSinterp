"""Step 5e tests: diagnostic layers symblock, locblock, locslot (handover 8.1 rows Sym-block, Loc-block, Loc-slot;
step 5e specification B2, B3, B5; Section 8 row "Paper decisions of 10 Nov and step 5e scope" with its
implementation decisions). The layers are computed in memory from forecasts/, because scripts/run_step5e.sh runs
pytest before it writes the conformal files; where the files exist they are compared with the in-memory result.
The results tests at the end skip until the step 5e evaluate runs have written results/."""
import hashlib
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from esann27 import conformal as cf
from esann27 import data
from esann27 import evaluate as ev

from .conftest import reference_outputs, require

ROOT = Path(__file__).resolve().parent.parent
FDIR, CDIR, RESULTS = ROOT / "forecasts", ROOT / "conformal", ROOT / "results"
MANIFEST_9 = ROOT / "tests" / "data" / "outputs_sha256_2026-11-09.txt"
MANIFEST = ROOT / "tests" / "data" / "outputs_sha256_2026-11-10.txt"
SETTINGS = ("hourly", "interp", "native")
MODELS = ("lear", "xgb")
DIAG = ("symblock", "locblock", "locslot")
WIN_LO, WIN_HI = "2026-03-30", "2026-04-26"      # target days whose 28-day window holds 29 Mar 2026 (92 quarters)
XGBM = sorted(f"conformal/{s}_xgbm_{r}{x}.parquet" for s in ("interp", "native") for r in ("aci", "aciblock")
              for x in ("", "_alpha"))
NEW_5E = sorted(f"conformal/{s}_{m}_{r}{x}.parquet" for s in SETTINGS for m in MODELS for r in DIAG
                for x in cf.regime_files(r))
# handover 8.2 "Step 5e macros", written out here as the reference list
MACROS_5E = sorted([f"{n}{M}" for M in ("LEAR", "XGB") for n in (
    "gapCrpsSymBlock", "dmPSymBlock", "gapCrpsLocBlock", "dmPLocBlock", "gapDscSymBlock", "gapDscLocBlock",
    "widthLocBlockInterp", "widthLocBlockNative", "profMornQzeroInterp", "profMornQthreeInterp",
    "profEveQzeroInterp", "profEveQthreeInterp")] + [f"gapCrps{R}{M}{x}" for M in ("LEAR", "XGB")
                                                     for R in ("SymBlock", "LocBlock") for x in ("Lo", "Hi")])


def _need(*paths):
    miss = [p.name for p in paths if not p.exists()]
    if miss:
        pytest.skip(f"files missing: {miss}")


def _dn(dates):
    return pd.to_datetime(pd.Series(dates)).values.astype("datetime64[D]").astype(np.int64)


def _key(f, kind):
    """Group label recomputed here from datetime_local: block 4 (h // 4) + p, slot 4h + p."""
    t = pd.to_datetime(f["datetime_local"])
    h, p = t.dt.hour.to_numpy(), t.dt.minute.to_numpy() // 15
    return 4 * (h // 4) + p if kind == "block" else 4 * h + p


def _bits_equal(a, b):
    if list(a.columns) != list(b.columns) or len(a) != len(b) or not a.dtypes.equals(b.dtypes):
        return False
    for c in a.columns:
        x, y = a[c].to_numpy(), b[c].to_numpy()
        if x.dtype.kind == "f":
            if not np.array_equal(x.view(np.int64), y.view(np.int64)):
                return False
        elif not a[c].reset_index(drop=True).equals(b[c].reset_index(drop=True)):
            return False
    return True


@pytest.fixture(scope="session")
def fc():
    missing = [f"{s}_{m}" for m in MODELS for s in SETTINGS if not (FDIR / f"{s}_{m}.parquet").exists()]
    if missing:
        pytest.skip(f"forecasts missing: {missing}")
    return {(m, s): pd.read_parquet(FDIR / f"{s}_{m}.parquet") for m in MODELS for s in SETTINGS}


@pytest.fixture(scope="session")
def diag(fc):
    return {(m, s, r): cf.run_diag(f, r) for (m, s), f in fc.items() for r in DIAG}


# ---------------------------------------------------------------- regime tables, keys, CLI
def test_regime_tables():
    assert cf.DIAG_REGIMES == {"symblock": ("sym", "block"), "locblock": ("loc", "block"), "locslot": ("loc", "slot")}
    assert not set(cf.DIAG_REGIMES) & set(cf.ALL_REGIMES)                  # --regime all and 5c tables unchanged
    assert list(cf.REGIMES) == ["split", "rolling", "acifrozen", "aci", "acipooled"]
    assert ev.DIAG == DIAG and not set(DIAG) & set(ev.LAYERS)
    assert cf.regime_files("symblock") == ["", "_alpha"] and cf.regime_files("aci") == ["", "_alpha"]
    assert cf.regime_files("locblock") == cf.regime_files("locslot") == ["", "_alpha", "_shift"]
    assert len(NEW_5E) == 48


def test_slot_key_synthetic():
    t = pd.date_range("2025-11-03", periods=2 * 96, freq="15min")
    df = pd.DataFrame({"datetime_local": t.strftime("%Y-%m-%d %H:%M:%S"), "hour": t.hour, "quarter": t.minute // 15})
    g = cf.group_key(df, "slot")
    assert g.tolist() == list(range(96)) * 2
    assert np.array_equal(cf.group_key(df.assign(quarter=df["quarter"] + 1), "slot"), g)       # 1-based quarter
    assert np.array_equal(cf.group_key(df, "block"), 4 * (df["hour"].to_numpy() // 4) + df["quarter"].to_numpy())
    with pytest.raises(AssertionError):
        cf.group_key(df.assign(hour=(df["hour"] + 1) % 24), "slot")
    with pytest.raises(AssertionError):
        cf.group_key(df.assign(quarter=(df["quarter"] + 2) % 4), "slot")


def test_cli_files_and_determinism(tmp_path, capsys):
    """symblock writes output and alpha, locblock and locslot also the shift frame; a second run gives
    byte-identical files (conformal.py has no --n-jobs: decision of 10 Nov)."""
    _need(FDIR / "interp_lear.parquet")
    for run in ("a", "b"):
        for r in DIAG:
            cf.main(["--setting", "interp", "--model", "lear", "--regime", r, "--src", str(FDIR),
                     "--out", str(tmp_path / run)])
    capsys.readouterr()
    names = sorted(f"interp_lear_{r}{x}.parquet" for r in DIAG for x in cf.regime_files(r))
    for run in ("a", "b"):
        assert sorted(p.name for p in (tmp_path / run).iterdir()) == names
    assert all((tmp_path / "a" / n).read_bytes() == (tmp_path / "b" / n).read_bytes() for n in names)


# ---------------------------------------------------------------- symmetric score and offsets (B2 rows 1-3)
def test_symmetric_score(fc):
    """S_k = max of the two one-sided scores row by row; both bounds of a pair move by the same issued offset on
    every quarter-day before sorting."""
    for (m, s), f in fc.items():
        f = f.sort_values("datetime_local").reset_index(drop=True)
        y = f["actual"].to_numpy(float)
        QLO, QHI, e_lo, e_hi, S = cf.pair_scores(f)
        for k, (lo, hi, _) in cf.PAIRS.items():
            assert np.array_equal(e_lo[:, k - 1], f[lo].to_numpy(float) - y)
            assert np.array_equal(e_hi[:, k - 1], y - f[hi].to_numpy(float))
            assert np.array_equal(S[:, k - 1], np.maximum(f[lo].to_numpy(float) - y, y - f[hi].to_numpy(float)))
        out, al, b = cf.calibrate_sym(f, with_bounds=True)
        ev_rows = f[f["date"] >= cf.EVAL_START].reset_index(drop=True)
        assert ev_rows["datetime_local"].equals(out["datetime_local"])
        g = _key(ev_rows, "block")
        for k, (lo, hi, _) in cf.PAIRS.items():
            off = al[al["k"] == k].set_index(["date", "group"])["offset"]
            o = off.reindex(pd.MultiIndex.from_arrays([ev_rows["date"].to_numpy(), g])).to_numpy()
            assert np.isfinite(o).all()
            assert np.array_equal(b["lower"][:, k - 1], ev_rows[lo].to_numpy(float) - o), (m, s, k)
            assert np.array_equal(b["upper"][:, k - 1], ev_rows[hi].to_numpy(float) + o), (m, s, k)
        # the median passes through and the rows are sorted after the offsets: the issued row is the sorted vector of
        # the four lower bounds, the base q50 and the four upper bounds (a negative offset can cross q50)
        unsorted = np.column_stack([b["lower"][:, k - 1] for k in (1, 2, 3, 4)] + [ev_rows["q50"].to_numpy(float)]
                                   + [b["upper"][:, k - 1] for k in (4, 3, 2, 1)])
        assert np.array_equal(out[data.QCOLS].to_numpy(), np.sort(unsorted, axis=1))


def test_symmetric_offsets(fc, diag):
    """For 2 x 24 (group, k, day) triples spread over the year (two in the DST window), offset, n, i and err of the
    alpha log equal a recomputation here from the forecast file."""
    days = [data.day_str(d) for d in cf.eval_days()]
    picks = days[::15] + [WIN_LO, "2026-04-05"]
    checked = 0
    for m, s in (("lear", "interp"), ("xgb", "native")):
        f = fc[m, s].sort_values("datetime_local").reset_index(drop=True)
        al = diag[m, s, "symblock"][1].set_index(["date", "group", "k"])
        g, dn, y = _key(f, "block"), _dn(f["date"]), f["actual"].to_numpy(float)
        for j, D in enumerate(picks):
            gg, k = (7 * j + (3 if m == "xgb" else 0)) % 24, j % 4 + 1
            if D in (WIN_LO, "2026-04-05"):
                gg = j % 4                                               # a group of hours 0-3: n = 111
            lo, hi, a_k = cf.PAIRS[k]
            d0 = _dn([D])[0]
            win = (g == gg) & (dn >= d0 - 28) & (dn <= d0 - 1)
            today = (g == gg) & (dn == d0)
            S = np.sort(np.maximum(f[lo].to_numpy(float) - y, y - f[hi].to_numpy(float))[win])
            rec = al.loc[(D, gg, k)]
            n = len(S)
            assert rec["n"] == n and n == (111 if WIN_LO <= D <= WIN_HI and gg < 4 else 112)
            i = math.ceil((1 - rec["a"]) * (n + 1) - 1e-9)
            assert rec["i"] == i
            off = S[min(max(i, 1), n) - 1]
            assert rec["offset"] == off
            yl, ql, qh = y[today], f[lo].to_numpy(float)[today], f[hi].to_numpy(float)[today]
            err = 0.0 if i > n else (1.0 if i < 1 else float(np.mean((yl < ql - off) | (yl > qh + off))))
            assert rec["err"] == err and rec["nq"] == today.sum()
            checked += 1
    assert checked >= 20 * 2


def test_symblock_log(diag):
    """32,352 records (337 x 24 x 4) per file, one per (date, group, k), n = 112 and 111 for g = 0-3 on
    30 Mar - 26 Apr 2026 (exactly 448 records), two-sided Prop. 4.1 per (setting, model, group, k)."""
    bounds = {k: round(cf.sym_bound(k), 5) for k in cf.PAIRS}
    assert bounds == {1: 0.05045, 2: 0.03858, 3: 0.03858, 4: 0.05045}
    for m in MODELS:
        for s in SETTINGS:
            al = diag[m, s, "symblock"][1]
            assert len(al) == 32352 and not al.duplicated(["date", "group", "k"]).any()
            assert set(al["side"]) == {"sym"} and sorted(al["group"].unique()) == list(range(24))
            win = (al["date"] >= WIN_LO) & (al["date"] <= WIN_HI) & (al["group"] < 4)
            assert (al.loc[win, "n"] == 111).all() and (al.loc[~win, "n"] == 112).all() and int(win.sum()) == 448
            first = al.groupby(["group", "k"])["a"].first()
            assert np.allclose(first.to_numpy(), [cf.PAIRS[k][2] for _, k in first.index], atol=0, rtol=0)
            e = al.groupby(["group", "k"])["err"].agg(["mean", "size"]).reset_index()
            assert (e["size"] == 337).all()
            for k in cf.PAIRS:
                dev = (e.loc[e["k"] == k, "mean"] - cf.PAIRS[k][2]).abs().max()
                assert dev <= cf.sym_bound(k), (m, s, k, dev)


# ---------------------------------------------------------------- recentering (B2 rows 4-5)
def _shift_ref(f, kind):
    """shift_{D,g} and n_shift recomputed here: numpy.median of y - q50 over the group's rows dated D-28..D-1."""
    f = f.sort_values("datetime_local").reset_index(drop=True)
    g, dn = _key(f, kind), _dn(f["date"])
    r = f["actual"].to_numpy(float) - f["q50"].to_numpy(float)
    shift, nsh = np.zeros(len(f)), np.zeros(len(f), dtype=np.int64)
    for gg in np.unique(g):
        idx = np.flatnonzero(g == gg)
        dd = dn[idx]
        for d in np.unique(dd):
            today, win = idx[dd == d], idx[(dd >= d - 28) & (dd <= d - 1)]
            nsh[today] = len(win)
            shift[today] = np.median(r[win]) if len(win) else 0.0
    return shift, nsh


@pytest.mark.parametrize("m,s,r", [("lear", "interp", "locblock"), ("xgb", "native", "locslot"),
                                   ("lear", "hourly", "locslot"), ("xgb", "interp", "locblock")])
def test_recentering_recomputed(fc, diag, m, s, r):
    sf = diag[m, s, r][2]
    shift, nsh = _shift_ref(fc[m, s], cf.DIAG_REGIMES[r][1])
    assert np.array_equal(sf["shift"].to_numpy(), shift) and np.array_equal(sf["n_shift"].to_numpy(), nsh)


def test_recentering_counts(diag):
    """Shift frame: 35,036 rows, SHIFT_COLS; shift = 0 and n_shift = 0 on 1 Oct 2025; n_shift grows by 4 per day
    (block) or 1 per day (slot) through the burn-in, 112 / 28 from 29 Oct, 111 / 27 for the groups of hour 2 on
    30 Mar - 26 Apr 2026."""
    for m in MODELS:
        for s in SETTINGS:
            for r in ("locblock", "locslot"):
                sf = diag[m, s, r][2]
                assert len(sf) == 35036 and list(sf.columns) == cf.SHIFT_COLS
                assert sf["n_shift"].dtype == np.int64 and sf["shift"].dtype == np.float64
                block = r == "locblock"
                per_day = 4 if block else 1
                hour2 = sf["group"] < 4 if block else sf["group"].between(8, 11)
                d = _dn(sf["date"]) - data.day_number("2025-10-01")
                first = d == 0
                assert (sf.loc[first, "shift"] == 0).all() and (sf.loc[first, "n_shift"] == 0).all()
                burn = d <= 27
                assert (sf.loc[burn, "n_shift"].to_numpy() == per_day * d[burn]).all()
                win = (sf["date"] >= WIN_LO) & (sf["date"] <= WIN_HI) & hour2
                full = ~burn & ~win.to_numpy()
                assert (sf.loc[full, "n_shift"] == 28 * per_day).all()
                assert (sf.loc[win, "n_shift"] == 28 * per_day - 1).all() and win.sum() > 0
                assert sf["group"].nunique() == (24 if block else 96)


def test_recentering_no_lookahead(fc):
    """Residuals of D or later never enter: y perturbed from day D on leaves every shift up to D unchanged."""
    f = fc["lear", "interp"].sort_values("datetime_local").reset_index(drop=True)
    for kind in ("block", "slot"):
        base = cf.recenter(f, kind)[1]
        for D in ("2025-10-15", "2026-03-29", "2026-07-01"):
            g = f.copy()
            late = (g["date"] >= D).to_numpy()
            g.loc[late, "actual"] = g.loc[late, "actual"] + 500.0 * np.sin(np.arange(late.sum()))
            pert = cf.recenter(g, kind)[1]
            upto = (base["date"] <= D).to_numpy()
            assert np.array_equal(pert["shift"].to_numpy()[upto], base["shift"].to_numpy()[upto]), (kind, D)
            assert not np.array_equal(pert["shift"].to_numpy()[~upto], base["shift"].to_numpy()[~upto])


def test_recentering_frame(fc):
    """Every quantile column of a row moves by the same shift, point and actual unchanged, rows stay sorted;
    interp and native shifts equal on 2 Oct 2025 (identical 1 Oct residuals)."""
    for m in MODELS:
        for kind in ("block", "slot"):
            sh = {}
            for s in ("interp", "native", "hourly"):
                f = fc[m, s].sort_values("datetime_local").reset_index(drop=True)
                out, sf = cf.recenter(f, kind)
                shift = sf["shift"].to_numpy()
                for c in data.QCOLS:
                    assert np.array_equal(out[c].to_numpy(), f[c].to_numpy() + shift)
                for c in ("datetime_local", "date", "hour", "quarter", "actual", "point"):
                    assert _bits_equal(out[[c]], f[[c]])
                q = out[data.QCOLS].to_numpy()
                assert (np.diff(q, axis=1) >= 0).all()
                sh[s] = sf
            d2 = (sh["interp"]["date"] == "2025-10-02").to_numpy()
            assert d2.sum() == 96
            assert np.array_equal(sh["interp"]["shift"].to_numpy()[d2], sh["native"]["shift"].to_numpy()[d2])


def test_loc_reduction(fc, tmp_path):
    """The locblock code path with the shift forced to 0 reproduces conformal/<setting>_<model>_aci.parquet and
    its alpha log bit for bit."""
    for m, s in (("lear", "hourly"), ("lear", "interp"), ("xgb", "native")):
        _need(CDIR / f"{s}_{m}_aci.parquet", CDIR / f"{s}_{m}_aci_alpha.parquet")
        out, al, sf = cf.run_diag(fc[m, s], "locblock", zero_shift=True)
        assert (sf["shift"] == 0).all()
        out.to_parquet(tmp_path / f"{s}_{m}_aci.parquet", index=False)
        al.to_parquet(tmp_path / f"{s}_{m}_aci_alpha.parquet", index=False)
        for x in ("", "_alpha"):
            assert cf.compare_files(tmp_path / f"{s}_{m}_aci{x}.parquet", CDIR / f"{s}_{m}_aci{x}.parquet") == "bytes"


def test_loc_logs(diag):
    """locblock and locslot alpha logs: 10,784 records, the aci structure (4 position groups, per side, n = 672 and
    671 in the DST window), Prop. 4.1 per side."""
    for m in MODELS:
        for s in SETTINGS:
            for r in ("locblock", "locslot"):
                al = diag[m, s, r][1]
                assert len(al) == 10784 and list(al.columns) == cf.ALPHA_COLS
                assert sorted(al["group"].unique()) == [0, 1, 2, 3] and set(al["side"]) == {"lo", "hi"}
                win = (al["date"] >= WIN_LO) & (al["date"] <= WIN_HI)
                assert (al.loc[win, "n"] == 671).all() and (al.loc[~win, "n"] == 672).all() and win.sum() == 896
                e = al.groupby(["group", "k", "side"])["err"].agg(["mean", "size"]).reset_index()
                assert (e["size"] == 337).all() and len(e) == 32
                for k in cf.PAIRS:
                    dev = (e.loc[e["k"] == k, "mean"] - cf.PAIRS[k][2] / 2).abs().max()
                    assert dev <= cf.aci_bound(k), (m, s, r, k)


# ---------------------------------------------------------------- output schema, files (B2 rows 8-9)
def test_output_schema(diag):
    for m in MODELS:
        for s in SETTINGS:
            _need(CDIR / f"{s}_{m}_aci.parquet")
            ref = pd.read_parquet(CDIR / f"{s}_{m}_aci.parquet")
            for r in DIAG:
                out = diag[m, s, r][0]
                assert len(out) == 32348 and list(out.columns) == list(ref.columns) and out.dtypes.equals(ref.dtypes)
                assert out["datetime_local"].equals(ref["datetime_local"]) and out["actual"].equals(ref["actual"])
                q = out[data.QCOLS].to_numpy()
                assert (np.diff(q, axis=1) >= 0).all() and np.array_equal(np.sort(q, axis=1), q)
                assert np.isfinite(q).all()


def test_files_equal_memory(diag):
    """Each of the 48 step 5e files in conformal/ equals the in-memory result (skips before the run)."""
    present = [n for n in NEW_5E if (ROOT / n).exists()]
    if not present:
        pytest.skip("step 5e conformal files not written yet")
    assert present == NEW_5E
    for m in MODELS:
        for s in SETTINGS:
            for r in DIAG:
                out, al, sf = diag[m, s, r]
                assert _bits_equal(pd.read_parquet(CDIR / f"{s}_{m}_{r}.parquet"), out)
                assert _bits_equal(pd.read_parquet(CDIR / f"{s}_{m}_{r}_alpha.parquet"), al)
                if sf is not None:
                    assert _bits_equal(pd.read_parquet(CDIR / f"{s}_{m}_{r}_shift.parquet"), sf)


def _hash_tree():
    out = {}
    for d in (FDIR, CDIR):
        for p in sorted(d.rglob("*")):
            if p.is_file() and p.name != ".DS_Store":
                out[p.relative_to(ROOT).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


@reference_outputs
def test_existing_outputs_keep_sha256():
    """Manifest of 10 Nov = the manifest of 9 Nov plus the 8 xgbm conformal files (decision (1) of 10 Nov). Every
    file in it keeps its sha256, the only new files are the 48 step 5e files, and once they all exist the set of
    files is the manifest plus the 48."""
    _need(MANIFEST, MANIFEST_9, FDIR, CDIR)
    t9, t10 = MANIFEST_9.read_text(), MANIFEST.read_text()
    assert t10.startswith(t9) and sorted(l.split("  ", 1)[1] for l in t10[len(t9):].splitlines()) == XGBM
    ref = dict(reversed(l.split("  ", 1)) for l in t10.splitlines() if l)
    now = _hash_tree()
    bad = [p for p in ref if now.get(p) != ref[p]]
    assert not bad, bad
    new = sorted(set(now) - set(ref))
    assert set(new) <= set(NEW_5E), new
    if len(new) == len(NEW_5E):
        assert set(now) == set(ref) | set(NEW_5E)


def test_regenerated_aci_aciblock(tmp_path, capsys):
    """aci and aciblock of both models, three settings, regenerated by the CLI equal conformal/ byte for byte."""
    expected = sorted(f"{s}_{m}_{r}{x}.parquet" for m in MODELS for s in SETTINGS for r in ("aci", "aciblock")
                      for x in ("", "_alpha"))
    _need(*[CDIR / n for n in expected])
    for m in MODELS:
        for s in SETTINGS:
            for r in ("aci", "aciblock"):
                cf.main(["--setting", s, "--model", m, "--regime", r, "--src", str(FDIR), "--out", str(tmp_path)])
    capsys.readouterr()
    assert sorted(p.name for p in tmp_path.glob("*.parquet")) == expected
    assert all(cf.compare_files(tmp_path / n, CDIR / n) == "bytes" for n in expected)


# ---------------------------------------------------------------- evaluate: rule, formats, macros (B3)
def _met(gaps):
    rows = []
    for m, g in gaps.items():
        for lay, v in g.items():
            rows += [{"model": m, "setting": "interp", "layer": lay, "CRPS": 15.0},
                     {"model": m, "setting": "native", "layer": lay, "CRPS": 15.0 + v}]
    return pd.DataFrame(rows)


def test_reading_rule_synthetic():
    rec = {"aci": -0.30, "aciblock": -0.06, "symblock": -0.29, "locblock": -0.05}
    wid = {"aci": -0.30, "aciblock": -0.06, "symblock": -0.07, "locblock": -0.28}
    met = _met({"lear": rec, "xgb": rec})
    sh = {m: ev.wl_shares(met, m) for m in MODELS}
    assert math.isclose(sh["lear"]["W"], 0.01 / 0.24) and math.isclose(sh["lear"]["L"], 0.25 / 0.24)
    assert math.isclose(sh["lear"]["G4"], 0.30) and math.isclose(sh["lear"]["GL"], 0.05)
    assert ev.reading_label(sh) == "recentering"
    assert ev.reading_line(sh) == ("reading: W = 0.042 / 0.042, L = 1.042 / 1.042 (LEAR / XGB) -> recentering")
    met_w = _met({"lear": wid, "xgb": wid})
    assert ev.reading_label({m: ev.wl_shares(met_w, m) for m in MODELS}) == "width"
    met_x = _met({"lear": rec, "xgb": wid})
    assert ev.reading_label({m: ev.wl_shares(met_x, m) for m in MODELS}) == "mixed"
    edge = {"aci": -0.30, "aciblock": -0.0, "symblock": -0.20, "locblock": -0.10}      # W = 1/3, L = 2/3 exactly
    assert ev.reading_label({m: ev.wl_shares(_met({"lear": edge, "xgb": edge}), m) for m in MODELS}) in (
        "recentering", "mixed")
    part = {"lear": sh["lear"], "xgb": None}
    assert ev.reading_label(part) == "pending"
    assert ev.reading_line(part) == "reading: W = 0.042 / pending, L = 1.042 / pending (LEAR / XGB) -> pending"
    assert ev.wl_shares(met[met["layer"] != "locblock"], "lear") is None


def test_formats_and_names():
    assert ev.fmt_signed(9.3049) == "+9.3" and ev.fmt_signed(-10.46) == "\\ensuremath{-}10.5"
    assert ev.fmt_signed(-0.04) == "0.0" and ev.fmt_signed(13.45) == "+13.4"
    assert sorted(ev.DIAG_MACROS) == MACROS_5E and len(set(ev.DIAG_MACROS)) == 32
    assert sorted(ev.diag_macro_names("lear")) == [n for n in MACROS_5E if "LEAR" in n]
    assert all(re.fullmatch(r"[A-Za-z]+", n) for n in MACROS_5E)
    assert dict(ev.PROFILE_MACROS) == {"profMornQzeroInterp": 8, "profMornQthreeInterp": 11,
                                       "profEveQzeroInterp": 16, "profEveQthreeInterp": 19}


def test_fig2a_size(tmp_path):
    """fig2a.pdf: panel (a) alone, 12.2 x 4.0 cm, TrueType fonts only (pdf.fonttype 42)."""
    names = [f"{s}_xgb_{r}.parquet" for s in SETTINGS for r in ev.REGIMES]
    _need(*[CDIR / n for n in names], *[FDIR / f"{s}_{m}.parquet" for s in SETTINGS for m in ("xgb", "naive")])
    frames = ev.load_all("xgb", FDIR, CDIR)
    ev.figure2a(frames, "xgb", tmp_path / "fig2a.pdf")
    _check_pdf_size(tmp_path / "fig2a.pdf", 12.2, 4.0)


def _check_pdf_size(p, w_cm, h_cm):
    b = p.read_bytes()
    box = re.search(rb"/MediaBox \[\s*([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+)\s*\]", b)
    assert box, "no MediaBox"
    x0, y0, x1, y1 = (float(v) for v in box.groups())
    assert abs((x1 - x0) - w_cm / 2.54 * 72) < 0.01 and abs((y1 - y0) - h_cm / 2.54 * 72) < 0.01
    assert b"/Type3" not in b and b"/FontFile2" in b


# ---------------------------------------------------------------- results (after the step 5e evaluate runs)
@reference_outputs
def test_diag_results_files():
    """After the step 5e run: numbers.tex holds 148 macros once both models ran (decision (2) of 10 Nov), the 16
    macros of each model run equal the bootstrap.csv, metrics_<model>.csv and profile_<model>.csv values they are
    formatted from, the DM p-values a recomputation from the conformal files, and the cells of the step 5e section
    of summary_<model>.md; the reading line follows from the two metrics files."""
    require(*[CDIR / f"{s}_{mo}_{r}.parquet" for mo in MODELS for s in ("interp", "native") for r in ev.DIAG_NAME],
            *[RESULTS / f for mo in MODELS for f in (f"summary_{mo}.md", f"metrics_{mo}.csv", f"profile_{mo}.csv")],
            RESULTS / "bootstrap.csv")
    p = RESULTS / "numbers.tex"
    _need(p)
    m = ev.read_macros(p)
    done = [mo for mo in MODELS if f"gapCrpsSymBlock{ev.MODEL_LABEL[mo]}" in m]
    if not done:
        pytest.skip("step 5e not run yet")
    step5e = sorted(k for k in m if k in ev.DIAG_MACROS)
    assert step5e == sorted(n for mo in done for n in ev.diag_macro_names(mo))
    assert len(m) == 116 + 16 * len(done)
    bt = pd.read_csv(RESULTS / "bootstrap.csv", float_precision="round_trip")
    mets = {mo: pd.read_csv(RESULTS / f"metrics_{mo}.csv", float_precision="round_trip") for mo in MODELS
            if (RESULTS / f"metrics_{mo}.csv").exists()}
    sh = {mo: ev.wl_shares(mets[mo], mo) if mo in mets else None for mo in MODELS}
    for model in done:
        M = ev.MODEL_LABEL[model]
        met = mets[model]
        assert sorted(met.loc[met["layer"].isin(DIAG), "setting"].tolist()) == sorted(list(SETTINGS) * 3)
        e = met[met["layer"].isin(DIAG)]
        assert np.allclose(e["CRPS"], e["MCB"] - e["DSC"] + e["UNC"], atol=1e-9)
        b = bt[(bt["model"] == model) & bt["regime"].isin(DIAG)]
        assert len(b) == 9 and (b["lo5"] <= b["hi95"]).all()
        crps = lambda s, lay, q="CRPS": met[(met["setting"] == s) & (met["layer"] == lay)][q].iloc[0]
        sec = (RESULTS / f"summary_{model}.md").read_text().split(f"\n## {12 if model == 'lear' else 13}. {ev.SEC5E}\n")
        assert len(sec) == 2 and "\n## " not in sec[1]                     # the last section of the summary
        s5e = sec[1]
        for k in ev.diag_macro_names(model):
            assert f"| \\{k} | {m[k]} |" in s5e, k
        for reg, R in ev.DIAG_NAME.items():
            r = b[(b["regime"] == reg) & (b["quantity"] == "CRPS")].iloc[0]
            assert np.isclose(r["diff_native_minus_interp"], crps("native", reg) - crps("interp", reg), atol=1e-12)
            assert m[f"gapCrps{R}{M}"] == ev.fmt(r["diff_native_minus_interp"])
            assert m[f"gapCrps{R}{M}Lo"] == ev.fmt(r["lo5"]) and m[f"gapCrps{R}{M}Hi"] == ev.fmt(r["hi95"])
            rd = b[(b["regime"] == reg) & (b["quantity"] == "DSC")].iloc[0]
            assert m[f"gapDsc{R}{M}"] == ev.fmt(rd["diff_native_minus_interp"])
            f = {s: ev.load_layer(model, s, reg, FDIR, CDIR) for s in ("interp", "native")}
            dm = ev.dm_test((ev.daily_crps(f["native"]) - ev.daily_crps(f["interp"])).to_numpy())
            assert dm["T"] == 337 and m[f"dmP{R}{M}"] == ev.fmt_p(dm["p"])
        assert m[f"widthLocBlockInterp{M}"] == ev.fmt(crps("interp", "locblock", "width80"))
        assert m[f"widthLocBlockNative{M}"] == ev.fmt(crps("native", "locblock", "width80"))
        prof = pd.read_csv(RESULTS / f"profile_{model}.csv", float_precision="round_trip")
        pi = prof[prof["setting"] == "interp"].set_index("group")
        for name, g in ev.PROFILE_MACROS:
            assert m[f"{name}{M}"] == ev.fmt_signed(pi.loc[g, "mean_resid"])
        line = [l for l in s5e.splitlines() if l.startswith("reading: ")]
        assert len(line) == 1 and not any(l.startswith("rule: ") for l in s5e.splitlines())
        if all(sh[mo] is not None for mo in MODELS):
            assert line[0] == ev.reading_line(sh)


@reference_outputs
def test_profile_file():
    """results/profile_<model>.csv recomputed here from forecasts/<setting>_<model>.parquet (evaluation rows)."""
    require(*[FDIR / f"{s}_{mo}.parquet" for mo in MODELS for s in ("interp", "native")])
    for model in MODELS:
        p = RESULTS / f"profile_{model}.csv"
        if not p.exists():
            pytest.skip("profile files not written yet")
        prof = pd.read_csv(p, float_precision="round_trip", dtype={"block": str})
        assert list(prof.columns) == ["model", "setting", "group", "block", "position", "mean_resid", "mad_resid", "n"]
        assert len(prof) == 48 and (prof["model"] == model).all()
        for s in ("interp", "native"):
            f = pd.read_parquet(FDIR / f"{s}_{model}.parquet")
            f = f[f["date"] >= cf.EVAL_START]
            assert len(f) == 32348
            g, r = _key(f, "block"), (f["actual"] - f["q50"]).to_numpy()
            ps = prof[prof["setting"] == s].set_index("group")
            assert ps.index.tolist() == list(range(24))
            for gg in range(24):
                v = r[g == gg]
                row = ps.loc[gg]
                assert row["n"] == len(v) == (1347 if gg < 4 else 1348)
                assert row["block"] == f"{4 * (gg // 4):02d}-{4 * (gg // 4) + 3:02d}" and row["position"] == gg % 4
                assert math.isclose(row["mean_resid"], float(np.mean(v)), rel_tol=0, abs_tol=1e-9)
                assert row["mad_resid"] == float(np.median(np.abs(v - np.median(v))))


def test_fig2a_file():
    p = RESULTS / "fig2a.pdf"
    if not p.exists():
        pytest.skip("fig2a.pdf not written yet")
    _check_pdf_size(p, 12.2, 4.0)
