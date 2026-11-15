"""Step 5c tests: regime aciblock, ACI-CQR with 24 Mondrian groups (handover 8.1 row ACI-block,
step 5c specification B2 and B3). The aciblock runs are computed in memory from forecasts/, because the
step 5c script runs pytest before it writes conformal/<setting>_<model>_aciblock.parquet."""
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
SETTINGS = ("hourly", "interp", "native")
MODELS = ("lear", "xgb")
REG = "aciblock"
DST = "2026-03-29"


def _dn(datestr):
    return pd.to_datetime(pd.Series(datestr)).values.astype("datetime64[D]").astype(np.int64)


@pytest.fixture(scope="session")
def fc():
    missing = [f"{s}_{m}" for m in MODELS for s in SETTINGS if not (FDIR / f"{s}_{m}.parquet").exists()]
    if missing:
        pytest.skip(f"forecasts missing: {missing}")
    return {(m, s): pd.read_parquet(FDIR / f"{s}_{m}.parquet") for m in MODELS for s in SETTINGS}


@pytest.fixture(scope="session")
def block(fc):
    return {k: cf.calibrate(f, REG) for k, f in fc.items()}


def _need(*paths):
    miss = [p.name for p in paths if not p.exists()]
    if miss:
        pytest.skip(f"files missing: {miss}")


# ---------------------------------------------------------------- regime tables and group key
def test_regime_tables():
    assert list(cf.REGIMES) == ["split", "rolling", "acifrozen", "aci", "acipooled"]     # --regime all, 5.1 tests
    assert cf.ALL_REGIMES[REG] == ("rolling", True, "block") and REG not in cf.REGIMES
    assert cf.ALL_REGIMES["aci"] == ("rolling", True, "position")      # aciblock differs from aci in the key only
    assert ev.REGIMES == tuple(cf.REGIMES) and ev.REG_BLOCK == REG and REG not in ev.LAYERS


def test_group_key_synthetic():
    t = pd.date_range("2025-11-03", periods=2 * 96, freq="15min")
    df = pd.DataFrame({"datetime_local": t.strftime("%Y-%m-%d %H:%M:%S"), "hour": t.hour, "quarter": t.minute // 15})
    g = cf.group_key(df, "block")
    assert g[:16].tolist() == [0, 1, 2, 3] * 4 and g[16] == 4 and g[95] == 23 and g[96] == 0
    assert np.array_equal(cf.group_key(df.assign(quarter=df["quarter"] + 1), "block"), g)      # 1-based quarter
    assert np.array_equal(cf.group_key(df, "position"), df["quarter"].to_numpy())
    assert (cf.group_key(df, "pooled") == cf.POOLED).all()
    bad = df.copy()
    bad.loc[5, "hour"] = 7
    with pytest.raises(AssertionError):
        cf.group_key(bad, "block")
    bad = df.copy()
    bad.loc[5, "quarter"] = 3
    with pytest.raises(AssertionError):
        cf.group_key(bad, "block")
    with pytest.raises(ValueError):
        cf.group_key(df, "hour")


def test_group_key_on_evaluation_rows(fc, block):
    for (m, s), (out, _) in block.items():
        g = cf.group_key(out, "block")
        assert sorted(np.unique(g).tolist()) == list(range(24)), (m, s)
        assert np.array_equal(g % 4, out["quarter"].to_numpy())
        assert np.array_equal(g // 4, out["hour"].to_numpy() // 4)
        # independent of group_key: hour and minute read from the datetime_local string
        hh = out["datetime_local"].str[11:13].astype(int).to_numpy()
        mm = out["datetime_local"].str[14:16].astype(int).to_numpy()
        assert np.array_equal(g, 4 * (hh // 4) + mm // 15)


# ---------------------------------------------------------------- counts (spec B2, checked by the authors)
def test_rows_per_group_and_day(fc, block):
    for (m, s), f in fc.items():
        c = pd.DataFrame({"date": f["date"].to_numpy(), "g": cf.group_key(f, "block")}).groupby(["date", "g"]).size()
        assert len(c) == 365 * 24
        dst = c.index.get_level_values("date") == DST
        low = dst & (c.index.get_level_values("g") <= 3)
        assert (c[low] == 3).all() and low.sum() == 4 and (c[~low] == 4).all(), (m, s)
        assert (f["date"] == "2025-10-26").sum() == 96 and (c.loc["2025-10-26"] == 4).all()
    for (m, s), (_, al) in block.items():
        low = (al["date"] == DST).to_numpy() & (al["group"].to_numpy() <= 3)
        assert (al["nq"].to_numpy()[low] == 3).all() and (al["nq"].to_numpy()[~low] == 4).all()


def test_evaluation_quarters_per_group(block):
    for (m, s), (out, _) in block.items():
        c = pd.Series(cf.group_key(out, "block")).value_counts().sort_index()
        assert c.loc[0:3].tolist() == [1347] * 4 and (c.loc[4:23] == 1348).all() and c.sum() == 32348, (m, s)


def test_window_size_n(block):
    lo, hi = data.day_number("2026-03-30"), data.day_number("2026-04-26")
    for (m, s), (_, al) in block.items():
        D = _dn(al["date"])
        short = (D >= lo) & (D <= hi) & (al["group"].to_numpy() <= 3)
        n = al["n"].to_numpy()
        assert (n[short] == 111).all() and (n[~short] == 112).all(), (m, s)
        assert int((n == 111).sum()) == 896


def test_alpha_log(block):
    ev_days = [data.day_str(d) for d in cf.eval_days()]
    for (m, s), (_, al) in block.items():
        assert list(al.columns) == cf.ALPHA_COLS
        assert len(al) == 337 * 24 * 4 * 2 == 64704
        assert not al.duplicated(["date", "group", "k", "side"]).any()
        assert al["date"].unique().tolist() == ev_days and (al.groupby("date").size() == 192).all()
        assert sorted(al["group"].unique().tolist()) == list(range(24))
        assert sorted(al["k"].unique().tolist()) == [1, 2, 3, 4] and sorted(al["side"].unique()) == ["hi", "lo"]


# ---------------------------------------------------------------- windows and look-ahead
def test_window_dates(block):
    for (m, s), (_, al) in block.items():
        D, w0, w1 = _dn(al["date"]), _dn(al["win_first"]), _dn(al["win_last"])
        assert (w0 == D - 28).all() and (w1 == D - 1).all(), (m, s)


def _scores(df, k, side):
    lo, hi, _ = cf.PAIRS[k]
    return (df[lo] - df["actual"]) if side == "lo" else (df["actual"] - df[hi])


@pytest.mark.parametrize("key", [("lear", "interp"), ("xgb", "native"), ("xgb", "hourly")])
def test_offsets_independent_recompute(fc, block, key):
    """Offsets recomputed with plain pandas filters on the date strings and a group label read from the
    datetime_local string: every score used for day D comes from D-28 .. D-1."""
    f = fc[key]
    al = block[key][1].set_index(["date", "group", "k", "side"])
    hh = f["datetime_local"].str[11:13].astype(int)
    mm = f["datetime_local"].str[14:16].astype(int)
    gid = 4 * (hh // 4) + mm // 15
    for D in ("2025-10-29", "2025-11-26", DST, "2026-03-30", "2026-04-26", "2026-04-27", "2026-09-30"):
        Dt = pd.Timestamp(D)
        a0 = (Dt - pd.Timedelta(days=28)).strftime("%Y-%m-%d")
        a1 = (Dt - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        win = f[(f["date"] >= a0) & (f["date"] <= a1)]
        assert win["date"].max() < D and win["date"].nunique() == 28
        for g in range(24):
            wg = win[gid.loc[win.index] == g]
            for k in cf.PAIRS:
                for side in cf.SIDES:
                    rec = al.loc[(D, g, k, side)]
                    srt = np.sort(_scores(wg, k, side).to_numpy())
                    assert rec["n"] == len(srt)
                    i = cf.order_index(rec["a"], len(srt))
                    assert rec["offset"] == srt[min(max(i, 1), len(srt)) - 1], (key, D, g, k, side)


@pytest.mark.parametrize("cut", ["2025-10-29", "2026-03-29"])
def test_no_lookahead(fc, block, cut):
    f = fc["xgb", "interp"]
    pert = f.copy()
    on = (pert["date"] >= cut).to_numpy()
    rng = np.random.default_rng(2027)
    pert.loc[on, "actual"] = pert.loc[on, "actual"] * 1.9 - 60.0 + 80.0 * rng.random(on.sum())
    out0, a0 = block["xgb", "interp"]
    out1, a1 = cf.calibrate(pert, REG)
    upto = (out0["date"] <= cut).to_numpy()
    assert np.array_equal(out0.loc[upto, data.QCOLS].to_numpy(), out1.loc[upto, data.QCOLS].to_numpy())
    au = (a0["date"] <= cut).to_numpy()
    for c in ("a", "n", "i", "offset"):
        assert a0.loc[au, c].equals(a1.loc[au, c]), c
    assert not np.array_equal(out0.loc[~upto, data.QCOLS].to_numpy(), out1.loc[~upto, data.QCOLS].to_numpy())


# ---------------------------------------------------------------- reduction and existing regimes
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("setting", SETTINGS)
def test_reduction_position_key_is_aci(fc, tmp_path, model, setting):
    """The aciblock code path with the position key reproduces the existing aci files bit for bit."""
    f_out, f_al = CDIR / f"{setting}_{model}_aci.parquet", CDIR / f"{setting}_{model}_aci_alpha.parquet"
    _need(f_out, f_al)
    out, al = cf.calibrate(fc[model, setting], REG, scheme="position")
    assert out.equals(pd.read_parquet(f_out)) and al.equals(pd.read_parquet(f_al))
    out.to_parquet(tmp_path / "o.parquet", index=False)
    al.to_parquet(tmp_path / "a.parquet", index=False)
    assert cf.compare_files(tmp_path / "o.parquet", f_out) in ("bytes", "frame")
    assert cf.compare_files(tmp_path / "a.parquet", f_al) in ("bytes", "frame")


@pytest.mark.parametrize("model", MODELS)
def test_existing_regimes_regenerated(tmp_path, model, capsys):
    """The five existing regimes regenerated by the CLI (--regime all) into a temporary folder equal the files in
    conformal/: 15 output and 15 alpha files per model."""
    expected = sorted(f"{s}_{model}_{r}{x}.parquet" for s in SETTINGS for r in cf.REGIMES for x in ("", "_alpha"))
    _need(*[CDIR / n for n in expected], *[FDIR / f"{s}_{model}.parquet" for s in SETTINGS])
    for s in SETTINGS:
        cf.main(["--setting", s, "--model", model, "--regime", "all", "--src", str(FDIR), "--out", str(tmp_path)])
    capsys.readouterr()
    assert sorted(p.name for p in tmp_path.glob("*.parquet")) == expected        # no aciblock under --regime all
    modes = {n: cf.compare_files(tmp_path / n, CDIR / n) for n in expected}
    assert all(v in ("bytes", "frame") for v in modes.values()), {n: v for n, v in modes.items() if not v}


def test_compare_files(tmp_path):
    df = pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})
    df.to_parquet(tmp_path / "1.parquet", index=False)
    df.to_parquet(tmp_path / "2.parquet", index=False)
    assert cf.compare_files(tmp_path / "1.parquet", tmp_path / "2.parquet") == "bytes"
    df.to_parquet(tmp_path / "3.parquet", index=False, compression="gzip")                 # same frame, other bytes
    assert cf.compare_files(tmp_path / "3.parquet", tmp_path / "1.parquet") == "frame"
    df.assign(a=[-0.0, 2.0]).to_parquet(tmp_path / "5.parquet", index=False)
    df.assign(a=[0.0, 2.0]).to_parquet(tmp_path / "6.parquet", index=False)
    assert cf.compare_files(tmp_path / "5.parquet", tmp_path / "6.parquet") == ""           # -0.0 vs 0.0: bits differ
    df.assign(a=[1, 2]).to_parquet(tmp_path / "7.parquet", index=False)
    assert cf.compare_files(tmp_path / "7.parquet", tmp_path / "1.parquet") == ""           # dtype differs


# ---------------------------------------------------------------- ACI bound, alpha paths, clipping, sorting
def test_prop41_per_side(block):
    assert [round(cf.aci_bound(k), 5) for k in cf.PAIRS] == [0.05638, 0.05045, 0.04451, 0.03858]
    for (m, s), (_, al) in block.items():
        e = al.groupby(["group", "k", "side"])["err"].agg(["mean", "size"])
        assert len(e) == 24 * 4 * 2 and (e["size"] == 337).all()
        for (g, k, side), r in e.iterrows():
            dev = abs(r["mean"] - cf.PAIRS[k][2] / 2)
            assert dev <= cf.aci_bound(k), (m, s, g, k, side, dev)


def test_alpha_paths_consistent(block):
    for (m, s), (_, al) in block.items():
        for (g, k, side), a in al.groupby(["group", "k", "side"]):
            a = a.sort_values("date")
            t = cf.PAIRS[k][2] / 2
            av, ev_ = a["a"].to_numpy(), a["err"].to_numpy()
            assert av[0] == t
            assert np.array_equal(av[1:], av[:-1] + cf.GAMMA * (t - ev_[:-1])), (m, s, g, k, side)
            i = np.array([cf.order_index(x, n) for x, n in zip(av, a["n"])])
            assert np.array_equal(i, a["i"].to_numpy())
            assert [cf.formal_state(x, n) for x, n in zip(i, a["n"])] == a["formal"].tolist()
            assert np.array_equal(a["clipped"].to_numpy(), a["formal"].to_numpy() != "bounded")
            assert (ev_[a["formal"].to_numpy() == "unbounded"] == 0).all()
            assert (ev_[a["formal"].to_numpy() == "empty"] == 1).all()
            assert ((ev_ >= 0) & (ev_ <= 1)).all()


def test_clipped_offset_never_enters_update(fc):
    """Swap the issued rule (clip -> always the smallest / largest score): a, n, i, formal and err unchanged,
    issued offsets changed. gamma = 2 drives the levels outside (0, 1), so both kinds of clipping occur."""
    f = fc["lear", "interp"]
    out0, a0 = cf.calibrate(f, REG, gamma=2.0)
    assert a0["clipped"].any() and (a0["formal"] == "unbounded").any() and (a0["formal"] == "empty").any()
    for rule in (lambda i, n: 1, lambda i, n: n):
        out1, a1 = cf.calibrate(f, REG, gamma=2.0, issued_index=rule)
        for c in ("a", "n", "i", "formal", "err"):
            assert a0[c].equals(a1[c]), c
        assert not a0["offset"].equals(a1["offset"])


def test_sorted_idempotent_and_rebuild(fc, block):
    for (m, s), (out, al) in block.items():
        Q = out[data.QCOLS].to_numpy()
        assert (np.diff(Q, axis=1) >= 0).all() and np.array_equal(np.sort(Q, axis=1), Q)
        assert np.array_equal(np.sort(np.sort(Q, axis=1), axis=1), Q)
        raw = fc[m, s][fc[m, s]["date"] >= cf.EVAL_START].reset_index(drop=True)
        key = pd.DataFrame({"date": raw["date"].to_numpy(), "group": cf.group_key(raw, "block")})
        U = raw[data.QCOLS].to_numpy(float).copy()
        col = {c: j for j, c in enumerate(data.QCOLS)}
        for k, (lo, hi, _) in cf.PAIRS.items():
            for side, c, sign in (("lo", lo, -1.0), ("hi", hi, 1.0)):
                off = al[(al["k"] == k) & (al["side"] == side)][["date", "group", "offset"]]
                o = key.merge(off, on=["date", "group"], how="left", validate="many_to_one")["offset"].to_numpy()
                U[:, col[c]] = raw[c].to_numpy(float) + sign * o
        assert np.array_equal(U[:, 4], raw["q50"].to_numpy())
        assert np.array_equal(np.sort(U, axis=1), Q), (m, s)


# ---------------------------------------------------------------- schema, CLI, report
def test_output_schema(block, tmp_path):
    for (m, s), (out, al) in block.items():
        ro_p, ra_p = CDIR / f"{s}_{m}_aci.parquet", CDIR / f"{s}_{m}_aci_alpha.parquet"
        _need(ro_p, ra_p)
        out.to_parquet(tmp_path / "o.parquet", index=False)
        al.to_parquet(tmp_path / "a.parquet", index=False)
        o, a = pd.read_parquet(tmp_path / "o.parquet"), pd.read_parquet(tmp_path / "a.parquet")
        ro, ra = pd.read_parquet(ro_p), pd.read_parquet(ra_p)
        assert len(o) == 32348 and list(o.columns) == list(ro.columns) and o.dtypes.equals(ro.dtypes), (m, s)
        for c in ("datetime_local", "date", "hour", "quarter", "actual", "point"):
            assert o[c].equals(ro[c]), (m, s, c)
        assert list(a.columns) == list(ra.columns) and a.dtypes.equals(ra.dtypes), (m, s)


def test_cli_aciblock(block, tmp_path, capsys):
    cf.main(["--setting", "native", "--model", "xgb", "--regime", REG, "--src", str(FDIR), "--out", str(tmp_path)])
    rep = capsys.readouterr().out
    assert sorted(p.name for p in tmp_path.glob("*.parquet")) == [f"native_xgb_{REG}.parquet",
                                                                   f"native_xgb_{REG}_alpha.parquet"]
    out, al = block["xgb", "native"]
    assert pd.read_parquet(tmp_path / f"native_xgb_{REG}.parquet").equals(out)
    assert pd.read_parquet(tmp_path / f"native_xgb_{REG}_alpha.parquet").equals(al)
    assert "pair k = 1, per-side miscoverage" in rep and "unbounded_k1" in rep and "PICP60" in rep
    det = [l for l in rep.splitlines() if l.strip().startswith(REG)]
    assert len(det) == 1 + 24                                     # summary row + 24 groups of the side table


def test_side_table_block(block):
    out, al = block["lear", "native"]
    st = cf.side_table(al, out, 1, "block")
    assert st["group"].tolist() == [str(g) for g in range(24)]
    g = cf.group_key(out, "block")
    for gg in (0, 7, 23):
        sel = out[g == gg]
        r = st[st["group"] == str(gg)].iloc[0]
        assert r["iss_lo"] == np.mean(sel["actual"] < sel["q10"]) and r["iss_hi"] == np.mean(sel["actual"] > sel["q90"])
        a = al[(al["k"] == 1) & (al["group"] == gg)]
        assert r["err_lo"] == a.loc[a["side"] == "lo", "err"].mean()


# ---------------------------------------------------------------- evaluate: bootstrap draws, macros, rule
def test_bootstrap_same_draws(block):
    _need(*[CDIR / f"{s}_lear_{r}.parquet" for s in SETTINGS for r in ev.REGIMES],
          *[FDIR / f"{s}_naive.parquet" for s in SETTINGS])
    frames = ev.load_all("lear", FDIR, CDIR)
    fr = {**frames, **{("lear", s, REG): block["lear", s][0] for s in SETTINGS}}
    D = ev.boot_days(337, 3)
    b0 = ev.bootstrap(frames, "lear", 3, 1)
    b1 = ev.bootstrap(fr, "lear", 3, 2, regimes=("raw", "aci", REG), D=D)
    b2 = ev.bootstrap(fr, "lear", 3, 1, regimes=(REG,), D=D)
    pd.testing.assert_frame_equal(b0, b1[b1["regime"] != REG].reset_index(drop=True))     # raw, aci unchanged
    pd.testing.assert_frame_equal(b2, b1[b1["regime"] == REG].reset_index(drop=True))
    with pytest.raises(ValueError):
        ev.bootstrap(frames, "lear", 3, 1, D=ev.boot_days(337, 2))


def _block_inputs(model, p=0.0004, crps_ci=(-0.48, -0.29)):
    met = pd.DataFrame([{"model": model, "setting": s, "layer": lay, "CRPS": c}
                        for s, lay, c in (("interp", "aci", 15.05), ("native", "aci", 14.67),
                                          ("interp", REG, 15.2149), ("native", REG, 14.8251),
                                          ("hourly", "aci", 16.0), ("hourly", REG, 16.2))])
    boot_b = pd.DataFrame([{"regime": REG, "quantity": q, "diff_native_minus_interp": d, "lo5": lo, "hi95": hi}
                           for q, d, lo, hi in (("CRPS", -0.3849, *crps_ci), ("MCB", 0.004, -0.03, 0.04),
                                                ("DSC", 0.38, 0.29, 0.47))])
    return met, boot_b, {"p": p}


@pytest.mark.parametrize("model", MODELS)
def test_block_macros(model):
    M = ev.MODEL_LABEL[model]
    met, boot_b, dm = _block_inputs(model)
    m = ev.block_macros(model, met, boot_b, dm)
    names = [f"gap{q}Block{M}{x}" for q in ("Crps", "Mcb", "Dsc") for x in ("", "Lo", "Hi")]
    names += [f"dmPBlock{M}", f"dCrpsBlockInterp{M}", f"dCrpsBlockNative{M}"]
    assert sorted(m) == sorted(names) and len(m) == 12
    assert m[f"gapCrpsBlock{M}"] == "\\ensuremath{-0.38}" and m[f"gapCrpsBlock{M}Lo"] == "\\ensuremath{-0.48}"
    assert m[f"gapMcbBlock{M}"] == "0.00" and m[f"gapDscBlock{M}Hi"] == "0.47"
    assert m[f"dmPBlock{M}"] == "\\ensuremath{<}0.001"
    assert m[f"dCrpsBlockInterp{M}"] == "0.16" and m[f"dCrpsBlockNative{M}"] == "0.16"
    assert all(re.fullmatch(r"[A-Za-z]+", k) for k in m)
    other = ev.MODEL_LABEL["xgb" if model == "lear" else "lear"]
    assert not any(k.endswith(other) or k.endswith(other + "Lo") or k.endswith(other + "Hi") for k in m)
    with pytest.raises(ValueError):
        ev.block_macros(model, met, boot_b.assign(regime="aci"), dm)


def test_block_rule():
    met, boot_b, dm = _block_inputs("lear")
    r = ev.block_rule(boot_b, dm)
    assert r["holds"] and r["line"] == "rule: gap holds = DM p < 0.05 and lo5..hi95 excludes 0 -> yes"
    assert not ev.block_rule(boot_b, {"p": 0.05})["holds"]                     # p < 0.05 is strict
    _, b2, _ = _block_inputs("lear", crps_ci=(-0.2, 0.01))
    r2 = ev.block_rule(b2, dm)
    assert not r2["excludes"] and r2["line"].endswith("-> no")
    _, b3, _ = _block_inputs("lear", crps_ci=(0.01, 0.2))
    assert ev.block_rule(b3, dm)["holds"]                                      # an interval above 0 excludes 0 too


# ---------------------------------------------------------------- results of the step 5c run
def _block_names(M):
    return sorted([f"gap{q}Block{M}{x}" for q in ("Crps", "Mcb", "Dsc") for x in ("", "Lo", "Hi")]
                  + [f"dmPBlock{M}", f"dCrpsBlockInterp{M}", f"dCrpsBlockNative{M}"])


@reference_outputs
def test_block_results_files():
    """After the step 5c run: the 12 macros of each model run equal the bootstrap.csv and metrics_<model>.csv
    values they are formatted from and the cells of Section 11 of summary_<model>.md, the DM p-value equals an
    independent recomputation from the conformal files, and the rule line follows from the printed inputs."""
    require(*[CDIR / f"{s}_{mo}_{REG}.parquet" for mo in MODELS for s in ("interp", "native")],
            *[RESULTS / f for mo in MODELS for f in (f"summary_{mo}.md", f"metrics_{mo}.csv")],
            RESULTS / "bootstrap.csv")
    p = RESULTS / "numbers.tex"
    _need(p)
    m = ev.read_macros(p)
    done = [mo for mo in MODELS if f"gapCrpsBlock{ev.MODEL_LABEL[mo]}" in m]
    if not done:
        pytest.skip("step 5c not run yet")
    # the 14 step 5d macros are counted by tests/test_retune.py (decision (1) of 9 Nov, handover Section 8 row
    # "XGB monthly re-tuning")
    retune = [k for k in m if re.search(r"Retune(Interp|Native)?XGB(Lo|Hi)?$", k)]
    # the 32 step 5e macros (gapCrpsSymBlockM, widthLocBlockInterpM, ...) are counted and checked by
    # tests/test_diag5e.py and left out here (decision (1) of 10 Nov, handover Section 8 row "Paper decisions of
    # 10 Nov and step 5e scope")
    step5e = [k for k in m if k in ev.DIAG_MACROS]
    assert len(m) - len(retune) - len(step5e) == 78 + 12 * len(done)
    bt = pd.read_csv(RESULTS / "bootstrap.csv", float_precision="round_trip")
    for model in done:
        M = ev.MODEL_LABEL[model]
        names = _block_names(M)
        assert all(k in m for k in names)
        assert sorted(k for k in m if re.search(rf"Block(Interp|Native)?{M}(Lo|Hi)?$", k)
                      and k not in step5e) == names
        sec = (RESULTS / f"summary_{model}.md").read_text().split(f"## 11. Step 5c: ACI-CQR with 24 groups ({REG})")
        assert len(sec) == 2
        s11 = sec[1].split("\n## ")[0]          # Section 11 up to the next heading (Section 12 of step 5d follows)
        for k in names:                                                         # the macro table of Section 11
            assert f"| \\{k} | {m[k]} |" in s11, k
        b = bt[(bt["model"] == model) & (bt["regime"] == REG)].set_index("quantity")
        assert sorted(b.index) == ["CRPS", "DSC", "MCB"]
        for q, Q in (("CRPS", "Crps"), ("MCB", "Mcb"), ("DSC", "Dsc")):
            r = b.loc[q]
            assert m[f"gap{Q}Block{M}"] == ev.fmt(r["diff_native_minus_interp"])
            assert m[f"gap{Q}Block{M}Lo"] == ev.fmt(r["lo5"]) and m[f"gap{Q}Block{M}Hi"] == ev.fmt(r["hi95"])
            row = [l for l in s11.splitlines() if l.startswith(f"| {REG} | {q} |")]
            assert len(row) == 1
            assert row[0].split(" | ")[4:7] == [f"{r['diff_native_minus_interp']:.4f}", f"{r['lo5']:.4f}",
                                                f"{r['hi95']:.4f}"]
        met = pd.read_csv(RESULTS / f"metrics_{model}.csv", float_precision="round_trip")
        crps = lambda s, lay: met[(met["model"] == model) & (met["setting"] == s) & (met["layer"] == lay)]["CRPS"].iloc[0]
        assert m[f"dCrpsBlockInterp{M}"] == ev.fmt(crps("interp", REG) - crps("interp", "aci"))
        assert m[f"dCrpsBlockNative{M}"] == ev.fmt(crps("native", REG) - crps("native", "aci"))
        fb = {s: ev.load_layer(model, s, REG, FDIR, CDIR) for s in ("interp", "native")}
        dm = ev.dm_test((ev.daily_crps(fb["native"]) - ev.daily_crps(fb["interp"])).to_numpy())
        assert m[f"dmPBlock{M}"] == ev.fmt_p(dm["p"]) and dm["T"] == 337
        dmrow = [l for l in s11.splitlines() if l.startswith(f"| {REG} | ")
                 and l.split(" | ")[1] not in ("CRPS", "MCB", "DSC")]
        assert len(dmrow) == 1 and dmrow[0].split(" | ")[4] == f"{dm['p']:.4g}" and dmrow[0].endswith(" | 337 |")
        rule = [l for l in s11.splitlines() if l.startswith("rule: ")]
        holds = dm["p"] < 0.05 and (b.loc["CRPS", "lo5"] > 0 or b.loc["CRPS", "hi95"] < 0)
        assert rule == [f"rule: gap holds = DM p < 0.05 and lo5..hi95 excludes 0 -> {'yes' if holds else 'no'}"]
