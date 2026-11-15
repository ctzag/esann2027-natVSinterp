"""Step 5d tests: XGB with monthly re-tuning, model xgbm (handover Section 8 row "XGB monthly re-tuning" with its
implementation decisions of 9 Nov, step 5d specification B3 and B4).

The re-tunes of 1 Oct and 1 Nov 2025 and a few October forecasts are computed in memory, because the step 5d script
runs pytest before it writes forecasts_retune/. The tests on the files of the run (schema, the 12 JSON files per
setting, October bit equality over the whole month, the results of evaluate) skip until those files exist."""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from esann27 import conformal as cf
from esann27 import data, retune, run
from esann27 import evaluate as ev
from esann27.models import xgb

from .conftest import N_JOBS, reference_outputs, require
from .test_models import VALUE_COLS, _perturb_from

ROOT = Path(__file__).resolve().parent.parent
FDIR, CDIR, RDIR, RESULTS = ROOT / "forecasts", ROOT / "conformal", ROOT / "forecasts_retune", ROOT / "results"
MANIFEST = Path(__file__).resolve().parent / "data" / "outputs_sha256_2026-11-09.txt"
SETS = xgb.RETUNE_SETTINGS
OCT1, NOV1 = "2025-10-01", "2025-11-01"
FROZEN_ONLY = ("tuning_wall_s", "shared_by", "window_sha256_interp")    # keys of the frozen file not in a re-tune
PER_SETTING = ("tuned_on", "setting")                                    # differ between the settings by construction
OCT_DAYS = ("2025-10-01", "2025-10-26", "2025-10-31")                    # 26 Oct: DST fall-back, duplicated hour averaged (96 quarters, Section 5)
NEW_CONFORMAL = sorted(f"{s}_xgbm_{r}{x}.parquet" for s in SETS for r in ("aci", "aciblock") for x in ("", "_alpha"))


def _norm(x):
    return json.loads(json.dumps(x))


def _strip(hp, keys):
    return {k: v for k, v in hp.items() if k not in keys}


def _selected(hp):
    """(depth, eta, rounds) of the 10 models of one hyperparameter dictionary."""
    d = hp["point"]["max_depth"]
    out = {"point": (d, hp["point"]["eta"], hp["point"]["rounds"])}
    out.update({c: (d, hp["quantiles"][c]["eta"], hp["quantiles"][c]["rounds"]) for c in data.QCOLS})
    return out


def _bits_equal(a, b):
    """Same columns, dtypes and rows, float columns equal in their bit patterns."""
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


def _need(*paths):
    miss = [str(p.relative_to(ROOT)) for p in paths if not p.exists()]
    if miss:
        pytest.skip(f"files missing: {miss}")


@pytest.fixture(scope="module")
def rt(frames):
    """Full re-tunes (step 6a procedure) of 1 Oct and 1 Nov 2025 in both settings, n_jobs = N_JOBS."""
    return {(s, r): xgb.retune(frames[s], s, r, n_jobs=N_JOBS)[0] for s in SETS for r in (OCT1, NOV1)}


def _write_hp(hp_dir, hp):
    hp_dir.mkdir(parents=True, exist_ok=True)
    xgb.retune_path(hp_dir, hp["setting"], hp["retune_date"]).write_text(xgb.retune_json_text(hp))
    return hp_dir


# ---------------------------------------------------------------- schedule and window dates
def test_schedule():
    """Twelve re-tune dates, the first day of each test month; day D uses the re-tune of the first day of its month."""
    assert xgb.RETUNE_DATES == ("2025-10-01", "2025-11-01", "2025-12-01") + tuple(f"2026-{m:02d}-01" for m in range(1, 10))
    days = data.test_days()
    assert len(days) == 365
    for dn in days:
        assert xgb.retune_date_of(dn) == data.day_str(dn)[:8] + "01"
    with pytest.raises(ValueError):
        xgb.retune_date_of(data.day_number("2025-09-30"))
    assert xgb.retune_path("x", "native", NOV1) == Path("x") / "native_2025-11-01.json"


def test_window_dates(frames):
    """Re-tune at R: the hour-fit window is R-364..R-1, the tuning split trains on R-364..R-57 and validates on
    R-56..R-1."""
    for r in ("2025-11-01", "2026-03-01", "2026-09-01"):
        dn = data.day_number(r)
        f = data.hour_fit(frames["native"], dn, 18)
        days = f.train["day"].to_numpy()
        assert days.min() == dn - 364 and days.max() == dn - 1
        cut = dn - xgb.VAL_DAYS
        assert set(days[days < cut]) == set(range(dn - 364, dn - 56))
        assert set(days[days >= cut]) == set(range(dn - 56, dn))
        Xtr, ytr, Xva, yva = xgb.tuning_split(f)
        assert len(ytr) == 4 * 308 and len(yva) == 4 * 56 and Xtr.shape[1] == Xva.shape[1] == 20


def test_retune_never_touches_R_or_later(frames):
    """Prices from R on and regressors after R changed: the re-tune at R is unchanged (reduced grid). Changing day
    R-1 changes it (sanity: the last validation day is inside)."""
    r = "2026-03-01"
    dn = data.day_number(r)
    df = frames["native"]
    kw = dict(depths=(3,), etas=(0.1,), max_rounds=60, hours=(0, 18), verbose=False, day=r)
    a, b, c = (xgb.tune(x, **kw) for x in (df, _perturb_from(df, dn), _perturb_from(df, dn - 1)))
    for x in (a, b, c):
        x.pop("tuning_wall_s")
    assert a == b
    assert c["grid"] != a["grid"]
    assert a["first_window"] == ["2025-03-02", "2026-02-28"] and a["val_days"] == ["2026-01-04", "2026-02-28"]


# ---------------------------------------------------------------- re-tunes of 1 Oct and 1 Nov 2025
@reference_outputs
def test_retune_oct_equals_frozen(rt):
    """The 1 Oct 2025 re-tune uses the first window: its hyperparameters (and grid, dates, digest, features) equal
    hyperparameters_15min.json, in both settings."""
    fz = xgb.load_hp("15min")
    for s in SETS:
        hp = rt[s, OCT1]
        assert set(hp) == (set(fz) - set(FROZEN_ONLY)) | {"setting", "retune_date"}
        assert hp["setting"] == s and hp["retune_date"] == OCT1 and hp["tuned_on"] == data.SETTINGS[s]
        assert _norm(_strip(hp, PER_SETTING + ("retune_date",))) == _norm(_strip(fz, FROZEN_ONLY + ("tuned_on",)))
        assert _norm(hp["point"]) == fz["point"] and _norm(hp["quantiles"]) == fz["quantiles"]
        assert hp["train_days"] == ["2024-10-02", "2025-08-05"] and hp["val_days"] == ["2025-08-06", "2025-09-30"]


def test_interp_native_before_switch(rt, frames):
    """1 Oct 2025: the re-tune is identical across the two settings. 1 Nov 2025 (validation 6 Sep - 31 Oct 2025,
    31 native days in the native file): training rows identical, validation prices differ in October only; whether
    the re-tune differs is recorded (printed, and in Section 12 of summary_xgb.md)."""
    assert _strip(rt["interp", OCT1], PER_SETTING) == _strip(rt["native", OCT1], PER_SETTING)
    a, b = rt["interp", NOV1], rt["native", NOV1]
    for hp in (a, b):
        assert hp["first_window"] == ["2024-11-02", "2025-10-31"]
        assert hp["train_days"] == ["2024-11-02", "2025-09-05"] and hp["val_days"] == ["2025-09-06", "2025-10-31"]
    dn, go = data.day_number(NOV1), data.day_number("2025-10-01")
    cols = xgb.columns(True) + [data.PRICE]
    for h in (0, 18):
        fi, fn = data.hour_fit(frames["interp"], dn, h).train, data.hour_fit(frames["native"], dn, h).train
        assert np.array_equal(fi["day"].to_numpy(), fn["day"].to_numpy())
        pre = (fi["day"] < go).to_numpy()
        assert np.array_equal(fi.loc[pre, cols].to_numpy(float), fn.loc[pre, cols].to_numpy(float))
        assert not np.array_equal(fi.loc[~pre, data.PRICE].to_numpy(), fn.loc[~pre, data.PRICE].to_numpy())
    full = _strip(a, PER_SETTING) != _strip(b, PER_SETTING)
    sel = _selected(a) != _selected(b)
    print(f"\n1 Nov 2025 re-tune, interp vs native: JSON {'differs' if full else 'identical'}, selected "
          f"configuration (depth, eta, rounds of the 10 models) {'differs' if sel else 'identical'}")


def test_determinism_n_jobs(rt, frames, target, tmp_path):
    """One re-tune date (1 Nov 2025, native) rerun with another n_jobs: byte-identical JSON; its forecasts of
    1 Nov 2025 with n_jobs 1 and N_JOBS: byte-identical parquet files."""
    s, r = "native", NOV1
    other = 4 if N_JOBS != 4 else 2
    hp2, _ = xgb.retune(frames[s], s, r, n_jobs=other)
    assert xgb.retune_json_text(hp2) == xgb.retune_json_text(rt[s, r])
    hp_dir = _write_hp(tmp_path / "hyperparameters", hp2)
    opts = {"setting": s, "hp_dir": str(hp_dir)}
    blobs = []
    for nj in (1, N_JOBS):
        fc, log, _ = run.run("xgbm", frames[s], target, [data.day_number(r)], n_jobs=nj, opts=opts, verbose=False)
        assert len(fc) == 96 and (log["retune_date"] == r).all()
        p = tmp_path / f"fc_{nj}.parquet"
        fc.to_parquet(p, index=False)
        blobs.append(p.read_bytes())
    assert blobs[0] == blobs[1]


def test_october_equals_xgb_in_memory(rt, frames, target, tmp_path):
    """xgbm on 1, 26 (DST) and 31 Oct 2025 with the in-memory 1 Oct re-tune equals forecasts/<setting>_xgb.parquet
    bit for bit (every column), interp and native."""
    for s in SETS:
        _need(FDIR / f"{s}_xgb.parquet")
        hp_dir = _write_hp(tmp_path / s, rt[s, OCT1])
        days = [data.day_number(d) for d in OCT_DAYS]
        fc, log, _ = run.run("xgbm", frames[s], target, days, n_jobs=N_JOBS,
                             opts={"setting": s, "hp_dir": str(hp_dir)}, verbose=False)
        ref = pd.read_parquet(FDIR / f"{s}_xgb.parquet")
        ref = ref[ref["date"].isin(OCT_DAYS)].reset_index(drop=True)
        fc.to_parquet(tmp_path / f"{s}.parquet", index=False)              # as written by esann27.run
        fc = pd.read_parquet(tmp_path / f"{s}.parquet")
        assert len(fc) == 3 * 96 and (log["retune_date"] == OCT1).all()
        assert _bits_equal(fc, ref), s


def test_xgbm_needs_its_own_files(frames, tmp_path):
    """fit_predict_retuned reads <hp_dir>/<setting>_<R>.json of the test day's month and checks its labels."""
    f = data.hour_fit(frames["native"], data.day_number("2025-11-15"), 18)
    with pytest.raises(FileNotFoundError):
        xgb.fit_predict_retuned(f, setting="native", hp_dir=str(tmp_path))
    hp = dict(xgb.load_hp("15min"), setting="interp", retune_date=NOV1)
    (tmp_path / "native_2025-11-01.json").write_text(xgb.retune_json_text(hp))
    with pytest.raises(ValueError):
        xgb.fit_predict_retuned(f, setting="native", hp_dir=str(tmp_path))
    with pytest.raises(ValueError):
        xgb.fit_predict_retuned(data.hour_fit(frames["hourly"], f.day, 18), setting="native", hp_dir=str(tmp_path))


# ---------------------------------------------------------------- write protection, log
def test_forecasts_dir_refused(tmp_path):
    with pytest.raises(SystemExit):
        retune.main(["--setting", "native", "--out", str(FDIR), "--stage", "tune", "--dates", OCT1])
    with pytest.raises(SystemExit):
        retune.main(["--setting", "native", "--out", str(FDIR / "sub"), "--stage", "forecast"])
    with pytest.raises(SystemExit):
        run.main(["--setting", "native", "--model", "xgbm", "--out", str(FDIR), "--start", OCT1, "--end", "2025-10-02"])
    with pytest.raises(SystemExit):
        run.main(["--setting", "hourly", "--model", "xgbm", "--out", str(tmp_path), "--start", OCT1,
                  "--end", "2025-10-02"])
    with pytest.raises(SystemExit):
        retune.main(["--setting", "native", "--out", str(tmp_path), "--stage", "tune", "--dates", "2025-10-02"])
    assert not any(tmp_path.iterdir())


def test_retune_log(rt, tmp_path):
    rows = retune.log_rows(rt["native", NOV1], 12.5)
    assert [r["model"] for r in rows] == ["point"] + data.QCOLS and all(r["wall_s"] == 12.5 for r in rows)
    sel = _selected(rt["native", NOV1])
    for r in rows:
        assert (r["max_depth"], r["eta"], r["rounds"]) == sel[r["model"]]
    p = tmp_path / "retune_log.csv"
    retune.update_log(p, rows + retune.log_rows(rt["interp", OCT1], 3.0))
    log = retune.update_log(p, retune.log_rows(rt["native", NOV1], 7.0))          # replaces native 1 Nov
    assert len(log) == 20 and list(log.columns) == retune.LOG_COLS
    assert log["setting"].tolist() == ["interp"] * 10 + ["native"] * 10
    assert (log.loc[log["setting"] == "native", "wall_s"] == 7.0).all()
    back = pd.read_csv(p, float_precision="round_trip", dtype={"retune_date": str})
    assert back["val_loss_sum"].tolist() == log["val_loss_sum"].tolist()


# ---------------------------------------------------------------- existing outputs (B3 last row)
def _hash_tree():
    out = {}
    for d in (FDIR, CDIR):
        for p in sorted(d.rglob("*")):
            if p.is_file() and p.name != ".DS_Store":
                out[p.relative_to(ROOT).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


@reference_outputs
def test_existing_outputs_keep_sha256():
    """Every file of forecasts/ and conformal/ in the manifest taken on 9 Nov before step 5d keeps its sha256; the
    only new files allowed are the 8 xgbm conformal files and the 48 step 5e conformal files (decision (1) of
    10 Nov, handover Section 8 row "Paper decisions of 10 Nov and step 5e scope")."""
    _need(MANIFEST, FDIR, CDIR)
    ref = dict(reversed(l.split("  ", 1)) for l in MANIFEST.read_text().splitlines() if l)
    now = _hash_tree()
    bad = [p for p in ref if now.get(p) != ref[p]]
    assert not bad, bad
    new = sorted(set(now) - set(ref))
    step5e = {f"conformal/{s}_{m}_{r}{x}.parquet" for s in ("hourly", "interp", "native") for m in ("lear", "xgb")
              for r in cf.DIAG_REGIMES for x in cf.regime_files(r)}
    assert set(new) <= {f"conformal/{n}" for n in NEW_CONFORMAL} | step5e, new


def test_existing_xgb_aci_regenerated(tmp_path, capsys):
    """xgb aci and aciblock regenerated from forecasts/ by the CLI equal conformal/ byte for byte (interp, native)."""
    expected = sorted(f"{s}_xgb_{r}{x}.parquet" for s in SETS for r in ("aci", "aciblock") for x in ("", "_alpha"))
    _need(*[CDIR / n for n in expected], *[FDIR / f"{s}_xgb.parquet" for s in SETS])
    for s in SETS:
        for r in ("aci", "aciblock"):
            cf.main(["--setting", s, "--model", "xgb", "--regime", r, "--src", str(FDIR), "--out", str(tmp_path)])
    capsys.readouterr()
    assert sorted(p.name for p in tmp_path.glob("*.parquet")) == expected
    assert all(cf.compare_files(tmp_path / n, CDIR / n) == "bytes" for n in expected)


# ---------------------------------------------------------------- evaluate: macros, rule, merge (B4)
def _retune_inputs(p_raw=0.004, p_aci=0.03, ci_raw=(-0.6, -0.3), ci_aci=(-0.5, -0.2)):
    met_r = pd.DataFrame([{"model": "xgbm", "setting": s, "layer": "raw", "CRPS": c}
                          for s, c in (("interp", 15.2049), ("native", 14.7951))])
    met_x = pd.DataFrame([{"model": "xgb", "setting": s, "layer": "raw", "CRPS": c}
                          for s, c in (("interp", 15.33), ("native", 14.83))])
    rows = []
    for lay, d, (lo, hi) in (("raw", -0.4098, ci_raw), ("aci", -0.3349, ci_aci), ("aciblock", -0.0951, (-0.19, 0.004))):
        for q, dd, l2, h2 in (("CRPS", d, lo, hi), ("MCB", -0.1, -0.2, 0.0), ("DSC", 0.3, 0.2, 0.4)):
            rows.append({"regime": lay, "quantity": q, "diff_native_minus_interp": dd, "lo5": l2, "hi95": h2})
    dm = {"raw": {"p": p_raw}, "aci": {"p": p_aci}, "aciblock": {"p": 0.2}}
    return met_r, met_x, pd.DataFrame(rows), dm


def test_retune_macros():
    met_r, met_x, boot_r, dm = _retune_inputs()
    m = ev.retune_macros(met_r, met_x, boot_r, dm)
    assert sorted(m) == sorted(ev.RETUNE_MACROS) and len(m) == 14
    assert m["gapCrpsRawRetuneXGB"] == "\\ensuremath{-0.41}" and m["gapCrpsRawRetuneXGBLo"] == "\\ensuremath{-0.60}"
    assert m["gapCrpsBlockRetuneXGBHi"] == "0.00" and m["gapCrpsAciRetuneXGB"] == "\\ensuremath{-0.33}"
    assert m["dmPRawRetuneXGB"] == "0.004" and m["dmPBlockRetuneXGB"] == "0.200"
    assert m["dCrpsRetuneInterpXGB"] == "\\ensuremath{-0.13}" and m["dCrpsRetuneNativeXGB"] == "\\ensuremath{-0.03}"
    assert all(re.fullmatch(r"[A-Za-z]+", k) for k in m)
    with pytest.raises(ValueError):
        ev.retune_macros(met_r, met_x, boot_r[boot_r["regime"] != "aci"], dm)


def test_retune_rule():
    met_r, met_x, boot_r, dm = _retune_inputs()
    r = ev.retune_rule(boot_r, dm)
    assert r["holds"] and r["line"] == ("rule: robust = DM p < 0.05 and lo5..hi95 excludes 0, raw and ACI-CQR "
                                        "-> yes")
    assert not ev.retune_rule(boot_r, {**dm, "aci": {"p": 0.05}})["holds"]                    # strict, ACI-CQR too
    _, _, b2, _ = _retune_inputs(ci_raw=(-0.6, 0.01))
    r2 = ev.retune_rule(b2, dm)
    assert not r2["holds"] and r2["line"].endswith("-> no") and not r2["parts"]["raw"]["excludes"]
    _, _, b3, _ = _retune_inputs(ci_aci=(0.01, 0.2))
    assert ev.retune_rule(b3, dm)["holds"]                                                     # above 0 excludes 0


def test_write_macros_keeps_every_macro(tmp_path):
    """B4.4: a later run that writes only its own macros keeps the others, and writing the same values again leaves
    numbers.tex byte-identical."""
    p = tmp_path / "numbers.tex"
    lear = {"gapCrpsRawLEAR": "\\ensuremath{-0.39}", "dmPRawLEAR": "\\ensuremath{<}0.001"}
    ret = {k: "0.01" for k in ev.RETUNE_MACROS}
    ev.write_macros(p, lear)
    ev.write_macros(p, ret)
    first = p.read_bytes()
    ev.write_macros(p, lear)
    assert p.read_bytes() == first and len(ev.read_macros(p)) == 16


# ---------------------------------------------------------------- files of the step 5d run (skip before it)
def test_schema_files():
    """35,036 rows, columns and dtypes as forecasts/<setting>_xgb.parquet, the same datetime_local order and actuals,
    quantiles sorted; the log has the columns of xgb plus retune_date, the day-wall file 365 days."""
    for s in SETS:
        _need(RDIR / f"{s}_xgbm.parquet", FDIR / f"{s}_xgb.parquet")
        a, b = pd.read_parquet(RDIR / f"{s}_xgbm.parquet"), pd.read_parquet(FDIR / f"{s}_xgb.parquet")
        assert len(a) == 35036 and list(a.columns) == list(b.columns) and a.dtypes.equals(b.dtypes), s
        for c in ("datetime_local", "date", "hour", "quarter", "actual"):
            assert a[c].equals(b[c]), (s, c)
        Q = a[data.QCOLS].to_numpy(float)
        assert np.isfinite(Q).all() and (np.diff(Q, axis=1) >= 0).all() and np.isfinite(a["point"]).all()
        la, lb = pd.read_parquet(RDIR / f"{s}_xgbm_log.parquet"), pd.read_parquet(FDIR / f"{s}_xgb_log.parquet")
        assert list(la.columns) == list(lb.columns) + ["retune_date"] and len(la) == len(lb)
        assert (la["retune_date"] == la["date"].str[:8] + "01").all()
        assert not la[[f"fallback_{c}" for c in data.QCOLS]].to_numpy().any()
        assert len(pd.read_parquet(RDIR / f"{s}_xgbm_daywall.parquet")) == 365


def test_october_equals_xgb_files():
    """All quarters of 1-31 Oct 2025 (2,976 = 31 x 96): xgbm = xgb bit for bit, every column, interp and native."""
    for s in SETS:
        _need(RDIR / f"{s}_xgbm.parquet", FDIR / f"{s}_xgb.parquet")
        a, b = pd.read_parquet(RDIR / f"{s}_xgbm.parquet"), pd.read_parquet(FDIR / f"{s}_xgb.parquet")
        a, b = (x[x["date"].str[:7] == "2025-10"].reset_index(drop=True) for x in (a, b))
        assert len(a) == 2976 and _bits_equal(a, b), s


def test_schedule_files(request):
    """12 JSON files per setting, one per re-tune date, labels and window dates right, 1 Oct and 1 Nov equal to the
    in-memory re-tunes byte for byte, retune_log.csv consistent with the JSON files."""
    hp_dir = RDIR / "hyperparameters"
    _need(hp_dir, RDIR / "retune_log.csv")
    rt = request.getfixturevalue("rt")                  # only once the files exist (the re-tunes cost about 1 min)
    keys = (set(xgb.load_hp("15min")) - set(FROZEN_ONLY)) | {"setting", "retune_date"}
    log = pd.read_csv(RDIR / "retune_log.csv", float_precision="round_trip", dtype={"retune_date": str})
    assert list(log.columns) == retune.LOG_COLS and len(log) == len(SETS) * 12 * 10
    for s in SETS:
        assert sorted(p.name for p in hp_dir.glob(f"{s}_*.json")) == [f"{s}_{r}.json" for r in xgb.RETUNE_DATES]
        for r in xgb.RETUNE_DATES:
            p = xgb.retune_path(hp_dir, s, r)
            hp = json.loads(p.read_text())
            dn = data.day_number(r)
            assert set(hp) == keys and hp["setting"] == s and hp["retune_date"] == r
            assert hp["tuned_on"] == data.SETTINGS[s] and hp["structure"] == "15min"
            assert hp["first_window"] == [data.day_str(dn - 364), data.day_str(dn - 1)]
            assert hp["train_days"] == [data.day_str(dn - 364), data.day_str(dn - 57)]
            assert hp["val_days"] == [data.day_str(dn - 56), data.day_str(dn - 1)]
            g = log[(log["setting"] == s) & (log["retune_date"] == r)].set_index("model")
            assert list(g.index) == ["point"] + data.QCOLS and g["wall_s"].nunique() == 1
            for m, cfg in _selected(hp).items():
                assert (g.loc[m, "max_depth"], g.loc[m, "eta"], g.loc[m, "rounds"]) == cfg, (s, r, m)
            if (s, r) in rt:
                assert p.read_text() == xgb.retune_json_text(rt[s, r]), (s, r)


@reference_outputs
def test_retune_results_files():
    """After the step 5d run: numbers.tex has the 116 macros, the 14 step 5d macros equal the CSV values they are
    formatted from and the macro table of Section 12, the DM p-values equal a recomputation from the files, the rule
    line follows from the printed inputs, the hyperparameter CSV is retune_log.csv without wall times."""
    require(RDIR / "retune_log.csv", *[RDIR / f"{s}_xgbm.parquet" for s in SETS],
            *[CDIR / f"{s}_xgbm_{l}.parquet" for s in SETS for l in ev.RETUNE_LAYERS if l != "raw"],
            *[RESULTS / f for f in ("metrics_xgb.csv", "monthly.csv", "summary_xgb.md", "summary_lear.md")])
    p = RESULTS / "numbers.tex"
    _need(p)
    m = ev.read_macros(p)
    if "gapCrpsRawRetuneXGB" not in m:
        pytest.skip("step 5d not run yet")
    # the 32 step 5e macros are counted by tests/test_diag5e.py (decision (1) of 10 Nov, handover Section 8 row
    # "Paper decisions of 10 Nov and step 5e scope")
    step5e = [k for k in m if k in ev.DIAG_MACROS]
    assert len(m) - len(step5e) == 116 and all(k in m for k in ev.RETUNE_MACROS)
    assert sorted(k for k in m if "Retune" in k) == sorted(ev.RETUNE_MACROS)
    for f in ev.RETUNE_FILES:
        _need(RESULTS / f)
    bt = pd.read_csv(RESULTS / "retune_xgb_bootstrap.csv", float_precision="round_trip")
    met = pd.read_csv(RESULTS / "retune_xgb_metrics.csv", float_precision="round_trip")
    mx = pd.read_csv(RESULTS / "metrics_xgb.csv", float_precision="round_trip")
    assert sorted(zip(bt["regime"], bt["quantity"])) == sorted((l, q) for l in ev.RETUNE_LAYERS
                                                                for q in ("CRPS", "MCB", "DSC"))
    assert len(met) == 6 and (met["model"] == "xgbm").all()
    assert np.allclose(met["CRPS"], met["MCB"] - met["DSC"] + met["UNC"], atol=1e-9)
    s12 = (RESULTS / "summary_xgb.md").read_text().split(ev.SEC12)
    assert len(s12) == 2 and ev.SEC12 not in (RESULTS / "summary_lear.md").read_text()
    s12 = s12[1]
    for k in ev.RETUNE_MACROS:
        assert f"| \\{k} | {m[k]} |" in s12, k
    dm = {}
    for lay in ev.RETUNE_LAYERS:
        f = {s: ev.load_layer("xgbm", s, lay, RDIR, CDIR) for s in SETS}
        dm[lay] = ev.dm_test((ev.daily_crps(f["native"]) - ev.daily_crps(f["interp"])).to_numpy())
        assert dm[lay]["T"] == 337 and m[f"dmP{ev.RETUNE_NAME[lay]}RetuneXGB"] == ev.fmt_p(dm[lay]["p"])
        r = bt[(bt["regime"] == lay) & (bt["quantity"] == "CRPS")].iloc[0]
        base = f"gapCrps{ev.RETUNE_NAME[lay]}RetuneXGB"
        assert m[base] == ev.fmt(r["diff_native_minus_interp"])
        assert m[base + "Lo"] == ev.fmt(r["lo5"]) and m[base + "Hi"] == ev.fmt(r["hi95"])
        crps = lambda s, l2: met[(met["setting"] == s) & (met["layer"] == l2)]["CRPS"].iloc[0]
        assert np.isclose(r["diff_native_minus_interp"], crps("native", lay) - crps("interp", lay), atol=1e-12)
    cx = lambda s: mx[(mx["model"] == "xgb") & (mx["setting"] == s) & (mx["layer"] == "raw")]["CRPS"].iloc[0]
    cm = lambda s: met[(met["setting"] == s) & (met["layer"] == "raw")]["CRPS"].iloc[0]
    assert m["dCrpsRetuneInterpXGB"] == ev.fmt(cm("interp") - cx("interp"))
    assert m["dCrpsRetuneNativeXGB"] == ev.fmt(cm("native") - cx("native"))
    holds = all(dm[l]["p"] < 0.05 and (bt[(bt["regime"] == l) & (bt["quantity"] == "CRPS")]["lo5"].iloc[0] > 0 or
                                       bt[(bt["regime"] == l) & (bt["quantity"] == "CRPS")]["hi95"].iloc[0] < 0)
                for l in ("raw", "aci"))
    rule = [l for l in s12.splitlines() if l.startswith("rule: ")]
    assert rule == ["rule: robust = DM p < 0.05 and lo5..hi95 excludes 0, raw and ACI-CQR -> "
                    + ("yes" if holds else "no")]
    log = pd.read_csv(RDIR / "retune_log.csv", float_precision="round_trip", dtype={"retune_date": str})
    hp = pd.read_csv(RESULTS / "retune_xgb_hyperparameters.csv", float_precision="round_trip", dtype={"retune_date": str})
    pd.testing.assert_frame_equal(hp, log.drop(columns=["wall_s"]))
    mon = pd.read_csv(RESULTS / "retune_xgb_monthly.csv", float_precision="round_trip")
    assert sorted(mon["model"].unique()) == ["xgb", "xgbm"] and (mon["layer"] == "raw").all()
    assert len(mon) == 2 * len(SETS) * 12
    old = pd.read_csv(RESULTS / "monthly.csv", float_precision="round_trip")
    ref = old[(old["model"] == "xgb") & (old["layer"] == "raw") & old["setting"].isin(SETS)]
    mx_ = mon[mon["model"] == "xgb"].merge(ref, on=["setting", "month"], suffixes=("", "_ref"))
    assert len(mx_) == 24 and (mx_["CRPS"] == mx_["CRPS_ref"]).all()
