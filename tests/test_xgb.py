"""XGB tests (step 6a): shared 15-min hyperparameters, early stopping on the summed curve, determinism,
no look-ahead, native share 0, design, schema."""
import json

import numpy as np
import pytest
import xgboost

from esann27 import data, run
from esann27.models import xgb

from .conftest import DAY0, N_JOBS, reference_hashes
from .test_models import VALUE_COLS, _perturb_from


@pytest.fixture(scope="module")
def xgb_day0(frames, target):
    dn = [data.day_number(DAY0)]
    return {s: run.run("xgb", frames[s], target, dn, n_jobs=N_JOBS, verbose=False) for s in data.SETTINGS}


def test_frozen_hyperparameters():
    for st, settings in xgb.STRUCTURES.items():
        hp = xgb.load_hp(st)
        assert hp["structure"] == st and hp["xgboost_version"] == xgboost.__version__ == "3.4.1"
        assert hp["first_window"] == ["2024-10-02", "2025-09-30"] and hp["val_days"] == ["2025-08-06", "2025-09-30"]
        assert hp["fixed"] == xgb.FIXED and hp["point_objective"] == "reg:squarederror"
        assert len(hp["grid"]["point"]) == 6 and len(hp["grid"]["quantiles"]) == 18
        best = min(hp["grid"]["point"], key=lambda g: g["score_mae_sum"])
        assert {k: best[k] for k in ("max_depth", "eta", "rounds")} == {k: hp["point"][k] for k in ("max_depth", "eta", "rounds")}
        assert list(hp["quantiles"]) == data.QCOLS
        for g in hp["grid"]["point"] + hp["grid"]["quantiles"]:
            assert 1 <= g["rounds"] <= xgb.MAX_ROUNDS
    assert xgb.structure_of(True) == "15min" and xgb.structure_of(False) == "hourly"


@reference_hashes
def test_hyperparameters_shared_native_interp(frames):
    """native and interp: identical first window (also the digest frozen in the JSON), and the tuning code returns
    identical results on the two files (reduced grid)."""
    hp = xgb.load_hp("15min")
    dn, di = xgb.window_digest(frames["native"]), xgb.window_digest(frames["interp"])
    assert dn == di == hp["window_sha256"] == hp["window_sha256_interp"]
    kw = dict(depths=(3,), etas=(0.1,), max_rounds=60, hours=(0, 18), verbose=False)
    a, b = xgb.tune(frames["native"], **kw), xgb.tune(frames["interp"], **kw)
    for r in (a, b):
        r.pop("tuning_wall_s"), r.pop("tuned_on")
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_early_stop_matches_xgboost(frames):
    """early_stop on a recorded curve gives the best iteration of xgboost's own EarlyStopping (patience 50)."""
    split = xgb.tuning_split(data.hour_fit(frames["native"], data.day_number(xgb.TUNE_DAY), 18))
    for tau in (None, 0.9):
        curve = xgb.val_curve(split, 3, 0.1, tau)
        rounds, best, _ = xgb.early_stop(curve)
        Xtr, ytr, Xva, yva = split
        dva = xgboost.DMatrix(Xva, label=yva, nthread=1)

        def metric(p, _):
            r = yva - p.astype(float)
            return "loss", float(np.mean(np.abs(r))) if tau is None else float(np.mean(np.maximum(tau * r, (tau - 1) * r)))
        bst = xgboost.train(dict(xgb._params(3, 0.1, tau), disable_default_eval_metric=1),
                            xgboost.DMatrix(Xtr, label=ytr, nthread=1), num_boost_round=1000, evals=[(dva, "val")],
                            custom_metric=metric, early_stopping_rounds=50, verbose_eval=False)
        assert bst.best_iteration + 1 == rounds, (tau, bst.best_iteration, rounds)
        assert best == curve[rounds - 1]


def test_design_raw_features(frames):
    dn = data.day_number(DAY0)
    for s, p in (("hourly", 17), ("interp", 20), ("native", 20)):
        f = data.hour_fit(frames[s], dn, 18)
        X, y, Xte = xgb.design(f)
        assert X.shape[1] == p and Xte.shape[1] == p, s
        assert np.array_equal(y, f.train[data.PRICE].to_numpy())                   # no VST
        assert np.array_equal(X[:, 0], f.train["price_lag_d1"].to_numpy())        # no standardization
    X, _, _ = xgb.design(data.hour_fit(frames["native"], dn, 18), qdum=False)
    assert X.shape[1] == 17


def test_native_share_zero_identical(xgb_day0):
    fi, fn = xgb_day0["interp"][0], xgb_day0["native"][0]
    assert len(fi) == 96
    for c in VALUE_COLS:
        assert np.array_equal(fi[c].to_numpy(), fn[c].to_numpy()), c


def test_determinism_n_jobs(frames, target, xgb_day0):
    """One hourly day, n_jobs = 1 vs 8: byte-identical forecasts."""
    dn = [data.day_number(DAY0)]
    a, la, _ = run.run("xgb", frames["hourly"], target, dn, n_jobs=1, verbose=False)
    b, lb, _ = run.run("xgb", frames["hourly"], target, dn, n_jobs=8, verbose=False)
    c = xgb_day0["hourly"][0]
    for col in VALUE_COLS:
        assert np.array_equal(a[col].to_numpy(), b[col].to_numpy()), col
        assert np.array_equal(a[col].to_numpy(), c[col].to_numpy()), col


@pytest.mark.parametrize("setting", ["native", "hourly"])
def test_no_lookahead(frames, setting):
    df = frames[setting]
    dn = data.day_number("2026-03-02")
    pert = _perturb_from(df, dn)
    for h in (0, 8, 18):
        a, b = xgb.fit_predict(data.hour_fit(df, dn, h)), xgb.fit_predict(data.hour_fit(pert, dn, h))
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]), h
    # the perturbation reaches the model when day D is inside the window (sanity of the test)
    a, b = xgb.fit_predict(data.hour_fit(df, dn + 1, 18)), xgb.fit_predict(data.hour_fit(pert, dn + 1, 18))
    assert not np.array_equal(a[1], b[1])


def test_output_schema_sorted(xgb_day0):
    cols = ["datetime_local", "date", "hour", "quarter", "actual", "point"] + data.QCOLS
    for s, (fc, log, _) in xgb_day0.items():
        assert list(fc.columns) == cols and len(fc) == 96 and (fc["date"] == DAY0).all()
        assert not fc[VALUE_COLS + ["actual"]].isna().any().any()
        assert (np.diff(fc[data.QCOLS].to_numpy(), axis=1) >= 0).all()
        assert len(log) == 24 and {"t_point_s", "t_quant_s", "wall_s", "n_train", "p"} <= set(log.columns)
        assert not log[[f"fallback_{c}" for c in data.QCOLS]].to_numpy().any()
        if s == "hourly":
            assert (fc.groupby("hour")[VALUE_COLS].nunique() == 1).all().all()
