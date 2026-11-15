"""Model-stage tests: look-ahead, native share 0, lambda mapping, determinism, sorting, output schema."""
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scipy.optimize import linprog
from sklearn.linear_model import QuantileRegressor

from esann27 import data, run
from esann27.models import lear, naive

from .conftest import DAY0

VALUE_COLS = ["point"] + data.QCOLS


def _perturb_from(df, dn):
    """Copy of the frame with every price from day D on changed, and every column of the rows after D."""
    p = df.copy()
    p.attrs = dict(df.attrs)
    rng = np.random.default_rng(1)
    on = (p["day"] >= dn).to_numpy()
    p.loc[on, data.PRICE] = p.loc[on, data.PRICE] * 1.7 + 40 + 30 * rng.random(on.sum())
    after = (p["day"] > dn).to_numpy()
    for c in data.CONTINUOUS:
        p.loc[after, c] = p.loc[after, c] * 1.3 + 11
    return p


@pytest.mark.parametrize("setting", ["native", "hourly"])
def test_no_lookahead_model_stage(frames, setting):
    df = frames[setting]
    dn = data.day_number("2026-03-02")
    pert = _perturb_from(df, dn)
    assert not np.array_equal(pert[data.PRICE].to_numpy(), df[data.PRICE].to_numpy())
    for h in data.hours_of_day(df, dn):
        a, b = naive.fit_predict(data.hour_fit(df, dn, h)), naive.fit_predict(data.hour_fit(pert, dn, h))
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    a, b = lear.fit_predict(data.hour_fit(df, dn, 18)), lear.fit_predict(data.hour_fit(pert, dn, 18))
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


@pytest.mark.parametrize("model", ["naive", "lear"])
def test_native_share_zero_identical(day0_runs, model):
    """1 Oct 2025: windows and test-row features coincide in interp and native, so the forecasts must too."""
    fi, li, _ = day0_runs["interp", model]
    fn, ln, _ = day0_runs["native", model]
    assert len(fi) == 96
    for c in VALUE_COLS:
        assert np.array_equal(fi[c].to_numpy(), fn[c].to_numpy()), c


def _rq_lasso_lp(X, y, tau, lam):
    """Direct LP of sum rho_tau(y - b0 - X b) + (lam/2) sum |b_j| (quantreg rq.fit.lasso objective)."""
    n, p = X.shape
    # variables: b0 (free), b+ (p), b- (p), u (n), v (n)
    c = np.concatenate([[0.0], np.full(2 * p, lam / 2), np.full(n, tau), np.full(n, 1 - tau)])
    A = sparse.hstack([sparse.csc_array(np.ones((n, 1))), sparse.csc_array(X), sparse.csc_array(-X),
                       sparse.eye_array(n, format="csc"), -sparse.eye_array(n, format="csc")], format="csc")
    bounds = [(None, None)] + [(0, None)] * (2 * p + 2 * n)
    r = linprog(c, A_eq=A, b_eq=y, bounds=bounds, method="highs")
    assert r.success
    return r.fun


def test_lambda_mapping(frames):
    f = data.hour_fit(frames["native"], data.day_number(DAY0), 18)
    X, y, _, _, _ = lear.design(f)
    n, p = X.shape
    assert (n, p) == (1456, 20)
    for tau in (0.1, 0.5, 0.9):
        for lam in (0.1, 10.0, 1000.0):
            m = QuantileRegressor(quantile=tau, alpha=lam / (2 * n), solver="highs").fit(X, y)
            obj = lear.pinball_sum(y - m.predict(X), tau) + lam / 2 * np.sum(np.abs(m.coef_))
            ref = _rq_lasso_lp(X, y, tau, lam)
            assert abs(obj - ref) / abs(ref) < 1e-9, (tau, lam, obj, ref)


def test_design_dimensions(frames):
    dn = data.day_number(DAY0)
    for s, p in (("hourly", 17), ("interp", 20), ("native", 20)):
        X, y, Xte, _, _ = lear.design(data.hour_fit(frames[s], dn, 18))
        assert X.shape[1] == p and Xte.shape[1] == p, s
        assert not np.isnan(X).any() and not np.isnan(Xte).any()
    X, *_ = lear.design(data.hour_fit(frames["native"], dn, 18), qdum=False)
    assert X.shape[1] == 17


def test_determinism_n_jobs(frames, target, day0_runs):
    """One test day, n_jobs = 1 vs 4 vs the fixture's N_JOBS: byte-identical values (hourly setting, LEAR and naive)."""
    dn = [data.day_number(DAY0)]
    for model in ("naive", "lear"):
        a, la, _ = run.run(model, frames["hourly"], target, dn, n_jobs=1, verbose=False)
        b, lb, _ = run.run(model, frames["hourly"], target, dn, n_jobs=4, verbose=False)
        c, _, _ = day0_runs["hourly", model]
        for col in VALUE_COLS:
            assert np.array_equal(a[col].to_numpy(), b[col].to_numpy()), (model, col)
            assert np.array_equal(a[col].to_numpy(), c[col].to_numpy()), (model, col)
        if model == "lear":
            sel = [k for k in la.columns if k.startswith(("lambda_", "m_", "lasso_"))]
            pd.testing.assert_frame_equal(la[sel], lb[sel])


def test_quantiles_sorted(day0_runs):
    for key, (fc, _, _) in day0_runs.items():
        Q = fc[data.QCOLS].to_numpy()
        assert (np.diff(Q, axis=1) >= 0).all(), key


def test_output_schema(day0_runs):
    cols = ["datetime_local", "date", "hour", "quarter", "actual", "point"] + data.QCOLS
    for (s, m), (fc, log, wall) in day0_runs.items():
        assert list(fc.columns) == cols
        assert len(fc) == 96 and (fc["date"] == DAY0).all()
        assert fc["datetime_local"].is_monotonic_increasing and fc["datetime_local"].is_unique
        assert sorted(fc["hour"].unique()) == list(range(24)) and sorted(fc["quarter"].unique()) == [0, 1, 2, 3]
        assert not fc[VALUE_COLS + ["actual"]].isna().any().any()
        assert len(log) == 24 and {"date", "hour", "n_train", "n_test", "wall_s"} <= set(log.columns)
        if m == "lear":
            need = {"lasso_alpha", "lasso_nnz", "t_lasso_s", "t_lqr_s"}
            for c in data.QCOLS:
                need |= {f"lambda_{c}", f"m_{c}", f"status_{c}", f"fallback_{c}", f"nfail_{c}"}
            assert need <= set(log.columns)
        if s == "hourly":   # one value per hour, repeated over its 4 quarters
            assert (fc.groupby("hour")[VALUE_COLS].nunique() == 1).all().all()


def test_select_stan_reproduces_infit(frames, target, tmp_path):
    """Lambda path + esann27.select rule stan == the in-fit BIC selection, byte for byte (one hourly day)."""
    from esann27 import select
    dn = [data.day_number(DAY0)]
    fc, log, _, path, bic = run.run("lear", frames["hourly"], target, dn, n_jobs=4, verbose=False, with_path=True)
    assert len(path) == 96 and list(path.columns[4:]) == run.PATH_COLS and len(run.PATH_COLS) == 180
    assert len(bic) == 24 * 9 * 20 and list(bic.columns) == ["date", "hour", "level", "k", "lam", "loss", "m", "status"]
    assert np.allclose(bic["lam"].to_numpy()[:20], 10.0 ** (-1 + 4 * np.arange(20) / 19), rtol=1e-14, atol=0)   # linspace grid: 1 ulp at k = 13, 17
    assert (path.groupby("hour")[run.PATH_COLS].nunique() == 1).all().all()      # hourly repeated over 4 quarters
    # the path files round-trip through parquet, as in select's CLI
    for name, frame in (("path", path), ("lqr", bic), ("log", log), ("", fc)):
        frame.to_parquet(tmp_path / f"hourly_lear{'_' + name if name else ''}.parquet", index=False)
    inputs = {f: (tmp_path / f).read_bytes() for f in ("hourly_lear_path.parquet", "hourly_lear_lqr.parquet",
                                                         "hourly_lear_log.parquet", "hourly_lear.parquet")}
    for rule in ("stan", "bic", "pbic"):            # CLI in the order of run_step4.sh: no output overwrites an input
        select.main(["--setting", "hourly", "--rule", rule, "--dir", str(tmp_path)])
        assert (tmp_path / f"hourly_lear_{rule}.parquet").exists()
    for f, b in inputs.items():
        assert (tmp_path / f).read_bytes() == b, f
    st = pd.read_parquet(tmp_path / "hourly_lear_stan.parquet")
    assert list(st.columns) == list(fc.columns)
    for c in fc.columns:
        assert np.array_equal(st[c].to_numpy(), fc[c].to_numpy()), c
    # the other rules: valid forecasts, penalty ordering stan <= bic <= pbic shows in the selected m
    med = {}
    for rule in ("stan", "bic", "pbic"):
        out, sel = select.apply(path, bic, log, fc, rule)
        assert (np.diff(out[data.QCOLS].to_numpy(), axis=1) >= 0).all()
        assert np.array_equal(out["point"].to_numpy(), fc["point"].to_numpy())
        med[rule] = sel["m_sel"].mean()
    assert med["stan"] >= med["bic"] >= med["pbic"]
