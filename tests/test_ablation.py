"""Step 5b tests: ablation windows (regular share reproduces the regular run, replaced days = interp file),
test rows identical across levels, no look-ahead, levels and macros."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from esann27 import ablation as A
from esann27 import data
from esann27.models import lear

from .conftest import N_JOBS
from .test_models import VALUE_COLS, _perturb_from

ROOT = Path(__file__).resolve().parent.parent
DAY = "2026-09-03"                       # first ablation day, n_D = 337


@pytest.fixture(scope="module")
def pair(frames):
    A.check_aligned(frames["native"], frames["interp"])
    return frames["native"], frames["interp"]


def test_levels_and_days():
    days = data.test_days(A.ABL_START, A.ABL_END)
    assert len(days) == 28 and data.day_str(days[0]) == DAY and data.day_str(days[-1]) == "2026-09-30"
    nd = [A.n_native(d) for d in days]
    assert nd[0] == 337 and nd[-1] == 364
    assert np.mean(nd) / 364 == pytest.approx(0.962912, abs=1e-6)
    assert [A.m_of(p) for p in A.LEVELS] == [0, 91, 182, 273]


def test_regular_share_reproduces_regular_window(pair):
    nat, itp = pair
    for d in (DAY, "2026-09-17", "2026-09-30"):
        dn = data.day_number(d)
        f = A.ablation_frame(nat, itp, dn, A.n_native(dn))
        ref = nat[(nat["day"] >= dn - 364) & (nat["day"] <= dn)]
        assert f.equals(ref) and f.dtypes.equals(ref.dtypes), d
        for h in (0, 18):
            a, b = data.hour_fit(f, dn, h), data.hour_fit(nat, dn, h)
            assert a.train.equals(b.train) and a.test.equals(b.test)
            assert np.array_equal(a.window_days, b.window_days) and a.is15 == b.is15


def test_regular_share_reproduces_forecast_file(pair, target):
    """One ablation day at m = n_D through the full step 4 path (LEAR, lambda path, rule pbic) equals
    forecasts/native_lear.parquet byte for byte."""
    p = ROOT / "forecasts" / "native_lear.parquet"
    if not p.exists():
        pytest.skip("forecasts/native_lear.parquet missing")
    nat, itp = pair
    dn = data.day_number(DAY)
    fc = A.run_level(nat, itp, target, [dn], A.n_native(dn), n_jobs=N_JOBS)[0]
    ref = pd.read_parquet(p)
    ref = ref[ref["date"] == DAY].sort_values("datetime_local").reset_index(drop=True)
    assert len(fc) == len(ref) == 96 and list(fc.columns) == list(ref.columns)
    for c in ref.columns:
        assert np.array_equal(fc[c].to_numpy(), ref[c].to_numpy()), c


@pytest.mark.parametrize("pi", A.LEVELS)
def test_replaced_days_equal_interp(pair, pi):
    nat, itp = pair
    dn = data.day_number(DAY)
    m = A.m_of(pi)
    f = A.ablation_frame(nat, itp, dn, m)
    w = (f["day"] >= dn - 364) & (f["day"] < dn)
    old, new = w & (f["day"] < dn - m), w & (f["day"] >= dn - m)
    assert old.sum() + new.sum() == w.sum() and f.loc[w, "day"].nunique() == 364
    assert f.loc[new, "day"].nunique() == m and f.loc[old, "day"].nunique() == 364 - m
    for c in A.REPLACED:
        assert np.array_equal(f.loc[old, c].to_numpy(), itp.loc[f.index[old], c].to_numpy()), c
        assert np.array_equal(f.loc[new, c].to_numpy(), nat.loc[f.index[new], c].to_numpy()), c
    other = [c for c in nat.columns if c not in A.REPLACED]
    assert f[other].equals(nat.loc[f.index, other])
    if m < A.n_native(dn):                         # some native window rows really are replaced
        assert not np.array_equal(f.loc[old, data.PRICE].to_numpy(), nat.loc[f.index[old], data.PRICE].to_numpy())


def test_test_rows_identical_across_levels(pair):
    nat, itp = pair
    dn = data.day_number(DAY)
    ref = nat[nat["day"] == dn]
    tests = {pi: pd.concat([f.test for f in A.fits_for(nat, itp, dn, A.m_of(pi))]) for pi in A.LEVELS}
    for pi, t in tests.items():
        assert t.sort_index().equals(ref), pi


def test_no_lookahead(pair):
    nat, itp = pair
    dn = data.day_number("2026-09-10")
    pn, pi_ = _perturb_from(nat, dn), _perturb_from(itp, dn)
    m = A.m_of(0.5)
    a = [f for f in A.fits_for(nat, itp, dn, m) if f.hour == 18][0]
    b = [f for f in A.fits_for(pn, pi_, dn, m) if f.hour == 18][0]
    assert not np.array_equal(a.test[data.PRICE].to_numpy(), b.test[data.PRICE].to_numpy())
    ra, rb = lear.fit_predict(a, path=True), lear.fit_predict(b, path=True)
    assert np.array_equal(ra[0], rb[0]) and np.array_equal(ra[1], rb[1])
    assert np.array_equal(ra[3]["q"], rb[3]["q"], equal_nan=True)


def test_macros_of():
    tab = pd.DataFrame({"level": ["regular", "0", "0.25", "0.5", "0.75"], "gap_crps": [0.0, 0.4249, 0.3, 0.1551, 0.05],
                        "dm_p": [np.nan, 0.0004, 0.01, 0.1234, 0.5], "DSC": [10.0, 9.6, 9.7, 9.8, 9.9]})
    m = A.macros_of(tab)
    assert m == {"ablGapZero": "0.42", "ablGapHalf": "0.16", "ablDmPZero": "\\ensuremath{<}0.001",
                 "ablDmPHalf": "0.123", "ablDscGapZero": "0.40"}
