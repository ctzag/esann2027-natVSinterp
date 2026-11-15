"""Data-stage tests: hashes, row counts, completeness, target."""
import shutil
import warnings

import numpy as np
import pandas as pd
import pytest

from esann27 import data

from .conftest import needs_data, reference_hashes


@reference_hashes
def test_reference_hashes(frames):
    for s, df in frames.items():
        assert df.attrs["hash_checked"], s
    assert data.check_hash(data.DATA_DIR / data.TARGET_FILE)


@needs_data
@reference_hashes
def test_hash_mismatch_raises(tmp_path, monkeypatch):
    name = data.SETTINGS["hourly"]
    shutil.copyfile(data.DATA_DIR / name, tmp_path / name)     # content only: the source is read-only on purpose
    b = bytearray((tmp_path / name).read_bytes())
    b[-2] = ord("0") if b[-2] != ord("0") else ord("1")
    (tmp_path / name).write_bytes(bytes(b))
    monkeypatch.setattr(data, "DATA_DIR", tmp_path)
    with pytest.raises(data.HashMismatch):
        data.load_setting(path=tmp_path / name)


def test_hash_mismatch_opt_out(tmp_path, monkeypatch):
    """ESANN27_ALLOW_HASH_MISMATCH=1: one warning per file and process, no exception. Unset or any other value:
    HashMismatch."""
    name, ref = "f.csv", "0" * 64
    (tmp_path / name).write_text("a\n1\n")
    monkeypatch.setattr(data, "DATA_DIR", tmp_path)
    monkeypatch.setattr(data, "HASHES", {name: ref})
    monkeypatch.setattr(data, "_WARNED", set())
    for v in (None, "0", "true"):
        if v is None:
            monkeypatch.delenv(data.ALLOW_ENV, raising=False)
        else:
            monkeypatch.setenv(data.ALLOW_ENV, v)
        with pytest.raises(data.HashMismatch):
            data.check_hash(tmp_path / name)
    monkeypatch.setenv(data.ALLOW_ENV, "1")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert data.check_hash(tmp_path / name) and data.check_hash(tmp_path / name)
    assert len(w) == 1
    msg = str(w[0].message)
    assert name in msg and data.sha256(tmp_path / name) in msg and ref in msg


def test_test_quarters(target):
    t = target[(target["date"] >= data.TEST_START) & (target["date"] < data.TEST_END)]
    assert len(t) == 35036
    per_day = t.groupby("day").size()
    assert len(per_day) == 365 and set(per_day.unique()) == {92, 96}      # 29 Mar 2026: 92 quarters


@pytest.mark.parametrize("setting,lo,hi,per_hour", [("hourly", 363, 364, 1), ("interp", 1452, 1456, 4),
                                                      ("native", 1452, 1456, 4)])
def test_window_rows(frames, setting, lo, hi, per_hour):
    df = frames[setting]
    cnt = df.groupby(["day", "hour"]).size().unstack(fill_value=0)
    cnt = cnt.reindex(range(cnt.index.min(), cnt.index.max() + 1), fill_value=0)
    win = cnt.rolling(data.WINDOW, min_periods=data.WINDOW).sum().shift(1)    # days D-364 .. D-1
    days = data.test_days()
    sizes = []
    for dn in days:
        for h in data.hours_of_day(df, dn):
            sizes.append(win.loc[dn, h])
            assert (cnt.loc[dn, h]) == per_hour
    sizes = np.array(sizes)
    assert sizes.min() >= lo and sizes.max() <= hi
    assert len(sizes) == 365 * 24 - 1                                          # 29 Mar 2026 has 23 hours
    # spot check against hour_fit
    for d, h in (("2025-10-01", 0), ("2026-03-29", 3), ("2026-09-30", 23)):
        f = data.hour_fit(df, data.day_number(d), h)
        assert len(f.train) == win.loc[data.day_number(d), h]
        assert f.train["day"].min() >= data.day_number(d) - 364 and f.train["day"].max() < data.day_number(d)


def test_no_nan_in_features(frames):
    cols = data.CONTINUOUS + data.DOW + data.QDUM + [data.PRICE]
    for s, df in frames.items():
        assert not df[cols].isna().any().any(), s


def test_target_is_native_price(frames, target, day0_runs):
    nat = frames["native"]
    assert np.array_equal(target["actual"].to_numpy(), nat[data.PRICE].to_numpy())
    assert (target["datetime_local"].to_numpy() == nat["datetime_local"].to_numpy()).all()
    for (s, m), (fc, _, _) in day0_runs.items():
        t = target.set_index("datetime_local").loc[fc["datetime_local"], "actual"].to_numpy()
        assert np.array_equal(fc["actual"].to_numpy(), t), (s, m)
