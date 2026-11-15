"""Data access for the model stage (handover Section 8, row "LEAR and naive implementation (step 2)", (a)).

Every CSV is read with float_precision="round_trip" (handover Section 5). The three ESANN input files in
out_planA_native/ are checked against their v3 sha256 on every load. With the environment variable
ESANN27_ALLOW_HASH_MISMATCH=1 a mismatch emits a warning (once per file and process) instead of raising HashMismatch.
"""
import hashlib
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "out_planA_native"

# sha256 of the 03 v3 build with --native-inputs (the reference hash list of the v3 build, equal on arm64 and x86-64)
HASHES = {
    "thesis_features_1hour_full.csv": "239032ecff3d1621fdd14dc9e084c14fa100d55650e5e660c302ec98ac805efc",
    "thesis_features_interpolated_mp_full.csv": "9cfdafd2c2d782371678ee641e8199640a4f324d2c39a599bb4e88afc4cdeaa7",
    "thesis_features_15min_mp_full.csv": "d9d07bbe6337a7ff168f84a5eb32dcb62a498863a384258b99c13d45f70f501b",
}
SETTINGS = {
    "hourly": "thesis_features_1hour_full.csv",
    "interp": "thesis_features_interpolated_mp_full.csv",
    "native": "thesis_features_15min_mp_full.csv",
}
TARGET_FILE = "thesis_features_15min_mp_full.csv"      # native price of D, scoring target in every setting

TEST_START, TEST_END = "2025-10-01", "2026-10-01"      # test year, end exclusive
EVAL_START = "2025-10-29"                              # common evaluation period 29 Oct 2025 - 30 Sep 2026
WINDOW = 364
TAUS = np.round(np.arange(1, 10) / 10, 1)              # 0.1 ... 0.9
QCOLS = [f"q{int(round(t * 100))}" for t in TAUS]      # q10 ... q90

PRICE_FEATURES = ["price_lag_d1", "price_lag_d7", "Ymax_d1", "Ymin_d1", "Peod_d1"]
# the 11 continuous regressors, in the column order of Stan's model_lear_*.R
CONTINUOUS = PRICE_FEATURES + ["residual_load_mw", "solar_mw", "wind_onshore_mw", "wind_offshore_mw",
                               "srmc_ccgt", "srmc_coal"]
DOW = ["dow_tue", "dow_wed", "dow_thu", "dow_fri", "dow_sat", "dow_sun"]
QDUM = ["q15", "q30", "q45"]                           # quarter-of-hour dummies, reference HH:00
PRICE = "price_eur_mwh"


ALLOW_ENV = "ESANN27_ALLOW_HASH_MISMATCH"             # "1": hash mismatch -> warning (code release)
_WARNED = set()                                         # files already warned about in this process


class HashMismatch(RuntimeError):
    pass


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_hash(path):
    """Raise HashMismatch if a reference file differs from its v3 hash (a warning instead under
    ESANN27_ALLOW_HASH_MISMATCH=1). Returns True if a reference hash exists."""
    path = Path(path).resolve()
    if path.parent == DATA_DIR.resolve() and path.name in HASHES:
        got = sha256(path)
        if got != HASHES[path.name]:
            msg = f"{path.name}: sha256 {got} != reference {HASHES[path.name]}"
            if os.environ.get(ALLOW_ENV) != "1":
                raise HashMismatch(msg)
            if path not in _WARNED:
                _WARNED.add(path)
                warnings.warn(f"{msg} ({ALLOW_ENV}=1: continuing)", stacklevel=2)
            return "mismatch-allowed"
        return True
    return False


def read_csv(path, check=True):
    """Feature file with derived columns date (datetime64), day (int, days since epoch), hour, minute, quarter
    (0..3) and the quarter dummies. check=True verifies the hash of the reference files; files outside
    out_planA_native/ (Stan's CSVs for R0) have no reference hash and are flagged in df.attrs."""
    hashed = check_hash(path) if check else False
    df = pd.read_csv(path, float_precision="round_trip")
    s = df["datetime_local"].astype(str)
    df["date"] = pd.to_datetime(s.str[:10])
    df["day"] = df["date"].values.astype("datetime64[D]").astype(np.int64)
    df["hour"] = s.str[11:13].astype(int)
    df["minute"] = s.str[14:16].astype(int)
    df["quarter"] = df["minute"] // 15
    for k, name in enumerate(QDUM, start=1):
        df[name] = (df["quarter"] == k).astype(int)
    df.attrs["path"], df.attrs["hash_checked"] = str(path), hashed
    df.attrs["is15"] = bool((df["minute"] != 0).any())
    return df


def setting_path(setting):
    return DATA_DIR / SETTINGS[setting]


def load_setting(setting=None, path=None, check=True):
    return read_csv(path if path is not None else setting_path(setting), check=check)


def load_target(path=None, check=True):
    """Native target: datetime_local, date, day, hour, quarter, actual."""
    t = read_csv(path if path is not None else DATA_DIR / TARGET_FILE, check=check)
    return t[["datetime_local", "date", "day", "hour", "quarter", PRICE]].rename(columns={PRICE: "actual"})


def day_number(d):
    return int(np.datetime64(pd.Timestamp(d).date(), "D").astype(np.int64))


def test_days(start=TEST_START, end=TEST_END):
    """Test days as day numbers, [start, end)."""
    return list(range(day_number(start), day_number(end)))


def day_str(dn):
    return str(np.datetime64(int(dn), "D"))


@dataclass
class HourFit:
    """Rows of one hour-fit: training window D-W..D-1 at clock hour h, and the test rows of D at hour h."""
    day: int
    hour: int
    train: pd.DataFrame
    test: pd.DataFrame
    window_days: np.ndarray       # every calendar day of the window (for the day folds)
    is15: bool                    # 15-min setting (quarter dummies apply) or hourly


def hour_fit(df, dn, h, window=WINDOW):
    lo = dn - window
    w = df[(df["day"] >= lo) & (df["day"] < dn) & (df["hour"] == h)]
    t = df[(df["day"] == dn) & (df["hour"] == h)]
    return HourFit(dn, h, w, t, np.arange(lo, dn), df.attrs["is15"])


def hours_of_day(df, dn):
    return sorted(df.loc[df["day"] == dn, "hour"].unique().tolist())
