import os
import platform
import sys
from pathlib import Path

import pytest

from esann27 import data, run

N_JOBS = min(8, os.cpu_count() or 1)
DAY0 = "2025-10-01"          # native share 0

# ---------------------------------------------------------------- skip conditions (code release, spec A2)
ROOT = Path(__file__).resolve().parent.parent
HASH_OPT_OUT = os.environ.get(data.ALLOW_ENV) == "1"
MAC_ARM64 = sys.platform == "darwin" and platform.machine() == "arm64"
DATA_REASON = "built data absent, see README step 2"


def data_present():
    """The three input files of out_planA_native/ (the files with a reference hash) exist."""
    return all((data.DATA_DIR / n).exists() for n in data.HASHES)


def require_data():
    if not data_present():
        pytest.skip(DATA_REASON)


def require(*paths):
    """Skip unless every path exists; the reason names the folder of each missing path."""
    miss = {}
    for p in map(Path, paths):
        if not p.exists():
            rel = p.relative_to(ROOT).parts if p.is_relative_to(ROOT) else p.parts
            miss.setdefault(rel[0], []).append("/".join(rel[1:]))
    if miss:
        pytest.skip("; ".join(f"{d}/ absent" + (f": {', '.join(n for n in names if n)}" if any(names) else "")
                              for d, names in miss.items()))


needs_data = pytest.mark.skipif(not data_present(), reason=DATA_REASON)
# data hashes: the data build is byte-identical across platforms, so only the opt-out skips these
reference_hashes = pytest.mark.skipif(HASH_OPT_OUT, reason=f"{data.ALLOW_ENV}=1: no comparison with reference hashes")
# committed files or hard-coded values produced on the Mac (forecasts and XGB tuning differ across machines)
reference_outputs = pytest.mark.skipif(HASH_OPT_OUT or not MAC_ARM64,
                                       reason="reference outputs were produced on macOS arm64")


@pytest.fixture(scope="session")
def frames():
    require_data()
    return {s: data.load_setting(s) for s in data.SETTINGS}


@pytest.fixture(scope="session")
def target():
    require_data()
    return data.load_target()


@pytest.fixture(scope="session")
def day0_runs(frames, target):
    """naive and LEAR forecasts for 1 Oct 2025 in the three settings (n_jobs = N_JOBS)."""
    dn = [data.day_number(DAY0)]
    out = {}
    for s, df in frames.items():
        for m in ("naive", "lear"):
            out[s, m] = run.run(m, df, target, dn, n_jobs=N_JOBS, verbose=False)
    return out
