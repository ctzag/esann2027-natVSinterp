"""Step 5d: XGB with monthly re-tuning, model `xgbm` (handover Section 8 row "XGB monthly re-tuning" with its
implementation decisions of 9 Nov, step 5d specification B1).

  python -m esann27.retune --setting interp --n-jobs 12              # 12 re-tunes, then the 365 test days
  python -m esann27.retune --setting native --stage tune --dates 2026-03-01 --n-jobs 12

Re-tune dates R: the first test day of each month, 1 Oct 2025 ... 1 Sep 2026 (xgb.RETUNE_DATES). Each re-tune is
xgb.tune, the step 6a procedure unchanged, on the window R-364..R-1 of the setting's own file: its first 308 days
train, its last 56 days validate, on the setting's own price (step 6a rule (3)). Test day D uses the re-tune of the
latest R <= D (xgb.retune_date_of). Everything else is the daily refit of `xgb`: xgb.fit_predict with the re-tuned
dictionary, called through esann27.run with the worker settings of step 6b.

Outputs in --out (default forecasts_retune/, git-ignored). forecasts/ is read-only on purpose and refused here and
in esann27.run for xgbm:
  hyperparameters/<setting>_<YYYY-MM-DD>.json  one per re-tune date: the keys of hyperparameters_15min.json without
                                               tuning_wall_s, shared_by, window_sha256_interp, plus setting and
                                               retune_date, so the file is byte-identical for any n_jobs
  retune_log.csv                               per setting, re-tune date and model (point, q10..q90): max_depth,
                                               eta, rounds, val_loss_sum (summed validation loss at the selected
                                               rounds: MAE for point, pinball for the quantiles), wall_s (wall time
                                               of the whole re-tune of that setting and date, on its 10 rows)
  <setting>_xgbm.parquet, _log.parquet (with retune_date), _daywall.parquet    written by esann27.run
"""
import argparse
import time
from pathlib import Path

import pandas as pd

from . import data, run
from .models import xgb

FORECASTS = data.ROOT / "forecasts"
LOG_COLS = ["setting", "retune_date", "model", "max_depth", "eta", "rounds", "val_loss_sum", "wall_s"]
MODEL_ORDER = ["point"] + data.QCOLS


def refuse_forecasts(out):
    """SystemExit if `out` is forecasts/ or inside it (spec B1.4: nothing of step 5d writes there)."""
    o, f = Path(out).resolve(), FORECASTS.resolve()
    if o == f or f in o.parents:
        raise SystemExit(f"{out}: forecasts/ is read-only on purpose, step 5d writes to forecasts_retune/")


def log_rows(hp, wall):
    """The 10 retune_log.csv rows of one re-tune (point, q10..q90)."""
    base = {"setting": hp["setting"], "retune_date": hp["retune_date"], "max_depth": hp["point"]["max_depth"]}
    rows = [{**base, "model": "point", "eta": hp["point"]["eta"], "rounds": hp["point"]["rounds"],
             "val_loss_sum": hp["point"]["score_mae_sum"], "wall_s": wall}]
    for c in data.QCOLS:
        q = hp["quantiles"][c]
        rows.append({**base, "model": c, "eta": q["eta"], "rounds": q["rounds"], "val_loss_sum": q["score_pinball_sum"],
                     "wall_s": wall})
    return [{k: r[k] for k in LOG_COLS} for r in rows]


def update_log(path, rows):
    """Replace the rows of the same (setting, retune_date) in retune_log.csv, keep the others, sort by setting,
    date and model (point, q10..q90)."""
    path = Path(path)
    new = pd.DataFrame(rows, columns=LOG_COLS)
    if path.exists():
        old = pd.read_csv(path, float_precision="round_trip", dtype={"retune_date": str, "setting": str, "model": str})
        keys = set(zip(new["setting"], new["retune_date"]))
        old = old[[k not in keys for k in zip(old["setting"], old["retune_date"])]]
        new = pd.concat([old, new], ignore_index=True)
    order = {m: i for i, m in enumerate(MODEL_ORDER)}
    new = (new.assign(_m=new["model"].map(order)).sort_values(["setting", "retune_date", "_m"], kind="stable")
           .drop(columns="_m").reset_index(drop=True))
    new.to_csv(path, index=False)
    return new


def retune_all(df, setting, dates, n_jobs, out):
    """Re-tune `setting` at every date of `dates`, write the JSON files and update retune_log.csv."""
    hp_dir = Path(out) / "hyperparameters"
    hp_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in dates:
        hp, wall = xgb.retune(df, setting, r, n_jobs=n_jobs)
        p = xgb.retune_path(hp_dir, setting, r)
        p.write_text(xgb.retune_json_text(hp))
        rows += log_rows(hp, wall)
        qr = [hp["quantiles"][c]["rounds"] for c in data.QCOLS]
        print(f"  {setting} {r}: point depth {hp['point']['max_depth']}, eta {hp['point']['eta']}, rounds "
              f"{hp['point']['rounds']}; quantile rounds {min(qr)}-{max(qr)}; {wall:.1f} s -> {p}", flush=True)
    return update_log(Path(out) / "retune_log.csv", rows)


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m esann27.retune")
    p.add_argument("--setting", choices=list(xgb.RETUNE_SETTINGS), required=True)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--out", default="forecasts_retune")
    p.add_argument("--stage", choices=["all", "tune", "forecast"], default="all")
    p.add_argument("--dates", nargs="+", default=list(xgb.RETUNE_DATES), help="re-tune dates (default: all 12)")
    p.add_argument("--start", default=data.TEST_START, help="forecast stage: first test day")
    p.add_argument("--end", default=data.TEST_END, help="forecast stage: end, exclusive")
    a = p.parse_args(argv)
    refuse_forecasts(a.out)
    bad = [d for d in a.dates if d not in xgb.RETUNE_DATES]
    if bad:
        p.error(f"not re-tune dates: {bad} (allowed: {', '.join(xgb.RETUNE_DATES)})")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.stage in ("all", "tune"):
        df = data.load_setting(a.setting)
        print(f"re-tuning {a.setting} on {df.attrs['path']} at {len(a.dates)} dates, n_jobs {a.n_jobs}", flush=True)
        t0 = time.perf_counter()
        retune_all(df, a.setting, a.dates, a.n_jobs, out)
        print(f"re-tuning {a.setting} done, {time.perf_counter() - t0:.1f} s", flush=True)
    if a.stage in ("all", "forecast"):
        need = sorted({xgb.retune_date_of(dn) for dn in data.test_days(a.start, a.end)})
        miss = [r for r in need if not xgb.retune_path(out / "hyperparameters", a.setting, r).exists()]
        if miss:
            raise SystemExit(f"re-tune files missing for {a.setting}: {miss}")
        run.main(["--setting", a.setting, "--model", "xgbm", "--out", str(out), "--n-jobs", str(a.n_jobs),
                  "--start", a.start, "--end", a.end])


if __name__ == "__main__":
    main()
