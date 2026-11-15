"""Rolling-window runs of the naive, LEAR and XGB models (handover Section 8, step 2 row (a), (f), (g)).

  python -m esann27.run --setting native --model lear --start 2025-10-01 --end 2025-10-04 --n-jobs 8
  python -m esann27.run --setting hourly --model naive                 # full test year

24 hour-fits per test day (23 / 24 on DST days), each fitted on the window D-W..D-1 at clock hour h.
Parallel over hours (one pool, the hour-fits of each day in parallel, per-day wall time) or over days.
BLAS / OpenMP threads are limited to 1 inside every worker. Outputs:
  <out>/<setting>_<model>.parquet      one row per test quarter: datetime_local, date, hour, quarter,
                                       actual, point, q10..q90 (hourly forecasts repeated over 4 quarters)
  <out>/<setting>_<model>_log.parquet  one row per hour-fit
  LEAR also: <out>/<setting>_lear_path.parquet (180 unsorted quantiles q<level>_l<k> per test quarter) and
             <out>/<setting>_lear_lqr.parquet (long: date, hour, level, k, lam, loss, m, status), read by
             esann27.select (rule outputs <setting>_lear_<rule>.parquet)
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from threadpoolctl import threadpool_limits

from . import data, metrics
from .models import lear, naive, xgb

MODELS = {"naive": naive.fit_predict, "lear": lear.fit_predict, "xgb": xgb.fit_predict,
          "xgbm": xgb.fit_predict_retuned}        # step 5d: re-tuned monthly, <out>/hyperparameters/*.json


N_LAMBDA = 20
PATH_COLS = [f"{c}_l{k}" for c in data.QCOLS for k in range(N_LAMBDA)]     # q10_l0 ... q90_l19


def _one(model, fit, opts):
    with threadpool_limits(limits=1):
        res = MODELS[model](fit, **opts)
    tq = fit.test["quarter"].to_numpy()
    out = {"day": fit.day, "hour": fit.hour, "quarter": tq, "point": res[0], "Q": res[1], "log": res[2]}
    if len(res) > 3:
        out["path"] = res[3]
    return out


def _day(model, fits, opts):
    return [_one(model, f, opts) for f in fits]


def _place(results, target, is15, values, cols):
    """Rows on the target's quarters of the test days (hourly: one value repeated over 4 quarters).
    values(r) -> array (n_test, len(cols)) for result r."""
    rows = []
    for r in results:
        base = {"day": r["day"], "hour": r["hour"]}
        if is15:
            base["quarter"] = r["quarter"]
        fc = pd.DataFrame(base, index=range(len(r["point"])))
        fc = pd.concat([fc, pd.DataFrame(values(r), columns=cols)], axis=1)
        rows.append(fc)
    fc = pd.concat(rows, ignore_index=True)
    days = fc["day"].unique()
    t = target[target["day"].isin(days)]
    keys = ["day", "hour", "quarter"] if is15 else ["day", "hour"]
    out = t.merge(fc, on=keys, how="left", validate="one_to_one" if is15 else "many_to_one")
    if out[cols[0]].isna().any():
        raise RuntimeError("target quarters without a forecast")
    if len(out) != len(t):
        raise RuntimeError("forecast rows do not match the target quarters")
    out = out.sort_values("datetime_local").reset_index(drop=True)
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out


def assemble(results, target, is15):
    out = _place(results, target, is15, lambda r: np.column_stack([r["point"], r["Q"]]), ["point"] + data.QCOLS)
    return out[["datetime_local", "date", "hour", "quarter", "actual", "point"] + data.QCOLS]


def assemble_path(results, target, is15):
    """<setting>_lear_path: one row per test quarter, 180 unsorted back-transformed quantiles q<level>_l<k>."""
    out = _place(results, target, is15, lambda r: r["path"]["q"].reshape(len(r["point"]), -1), PATH_COLS)
    return out[["datetime_local", "date", "hour", "quarter"] + PATH_COLS]


def bic_frame(results):
    """<setting>_lear_lqr: long, per hour-fit, level and lambda the window check loss, m and the solver status."""
    from .models.lear import LAMBDA_GRID
    G = len(LAMBDA_GRID)
    parts = []
    for r in results:
        p = r["path"]
        nl = len(data.TAUS)
        parts.append(pd.DataFrame({
            "date": data.day_str(r["day"]), "hour": r["hour"],
            "level": np.repeat(data.TAUS, G), "k": np.tile(np.arange(G), nl), "lam": np.tile(LAMBDA_GRID, nl),
            "loss": p["loss"].ravel(), "m": p["m"].ravel(), "status": p["status"].ravel()}))
    return pd.concat(parts, ignore_index=True)


def log_frame(results):
    rows = []
    for r in results:
        row = {"date": data.day_str(r["day"]), "hour": r["hour"]}
        row.update(r["log"])
        rows.append(row)
    return pd.DataFrame(rows)


def run(model, df, target, days, window=data.WINDOW, n_jobs=1, parallel="hours", opts=None, verbose=True,
        with_path=False):
    """Returns (forecasts, log, wall time per day in seconds); with_path=True (LEAR) also the lambda path
    frame and the long check-loss / m / status frame."""
    opts = dict(opts or {})
    if with_path:
        opts["path"] = True
    is15 = df.attrs["is15"]
    fits_of = lambda dn: [data.hour_fit(df, dn, h, window) for h in data.hours_of_day(df, dn)]
    results, day_wall = [], {}
    with parallel_config(backend="loky", inner_max_num_threads=1):
        with Parallel(n_jobs=n_jobs) as par:
            if parallel == "hours":
                for dn in days:
                    t0 = time.perf_counter()
                    results += par(delayed(_one)(model, f, opts) for f in fits_of(dn))
                    day_wall[data.day_str(dn)] = time.perf_counter() - t0
                    if verbose:
                        print(f"  {data.day_str(dn)}  {day_wall[data.day_str(dn)]:8.2f} s", flush=True)
            elif parallel == "days":
                t0 = time.perf_counter()
                per_day = par(delayed(_day)(model, fits_of(dn), opts) for dn in days)
                for r in per_day:
                    results += r
                tot = time.perf_counter() - t0
                day_wall = {data.day_str(dn): tot / len(days) for dn in days}
            else:
                raise ValueError(parallel)
    out = (assemble(results, target, is15), log_frame(results), day_wall)
    if with_path:
        out += (assemble_path(results, target, is15), bic_frame(results))
    return out


def print_summary(fc, label):
    rows = []
    for name, (a, b) in {"test year": (data.TEST_START, data.TEST_END),
                         "evaluation": (data.EVAL_START, data.TEST_END)}.items():
        part = metrics.period(fc, a, b)
        if len(part):
            s = metrics.summary(part)
            rows.append({"run": label, "period": name, **{k: s[k] for k in
                         ("n", "MAE", "RMSE", "CRPS", "PICP80", "width80", "PICP60", "width60")}})
    t = pd.DataFrame(rows)
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(t.to_string(index=False))
    return t


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m esann27.run")
    p.add_argument("--setting", choices=list(data.SETTINGS), help="ESANN setting (file in out_planA_native/)")
    p.add_argument("--features", default=None, help="feature file instead of --setting (R0); no reference hash")
    p.add_argument("--target", default=None, help="target file (default: native 15-min mp file)")
    p.add_argument("--name", default=None, help="output label (default: the setting)")
    p.add_argument("--model", choices=list(MODELS), required=True)
    p.add_argument("--start", default=data.TEST_START)
    p.add_argument("--end", default=data.TEST_END, help="exclusive")
    p.add_argument("--window", type=int, default=data.WINDOW)
    p.add_argument("--no-quarter-dummies", action="store_true")
    p.add_argument("--folds", choices=["day", "row"], default="day")
    p.add_argument("--no-standardize", action="store_true")
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--parallel", choices=["hours", "days"], default="hours")
    p.add_argument("--out", default="forecasts")
    a = p.parse_args(argv)
    if (a.setting is None) == (a.features is None):
        p.error("give exactly one of --setting and --features")
    if a.model == "xgbm":                      # step 5d (handover Section 8 row "XGB monthly re-tuning")
        if a.setting not in xgb.RETUNE_SETTINGS or a.features or a.name or a.window != data.WINDOW:
            p.error(f"xgbm runs on --setting {' / '.join(xgb.RETUNE_SETTINGS)} only, default window and name")
        o, f = Path(a.out).resolve(), (data.ROOT / "forecasts").resolve()
        if o == f or f in o.parents:
            p.error("forecasts/ is read-only on purpose: xgbm writes to forecasts_retune/")
    name = a.name or a.setting or Path(a.features).stem
    df = data.load_setting(a.setting, a.features)
    target = data.load_target(a.target)
    days = data.test_days(a.start, a.end)
    if a.model == "naive":
        opts = {}
    elif a.model == "xgb":                     # frozen hyperparameters_<structure>.json (esann27/models/)
        opts = {"qdum": not a.no_quarter_dummies}
    elif a.model == "xgbm":                    # re-tuned: <out>/hyperparameters/<setting>_<date>.json (esann27.retune)
        opts = {"qdum": not a.no_quarter_dummies, "setting": a.setting,
                "hp_dir": str((Path(a.out) / "hyperparameters").resolve())}
    else:
        opts = {"qdum": not a.no_quarter_dummies, "standardize": not a.no_standardize, "folds": a.folds}
    hc = df.attrs["hash_checked"]
    hc = "MISMATCH allowed" if hc == "mismatch-allowed" else "checked" if hc else "NOT checked"
    print(f"{a.model} on {name} ({df.attrs['path']}, hash {hc}),"
          f" {len(days)} test days {a.start} .. {a.end} (excl.), window {a.window}, n_jobs {a.n_jobs} over {a.parallel},"
          f" options {opts}")
    t0 = time.perf_counter()
    res = run(a.model, df, target, days, a.window, a.n_jobs, a.parallel, opts, with_path=(a.model == "lear"))
    fc, log, day_wall = res[:3]
    total = time.perf_counter() - t0
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.model == "lear":                      # lambda path for esann27.select (handover Section 8, row (d))
        res[3].to_parquet(out / f"{name}_lear_path.parquet", index=False)
        res[4].to_parquet(out / f"{name}_lear_lqr.parquet", index=False)
    fc.to_parquet(out / f"{name}_{a.model}.parquet", index=False)
    log.to_parquet(out / f"{name}_{a.model}_log.parquet", index=False)
    pd.DataFrame({"date": list(day_wall), "wall_s": list(day_wall.values())}).to_parquet(
        out / f"{name}_{a.model}_daywall.parquet", index=False)
    print(f"wrote {out / f'{name}_{a.model}.parquet'} ({len(fc)} rows), total {total:.1f} s, "
          f"{total / len(days):.2f} s per day, cores used {a.n_jobs}")
    print_summary(fc, f"{name} {a.model}")


if __name__ == "__main__":
    main()
