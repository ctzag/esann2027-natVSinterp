"""XGBoost, point and quantiles (handover Section 8, rows "XGB tuning structure", "15-min model structure" and
"LEAR and naive implementation (step 2)" (a), (e)-(g); step 6).

Fit: same hour-fit structure, window, test rows and regressors as LEAR: the 11 continuous regressors and the 6
weekday dummies in the column order of Stan's model_xgb_*.R, plus q15 / q30 / q45 in the 15-min settings
(p = 20; hourly p = 17). As in Stan's script: raw target (no asinh), raw regressors (no standardization),
point objective reg:squarederror, gamma 0, lambda 1, quantile fallback = point forecast if a booster fails.
Quantiles: reg:quantileerror with quantile_alpha = tau, one booster per level, the 9 levels sorted per row.
Fixed: xgboost 3.4.1, tree_method hist, nthread 1, seed 2027, subsample = colsample_bytree = 0.8,
min_child_weight 5.

Tuning (once per structure, frozen): first window 2 Oct 2024 - 30 Sep 2025 (the window of test day
1 Oct 2025), its first 308 days train, its last 56 days validate, per clock hour (24 hour-fits as in the
rolling run, validated on the same hour's rows). For each configuration every hour-fit is boosted to
1,000 rounds and its validation loss recorded after every round (mean pinball, point model: mean absolute
error, in float64 from the float32 predictions). The 24 curves are summed and early stopping with patience 50
is applied to the summed curve (stop at the first round 50 rounds after the best one, strict improvement),
giving one round count per configuration; its score is the summed loss at that round. Point grid max_depth
{3, 5, 7} x eta {0.05, 0.1}; quantile models at the point depth, eta {0.05, 0.1} per level. Structures:
"15min" (tuned on the native file, the interp file's first window asserted identical) and "hourly".

  python -m esann27.models.xgb --structure 15min --n-jobs 8     # writes esann27/models/hyperparameters_15min.json

Step 5d (handover Section 8 row "XGB monthly re-tuning", model xgbm): tune() also runs on the window of a given
re-tune date (day=R, default the first test day, so the frozen tuning above is unchanged) and on a given setting
file. retune() wraps it for one setting and date, fit_predict_retuned() is fit_predict with the dictionary of the
latest re-tune date <= the test day. The command line of the re-tuning is esann27.retune.
"""
import argparse
import hashlib
import json
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import xgboost as xgb

from .. import data
from ..data import CONTINUOUS, DOW, PRICE, QCOLS, QDUM, TAUS

SEED = 2027
HP_DIR = Path(__file__).resolve().parent
FIXED = {"tree_method": "hist", "nthread": 1, "seed": SEED, "subsample": 0.8, "colsample_bytree": 0.8,
         "min_child_weight": 5, "gamma": 0.0, "lambda": 1.0}
POINT_OBJ = "reg:squarederror"                         # Stan's model_xgb_*.R
DEPTHS, ETAS = (3, 5, 7), (0.05, 0.1)
MAX_ROUNDS, PATIENCE, VAL_DAYS = 1000, 50, 56
TUNE_DAY = "2025-10-01"                                # first test day: its window is the tuning window
STRUCTURES = {"15min": ("native", "interp"), "hourly": ("hourly",)}


def structure_of(is15):
    return "15min" if is15 else "hourly"


def hp_path(structure):
    return HP_DIR / f"hyperparameters_{structure}.json"


@lru_cache(maxsize=None)
def load_hp(structure):
    p = hp_path(structure)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: run python -m esann27.models.xgb --structure {structure}")
    return json.loads(p.read_text())


def columns(is15, qdum=True):
    return CONTINUOUS + DOW + (QDUM if (qdum and is15) else [])


def design(fit, qdum=True):
    cols = columns(fit.is15, qdum)
    return (fit.train[cols].to_numpy(dtype=float), fit.train[PRICE].to_numpy(dtype=float),
            fit.test[cols].to_numpy(dtype=float))


def _params(depth, eta, tau=None):
    p = dict(FIXED, max_depth=int(depth), eta=float(eta))
    if tau is None:
        p["objective"] = POINT_OBJ
    else:
        p.update(objective="reg:quantileerror", quantile_alpha=float(tau))
    return p


def _train(params, dtrain, rounds):
    return xgb.train(params, dtrain, num_boost_round=int(rounds), verbose_eval=False)


def fit_predict(fit, qdum=True, hp=None):
    """point (n_test,), Q (n_test, 9) sorted, log dict. hp: the dict of hyperparameters_<structure>.json
    (default: the frozen file of the fit's structure)."""
    hp = hp if hp is not None else load_hp(structure_of(fit.is15))
    t0 = time.perf_counter()
    Xtr, y, Xte = design(fit, qdum)
    dtrain = xgb.DMatrix(Xtr, label=y, nthread=1)
    dtest = xgb.DMatrix(Xte, nthread=1)
    pt = hp["point"]
    point = _train(_params(pt["max_depth"], pt["eta"]), dtrain, pt["rounds"]).predict(dtest).astype(float)
    t1 = time.perf_counter()
    Q = np.empty((len(Xte), len(TAUS)))
    log = {"n_train": len(y), "n_test": len(Xte), "p": Xtr.shape[1]}
    for k, tau in enumerate(TAUS):
        c = QCOLS[k]
        qh = hp["quantiles"][c]
        try:
            Q[:, k] = _train(_params(pt["max_depth"], qh["eta"], tau), dtrain, qh["rounds"]).predict(dtest)
            log[f"fallback_{c}"] = False
        except xgb.core.XGBoostError:                  # Stan: tryCatch(error) -> point forecast
            Q[:, k] = point
            log[f"fallback_{c}"] = True
    Q = np.sort(Q, axis=1)
    t2 = time.perf_counter()
    log.update({"t_point_s": t1 - t0, "t_quant_s": t2 - t1, "wall_s": t2 - t0})
    return point, Q, log


# ----------------------------------------------------------------------------------------------- tuning

def tuning_split(fit):
    """First window of the hour-fit split by day: last VAL_DAYS days validate."""
    cut = fit.day - VAL_DAYS
    tr, va = fit.train[fit.train["day"] < cut], fit.train[fit.train["day"] >= cut]
    cols = columns(fit.is15)
    return (tr[cols].to_numpy(dtype=float), tr[PRICE].to_numpy(dtype=float),
            va[cols].to_numpy(dtype=float), va[PRICE].to_numpy(dtype=float))


def window_digest(df, dn=None, window=data.WINDOW):
    """sha256 of the model columns of the first window (all hours), for the identity of native and interp."""
    dn = data.day_number(TUNE_DAY) if dn is None else dn
    w = df[(df["day"] >= dn - window) & (df["day"] < dn)]
    cols = ["day", "hour", "quarter", PRICE] + columns(df.attrs["is15"])
    return hashlib.sha256(np.ascontiguousarray(w[cols].to_numpy(dtype=float)).tobytes()).hexdigest()


def val_curve(split, depth, eta, tau=None, max_rounds=MAX_ROUNDS):
    """Validation loss after each round 1..max_rounds (mean pinball at tau, or MAE for the point model)."""
    from threadpoolctl import threadpool_limits
    Xtr, ytr, Xva, yva = split
    curve = []

    def metric(predt, _):
        r = yva - predt.astype(float)
        v = float(np.mean(np.abs(r))) if tau is None else float(np.mean(np.maximum(tau * r, (tau - 1) * r)))
        curve.append(v)                                # exact double (evals_result would round to 6 decimals)
        return "loss", v

    with threadpool_limits(limits=1):
        dtr = xgb.DMatrix(Xtr, label=ytr, nthread=1)
        dva = xgb.DMatrix(Xva, label=yva, nthread=1)
        xgb.train(dict(_params(depth, eta, tau), disable_default_eval_metric=1), dtr, num_boost_round=max_rounds,
                  evals=[(dva, "val")], custom_metric=metric, verbose_eval=False)
    return np.array(curve)


def early_stop(curve, patience=PATIENCE):
    """xgboost EarlyStopping on a precomputed curve: (rounds of the best iteration, its loss, round stopped at)."""
    best, best_i, i = np.inf, -1, -1
    for i, v in enumerate(curve):
        if v < best:
            best, best_i = v, i
        elif i - best_i >= patience:
            break
    return best_i + 1, float(best), i + 1


def tune(df, n_jobs=1, depths=DEPTHS, etas=ETAS, max_rounds=MAX_ROUNDS, hours=None, verbose=True, day=TUNE_DAY):
    """Tuning of one structure on the window of test day `day` (default: the first window). Returns the
    hyperparameter dict (JSON content)."""
    from joblib import Parallel, delayed, parallel_config
    dn = data.day_number(day)
    hours = data.hours_of_day(df, dn) if hours is None else list(hours)
    splits = {h: tuning_split(data.hour_fit(df, dn, h)) for h in hours}
    t0 = time.perf_counter()

    def run(tasks):
        with parallel_config(backend="loky", inner_max_num_threads=1):
            return Parallel(n_jobs=n_jobs)(delayed(val_curve)(splits[h], d, e, t, max_rounds) for h, d, e, t in tasks)

    def summed(tasks, curves):
        acc = {}
        for (h, *k), c in zip(tasks, curves):
            k = tuple(k)
            acc[k] = acc.get(k, 0.0) + c
        return acc

    # point model
    tasks = [(h, d, e, None) for d in depths for e in etas for h in hours]
    acc = summed(tasks, run(tasks))
    grid_point = []
    for d in depths:
        for e in etas:
            r, s, stop = early_stop(acc[(d, e, None)])
            grid_point.append({"max_depth": d, "eta": e, "rounds": r, "stopped_at": stop, "score_mae_sum": s})
    bp = min(grid_point, key=lambda g: g["score_mae_sum"])        # first in grid order on ties
    if verbose:
        print(f"  point grid done ({time.perf_counter() - t0:.0f} s): depth {bp['max_depth']}, eta {bp['eta']},"
              f" rounds {bp['rounds']}", flush=True)

    # quantile models at the point depth, eta and rounds per level
    tasks = [(h, bp["max_depth"], e, float(t)) for t in TAUS for e in etas for h in hours]
    acc = summed(tasks, run(tasks))
    grid_q, quant = [], {}
    for c, t in zip(QCOLS, TAUS):
        rows = []
        for e in etas:
            r, s, stop = early_stop(acc[(bp["max_depth"], e, float(t))])
            rows.append({"level": c, "tau": float(t), "eta": e, "rounds": r, "stopped_at": stop, "score_pinball_sum": s})
        grid_q += rows
        b = min(rows, key=lambda g: g["score_pinball_sum"])
        quant[c] = {"tau": float(t), "eta": b["eta"], "rounds": b["rounds"], "score_pinball_sum": b["score_pinball_sum"]}
    dt = time.perf_counter() - t0
    w0 = data.day_str(dn - data.WINDOW)
    return {
        "structure": structure_of(df.attrs["is15"]),
        "tuned_on": Path(df.attrs["path"]).name,
        "first_window": [w0, data.day_str(dn - 1)],
        "train_days": [w0, data.day_str(dn - VAL_DAYS - 1)],
        "val_days": [data.day_str(dn - VAL_DAYS), data.day_str(dn - 1)],
        "hours": hours,
        "window_sha256": window_digest(df, dn),
        "features": columns(df.attrs["is15"]),
        "fixed": FIXED, "point_objective": POINT_OBJ, "max_rounds": max_rounds, "patience": PATIENCE,
        "score": "sum over the hour-fits of the mean validation loss per round (point: MAE, quantiles: pinball),"
                 " early stopping on the summed curve",
        "point": {"max_depth": bp["max_depth"], "eta": bp["eta"], "rounds": bp["rounds"],
                  "score_mae_sum": bp["score_mae_sum"]},
        "quantiles": quant,
        "grid": {"point": grid_point, "quantiles": grid_q},
        "xgboost_version": xgb.__version__,
        "tuning_wall_s": dt,
    }


# ----------------------------------------------------------------------- step 5d: monthly re-tuning (xgbm)

RETUNE_DATES = tuple(f"{m}-01" for m in np.arange(np.datetime64("2025-10", "M"), np.datetime64("2026-10", "M")))
RETUNE_SETTINGS = ("interp", "native")
RETUNE_DROP = ("tuning_wall_s",)       # wall time only in retune_log.csv: the JSON is byte-identical for any n_jobs


def retune_date_of(dn):
    """Latest re-tune date <= test day dn (the first day of its month)."""
    past = [r for r in RETUNE_DATES if data.day_number(r) <= dn]
    if not past:
        raise ValueError(f"{data.day_str(dn)} is before the first re-tune date {RETUNE_DATES[0]}")
    return past[-1]


def retune_path(hp_dir, setting, rdate):
    return Path(hp_dir) / f"{setting}_{rdate}.json"


def retune(df, setting, rdate, n_jobs=1, verbose=False):
    """Re-tune of one setting at re-tune date rdate: tune() unchanged on the window rdate-364..rdate-1 of the
    setting's own file df. Returns (JSON content: the keys of hyperparameters_15min.json without tuning_wall_s,
    shared_by, window_sha256_interp, plus setting and retune_date; wall time in seconds)."""
    if rdate not in RETUNE_DATES or setting not in RETUNE_SETTINGS:
        raise ValueError(f"no re-tune for {setting} at {rdate}")
    if Path(df.attrs["path"]).name != data.SETTINGS[setting]:
        raise ValueError(f"{df.attrs['path']} is not the file of setting {setting}")
    t0 = time.perf_counter()
    hp = tune(df, n_jobs=n_jobs, verbose=verbose, day=rdate)
    wall = time.perf_counter() - t0
    for k in RETUNE_DROP:
        hp.pop(k)
    hp["setting"], hp["retune_date"] = setting, rdate
    return hp, wall


def retune_json_text(hp):
    return json.dumps(hp, indent=1) + "\n"


@lru_cache(maxsize=None)
def _read_json(path, mtime_ns, size):
    return json.loads(Path(path).read_text())


def load_retuned(hp_dir, setting, rdate):
    p = retune_path(hp_dir, setting, rdate)
    st = p.stat()
    hp = _read_json(str(p), st.st_mtime_ns, st.st_size)
    if hp.get("setting") != setting or hp.get("retune_date") != rdate or hp.get("structure") != "15min":
        raise ValueError(f"{p}: labels {hp.get('setting')} / {hp.get('retune_date')} / {hp.get('structure')}")
    return hp


def fit_predict_retuned(fit, qdum=True, setting=None, hp_dir=None):
    """Model xgbm: fit_predict with the hyperparameters of the latest re-tune date <= the fit's test day, read from
    <hp_dir>/<setting>_<date>.json. The log gains retune_date."""
    if not fit.is15:
        raise ValueError("xgbm is defined for the 15-min settings interp and native only")
    r = retune_date_of(fit.day)
    point, Q, log = fit_predict(fit, qdum, hp=load_retuned(hp_dir, setting, r))
    log["retune_date"] = r
    return point, Q, log


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m esann27.models.xgb")
    p.add_argument("--structure", choices=list(STRUCTURES), required=True)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--force", action="store_true", help="overwrite a frozen hyperparameter file")
    a = p.parse_args(argv)
    out = hp_path(a.structure)
    if out.exists() and not a.force:
        raise SystemExit(f"{out} exists (frozen). --force to retune.")
    frames = {s: data.load_setting(s) for s in STRUCTURES[a.structure]}
    if a.structure == "15min":                         # native and interp share the tuning: identical first window
        dig = {s: window_digest(f) for s, f in frames.items()}
        assert dig["native"] == dig["interp"], dig
        print(f"first window identical in native and interp (sha256 {dig['native'][:16]}...)")
    df = frames[STRUCTURES[a.structure][0]]
    print(f"tuning {a.structure} on {df.attrs['path']}, n_jobs {a.n_jobs}")
    hp = tune(df, n_jobs=a.n_jobs)
    if a.structure == "15min":
        hp["shared_by"] = ["native", "interp"]
        hp["window_sha256_interp"] = window_digest(frames["interp"])
    out.write_text(json.dumps(hp, indent=1) + "\n")
    print(f"wrote {out} ({hp['tuning_wall_s']:.0f} s)")
    print(json.dumps({k: hp[k] for k in ("point", "quantiles")}, indent=1))


if __name__ == "__main__":
    main()
