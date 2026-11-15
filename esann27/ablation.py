"""Share ablation, step 5b (handover Section 9, step row 5b, rev 2.8.11).

  python -m esann27.ablation --n-jobs 12 [--start 2026-09-03 --end 2026-10-01] [--fdir forecasts]
                             [--abl-dir forecasts/ablation] [--out results]

Days D = 3-30 Sep 2026 (28). For each level pi in {0, 0.25, 0.5, 0.75} the training window D-364 .. D-1 of the
native setting keeps its native rows on the m = round(pi * 364) most recent window days; every older window day
takes the price and the five price features (price_lag_d1, price_lag_d7, Ymax_d1, Ymin_d1, Peod_d1) from the
interpolated file (the inputs are identical in both files under --native-inputs, asserted). Test rows of D from
the native file. LEAR exactly as step 4 (quarter dummies, day folds, standardization, lambda path), then rule
(iii) (pbic) with the esann27.select logic. Reference level: the regular native run (forecasts/native_lear.parquet)
at its own share pi_D = n_D / 364, n_D = D - 1 Oct 2025 (337 .. 364, mean 0.963).

Per level on the 28 days: CRPS, MCB, DSC, UNC (CORP as evaluate.py), PICP-80, width-80, DM of the level against
the regular run on the 28 daily CRPS differences (Newey-West lag 7, two-sided). Predicted (Proposition A, LEAR
burn-in inputs, pooled over positions): CRPS_A(pi) - CRPS_A(pi_bar), pi_bar = mean of pi_D over the days.

Outputs: <abl-dir>/native_pi<000|025|050|075>_lear{,_path,_lqr,_log,_pbic}.parquet, <out>/ablation.csv (one row
per level and the regular run), <out>/ablation_pred.csv (predicted curve on 101 shares and pi_bar), macros
\\ablGapZero, \\ablGapHalf (CRPS(pi) - CRPS(regular) at pi = 0, 0.5), \\ablDmPZero, \\ablDmPHalf,
\\ablDscGapZero (DSC(regular) - DSC(0)) merged into <out>/numbers.tex.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config

from . import data, metrics, run, select
from . import evaluate as ev

LEVELS = (0.0, 0.25, 0.5, 0.75)
ABL_START, ABL_END = "2026-09-03", "2026-10-01"           # end exclusive: 28 days
GO_LIVE = "2025-10-01"
REPLACED = [data.PRICE] + data.PRICE_FEATURES
INPUT_COLS = [c for c in data.CONTINUOUS if c not in data.PRICE_FEATURES] + data.DOW + data.QDUM
LEAR_OPTS = {"qdum": True, "standardize": True, "folds": "day"}      # step 4 defaults of esann27.run


def level_tag(pi):
    return f"pi{int(round(100 * pi)):03d}"


def n_native(dn, window=data.WINDOW):
    """Native window days of the regular run for test day dn."""
    return int(np.clip(dn - data.day_number(GO_LIVE), 0, window))


def m_of(pi, window=data.WINDOW):
    return int(round(pi * window))


def check_aligned(nat, itp):
    if not nat["datetime_local"].equals(itp["datetime_local"]):
        raise RuntimeError("native and interp files have different rows")


def ablation_frame(nat, itp, dn, m, window=data.WINDOW):
    """Rows of days dn-window .. dn of the native file; window days older than the m most recent take the
    price and the five price features of the interp file. Day dn (test rows) stays native."""
    lo = dn - window
    sel = ((nat["day"] >= lo) & (nat["day"] <= dn)).to_numpy()
    f = nat.loc[sel].copy()
    f.attrs = dict(nat.attrs)
    f.attrs["ablation_m"] = m
    old = (f["day"] < dn - m).to_numpy()
    src = itp.loc[sel]
    if not np.array_equal(src[INPUT_COLS].to_numpy(float), f[INPUT_COLS].to_numpy(float)):
        raise RuntimeError("inputs differ between the native and interp files (expected --native-inputs build)")
    f.loc[old, REPLACED] = src.loc[old, REPLACED].to_numpy()
    return f


def fits_for(nat, itp, dn, m, window=data.WINDOW):
    f = ablation_frame(nat, itp, dn, m, window)
    return [data.hour_fit(f, dn, h, window) for h in data.hours_of_day(f, dn)]


def run_level(nat, itp, target, days, m, n_jobs=1):
    """LEAR with the lambda path on the ablation windows of all days, then rule pbic. Returns (pbic forecasts,
    raw lear frame, path, lqr, log, wall seconds)."""
    t0 = time.perf_counter()
    fits = [f for dn in days for f in fits_for(nat, itp, dn, m)]
    opts = dict(LEAR_OPTS, path=True)
    with parallel_config(backend="loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=n_jobs)(delayed(run._one)("lear", f, opts) for f in fits)
    fc = run.assemble(results, target, True)
    path = run.assemble_path(results, target, True)
    lqr = run.bic_frame(results)
    log = run.log_frame(results)
    pbic, _ = select.apply(path, lqr, log, fc, "pbic")
    return pbic, fc, path, lqr, log, time.perf_counter() - t0


def level_metrics(df, ref_daily=None):
    y, Q = df["actual"].to_numpy(float), df[data.QCOLS].to_numpy(float)
    c = ev.corp(y, Q)
    s = metrics.summary(df)
    out = {"n_quarters": len(df), "MAE": s["MAE"], "CRPS": c["CRPS"], "MCB": c["MCB"], "DSC": c["DSC"],
           "UNC": c["UNC"], "PICP80": s["PICP80"], "width80": s["width80"]}
    daily = ev.daily_crps(df)
    if ref_daily is not None:
        if not daily.index.equals(ref_daily.index):
            raise RuntimeError("daily CRPS of the level and of the regular run cover different days")
        r = ev.dm_test((daily - ref_daily).to_numpy())
        out.update({"dm_mean": r["mean"], "dm_se": r["se"], "dm_stat": r["stat"], "dm_p": r["p"]})
    return out, daily


def pred_crps(inp, pi):
    """Proposition A, CRPS of the native setting at share pi, pooled (mean) over the four positions."""
    return float(np.mean([ev.prop_a_point(inp, q, pi)[2] for q in range(4)]))


def macros_of(tab):
    t = tab.set_index("level")
    return {"ablGapZero": ev.fmt(t.loc["0", "gap_crps"]), "ablGapHalf": ev.fmt(t.loc["0.5", "gap_crps"]),
            "ablDmPZero": ev.fmt_p(t.loc["0", "dm_p"]), "ablDmPHalf": ev.fmt_p(t.loc["0.5", "dm_p"]),
            "ablDscGapZero": ev.fmt(t.loc["regular", "DSC"] - t.loc["0", "DSC"])}


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m esann27.ablation")
    p.add_argument("--start", default=ABL_START)
    p.add_argument("--end", default=ABL_END, help="exclusive")
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--fdir", default="forecasts", help="regular native_lear.parquet and hourly_lear.parquet")
    p.add_argument("--abl-dir", default=None, help="ablation forecasts (default <fdir>/ablation)")
    p.add_argument("--out", default="results")
    a = p.parse_args(argv)
    fdir, out = Path(a.fdir), Path(a.out)
    adir = Path(a.abl_dir) if a.abl_dir else fdir / "ablation"
    adir.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    t_all = time.perf_counter()

    nat, itp = data.load_setting("native"), data.load_setting("interp")
    check_aligned(nat, itp)
    target = data.load_target()
    days = data.test_days(a.start, a.end)
    dates = [data.day_str(d) for d in days]
    pis_d = np.array([n_native(d) for d in days]) / data.WINDOW
    pi_bar = float(pis_d.mean())
    print(f"ablation on {len(days)} days {a.start} .. {a.end} (excl.), regular share {pis_d.min():.3f} .. "
          f"{pis_d.max():.3f}, mean {pi_bar:.6f}, n_jobs {a.n_jobs}", flush=True)

    # Proposition A first (burn-in inputs only)
    inp = ev.burnin_inputs("lear", fdir)
    c_bar = pred_crps(inp, pi_bar)
    grid = np.linspace(0.0, 1.0, ev.N_PI)
    pr = pd.DataFrame({"pi": np.append(grid, pi_bar)})
    pr["pred_crps"] = [pred_crps(inp, x) for x in pr["pi"]]
    pr["pred_gap"] = pr["pred_crps"] - c_bar
    pr["kind"] = ["grid"] * len(grid) + ["regular"]
    pr.to_csv(out / "ablation_pred.csv", index=False)

    reg = pd.read_parquet(fdir / "native_lear.parquet")
    reg = reg[reg["date"].isin(dates)].sort_values("datetime_local").reset_index(drop=True)
    reg_m, reg_daily = level_metrics(reg)
    rows = [{"level": "regular", "pi": pi_bar, "m": float(np.mean(pis_d * data.WINDOW)), "n_days": len(days),
             **reg_m, "dm_mean": np.nan, "dm_se": np.nan, "dm_stat": np.nan, "dm_p": np.nan,
             "pred_crps": c_bar, "wall_s_per_day": np.nan}]
    for pi in LEVELS:
        m = m_of(pi)
        fc, raw, path, lqr, log, wall = run_level(nat, itp, target, days, m, a.n_jobs)
        tag = f"native_{level_tag(pi)}_lear"
        raw.to_parquet(adir / f"{tag}.parquet", index=False)
        path.to_parquet(adir / f"{tag}_path.parquet", index=False)
        lqr.to_parquet(adir / f"{tag}_lqr.parquet", index=False)
        log.to_parquet(adir / f"{tag}_log.parquet", index=False)
        fc.to_parquet(adir / f"{tag}_pbic.parquet", index=False)
        if not (fc["datetime_local"].equals(reg["datetime_local"]) and fc["actual"].equals(reg["actual"])):
            raise RuntimeError("ablation rows or actuals differ from the regular run")
        lm, _ = level_metrics(fc, reg_daily)
        rows.append({"level": f"{pi:g}", "pi": pi, "m": m, "n_days": len(days), **lm,
                     "pred_crps": pred_crps(inp, pi), "wall_s_per_day": wall / len(days)})
        print(f"  pi = {pi:<4g} m = {m:3d}: CRPS {lm['CRPS']:.3f} (regular {reg_m['CRPS']:.3f}), DM p {lm['dm_p']:.4f},"
              f" {wall:.0f} s ({wall / len(days):.1f} s per day)", flush=True)
    tab = pd.DataFrame(rows)
    tab["gap_crps"] = tab["CRPS"] - reg_m["CRPS"]
    tab["gap_dsc_regular_minus"] = reg_m["DSC"] - tab["DSC"]
    tab["pred_gap"] = tab["pred_crps"] - c_bar
    order = ["level", "pi", "m", "n_days", "n_quarters", "MAE", "CRPS", "MCB", "DSC", "UNC", "PICP80", "width80",
             "gap_crps", "gap_dsc_regular_minus", "dm_mean", "dm_se", "dm_stat", "dm_p", "pred_crps", "pred_gap",
             "wall_s_per_day"]
    tab = tab[order]
    tab.to_csv(out / "ablation.csv", index=False)
    mac = macros_of(tab)
    ev.write_macros(out / "numbers.tex", mac)
    with pd.option_context("display.width", 250, "display.float_format", "{:.4f}".format, "display.max_columns", 30):
        print(tab.drop(columns=["n_days"]).to_string(index=False))
    print("macros:", mac)
    print(f"total {time.perf_counter() - t_all:.0f} s")


if __name__ == "__main__":
    main()
