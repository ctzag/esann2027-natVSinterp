"""LQR penalty selection from the stored lambda path (handover Section 8, row (d) and "LQR penalty selection rule").

  python -m esann27.select --setting native --rule pbic [--dir forecasts]

Reads <dir>/<setting>_lear_path.parquet, _lear_lqr.parquet, _lear_log.parquet (n and p per hour-fit) and
_lear.parquet (point forecast and target). Criterion per hour-fit and level: log(loss) + m * c(n, p) with
  stan  c = log(n) / (2n log p)     rule (i), Stan's script as coded (replication)
  bic   c = log(n) / (2n)           rule (ii)
  pbic  c = log(n) log(p) / (2n)    rule (iii), Uniejewski and Weron 2021 eq. (11), main
strict minimum (the smallest lambda wins ties), point forecast where no lambda solved, the 9 levels sorted per
row. Writes <dir>/<setting>_lear_<rule>.parquet in the standard schema.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import data

RULES = ("stan", "bic", "pbic")
G = 20
STD_COLS = ["datetime_local", "date", "hour", "quarter", "actual", "point"] + data.QCOLS


def criterion(loss, m, n, p, rule):
    """Vectorized; the operation order of rule stan is the one of lear.select_bic (byte-identical selection)."""
    ll = np.log(loss)
    if rule == "stan":
        return ll + m * np.log(n) / (2 * n * np.log(p))
    if rule == "bic":
        return ll + m * np.log(n) / (2 * n)
    if rule == "pbic":
        return ll + m * np.log(n) * np.log(p) / (2 * n)
    raise ValueError(rule)


def choose(bic, log, rule):
    """Selected grid index per (date, hour, level); -1 where no lambda solved. Returns a frame
    date, hour, level, k_sel, m_sel."""
    b = bic.merge(log[["date", "hour", "n_train", "p"]], on=["date", "hour"], how="left", validate="many_to_one")
    if b["n_train"].isna().any():
        raise RuntimeError("bic rows without an hour-fit in the log")
    b = b.sort_values(["date", "hour", "level", "k"]).reset_index(drop=True)
    ok = (b["status"].to_numpy() == 0) & np.isfinite(b["loss"].to_numpy())
    crit = np.full(len(b), np.inf)
    with np.errstate(divide="ignore"):
        crit[ok] = criterion(b["loss"].to_numpy()[ok], b["m"].to_numpy()[ok].astype(float),
                             b["n_train"].to_numpy()[ok], b["p"].to_numpy()[ok], rule)
    C = crit.reshape(-1, G)
    if not (b["k"].to_numpy().reshape(-1, G) == np.arange(G)).all():
        raise RuntimeError("bic file is not a complete 20-lambda grid per hour-fit and level")
    k = np.argmin(C, axis=1)                       # first minimum = smallest lambda among ties
    k = np.where(np.isfinite(C[np.arange(len(C)), k]), k, -1)
    keys = b.iloc[::G][["date", "hour", "level"]].reset_index(drop=True)
    msel = np.where(k >= 0, b["m"].to_numpy().reshape(-1, G)[np.arange(len(C)), np.maximum(k, 0)], -1)
    return keys.assign(k_sel=k, m_sel=msel)


def apply(path, bic, log, std, rule):
    """Forecast frame in the standard schema and the selection table."""
    sel = choose(bic, log, rule)
    wide = sel.pivot_table(index=["date", "hour"], columns="level", values="k_sel", aggfunc="first")
    wide = wide[list(data.TAUS)]
    wide.columns = [f"k_{c}" for c in data.QCOLS]
    wide = wide.reset_index()
    base = std[STD_COLS[:6]]
    pth = path.merge(base[["datetime_local", "point", "actual"]], on="datetime_local", how="inner", validate="one_to_one")
    pth = pth.merge(wide, on=["date", "hour"], how="left", validate="many_to_one")
    if len(pth) != len(std):
        raise RuntimeError("path and standard file cover different quarters")
    P = pth[[f"{c}_l{k}" for c in data.QCOLS for k in range(G)]].to_numpy().reshape(len(pth), len(data.QCOLS), G)
    K = pth[[f"k_{c}" for c in data.QCOLS]].to_numpy().astype(int)
    Q = np.take_along_axis(P, np.maximum(K, 0)[:, :, None], axis=2)[:, :, 0]
    point = pth["point"].to_numpy()
    Q = np.where(K >= 0, Q, point[:, None])        # fallback: point forecast where no lambda solved
    Q = np.sort(Q, axis=1)
    out = pth[["datetime_local", "date", "hour", "quarter", "actual", "point"]].copy()
    out[data.QCOLS] = Q
    out = out.sort_values("datetime_local").reset_index(drop=True)
    return out[STD_COLS], sel


def report(sel, label):
    rows = []
    for tau, g in sel.groupby("level"):
        k = g["k_sel"].to_numpy()
        rows.append({"level": f"q{int(round(tau * 100))}", "pairs": len(g), "at_min": np.mean(k == 0),
                     "at_max": np.mean(k == G - 1), "median_m": float(np.median(g["m_sel"][k >= 0])),
                     "fallback": int(np.sum(k < 0))})
    k = sel["k_sel"].to_numpy()
    rows.append({"level": "all", "pairs": len(sel), "at_min": np.mean(k == 0), "at_max": np.mean(k == G - 1),
                 "median_m": float(np.median(sel["m_sel"][k >= 0])), "fallback": int(np.sum(k < 0))})
    t = pd.DataFrame(rows)
    print(f"{label}: share of hour-fit x level pairs at the grid minimum (lambda 0.1) and maximum (1000), median m")
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(t.to_string(index=False))
    return t


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m esann27.select")
    p.add_argument("--setting", required=True, help="setting label of the LEAR run (file prefix)")
    p.add_argument("--rule", choices=RULES, required=True)
    p.add_argument("--dir", default="forecasts")
    a = p.parse_args(argv)
    d = Path(a.dir)
    f = lambda suffix: pd.read_parquet(d / f"{a.setting}_lear{suffix}.parquet")
    out, sel = apply(f("_path"), f("_lqr"), f("_log"), f(""), a.rule)
    dst = d / f"{a.setting}_lear_{a.rule}.parquet"
    if dst.name in {f"{a.setting}_lear{x}.parquet" for x in ("_path", "_lqr", "_log", "")}:
        raise SystemExit(f"refusing to overwrite an input file: {dst}")
    out.to_parquet(dst, index=False)
    print(f"wrote {dst} ({len(out)} rows), rule {a.rule}")
    report(sel, f"{a.setting} rule {a.rule}")
    from .run import print_summary
    print_summary(out, f"{a.setting} lear {a.rule}")


if __name__ == "__main__":
    main()
