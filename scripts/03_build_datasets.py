"""
03_build_datasets.py  --  build feature files in Stan's 22-column format

  Plan B : Stan's design extended to 30 Sep 2026 (NED forecasts, his conventions)
  Plan A : ENTSO-E day-ahead forecasts, gate-closure-consistent fuel prices

Usage (venv active, run from esann2027_data; raw files in .\\raw):
  python 03_build_datasets.py --plan B
  python 03_build_datasets.py --plan A                     # file starts 2 Oct 2024 (W=364 -> test 1 Oct 2025)
  python 03_build_datasets.py --plan A --start 2023-10-04  # robustness, W=728
  python 03_build_datasets.py --plan B --validate "C:\\...\\Thesis Data Stan"
  python 03_build_datasets.py --plan B --leaky-interp      # Stan's cross-midnight interpolation -> .\\out_planB_leaky
  python 03_build_datasets.py --plan A --native-inputs     # ESANN files -> .\\out_planA_native

v3: round_trip parsing, --native-inputs, platform-independent hour-start interpolation
  Every pd.read_csv passes float_precision="round_trip" (correctly rounded, platform independent).
  interp_hour_start avoids np.interp (FMA contraction on arm64, last-bit differences vs x86-64, handover Section 5).
  --native-inputs (Plan A only): in all four 15-min files load, solar, wind onshore and wind offshore are
  the ENTSO-E 15-min forecasts over the whole span (handover Section 8, option 2), so the interpolated and
  the native 15-min files differ only in the price and the features built from it
  (lags, Ymax_d1, Ymin_d1, Peod_d1). Prices, price lags, Ymax/Ymin/Peod, srmc and
  the hourly file are unchanged; residual_load_mw follows from the inputs.

Interpolation is day-bounded by default (v2, 12 Oct 2026): no interpolated value on day D uses anything
from day D+1, for prices and inputs, so the lags and previous-day max / min / last built from the
interpolated price are free of look-ahead. --leaky-interp restores the cross-midnight interpolation of
Stan's files, in which Peod_d1(D) = 0.25*P(D-1,23h) + 0.75*P(D,00h) (handover Section 4, issue 18).

Outputs in .\\out_planB or .\\out_planA (with --leaky-interp: .\\out_planB_leaky or .\\out_planA_leaky):
  thesis_features_1hour_full.csv           hourly
  thesis_features_interpolated_full.csv    15-min, hour-start linear interpolation (Stan's method)
  thesis_features_15min_full.csv           15-min native hybrid: interpolated before 1 Oct 2025, genuine after
  Plan A also writes the mean-preserving variants:
  thesis_features_interpolated_mp_full.csv
  thesis_features_15min_mp_full.csv

Conventions taken from Stan's files (verified on his CSVs):
  - local Europe/Amsterdam time; spring-forward days have 23 h / 92 quarters;
    fall-back duplicated hour collapsed: prices averaged, NED keeps the first occurrence
  - interpolation by position (01:00 -> 03:00 on spring-forward days in four steps), last hour carried forward
    (Stan: last hour of the file only; day-bounded default: last hour of every day)
  - price lags / previous-day max, min, last computed from each file's own price series;
    lags that point to a non-existent DST hour take the previous row's value
  - NED gaps forward-filled (first hour back-filled); MW = volume_kWh * (60/interval) / 1000; offshore = type 17
    Known gaps in the NED load forecast (raw archive of 12 Oct 2026, local time), all forward-filled in Plan B:
      2 Apr 02:00 - 5 Apr 10:00 2026 (321 quarters / 80 hours, Easter; frozen in Stan's files too)
      29 Apr 2026 02:00 - 09:00 (8 hours, hourly series only)
      26 May 02:00 - 27 May 13:00 2026 (141 quarters / 34 hours)
  - SRMC (thesis eq. 3.1/3.2, Table 3.2) from daily closes carried forward over weekends
Corrections relative to Stan's files (Plan B):
  - 30 Sep 2025 real hourly prices (Stan's file holds a forward-filled 92.74)
  - hourly price after go-live = mean of the four quarters (Stan downloaded a separate hourly series)
  - API2 coal from ICE (investing.com) for the whole period (Yahoo MTF=F ended 26 Dec 2025)
  - real price history for the first-week lags of the 15-min files
  - day-bounded interpolation (unless --leaky-interp)
Plan A differences: ENTSO-E forecasts (load A65, wind/solar A69), fuel settlement of D-2
  (the last close known at gate closure, 12:00 on D-1), duplicated fall-back hours averaged.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TZ = "Europe/Amsterdam"
GO_LIVE = pd.Timestamp("2025-10-01")
INPUTS = ["load_mw", "solar_mw", "wind_onshore_mw", "wind_offshore_mw"]
COLUMNS = ["datetime_local", "load_mw", "solar_mw", "wind_onshore_mw", "wind_offshore_mw",
           "residual_load_mw", "price_eur_mwh", "weekday_name", "is_weekend",
           "dow_tue", "dow_wed", "dow_thu", "dow_fri", "dow_sat", "dow_sun",
           "price_lag_d1", "price_lag_d7", "Ymax_d1", "Ymin_d1", "Peod_d1", "srmc_ccgt", "srmc_coal"]
NED_FILES = {"load_mw": "load", "solar_mw": "solar", "wind_onshore_mw": "wind_onshore",
             "wind_offshore_mw": "wind_offshore_t17"}
ENTSOE_WS = {"solar_mw": "solar_mw", "wind_onshore_mw": "wind_onshore_mw", "wind_offshore_mw": "wind_offshore_mw"}


# ---------------------------------------------------------------------------
# time helpers
# ---------------------------------------------------------------------------
def to_local(ts_utc):
    return pd.to_datetime(ts_utc, utc=True).dt.tz_convert(TZ).dt.tz_localize(None)


def local_grid(start, end, freq):
    """All existing local wall-clock stamps in [start, end), fall-back duplicates collapsed."""
    a = pd.Timestamp(start).tz_localize(TZ).tz_convert("UTC")
    b = pd.Timestamp(end).tz_localize(TZ).tz_convert("UTC")
    g = pd.date_range(a, b, freq=freq, inclusive="left").tz_convert(TZ).tz_localize(None)
    return pd.DatetimeIndex(g.unique())


def collapse(df, time_col, how):
    """Local-time series with fall-back duplicates averaged ('mean') or first kept ('first')."""
    g = df.groupby(time_col)
    return g.mean(numeric_only=True) if how == "mean" else g.first()


# ---------------------------------------------------------------------------
# raw loaders
# ---------------------------------------------------------------------------
def load_prices(raw):
    h = pd.read_csv(raw / "entsoe_dayahead_60min.csv", float_precision="round_trip")
    q = pd.read_csv(raw / "entsoe_dayahead_15min.csv", float_precision="round_trip")
    h["t"], q["t"] = to_local(h.timestamp_utc), to_local(q.timestamp_utc)
    h = collapse(h[["t", "price_eur_mwh"]], "t", "mean")["price_eur_mwh"]
    q = collapse(q[["t", "price_eur_mwh"]], "t", "mean")["price_eur_mwh"]
    q = q[q.index >= GO_LIVE]
    hourly = pd.concat([h[h.index < GO_LIVE], q.groupby(q.index.floor("h")).mean()]).sort_index()
    return hourly, q


def load_ned(raw):
    hourly, quarter = {}, {}
    for col, name in NED_FILES.items():
        for gran, store in (("60min", hourly), ("15min", quarter)):
            d = pd.read_csv(raw / f"ned_{name}_{gran}.csv", float_precision="round_trip")
            d["t"] = to_local(d.validfrom_utc)
            store[col] = collapse(d[["t", "mw"]], "t", "first")["mw"]
    return pd.DataFrame(hourly), pd.DataFrame(quarter)


def load_entsoe_forecasts(raw):
    lf = pd.read_csv(raw / "entsoe_load_forecast_da.csv", float_precision="round_trip")
    ws = pd.read_csv(raw / "entsoe_wind_solar_forecast_da.csv", float_precision="round_trip")
    lf["t"], ws["t"] = to_local(lf.timestamp_utc), to_local(ws.timestamp_utc)
    q = collapse(ws.drop(columns="timestamp_utc"), "t", "mean")[list(ENTSOE_WS)]
    q["load_mw"] = collapse(lf.drop(columns="timestamp_utc"), "t", "mean")["load_forecast_mw"]
    q = q[INPUTS]
    # missing day(s): same quarter of the previous day
    grid = local_grid(q.index.min().normalize(), q.index.max().normalize() + pd.Timedelta(days=1), "15min")
    q = q.reindex(grid)
    for _ in range(3):
        prev = q.shift(freq="1D").reindex(q.index)
        q = q.fillna(prev)
    q = q.ffill()
    return q.groupby(q.index.floor("h")).mean(), q


def load_fuels(raw, shift_days):
    y = pd.read_csv(raw / "yahoo_daily_close.csv", parse_dates=["date"], float_precision="round_trip").set_index("date")
    c = pd.read_csv(raw / "api2_investing.csv", float_precision="round_trip")
    date_col = next(k for k in c.columns if "date" in k.lower())
    price_col = next(k for k in c.columns if k.lower() in ("price", "close", "last"))
    c.index = pd.to_datetime(c[date_col], format="%m/%d/%Y")
    api2 = pd.to_numeric(c[price_col].astype(str).str.replace(",", ""), errors="coerce").sort_index()
    f = pd.concat([y, api2.rename("api2_usd_t")], axis=1)
    days = pd.date_range(f.index.min(), "2026-10-01", freq="D")
    f = f.reindex(days).ffill()
    srmc = pd.DataFrame({
        "srmc_ccgt": f.ttf_eur_mwh / 0.52 + f.eua_tracker * 0.202 / 0.52 + 2.5,
        "srmc_coal": f.api2_usd_t / (6.978 * f.eurusd) / 0.40 + f.eua_tracker * 0.341 / 0.40 + 5.0,
    })
    return srmc.shift(shift_days)          # value used for delivery day d = settlement of day d - shift


# ---------------------------------------------------------------------------
# interpolation hourly -> 15 min
# ---------------------------------------------------------------------------
def last_hour_of_day(stamps):
    """True where the hour of a local stamp is the last hour of its local day (23:00 on every day)."""
    hr = stamps.floor("h")
    return hr == pd.Series(hr, index=stamps).groupby(stamps.normalize()).transform("max").values


def interp_hour_start(hourly, qgrid, day_bounded=True):
    """Linear interpolation between hour starts (Stan's scheme), written out so that it rounds the same on
    every platform (v3, 6 Nov 2026). np.interp, which pandas .interpolate(method="linear") calls, evaluates
    slope * (x - xp) + fp in one C expression that clang contracts to a fused multiply-add on arm64, so the
    arm64 and x86-64 wheels differ in the last bit (handover Section 5). Here the product and the sum are
    separate numpy operations, which reproduces the x86-64 result of v2 exactly."""
    on = np.asarray(qgrid.minute == 0)
    pos = np.arange(len(qgrid), dtype=float)
    xp = pos[on]
    j = np.searchsorted(xp, pos, side="right") - 1
    mid = (j >= 0) & (j < len(xp) - 1) & ~on
    jl = j[mid]
    knots = hourly.reindex(qgrid[on])
    out = pd.DataFrame(index=qgrid, columns=hourly.columns, dtype=float)
    for col in hourly.columns:
        fp = knots[col].to_numpy(dtype=float)
        assert not np.isnan(fp).any(), f"NaN at an hour start in {col}"
        v = np.full(len(qgrid), np.nan)
        v[on] = fp
        slope = (fp[jl + 1] - fp[jl]) / (xp[jl + 1] - xp[jl])
        step = slope * (pos[mid] - xp[jl])
        v[mid] = step + fp[jl]
        out[col] = v
    if day_bounded:
        out.loc[last_hour_of_day(qgrid) & (qgrid.minute != 0)] = np.nan
    return out.ffill()


def interp_mean_preserving(hourly, qgrid, nonneg=(), day_bounded=True):
    """Hourly value at HH:30, linear between hour midpoints, then shifted so the four quarters
    average exactly to the hourly value.
    day_bounded: in the last hour of a day the next hour is replaced by the hour itself, so the
    second half of that hour is flat before the shift and nothing from the next day enters.
    The first hour of a day still uses the last hour of the previous day (known at gate closure)."""
    h = hourly.reindex(sorted(set(qgrid.floor("h"))))
    prev = h.shift(1).fillna(h)
    nxt = h.shift(-1).fillna(h)
    if day_bounded:
        last = last_hour_of_day(h.index)
        nxt.loc[last] = h.loc[last]
    pos = pd.Series(qgrid.minute // 15, index=qgrid)
    hr = qgrid.floor("h")
    out = pd.DataFrame(index=qgrid, columns=hourly.columns, dtype=float)
    for col in hourly.columns:
        P, Pp, Pn = h[col].reindex(hr).values, prev[col].reindex(hr).values, nxt[col].reindex(hr).values
        c = (pos.values * 15 + 7.5) / 60.0
        v = np.where(c < 0.5, Pp + (c + 0.5) * (P - Pp), P + (c - 0.5) * (Pn - P))
        s = pd.Series(v, index=qgrid)
        s = s + (pd.Series(P, index=qgrid) - s.groupby(hr).transform("mean"))
        out[col] = s.clip(lower=0) if col in nonneg else s
    return out


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------
def add_features(df, srmc):
    df = df.copy()
    df["residual_load_mw"] = df.load_mw - df.solar_mw - df.wind_onshore_mw - df.wind_offshore_mw
    p = df.price_eur_mwh
    for k, col in ((1, "price_lag_d1"), (7, "price_lag_d7")):
        df[col] = p.shift(freq=f"{k}D").reindex(df.index)
    dates = df.index.normalize()
    day = p.groupby(dates)
    prev = lambda s: s.shift(1, freq="D").reindex(dates).values
    df["Ymax_d1"], df["Ymin_d1"], df["Peod_d1"] = prev(day.max()), prev(day.min()), prev(day.last())
    lag_cols = ["price_lag_d1", "price_lag_d7", "Ymax_d1", "Ymin_d1", "Peod_d1"]
    df[lag_cols] = df[lag_cols].ffill()
    df["weekday_name"] = df.index.day_name()
    dow = df.index.dayofweek
    df["is_weekend"] = (dow >= 5).astype(int)
    for i, name in enumerate(["tue", "wed", "thu", "fri", "sat", "sun"], start=1):
        df[f"dow_{name}"] = (dow == i).astype(int)
    df["srmc_ccgt"] = srmc.srmc_ccgt.reindex(dates).values
    df["srmc_coal"] = srmc.srmc_coal.reindex(dates).values
    return df


def finish(df, start, end):
    df = df[(df.index >= pd.Timestamp(start)) & (df.index < pd.Timestamp(end))].copy()
    df.insert(0, "datetime_local", df.index.strftime("%Y-%m-%d %H:%M:%S"))
    return df[COLUMNS].reset_index(drop=True)


# ---------------------------------------------------------------------------
def build(plan, raw, start, end, day_bounded=True, native_inputs=False):
    hist = (pd.Timestamp(start) - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    hgrid, qgrid = local_grid(hist, end, "h"), local_grid(hist, end, "15min")
    price_h, price_q = load_prices(raw)
    if plan == "B":
        in_h, in_q = load_ned(raw)
        srmc = load_fuels(raw, shift_days=0)
    else:
        in_h, in_q = load_entsoe_forecasts(raw)
        srmc = load_fuels(raw, shift_days=2)

    H = in_h.reindex(hgrid)
    H[INPUTS] = H[INPUTS].ffill().bfill()
    H["price_eur_mwh"] = price_h.reindex(hgrid).ffill()

    Qnat = in_q.reindex(qgrid).ffill()
    Qnat["price_eur_mwh"] = price_q.reindex(qgrid)
    after = qgrid >= GO_LIVE

    files = {"thesis_features_1hour_full.csv": add_features(H, srmc)}
    variants = [("", interp_hour_start(H, qgrid, day_bounded))]
    if plan == "A":
        variants.append(("_mp", interp_mean_preserving(H, qgrid, nonneg=INPUTS, day_bounded=day_bounded)))
    for tag, I in variants:
        if native_inputs:      # v3 option 2: native 15-min forecasts over the whole span, prices untouched
            I.loc[:, INPUTS] = Qnat.loc[:, INPUTS].values
        N = I.copy()
        N.loc[after, INPUTS + ["price_eur_mwh"]] = Qnat.loc[after, INPUTS + ["price_eur_mwh"]].values
        files[f"thesis_features_interpolated{tag}_full.csv"] = add_features(I, srmc)
        files[f"thesis_features_15min{tag}_full.csv"] = add_features(N, srmc)
    return {k: finish(v, start, end) for k, v in files.items()}


def validate(out, stan_dir, overlap_end="2026-04-20"):
    print(f"\nValidation against Stan's files (overlap up to {overlap_end}):")
    for name in ("thesis_features_1hour_full.csv", "thesis_features_interpolated_full.csv",
                 "thesis_features_15min_full.csv"):
        s = pd.read_csv(Path(stan_dir) / name, parse_dates=["datetime_local"], float_precision="round_trip").set_index("datetime_local")
        o = out[name].assign(datetime_local=lambda d: pd.to_datetime(d.datetime_local)).set_index("datetime_local")
        o = o[o.index < pd.Timestamp(overlap_end)]
        print(f"  {name}: rows ours {len(o)} vs Stan {len(s)}, common {len(o.index.intersection(s.index))}")
        for col in COLUMNS[1:]:
            if col == "weekday_name":
                ok = (o[col].reindex(s.index) == s[col]).mean()
                print(f"    {col:18s} identical {100 * ok:6.2f}%")
                continue
            d = (o[col].reindex(s.index) - s[col]).abs()
            bad = d[d > 1e-6]
            days = sorted(set(bad.index.date))
            show = ", ".join(str(x) for x in days[:4]) + (" ..." if len(days) > 4 else "")
            print(f"    {col:18s} identical {100 * (1 - len(bad) / len(d)):6.2f}%  "
                  f"mean|d| {d.mean():8.4f}  max {d.max():9.2f}  days {len(days):3d}  {show}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plan", choices=["A", "B"], required=True)
    p.add_argument("--raw", default="raw")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default="2026-10-01", help="exclusive")
    p.add_argument("--out", default=None)
    p.add_argument("--validate", default=None, metavar="STAN_DIR")
    p.add_argument("--leaky-interp", action="store_true",
                   help="Stan's cross-midnight interpolation (look-ahead in Peod_d1, issue 18). Replication only")
    p.add_argument("--native-inputs", action="store_true",
                   help="Plan A only: ENTSO-E 15-min forecasts as inputs of all 15-min files over the whole span")
    a = p.parse_args()
    if a.native_inputs and a.plan != "A":
        p.error("--native-inputs is defined for Plan A only")
    start = a.start or ("2025-01-16" if a.plan == "B" else "2024-10-02")
    out_dir = Path(a.out or f"out_plan{a.plan}" + ("_leaky" if a.leaky_interp else "")
                   + ("_native" if a.native_inputs else ""))
    out_dir.mkdir(exist_ok=True)
    print(f"Plan {a.plan}, interpolation: "
          + ("LEAKY (cross-midnight, as in Stan's files)" if a.leaky_interp else "day-bounded")
          + (", native 15-min inputs" if a.native_inputs else "")
          + f" -> {out_dir}")
    files = build(a.plan, Path(a.raw), start, a.end, day_bounded=not a.leaky_interp,
                  native_inputs=a.native_inputs)
    for name, df in files.items():
        df.to_csv(out_dir / name, index=False)
        n_days = pd.to_datetime(df.datetime_local).dt.normalize().nunique()
        print(f"{name:42s} {len(df):7d} rows  {n_days} days  {df.datetime_local.iloc[0]} -> "
              f"{df.datetime_local.iloc[-1]}  NaN {int(df.isna().sum().sum())}")
    if a.validate:
        validate(files, a.validate)


if __name__ == "__main__":
    main()
