"""
01_download_raw.py  (v3: v2 plus vintage snapshots of the ENTSO-E day-ahead forecasts)

Downloads the raw inputs for the Dutch day-ahead dataset, 1 Sep 2023 - 30 Sep 2026.

  ENTSO-E  day-ahead prices NL                 native MTU: 60-min before / 15-min from 1 Oct 2025
  ENTSO-E  day-ahead total load forecast NL    (A65, process A01)
  ENTSO-E  day-ahead wind & solar forecast NL  (A69, process A01)
  Yahoo    daily TTF (TTF=F), EUA tracker (CO2.L), EUR/USD (EURUSD=X)
  manual   API2 Rotterdam coal from investing.com -> raw/api2_investing.csv (only checked here)

Design dates
  download start   1 Sep 2023   (covers the optional 728-day robustness window)
  core start      25 Sep 2024   (364-day window + 7-day lag buffer)
  test year        1 Oct 2025 - 30 Sep 2026

Usage (venv active, token set):
  export ENTSOE_API_KEY="..."                            # bash / zsh
  $env:ENTSOE_API_KEY = "..."                            # PowerShell
  python 01_download_raw.py                              # prices, forecasts, yahoo, coal check
  python 01_download_raw.py --sources forecasts          # any subset of: prices forecasts yahoo coal

Vintage test of the day-ahead forecasts (A65 load, A69 wind/solar, process A01) for one delivery day D.
These two options only read the API and write raw/entsoe_snapshot_*.csv; nothing else is downloaded or overwritten:
  python 01_download_raw.py --snapshot 2026-10-19          # on D-1 at ~11:45 CEST (and again at ~18:30 if wanted)
  python 01_download_raw.py --compare-snapshot 2026-10-19  # after D: every snapshot of D vs the values served now

Raw files are saved untouched with UTC timestamps under ./raw/. No cleaning or filling here.
The GATE CHECK at the end reports 15-min completeness of the forecasts for the core and
robustness windows: this decides whether the design holds as planned.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd

TZ = "Europe/Amsterdam"
DOWNLOAD_START = pd.Timestamp("2023-09-01", tz=TZ)
ROBUST_START = pd.Timestamp("2023-09-27", tz=TZ)
CORE_START = pd.Timestamp("2024-09-25", tz=TZ)
GO_LIVE = pd.Timestamp("2025-10-01", tz=TZ)
END = pd.Timestamp("2026-10-01", tz=TZ)          # exclusive: last interval 30 Sep 2026 23:45
RAW = Path("raw")
PASS_LEVEL = 0.995

YAHOO_TICKERS = {"TTF=F": "ttf_eur_mwh", "CO2.L": "eua_tracker", "EURUSD=X": "eurusd"}
WS_RENAME = {"Solar": "solar_mw", "Wind Onshore": "wind_onshore_mw", "Wind Offshore": "wind_offshore_mw"}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def month_chunks(start, end):
    edges = [start] + list(pd.date_range(start, end, freq="MS", inclusive="neither")) + [end]
    return list(zip(edges[:-1], edges[1:]))


def fetch_chunks(fn, label, start, end, pause=0.3):
    """Call fn(a, b) per calendar month, trim to [a, b), concatenate, report failed chunks."""
    from entsoe.exceptions import NoMatchingDataError
    parts, failed = [], []
    for a, b in month_chunks(start, end):
        try:
            x = fn(a, b)
        except NoMatchingDataError:
            failed.append(f"{a:%Y-%m} no data")
            continue
        except Exception as e:  # keep going, report at the end
            failed.append(f"{a:%Y-%m} {type(e).__name__}: {str(e)[:100]}")
            continue
        parts.append(x[(x.index >= a) & (x.index < b)])     # entsoe-py also returns the end point
        time.sleep(pause)
    if failed:
        print(f"  {label}: {len(failed)} month(s) failed or empty -> {failed[:8]}")
    if not parts:
        return None
    out = pd.concat(parts)
    return out[~out.index.duplicated(keep="last")].sort_index()


def step_runs(idx):
    """Median sampling step per local month, compressed into runs: [(first, last, minutes), ...]."""
    idx = pd.DatetimeIndex(idx)
    step = pd.Series(idx, index=idx).diff().dt.total_seconds().div(60)
    med = step.groupby(idx.tz_convert(TZ).strftime("%Y-%m")).median()
    runs = []
    for month, v in med.items():
        if runs and runs[-1][2] == v:
            runs[-1][1] = month
        else:
            runs.append([month, month, v])
    return runs


def completeness(idx_valid, start, end, freq="15min"):
    """Share of the regular grid [start, end) that has a non-missing value."""
    grid = pd.date_range(start.tz_convert("UTC"), end.tz_convert("UTC"), freq=freq, inclusive="left")
    return len(pd.DatetimeIndex(idx_valid).tz_convert("UTC").intersection(grid)) / len(grid)


def save(obj, path):
    df = obj.to_frame() if isinstance(obj, pd.Series) else obj.copy()
    df.index = pd.DatetimeIndex(df.index).tz_convert("UTC")
    df.index.name = "timestamp_utc"
    df.to_csv(path)


def describe_series(label, s):
    runs = ", ".join(f"{a}..{b}: {int(m)} min" for a, b, m in step_runs(s.index))
    print(f"  {label}: {s.notna().sum()} values, {s.index.min()} -> {s.index.max()}")
    print(f"    sampling step by month: {runs}")


# ---------------------------------------------------------------------------
# ENTSO-E
# ---------------------------------------------------------------------------
def client_from_env(key):
    from entsoe import EntsoePandasClient
    return EntsoePandasClient(api_key=key)


def download_prices(client):
    ser = fetch_chunks(lambda a, b: client.query_day_ahead_prices("NL", start=a, end=b),
                       "prices", DOWNLOAD_START, END)
    if ser is None:
        print("  prices: no data")
        return
    for res, mask, s, e in [("60min", ser.index < GO_LIVE, DOWNLOAD_START, GO_LIVE),
                            ("15min", ser.index >= GO_LIVE, GO_LIVE, END)]:
        part = ser[mask].rename("price_eur_mwh")
        save(part, RAW / f"entsoe_dayahead_{res}.csv")
        comp = completeness(part.dropna().index, s, e, res)
        print(f"  prices {res} ({s.date()} -> {e.date()}): {len(part)} rows, "
              f"{100 * comp:.2f}% of the {res} grid")


def download_forecasts(client):
    out = {}
    load = fetch_chunks(lambda a, b: client.query_load_forecast("NL", start=a, end=b, process_type="A01"),
                        "load forecast", DOWNLOAD_START, END)
    if load is not None:
        load = load.squeeze("columns") if isinstance(load, pd.DataFrame) else load
        load = load.rename("load_forecast_mw")
        save(load, RAW / "entsoe_load_forecast_da.csv")
        describe_series("load forecast", load)
        out["load_forecast_mw"] = load

    ws = fetch_chunks(lambda a, b: client.query_wind_and_solar_forecast("NL", start=a, end=b,
                                                                        psr_type=None, process_type="A01"),
                      "wind/solar forecast", DOWNLOAD_START, END)
    if ws is not None:
        ws = ws.to_frame() if isinstance(ws, pd.Series) else ws
        ws = ws.rename(columns=WS_RENAME)
        save(ws, RAW / "entsoe_wind_solar_forecast_da.csv")
        for c in ws.columns:
            describe_series(f"{c}", ws[c])
            out[c] = ws[c]
    return out


def gate_check(series):
    print("\nGATE CHECK: 15-min completeness of day-ahead forecasts")
    print(f"  {'series':22s} {'core 25 Sep 2024-30 Sep 2026':>30s} {'robustness 27 Sep 2023-24 Sep 2024':>36s}")
    for name, s in series.items():
        valid = s.dropna().index
        core = completeness(valid, CORE_START, END)
        rob = completeness(valid, ROBUST_START, CORE_START)
        flag = lambda x: "PASS" if x >= PASS_LEVEL else "CHECK"
        print(f"  {name:22s} {100 * core:24.2f}% {flag(core):5s} {100 * rob:30.2f}% {flag(rob):5s}")
    print("  (an hourly series shows ~25% here: that would mean 15-min forecasts are not available)")


# ---------------------------------------------------------------------------
# Yahoo Finance and coal
# ---------------------------------------------------------------------------
def download_yahoo():
    import yfinance as yf
    df = yf.download(list(YAHOO_TICKERS), start="2023-08-01", end="2026-10-01",
                     interval="1d", auto_adjust=False, progress=False)
    close = df["Close"].rename(columns=YAHOO_TICKERS)
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close.index.name = "date"
    close.to_csv(RAW / "yahoo_daily_close.csv")
    for t, name in YAHOO_TICKERS.items():
        s = close[name].dropna()
        try:
            cur = yf.Ticker(t).fast_info["currency"]
        except Exception:
            cur = "?"
        print(f"  Yahoo {t:9s} -> {name:12s} {len(s)} days, {s.index.min().date()} -> "
              f"{s.index.max().date()}, last {s.iloc[-1]:.2f} {cur}")


def check_coal():
    p = RAW / "api2_investing.csv"
    if not p.exists():
        print(f"  coal: {p} not found yet (manual download from investing.com, see instructions)")
        return
    df = pd.read_csv(p)
    date_col = next(c for c in df.columns if "date" in c.lower())
    price_col = next(c for c in df.columns if c.lower() in ("price", "close", "last"))
    d = pd.to_datetime(df[date_col], format="mixed", dayfirst=False)
    v = pd.to_numeric(df[price_col].astype(str).str.replace(",", ""), errors="coerce")
    print(f"  coal (investing.com): {v.notna().sum()} days, {d.min().date()} -> {d.max().date()}, "
          f"last {v[d.idxmax()]:.2f}, min {v.min():.2f}, max {v.max():.2f}")
    if d.min() > pd.Timestamp("2023-09-01") or d.max() < pd.Timestamp("2026-09-29"):
        print("    WARNING: range does not cover 1 Sep 2023 - 30 Sep 2026")


# ---------------------------------------------------------------------------
# vintage snapshots of the day-ahead forecasts
# ---------------------------------------------------------------------------
SNAP_SERIES = ["load_forecast_mw", "solar_mw", "wind_onshore_mw", "wind_offshore_mw"]


def gate_closure(day):
    """12:00 local time on the day before delivery day `day`."""
    return (pd.Timestamp(day) - pd.Timedelta(days=1) + pd.Timedelta(hours=12)).tz_localize(TZ)


def fetch_day(client, day):
    """A65 load and A69 wind/solar forecasts (process A01) for local delivery day `day`, as served now.
    Long format: timestamp_utc, series, mw. A series that is not published yet is simply absent."""
    from entsoe.exceptions import NoMatchingDataError
    a = pd.Timestamp(day).tz_localize(TZ)
    b = (pd.Timestamp(day) + pd.Timedelta(days=1)).tz_localize(TZ)     # 23, 24 or 25 h
    parts = []
    try:
        load = client.query_load_forecast("NL", start=a, end=b, process_type="A01")
        load = load.squeeze("columns") if isinstance(load, pd.DataFrame) else load
        parts.append(load.rename("load_forecast_mw"))
    except NoMatchingDataError:
        pass
    try:
        ws = client.query_wind_and_solar_forecast("NL", start=a, end=b, psr_type=None, process_type="A01")
        ws = (ws.to_frame() if isinstance(ws, pd.Series) else ws).rename(columns=WS_RENAME)
        parts += [ws[c] for c in ws.columns]
    except NoMatchingDataError:
        pass
    rows = []
    for s in parts:
        s = s[(s.index >= a) & (s.index < b)].dropna()
        rows.append(pd.DataFrame({"timestamp_utc": pd.DatetimeIndex(s.index).tz_convert("UTC"),
                                  "series": s.name, "mw": s.values}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["timestamp_utc", "series", "mw"])


def snapshot(client, day):
    taken = pd.Timestamp.now(tz="UTC")
    df = fetch_day(client, day)
    df["taken_utc"] = taken
    path = RAW / f"entsoe_snapshot_{day}_{taken.tz_convert(TZ):%Y%m%d_%H%M%S}.csv"
    df.to_csv(path, index=False)
    gate = gate_closure(day)
    n = df.groupby("series").size().reindex(SNAP_SERIES).fillna(0).astype(int).to_dict()
    print(f"Delivery day {day}: snapshot taken {taken.tz_convert(TZ):%Y-%m-%d %H:%M} local, "
          f"{'BEFORE' if taken < gate else 'after'} gate closure ({gate:%Y-%m-%d %H:%M})")
    print(f"  values per series: {n}   (a full day has 96, 92 or 100 on DST days; 0 = not published yet)")
    print(f"  saved -> {path}")
    print(f"After delivery, run:  python 01_download_raw.py --compare-snapshot {day}")


def compare_snapshot(client, day):
    files = sorted(RAW.glob(f"entsoe_snapshot_{day}_*.csv"))
    if not files:
        sys.exit(f"No snapshot for {day} in {RAW}")
    now = fetch_day(client, day)
    gate = gate_closure(day)
    print(f"Delivery day {day}: each snapshot vs the values served now (gate closure {gate:%Y-%m-%d %H:%M} local)\n")
    print(f"  {'snapshot taken (local)':22s} {'series':17s} {'n snap':>6s} {'n now':>6s} {'changed':>8s} "
          f"{'mean |change| MW':>17s} {'max |change| MW':>16s}")
    for f in files:
        old = pd.read_csv(f, parse_dates=["timestamp_utc", "taken_utc"])
        taken = (pd.to_datetime("".join(f.stem.split("_")[-2:]), format="%Y%m%d%H%M%S").tz_localize(TZ) if old.empty
                 else old.taken_utc.iloc[0].tz_convert(TZ))
        tag = f"{taken:%Y-%m-%d %H:%M} ({'pre' if taken < gate else 'post'}-gate)"
        for s in SNAP_SERIES:
            o = old[old.series == s].set_index("timestamp_utc").mw
            n = now[now.series == s].set_index("timestamp_utc").mw
            j = pd.concat([o.rename("snap"), n.rename("now")], axis=1).dropna()
            d = (j.snap - j.now).abs()
            ch = f"{100 * (d > 0.5).mean():7.1f}%" if len(j) else "      - "
            print(f"  {tag:22s} {s:17s} {len(o):6d} {len(n):6d} {ch} {d.mean() if len(j) else float('nan'):17.1f} "
                  f"{d.max() if len(j) else float('nan'):16.1f}")
    print("\n  n snap = 0 in a pre-gate snapshot: the forecast was not published at gate closure (look-ahead real).")
    print("  n snap = n now and changed 0% in a pre-gate snapshot: the version served now existed at gate closure.")
    print("  changed > 0%: the platform now serves a revised version of the day-ahead forecast.")


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sources", nargs="+", default=["prices", "forecasts", "yahoo", "coal"],
                   choices=["prices", "forecasts", "yahoo", "coal"])
    p.add_argument("--entsoe-key", default=os.environ.get("ENTSOE_API_KEY"))
    p.add_argument("--snapshot", metavar="YYYY-MM-DD",
                   help="save the day-ahead forecasts for delivery day D as served now (run on D-1 before 12:00)")
    p.add_argument("--compare-snapshot", metavar="YYYY-MM-DD",
                   help="compare every saved snapshot of delivery day D with the values served now")
    args = p.parse_args()
    RAW.mkdir(exist_ok=True)

    if args.snapshot or args.compare_snapshot:
        if not args.entsoe_key:
            sys.exit("ENTSOE_API_KEY not set")
        client = client_from_env(args.entsoe_key)
        if args.snapshot:
            snapshot(client, args.snapshot)
        else:
            compare_snapshot(client, args.compare_snapshot)
        return

    if {"prices", "forecasts"} & set(args.sources):
        if not args.entsoe_key:
            sys.exit("ENTSOE_API_KEY not set")
        client = client_from_env(args.entsoe_key)
    if "prices" in args.sources:
        print("ENTSO-E day-ahead prices ...")
        download_prices(client)
    if "forecasts" in args.sources:
        print("ENTSO-E day-ahead load and wind/solar forecasts ...")
        series = download_forecasts(client)
        if series:
            gate_check(series)
    if "yahoo" in args.sources:
        print("Yahoo Finance ...")
        download_yahoo()
    if "coal" in args.sources:
        print("Coal ...")
        check_coal()
    print(f"Done. Files in {RAW.resolve()}")


if __name__ == "__main__":
    main()
