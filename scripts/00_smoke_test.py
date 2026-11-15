"""
Smoke test for the ENTSO-E token: one day (30 Sep 2026) of 15-min NL day-ahead prices.
Expected: 96 rows, 00:00 -> 23:45.

Usage (venv active):
  export ENTSOE_API_KEY="your-token"        # bash / zsh
  $env:ENTSOE_API_KEY = "your-token"        # PowerShell
  python 00_smoke_test.py
"""
import inspect
import os
import sys

import entsoe
import pandas as pd
from entsoe import EntsoePandasClient

key = os.environ.get("ENTSOE_API_KEY")
if not key:
    sys.exit("ENTSOE_API_KEY is not set in this shell (bash/zsh: export ENTSOE_API_KEY=..., "
             "PowerShell: $env:ENTSOE_API_KEY = \"...\").")

client = EntsoePandasClient(api_key=key)
params = list(inspect.signature(client.query_day_ahead_prices).parameters)
print(f"entsoe-py {entsoe.__version__} | query_day_ahead_prices args: {params}")

start = pd.Timestamp("2026-09-30", tz="Europe/Amsterdam")
end = pd.Timestamp("2026-10-01", tz="Europe/Amsterdam")

if "resolution" in params:
    s = client.query_day_ahead_prices("NL", start=start, end=end, resolution="15min")
else:
    s = client.query_day_ahead_prices("NL", start=start, end=end)

print(f"rows: {len(s)} | first: {s.index[0]} | last: {s.index[-1]} | "
      f"step: {pd.Series(s.index).diff().median()}")
print(s.describe())
