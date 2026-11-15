# Native versus interpolated training data under conformal calibration

Code for the paper

> C. Tzagkarakis, S. van Est and R. Dekker, "Native versus interpolated training data under conformal calibration: Probabilistic time series forecasting after a change in temporal resolution", submitted to ESANN 2027, Bruges, Belgium, 21–23 April 2027.

The paper asks whether training on the short history of a new, finer target resolution improves probabilistic forecasts beyond what post hoc conformal calibration of a model trained on an interpolated archive provides. The testbed is the Dutch day-ahead electricity price, which moved from hourly to 15-minute resolution on 1 October 2025. The test year runs from 1 October 2025 to 30 September 2026.

This repository holds the code that downloads the data, builds the three training settings, fits the base models (seasonal naive, LEAR, XGBoost), applies the conformal layer (Split-CQR, Rolling-CQR, ACI-frozen, ACI-CQR, pooled ACI-CQR, ACI-block and the diagnostic layers Sym-block, Loc-block and Loc-slot) and evaluates them. It also holds the aggregate results behind every table, figure and result quoted in the paper. The configuration details (data spans, formulas, grids, selected hyperparameters, conformal and evaluation settings, software versions) are in [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md).

## Data

The repository contains no data. Most of the series may not be redistributed, so `scripts/01_download_raw.py` downloads them from the original sources and `scripts/03_build_datasets.py` builds the model inputs.

| Series | Source | Span downloaded | Redistribution |
|---|---|---|---|
| Day-ahead prices, Netherlands, 60-min to 30 Sep 2025 and 15-min from 1 Oct 2025 (Transparency Regulation item 12.1.d) | ENTSO-E Transparency Platform | 1 Sep 2023 – 30 Sep 2026 | Not on ENTSO-E's list of data available for free re-use |
| Day-ahead total load forecast, Netherlands (6.1.b) | ENTSO-E Transparency Platform | 1 Sep 2023 – 30 Sep 2026 | On the list, CC BY 4.0 |
| Day-ahead wind and solar generation forecasts, Netherlands (14.1.d) | ENTSO-E Transparency Platform | 1 Sep 2023 – 30 Sep 2026 | Not on the list |
| TTF front month (`TTF=F`), EUA tracker (`CO2.L`), EUR/USD (`EURUSD=X`), daily closes | Yahoo Finance | 1 Aug 2023 – 30 Sep 2026 | Not permitted by the provider's terms |
| Rotterdam coal (API2) front month, daily | investing.com, manual download | 1 Sep 2023 – 30 Sep 2026 | Not permitted by the provider's terms |

ENTSO-E data are used under the ENTSO-E Transparency Platform Terms and Conditions. Source of the ENTSO-E series: ENTSO-E Transparency Platform. Our files are derived from them (time zone conversion, aggregation to hourly means, interpolation, lags and marginal-cost features).

`data_manifest_sha256.txt` lists the sha256 of the six raw files and of the five built files (the three model inputs and the two hour-start files not used in the paper). The exact files are available from the corresponding author on reasonable request, subject to the terms of use of the data providers.

## Requirements

- Python 3.12. The paper's runs used Python 3.12.16 on macOS (Apple Silicon) with the packages pinned in `requirements_mac.txt` (versions in `env_versions_mac.txt`).
- pandas is pinned at 3.0.5. pandas 3.1.1 changes the last bits of several built columns (see Numerical reproducibility below).
- On macOS, xgboost needs OpenMP: `brew install libomp`.
- The download scripts also need `entsoe-py` and `yfinance` (`requirements_data.txt`). The downloads of the paper were made on 12 October 2026 with entsoe-py 0.8.1, yfinance 1.7.0 and pandas 2.2.3. The downloads have not been run under pandas 3.0.5.
- An ENTSO-E Transparency Platform account with a security token for the RESTful API.

```bash
python3.12 -m venv ~/venvs/esann27
source ~/venvs/esann27/bin/activate
pip install -r requirements_mac.txt -r requirements_data.txt
```

## Reproducing the paper

Run every command from the repository root.

### 1. Download the data

Set the token as an environment variable in your shell. Never write it into a file or a command line that is logged.

```bash
export ENTSOE_API_KEY="<your token>"
python scripts/00_smoke_test.py          # checks the token
python scripts/01_download_raw.py        # ENTSO-E prices and forecasts, Yahoo closes, into raw/
```

The coal series has no download API. On investing.com, open the historical data of the Rotterdam coal futures (front month), choose daily data from 1 September 2023 to 30 September 2026, download the CSV and save it unchanged as `raw/api2_investing.csv`. The build reads the columns `Date` (MM/DD/YYYY) and `Price`. Run `python scripts/01_download_raw.py --sources coal` to check its range.

### 2. Build the model inputs

```bash
python scripts/03_build_datasets.py --plan A --native-inputs    # writes out_planA_native/
python scripts/check_manifest.py                                # compares raw/ and out_planA_native/ with the manifest
```

The models read three of the five built files: `thesis_features_1hour_full.csv` (hourly setting), `thesis_features_interpolated_mp_full.csv` (interpolated setting) and `thesis_features_15min_mp_full.csv` (native setting, and the scoring target of every setting).

The package checks the sha256 of these three files on every load and stops if they differ from ours. A new download can differ from ours: ENTSO-E revises published forecasts and Yahoo revises past closes. If `check_manifest.py` reports a difference and you want to run on your own files, set

```bash
export ESANN27_ALLOW_HASH_MISMATCH=1
```

The package then warns instead of stopping. Results computed on such files can differ from the paper's.

### 3. Run the experiments

```bash
NJOBS=8 bash scripts/reproduce.sh
```

`reproduce.sh` runs, in order: the naive forecasts, LEAR with the λ path and the penalty selection, XGBoost with the frozen hyperparameters in `esann27/models/hyperparameters_*.json`, the conformal layer for every variant, the monthly XGBoost re-tuning (robustness check), the native-share ablation, and the evaluation. Logs go to `logs/`. On an Apple Silicon MacBook with 8 performance cores the full run takes roughly 12 hours, of which LEAR takes about 9 (measured with 8 workers for LEAR and 12 for XGBoost, the re-tuning and the ablation).

### 4. Where the paper's numbers come from

| Paper | File |
|---|---|
| Table 1 | `results/table1.tex` |
| Figure 1 | `results/fig1.pdf` |
| Figure 2 | `results/fig2a.pdf` |
| Every result quoted in the text | `results/numbers.tex` (one `\newcommand` per number) |
| All tables and tests in plain text | `results/summary_lear.md`, `results/summary_xgb.md` |

The `results/` folder of this repository holds the outputs of the runs reported in the paper. They were produced with an internal development repository whose code equals the code here, apart from the release changes listed in `docs/CONFIGURATION.md`, Section 10. Each summary names the internal commit of its run: 17792b2 for `summary_lear.md`, c8f107d for `summary_xgb.md`. Between the two commits only the Figure 2 panel titles and the commit label changed. We checked this on a fresh clone: rerunning the evaluation stage from this repository on the forecast and conformal files of the paper reproduces `numbers.tex`, the tables and the result CSVs exactly, and the figures apart from their creation dates. The summaries then differ only in the commit named in their header and in wall times.

### Numerical reproducibility

- The data build is byte-identical on macOS arm64 and Linux x86-64 for identical raw files (pandas 3.0.5, `float_precision="round_trip"` in every CSV read, an interpolation written without fused multiply-add).
- Forecasts are deterministic on one machine for any number of workers. Across machines they agree to a tolerance, not bit for bit. XGBoost's row and column subsampling draws differ between platforms for the same seed.
- Seeds are fixed at 2027.

## Repository layout

| Path | Content |
|---|---|
| `scripts/` | Data download (`00`, `01`), build (`03`), manifest check, `reproduce.sh` |
| `esann27/` | The package: `data.py`, `models/` (naive, LEAR, XGBoost, frozen hyperparameters), `run.py`, `select.py`, `conformal.py`, `retune.py`, `ablation.py`, `evaluate.py`, `metrics.py` |
| `tests/` | pytest suite. Tests that need data or outputs skip when these are absent |
| `results/` | Aggregate outputs of the paper's runs |
| `docs/CONFIGURATION.md` | Configuration supplement |

Code comments refer to internal design documents (the "handover" and the step specifications), which are not part of this repository. `docs/CONFIGURATION.md` states every setting they fix.

## Tests

```bash
python -m pytest -q -rs
```

Without `out_planA_native/`, `forecasts/` and `conformal/`, 121 of the 169 tests skip and report why (81 with the built data only). With `ESANN27_ALLOW_HASH_MISMATCH=1`, or on a machine other than macOS arm64, the tests against our reference hashes and outputs skip as well.

## Citation

See `CITATION.cff`. The paper is under review. This section will carry the proceedings reference once it exists.

## License

The code is released under the MIT License (`LICENSE`). The license covers the code only, not the data sources listed above, whose terms apply.

## Acknowledgments

This work has been funded by the EU under Horizon Europe's TwinODIS project (No. 101160009).

## Contact

Christos Tzagkarakis, Institute of Computer Science, FORTH, Heraklion, Greece. tzagarak@ics.forth.gr, or ctzagkarakis@gmail.com.
