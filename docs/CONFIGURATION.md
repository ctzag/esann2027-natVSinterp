# Configuration details

Configuration of the experiments in "Native versus interpolated training data under conformal calibration: Probabilistic time series forecasting after a change in temporal resolution" (ESANN 2027). This file states settings only. All results are in the paper and in `results/`.

Times are local (Europe/Amsterdam) unless marked UTC. "Quarter" is a 15-minute delivery period. D is the delivery day.

## 1. Data sources

| Series | Source, query | Raw file | Span |
|---|---|---|---|
| Day-ahead price, NL | ENTSO-E Transparency Platform, `query_day_ahead_prices("NL")` (entsoe-py). Native resolution 60 min to 30 Sep 2025, 15 min from 1 Oct 2025 | `raw/entsoe_dayahead_60min.csv`, `raw/entsoe_dayahead_15min.csv` | 1 Sep 2023 – 30 Sep 2026 |
| Day-ahead total load forecast, NL | ENTSO-E, A65, process A01 | `raw/entsoe_load_forecast_da.csv` | 1 Sep 2023 – 30 Sep 2026, 15 min |
| Day-ahead wind and solar forecasts, NL (solar, wind onshore, wind offshore) | ENTSO-E, A69, process A01 | `raw/entsoe_wind_solar_forecast_da.csv` | 1 Sep 2023 – 30 Sep 2026, 15 min |
| TTF front month (EUR/MWh), EUA tracker `CO2.L` (EUR, about one allowance per unit), EUR/USD | Yahoo Finance, `TTF=F`, `CO2.L`, `EURUSD=X`, daily close, `auto_adjust=False` | `raw/yahoo_daily_close.csv` | 1 Aug 2023 – 30 Sep 2026 |
| API2 Rotterdam coal, front month (USD/t) | investing.com, manual CSV download | `raw/api2_investing.csv` | 1 Sep 2023 – 30 Sep 2026 |

ENTSO-E files are stored with UTC timestamps. The sha256 of every raw file is in `data_manifest_sha256.txt`.

## 2. Built datasets and settings

`python scripts/03_build_datasets.py --plan A --native-inputs` writes five files to `out_planA_native/`, from 2 Oct 2024 to 30 Sep 2026 (729 days, lags from 10 extra days of history). The models use three of them.

| Setting | File | Rows | Training target | Inputs |
|---|---|---|---|---|
| Hourly | `thesis_features_1hour_full.csv` | 17,494 | ENTSO-E hourly price to 30 Sep 2025, mean of the four quarter prices from 1 Oct 2025 | Hourly means of the 15-min ENTSO-E forecasts |
| Interpolated | `thesis_features_interpolated_mp_full.csv` | 69,976 | The hourly price interpolated to quarters by eq. (1), on every day | 15-min ENTSO-E forecasts |
| Native | `thesis_features_15min_mp_full.csv` | 69,976 | As interpolated before 1 Oct 2025, the quarter price from 1 Oct 2025 | 15-min ENTSO-E forecasts |

The interpolated and the native file differ only in the price and the price-derived columns from 1 Oct 2025. The scoring target of every setting is the quarter price in the native file. The remaining two built files (`thesis_features_interpolated_full.csv`, `thesis_features_15min_full.csv`) use hour-start linear interpolation and are not used in the paper.

Daylight saving time: the spring-forward day has 23 hours (92 quarters). On the fall-back day the duplicated hour is averaged.

## 3. Information set and timing

- Forecasts for D are issued at gate closure, 12:00 CET on D−1.
- Prices: up to D−1. Lags and previous-day statistics are computed within day boundaries, so no value of D enters a feature of D.
- Load forecast: available before gate closure.
- Wind and solar forecasts: published about 2 to 3 hours after gate closure (snapshots of deliveries 18, 19 and 20 Oct 2026). All settings use the same series. This is the one exception to the gate-closure information set.
- Fuel, carbon and currency: daily close of D−2, forward-filled over weekends and holidays.

## 4. Mean-preserving interpolation

For quarter q = 0, …, 3 of hour H with hourly prices P_{H−1}, P_H, P_{H+1}:

P̃_{H,q} = P_H + a_q (P_{H+1} − P_{H−1}) + b_q (P_{H+1} − 2P_H + P_{H−1}), with a = (−3, −1, 1, 3)/16 and b = (1, −1, −1, 1)/16. (1)

This is linear interpolation between hour midpoints, shifted so that the four quarters average to P_H. In the last hour of a day P_{H+1} is replaced by P_H (day-bounded), so nothing from D+1 enters D. The first hour of a day uses the last hour of the previous day, which is known at gate closure.

## 5. Features

| Feature | Definition |
|---|---|
| `price_lag_d1`, `price_lag_d7` | Price of the same quarter (hourly setting: hour) 1 and 7 days earlier |
| `Ymax_d1`, `Ymin_d1`, `Peod_d1` | Maximum, minimum and last value of the price on D−1 |
| `solar_mw`, `wind_onshore_mw`, `wind_offshore_mw` | ENTSO-E day-ahead forecasts |
| `residual_load_mw` | Load forecast − solar − wind onshore − wind offshore (net load of the TSO-visible fleet) |
| `srmc_ccgt` | TTF/0.52 + EUA · 0.202/0.52 + 2.5 (EUR/MWh) |
| `srmc_coal` | API2/(6.978 · EURUSD)/0.40 + EUA · 0.341/0.40 + 5.0 (EUR/MWh) |
| `dow_tue` … `dow_sun` | Weekday dummies, Monday the reference |
| `q15`, `q30`, `q45` | Quarter-of-hour dummies, HH:00 the reference (15-min settings only) |

The marginal costs and the previous-day statistics are constant within a day. `load_mw` is stored but not used as a regressor. Regressors: 17 in the hourly setting, 20 in the 15-min settings.

## 6. Experimental protocol

| Element | Setting |
|---|---|
| Test year | 1 Oct 2025 – 30 Sep 2026, 365 days, 35,036 quarters |
| Window | The 364 days D−364 … D−1, refit every day |
| Fits per test day | 24, one per clock hour. 15-min settings: the four quarters pooled, 1,456 rows (1,452–1,456 around DST). Hourly setting: 363–364 rows |
| Output | Point forecast and quantiles 0.1, …, 0.9 for the 96 quarters of D, sorted per row. Hourly forecasts are repeated over the four quarters |
| Native share of the window | π_D = (number of window days on or after 1 Oct 2025)/364, from 0 on 1 Oct 2025 to 1 on 30 Sep 2026 |
| Conformal burn-in | 1–28 Oct 2025 |
| Evaluation period | 29 Oct 2025 – 30 Sep 2026, 337 days, 32,348 quarters, for every reported score |
| Seeds | 2027 everywhere |

## 7. Base models

### Seasonal naive

Point = `price_lag_d7`. Quantile τ = point + the empirical τ-quantile (numpy `linear`, R type 7) of price − `price_lag_d7` over the window rows of the same clock hour, the four quarters pooled. Used as a raw benchmark and as the rMAE denominator. Not conformalized.

### LEAR

| Step | Setting |
|---|---|
| Variance-stabilizing transform | med and MAD/Φ⁻¹(0.75) of the window target (MAD < 1e-6 → 1), asinh((x − med)/MAD) on the target and on the five price features |
| Standardization | The 11 continuous regressors z-scored with the window mean and population standard deviation (std < 1e-12 → 1), the same transform on the test rows. Dummies 0/1 |
| Point model | scikit-learn `LassoCV(alphas=100, eps=1e-4, max_iter=100000, tol=1e-7)`, minimum mean CV error, refit on the window. 10 folds of whole days: one `default_rng(2027)` per fit, a permutation of the 364 window days, fold of the day at permuted position i = i mod 10. Duan smearing back-transform with the in-sample residuals |
| Quantile models | `QuantileRegressor(quantile=τ, alpha=λ/(2n), solver="highs")`, intercept unpenalized, for λ_k = 10^(−1 + 4k/19), k = 0, …, 19 (n = rows of the fit). Plain inverse transform |
| Penalty selection | Per hour-fit and level: minimize log(Σ ρ_τ) + m · log(n) · log(p)/(2n) over the 20 λ, m = number of coefficients with \|β_j\| > 1e-8 (Uniejewski and Weron 2021, eq. 11). Ties to the smallest λ. Point forecast as fallback if no λ solves |
| Alternative rules (code only) | `stan`: m · log(n)/(2n · log p), `bic`: m · log(n)/(2n). `esann27.select --rule` |

### XGBoost

| Element | Setting |
|---|---|
| Regressors | As LEAR, raw target, no transform, no standardization |
| Objectives | Point `reg:squarederror`, quantiles `reg:quantileerror` with `quantile_alpha = τ`, one booster per level |
| Fixed | xgboost 3.4.1, `tree_method=hist`, `nthread=1`, `seed=2027`, `subsample=0.8`, `colsample_bytree=0.8`, `min_child_weight=5`, `gamma=0`, `lambda=1` |
| Tuning | Once per structure (15-min, hourly) on the window of 1 Oct 2025 (2 Oct 2024 – 30 Sep 2025): first 308 days train, last 56 days validate (6 Aug – 30 Sep 2025), 24 hour-fits as in the rolling run. Every configuration boosted to 1,000 rounds, validation loss (point: MAE, quantiles: pinball) summed over the 24 hour-fits per round, early stopping with patience 50 on the summed curve. Point grid max_depth {3, 5, 7} × η {0.05, 0.1}. Quantile models at the point depth, η {0.05, 0.1} per level. Ties to the first configuration in grid order. Hyperparameters then frozen for the test year |
| Files | `esann27/models/hyperparameters_15min.json`, `hyperparameters_hourly.json` (the full grid with scores) |

Selected values:

| Model | 15-min: max_depth, η, rounds | Hourly: max_depth, η, rounds |
|---|---|---|
| Point | 3, 0.05, 86 | 5, 0.05, 83 |
| q10 | 3, 0.05, 228 | 5, 0.10, 82 |
| q20 | 3, 0.05, 283 | 5, 0.05, 159 |
| q30 | 3, 0.05, 418 | 5, 0.05, 258 |
| q40 | 3, 0.10, 362 | 5, 0.05, 962 |
| q50 | 3, 0.05, 426 | 5, 0.05, 717 |
| q60 | 3, 0.05, 391 | 5, 0.05, 974 |
| q70 | 3, 0.10, 106 | 5, 0.05, 483 |
| q80 | 3, 0.05, 232 | 5, 0.10, 96 |
| q90 | 3, 0.05, 229 | 5, 0.05, 154 |

The interpolated and the native setting share the 15-min values because their first windows are identical. The 1,000-round cap binds for the hourly q40 and q60 and is kept, since the grid was fixed before tuning.

Monthly re-tuning (robustness check, model `xgbm`, interpolated and native settings): the tuning above repeated on the first test day of each month, 1 Oct 2025 … 1 Sep 2026, on the window of that day and on each setting's own file. Test day D uses the latest re-tune on or before D. `python -m esann27.retune`.

## 8. Conformal layer

| Element | Setting |
|---|---|
| Pairs | k = 1, …, 4: levels (0.1k, 1 − 0.1k), miscoverage α_k = 0.2k. The median passes through |
| Scores (asymmetric CQR) | E^lo = q̂_{0.1k} − y, E^hi = y − q̂_{1−0.1k}, one correction per side |
| Correction | The ⌈(1 − a)(n + 1) − 10⁻⁹⌉-th order statistic of the window scores, subtracted from the lower and added to the upper quantile, a the side's level. The 10⁻⁹ guards against floating-point error in (1 − a)(n + 1) |
| Sorting | The nine quantiles sorted per row after the corrections |
| Groups (Mondrian) | Quarter position p = 0, …, 3 (4 groups), scores pooled over the 24 hours. 24 scores per group per day (23 on 29 Mar 2026) |
| Window | Scores of the last K = 28 days, n = 672 per group (671 when 29 Mar 2026 is in the window) |
| Timing | Scores of D−1 are formed when the prices of D−1 are known (published on D−2) and enter the window and the level update for D |
| Adaptive level (ACI) | Per side, group and pair: a_{t+1} = a_t + γ(α_k/2 − err_t), γ = 0.05, a_1 = α_k/2, err_t the share of the group's quarters of day t beyond that side of the formal set |
| Formal and issued sets | i = ⌈(1 − a)(n + 1) − 10⁻⁹⌉. Formal side unbounded if i > n (err = 0), empty if i < 1 (err = 1). The ACI update uses the formal side. Scores use the issued side, with i clipped to [1, n] |
| ACI bound | Per side, \|(1/T) Σ err_t − α_k/2\| ≤ (max{α_k/2, 1 − α_k/2} + γ)/(Tγ) (Gibbs and Candès 2021, Prop. 4.1). T = 337: 0.056, 0.050, 0.045, 0.039 for k = 1, …, 4 |

Variants (`--regime` of `esann27.conformal`):

| Variant | Code | Scores | Level | Groups |
|---|---|---|---|---|
| Split-CQR | `split` | Burn-in only, frozen | Fixed | 4 positions |
| Rolling-CQR | `rolling` | Last 28 days | Fixed | 4 positions |
| ACI-frozen | `acifrozen` | Burn-in only, frozen | ACI | 4 positions |
| ACI-CQR | `aci` | Last 28 days | ACI | 4 positions |
| Pooled ACI-CQR | `acipooled` | Last 28 days | ACI | 1 group (96 scores per day) |
| ACI-block | `aciblock` | Last 28 days | ACI | 24 groups g = 4⌊h/4⌋ + p (h the clock hour), n = 112 per window (111 for g = 0–3 around 29 Mar 2026) |
| Sym-block (diagnostic) | `symblock` | Last 28 days, symmetric score max(q̂_{0.1k} − y, y − q̂_{1−0.1k}) | ACI two-sided, a_1 = α_k, update a_{t+1} = a_t + γ(α_k − err_t) | 24 groups as ACI-block |
| Loc-block (diagnostic) | `locblock` | Quantiles of D shifted by the median of y − q̂_{0.5} over the group's rows of D−28 … D−1, then `aci` | ACI | Shift by 24 groups, correction by 4 positions |
| Loc-slot (diagnostic, not in the paper) | `locslot` | As Loc-block with 96 slot groups 4h + p | ACI | Shift by 96 slots, correction by 4 positions |

Sym-block, Loc-block and Loc-slot are diagnostics defined after the ACI-block result. Two-sided bound of Sym-block at T = 337: 0.050 (k = 1, 4) and 0.039 (k = 2, 3).

## 9. Evaluation

| Element | Setting |
|---|---|
| Point | MAE, RMSE, rMAE = MAE / MAE of the naive in the same setting |
| CRPS | 2 × the mean pinball loss over the nine levels |
| CORP decomposition | Per level τ: isotonic τ-quantile regression of y on the issued quantile by pool-adjacent-violators, pooled lower τ-quantile (`np.quantile(method="inverted_cdf")`), ties in the forecast pooled first. MCB_τ = S_τ − S_τ(recalibrated), UNC_τ = S_τ of the unconditional τ-quantile, DSC_τ = UNC_τ − S_τ(recalibrated). Reported as 2 × the mean over τ, so CRPS = MCB − DSC + UNC |
| Coverage | PICP and mean width of [q10, q90] (80%) and [q20, q80] (60%), bounds inclusive |
| Diebold–Mariano | Daily CRPS difference native − interpolated, 337 days, Newey–West variance with lag 7, two-sided |
| Block bootstrap | Circular blocks of 7 days: `default_rng(2027).integers(0, 337, size=(1000, 49))` block starts, days (start + 0, …, 6) mod 337, 343 indices truncated to 337. CRPS, MCB, DSC recomputed per draw on the pooled quarters. 90% intervals from the 5th and 95th percentiles of the native − interpolated differences |
| Share regression | Daily CRPS gap interpolated − native on π_D, OLS with intercept, Newey–West lag 7 |
| Per-slot tests | Kupiec and Christoffersen per quarter-hour slot across days (96 slots × 337 days), Holm over the 96 tests per level (summaries only) |
| Share ablation | LEAR (penalty rule above), test days 3–30 Sep 2026. Levels π ∈ {0, 0.25, 0.5, 0.75}: the round(π · 364) most recent window days keep their native rows, older window days take theirs from the interpolated file. Test rows from the native file. Compared with the regular native run (π_D from 0.926 to 1) by DM on the 28 daily CRPS differences. `python -m esann27.ablation` |
| Proposition A | Inputs from the burn-in only: within-hour deviation scale by position (native and interpolated), the hourly setting's point-forecast residual scale and their empirical distribution. Mixture quantiles by bisection on a grid of 101 values of π. `results/pred_doseresponse.csv` |

## 10. Code release

The public repository is a single-commit copy of the internal development repository at commit `c8f107d`, with these changes.

- `esann27/data.py`: with `ESANN27_ALLOW_HASH_MISMATCH=1` a mismatch of an input hash gives one warning per file instead of an error and is recorded as "mismatch-allowed". `esann27/run.py` then logs "hash MISMATCH allowed". Default behavior unchanged.
- `tests/`: tests that need built data, forecasts, conformal outputs or result files skip when these are absent. Tests against reference data hashes skip when the variable above is set. Tests against outputs produced on macOS arm64 (committed hyperparameters, output manifests, results recomputed from forecasts) also skip on other platforms. One new test of the opt-out (169 tests).
- `scripts/00_smoke_test.py`, `scripts/01_download_raw.py`: the usage text and the missing-key message show the bash/zsh form (`export ENTSOE_API_KEY=...`) next to the PowerShell form. Text only.
- Comments, docstrings and three sentences of the summaries (`esann27/evaluate.py`, `results/summary_*.md`) that named internal file paths now name the documents neutrally, e.g. "step 5c specification". Text only.
- `scripts/reproduce.sh`, `scripts/check_manifest.py`, `data_manifest_sha256.txt`: new.
- Not released:
  - the NED download and audit (`02`, `05`)
  - the comparisons with the thesis data and between builds (`04`, `check_builds.py`, `diff_builds.py`)
  - the build tests that need Plan B or NED files (`test_03_identity.py`, `test_03_perturb.py`)
  - the run and comparison scripts of the development steps (`run_step*.sh`, `compare_step*.py`)
  - `results/summary.md` (a copy of `summary_xgb.md`), `results/fig2_lear.pdf`, `results/evaluate_lear.log`
  - internal notes

## 11. Software

Runs of the paper: macOS 27.0.1 arm64, Python 3.12.16, numpy 2.5.4, pandas 3.0.5, scipy 1.18.1, scikit-learn 1.9.1, xgboost 3.4.1, pyarrow 26.0.1, matplotlib 3.11.2, statsmodels 0.15.0 (`env_versions_mac.txt`, full list in `requirements_mac.txt`). Downloads: entsoe-py 0.8.1, yfinance 1.7.0, pandas 2.2.3 (12 Oct 2026). Figures with `pdf.fonttype = 42`.

## 12. Not run

Exponential score weights, AgACI, DtACI, EnbPI, SPCI, scaled conformal scores, a native-only training window, other grouping schemes, other base models, other bidding zones.
