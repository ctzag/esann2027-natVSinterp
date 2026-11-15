# Step 5.2 evaluation summary: XGB

`python -m esann27.evaluate --model xgb`, git c8f107d, run 2026-11-13 04:00. Evaluation period 2025-10-29 .. 2026-09-30 (337 days, 32,348 quarters), target = native price. CRPS = 2 x mean pinball over 9 levels. MCB / DSC / UNC on the CRPS scale (2 x mean over levels). Newey-West: Bartlett, lag 7, no small-sample correction, normal p. Bootstrap: circular blocks of 7 days, 49 blocks truncated to 337 days, 1000 draws, default_rng(2027), the same draws for both settings and both regimes.

## 1. Proposition A (burn-in 1-28 Oct 2025 only, computed and written before any evaluation metric)

| position | sigma_q native | interp sd | kappa_q | s_I | s_N |
|---|---|---|---|---|---|
| :00 | 22.0664 | 7.2719 | 0.3295 | 24.8979 | 32.4646 |
| :15 | 9.1120 | 2.8322 | 0.3108 | 23.9801 | 25.4961 |
| :30 | 8.9016 | 2.8450 | 0.3196 | 23.9816 | 25.4217 |
| :45 | 21.9797 | 7.2568 | 0.3302 | 24.8935 | 32.4057 |

s_H = 23.8123 EUR/MWh (std, ddof 1, of the 672 hourly residuals y - yhat of the hourly XGB point forecast against the hourly mean native price; residual mean -1.0452). F = ECDF of the residuals / s_H (not centered). Per-hour s_H (information only, not used): min 11.33 (h6), max 57.79 (h19).

Predicted raw PICP-80 (%) and CRPS of the native setting at pi = 0, 0.5, 1. gap = CRPS(pi) - CRPS(1) (item (i)), gap_interp_minus_native = CRPS(0) - CRPS(pi) (used for Fig. 1 and \predCrpsGapHalf).

| pi | position | picp80 | crps | gap | gap_interp_minus_native |
|---|---|---|---|---|---|
| 0.0000 | 0 | 72.3214 | 17.1305 | 0.1214 | 0.0000 |
| 0.5000 | 0 | 76.3393 | 17.0427 | 0.0336 | 0.0878 |
| 1.0000 | 0 | 79.9107 | 17.0091 | 0.0000 | 0.1214 |
| 0.0000 | 1 | 78.1250 | 13.3649 | 0.0068 | 0.0000 |
| 0.5000 | 1 | 79.0179 | 13.3602 | 0.0021 | 0.0047 |
| 1.0000 | 1 | 79.9107 | 13.3581 | 0.0000 | 0.0068 |
| 0.0000 | 2 | 78.2738 | 13.3253 | 0.0062 | 0.0000 |
| 0.5000 | 2 | 79.0179 | 13.3210 | 0.0018 | 0.0044 |
| 1.0000 | 2 | 79.9107 | 13.3191 | 0.0000 | 0.0062 |
| 0.0000 | 3 | 72.3214 | 17.0981 | 0.1199 | 0.0000 |
| 0.5000 | 3 | 76.3393 | 17.0111 | 0.0329 | 0.0870 |
| 1.0000 | 3 | 79.9107 | 16.9783 | 0.0000 | 0.1199 |
| 0.0000 | pooled | 75.2604 | 15.2297 | 0.0636 | 0.0000 |
| 0.5000 | pooled | 77.6786 | 15.1838 | 0.0176 | 0.0460 |
| 1.0000 | pooled | 79.9107 | 15.1662 | 0.0000 | 0.0636 |

Predicted raw PICP-80 by position over the evaluation days (Fig. 2(b) dashes): interp at pi = 0, native = mean over the 337 days of the prediction at pi_D.

| kind | position | pi | picp80 | crps |
|---|---|---|---|---|
| eval_native | 0 | 0.5385 | 76.6550 | 17.0457 |
| eval_interp | 0 | 0.0000 | 72.3214 | 17.1305 |
| eval_native | 1 | 0.5385 | 79.1026 | 13.3600 |
| eval_interp | 1 | 0.0000 | 78.1250 | 13.3649 |
| eval_native | 2 | 0.5385 | 79.1525 | 13.3208 |
| eval_interp | 2 | 0.0000 | 78.2738 | 13.3253 |
| eval_native | 3 | 0.5385 | 76.6789 | 17.0144 |
| eval_interp | 3 | 0.0000 | 72.3214 | 17.0981 |

## 2. Metrics per file (evaluation period)

| model | setting | layer | MAE | RMSE | rMAE | CRPS | MCB | DSC | UNC | PICP80 | width80 | PICP60 | width60 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| naive | Hourly | raw | 34.5251 | 54.9515 | 1.0000 | 28.7666 |  |  |  | 76.84 | 98.6328 | 54.27 | 50.7747 |
| naive | Interpolated | raw | 34.0425 | 54.4415 | 1.0000 | 28.3784 |  |  |  | 77.69 | 99.5431 | 55.24 | 51.1107 |
| naive | Native | raw | 33.9859 | 54.8241 | 1.0000 | 28.3856 |  |  |  | 77.96 | 100.6974 | 55.95 | 51.7408 |
| xgb | Hourly | raw | 20.0222 | 33.5136 | 0.5799 | 16.3375 | 1.0565 | 19.3353 | 34.6163 | 62.09 | 44.6660 | 40.65 | 25.2384 |
| xgb | Hourly | split | 20.0222 | 33.5136 | 0.5799 | 16.1867 | 0.8541 | 19.2837 | 34.6163 | 76.30 | 60.5175 | 57.55 | 35.6641 |
| xgb | Hourly | rolling | 20.0222 | 33.5136 | 0.5799 | 15.7470 | 0.4288 | 19.2982 | 34.6163 | 78.85 | 60.7273 | 59.23 | 35.8523 |
| xgb | Hourly | acifrozen | 20.0222 | 33.5136 | 0.5799 | 15.7198 | 0.4119 | 19.3084 | 34.6163 | 80.35 | 63.0142 | 60.76 | 36.7268 |
| xgb | Hourly | aci | 20.0222 | 33.5136 | 0.5799 | 15.8220 | 0.4077 | 19.2021 | 34.6163 | 80.14 | 65.1911 | 60.06 | 37.1733 |
| xgb | Hourly | acipooled | 20.0222 | 33.5136 | 0.5799 | 15.8558 | 0.4834 | 19.2440 | 34.6163 | 80.09 | 65.2195 | 60.00 | 37.1106 |
| xgb | Interpolated | raw | 19.6830 | 34.1810 | 0.5782 | 15.3336 | 0.6980 | 19.9808 | 34.6163 | 64.72 | 43.4070 | 42.71 | 24.6621 |
| xgb | Interpolated | split | 19.6830 | 34.1810 | 0.5782 | 15.2264 | 0.5874 | 19.9773 | 34.6163 | 76.15 | 54.5149 | 56.52 | 32.2682 |
| xgb | Interpolated | rolling | 19.6830 | 34.1810 | 0.5782 | 14.9774 | 0.3451 | 19.9840 | 34.6163 | 78.69 | 56.2044 | 58.51 | 34.0587 |
| xgb | Interpolated | acifrozen | 19.6830 | 34.1810 | 0.5782 | 14.9413 | 0.3364 | 20.0114 | 34.6163 | 80.09 | 57.9267 | 59.74 | 34.5170 |
| xgb | Interpolated | aci | 19.6830 | 34.1810 | 0.5782 | 15.0467 | 0.3435 | 19.9132 | 34.6163 | 80.00 | 60.5849 | 59.90 | 35.6436 |
| xgb | Interpolated | acipooled | 19.6830 | 34.1810 | 0.5782 | 15.0725 | 0.3607 | 19.9045 | 34.6163 | 79.91 | 60.5514 | 59.88 | 35.5533 |
| xgb | Native | raw | 19.1745 | 33.0119 | 0.5642 | 14.8324 | 0.5726 | 20.3565 | 34.6163 | 69.41 | 46.9334 | 46.70 | 26.5839 |
| xgb | Native | split | 19.1745 | 33.0119 | 0.5642 | 14.7453 | 0.4703 | 20.3412 | 34.6163 | 77.48 | 54.8652 | 58.02 | 32.6689 |
| xgb | Native | rolling | 19.1745 | 33.0119 | 0.5642 | 14.6092 | 0.3367 | 20.3438 | 34.6163 | 78.75 | 55.8116 | 58.40 | 33.4586 |
| xgb | Native | acifrozen | 19.1745 | 33.0119 | 0.5642 | 14.5703 | 0.3178 | 20.3639 | 34.6163 | 80.08 | 57.2768 | 59.58 | 33.8838 |
| xgb | Native | aci | 19.1745 | 33.0119 | 0.5642 | 14.6727 | 0.3333 | 20.2769 | 34.6163 | 80.09 | 59.9978 | 59.90 | 34.8575 |
| xgb | Native | acipooled | 19.1745 | 33.0119 | 0.5642 | 14.6796 | 0.3410 | 20.2777 | 34.6163 | 80.02 | 59.7181 | 59.88 | 34.7579 |

Check CRPS = MCB - DSC + UNC: max abs residual 3.55e-15.

## 3. Table 1 (results/table1.tex)

```latex
\begin{tabular}{|l|l|c|c|c|c|c|c|c|}
\hline
Setting & Model & MAE & CRPS & MCB & DSC & PICP & Width & DM \\
\hline
Hourly & LEAR & 19.84 & 16.14 & 0.63 & 19.10 & 71.1 & 52.86 & -- \\
Interpolated & LEAR & 18.92 & 15.08 & 0.55 & 20.09 & 71.9 & 51.65 & -- \\
Native & LEAR & 18.60 & 14.70 & 0.53 & 20.45 & 75.5 & 54.73 & \ensuremath{<}0.001 \\
\hline
Hourly & LEAR+ACI & 19.84 & 16.03 & 0.49 & 19.07 & 80.1 & 65.17 & -- \\
Interpolated & LEAR+ACI & 18.92 & 15.08 & 0.51 & 20.05 & 80.1 & 62.13 & -- \\
Native & LEAR+ACI & 18.60 & 14.78 & 0.56 & 20.40 & 80.0 & 61.41 & \ensuremath{<}0.001 \\
\hline
Hourly & XGB & 20.02 & 16.34 & 1.06 & 19.34 & 62.1 & 44.67 & -- \\
Interpolated & XGB & 19.68 & 15.33 & 0.70 & 19.98 & 64.7 & 43.41 & -- \\
Native & XGB & 19.17 & 14.83 & 0.57 & 20.36 & 69.4 & 46.93 & \ensuremath{<}0.001 \\
\hline
Hourly & XGB+ACI & 20.02 & 15.82 & 0.41 & 19.20 & 80.1 & 65.19 & -- \\
Interpolated & XGB+ACI & 19.68 & 15.05 & 0.34 & 19.91 & 80.0 & 60.58 & -- \\
Native & XGB+ACI & 19.17 & 14.67 & 0.33 & 20.28 & 80.1 & 60.00 & \ensuremath{<}0.001 \\
\hline
\end{tabular}
```

## 4. Table 2 (results/table2.tex), interpolated XGB

| regime | CRPS | PICP80 | width80 | unbounded |
|---|---|---|---|---|
| split | 15.2264 | 76.1531 | 54.5149 | 0.0000 |
| rolling | 14.9774 | 78.6942 | 56.2044 | 0.0000 |
| acifrozen | 14.9413 | 80.0884 | 57.9267 | 0.0000 |
| aci | 15.0467 | 80.0049 | 60.5849 | 0.1484 |

```latex
\begin{tabular}{|l|c|c|c|c|}
\hline
Regime & CRPS & PICP-80 & Width-80 & Unbounded \\
\hline
Split-CQR & 15.23 & 76.2 & 54.51 & 0.0 \\
Rolling-CQR & 14.98 & 78.7 & 56.20 & 0.0 \\
ACI-frozen & 14.94 & 80.1 & 57.93 & 0.0 \\
ACI-CQR & 15.05 & 80.0 & 60.58 & 0.1 \\
\hline
\end{tabular}
```

Unbounded = share (%) of quarter-days with a formal unbounded side at the 80% pair. All regimes:

| setting | regime | unbounded_pct |
|---|---|---|
| hourly | split | 0.000 |
| hourly | rolling | 0.000 |
| hourly | acifrozen | 0.000 |
| hourly | aci | 0.074 |
| hourly | acipooled | 0.000 |
| interp | split | 0.000 |
| interp | rolling | 0.000 |
| interp | acifrozen | 0.000 |
| interp | aci | 0.148 |
| interp | acipooled | 0.000 |
| native | split | 0.000 |
| native | rolling | 0.000 |
| native | acifrozen | 0.000 |
| native | aci | 0.074 |
| native | acipooled | 0.000 |

## 5. Diebold-Mariano (daily CRPS native - interp) and pi-regression (daily gap interp - native on pi_D)

| regime | dm_mean_native_minus_interp | dm_se | dm_stat | dm_p | intercept | intercept_se | slope | slope_se | slope_z | slope_p | T |
|---|---|---|---|---|---|---|---|---|---|---|---|
| raw | -0.5013 | 0.0721 | -6.9490 | 3.678e-12 | 0.1302 | 0.1061 | 0.6892 | 0.2232 | 3.0878 | 0.002016 | 337 |
| aci | -0.3740 | 0.0592 | -6.3216 | 2.589e-10 | 0.1302 | 0.0893 | 0.4528 | 0.1789 | 2.5312 | 0.01137 | 337 |

## 6. Bootstrap 90% intervals of native - interp

| regime | quantity | native | interp | diff_native_minus_interp | lo5 | hi95 | boot_mean | boot_sd | n_draws |
|---|---|---|---|---|---|---|---|---|---|
| raw | CRPS | 14.8324 | 15.3336 | -0.5013 | -0.6184 | -0.3957 | -0.5015 | 0.0698 | 1000 |
| raw | MCB | 0.5726 | 0.6980 | -0.1255 | -0.1658 | -0.0529 | -0.1149 | 0.0355 | 1000 |
| raw | DSC | 20.3565 | 19.9808 | 0.3758 | 0.2768 | 0.5036 | 0.3865 | 0.0728 | 1000 |
| aci | CRPS | 14.6727 | 15.0467 | -0.3740 | -0.4708 | -0.2847 | -0.3745 | 0.0579 | 1000 |
| aci | MCB | 0.3333 | 0.3435 | -0.0102 | -0.0303 | 0.0614 | 0.0057 | 0.0277 | 1000 |
| aci | DSC | 20.2769 | 19.9132 | 0.3637 | 0.2798 | 0.4857 | 0.3802 | 0.0665 | 1000 |

## 7. Monthly CRPS (results/monthly.csv)

| month | n_days | mean_pi | hourly_aci | hourly_acifrozen | hourly_acipooled | hourly_raw | hourly_rolling | hourly_split | interp_aci | interp_acifrozen | interp_acipooled | interp_raw | interp_rolling | interp_split | native_aci | native_acifrozen | native_acipooled | native_raw | native_rolling | native_split | gap_raw (interp-native) | gap_aci (interp-native) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2025-10 | 3 | 0.080 | 12.0424 | 12.0505 | 12.1691 | 12.6535 | 12.0616 | 12.0785 | 11.6754 | 11.6980 | 11.7684 | 12.1588 | 11.7176 | 11.7384 | 11.5012 | 11.4988 | 11.5569 | 11.5747 | 11.5127 | 11.5132 | 0.5841 | 0.1742 |
| 2025-11 | 30 | 0.125 | 11.4901 | 11.4611 | 11.5302 | 11.6294 | 11.5056 | 11.5227 | 11.1632 | 11.1403 | 11.2064 | 11.2359 | 11.1868 | 11.1747 | 11.0429 | 11.0160 | 11.0671 | 11.0154 | 11.0670 | 11.0442 | 0.2205 | 0.1203 |
| 2025-12 | 31 | 0.209 | 8.0863 | 8.0003 | 8.1070 | 8.3739 | 8.0475 | 8.2285 | 8.0478 | 7.9575 | 8.0715 | 8.4351 | 8.0334 | 8.1414 | 7.9180 | 7.8917 | 7.9349 | 8.2390 | 7.9484 | 8.1996 | 0.1961 | 0.1298 |
| 2026-01 | 31 | 0.294 | 11.7624 | 11.2960 | 11.8166 | 11.2846 | 11.5383 | 11.5429 | 11.5435 | 11.0308 | 11.5746 | 10.8901 | 11.2795 | 11.1051 | 11.2999 | 10.8179 | 11.3217 | 10.7144 | 11.0945 | 10.7525 | 0.1757 | 0.2436 |
| 2026-02 | 28 | 0.375 | 8.6705 | 8.6966 | 8.7019 | 8.8250 | 8.7312 | 8.7896 | 8.7736 | 8.7690 | 8.7913 | 8.8892 | 8.8153 | 8.7632 | 8.6830 | 8.6745 | 8.6860 | 8.6983 | 8.7412 | 8.7073 | 0.1908 | 0.0906 |
| 2026-03 | 31 | 0.456 | 17.2799 | 17.1541 | 17.3127 | 17.8697 | 17.2183 | 17.5259 | 16.1348 | 16.0543 | 16.1633 | 16.5331 | 16.0635 | 16.3555 | 15.6504 | 15.6162 | 15.6542 | 16.0201 | 15.5913 | 15.7984 | 0.5130 | 0.4844 |
| 2026-04 | 30 | 0.540 | 18.5764 | 18.4988 | 18.6275 | 18.8988 | 18.4937 | 18.5002 | 17.2850 | 17.1871 | 17.3226 | 17.4105 | 17.2168 | 17.1530 | 16.6165 | 16.5104 | 16.6184 | 16.5560 | 16.5356 | 16.4608 | 0.8545 | 0.6685 |
| 2026-05 | 31 | 0.624 | 16.9683 | 16.8776 | 16.9866 | 17.4917 | 16.8911 | 17.4194 | 16.1257 | 16.0224 | 16.1557 | 16.3885 | 16.0632 | 16.4480 | 15.4993 | 15.3816 | 15.5107 | 15.6635 | 15.4323 | 15.6327 | 0.7250 | 0.6265 |
| 2026-06 | 30 | 0.707 | 24.3624 | 24.3104 | 24.3947 | 25.4152 | 24.3590 | 25.1567 | 22.5989 | 22.4882 | 22.6159 | 23.1898 | 22.5635 | 23.0534 | 22.0206 | 21.9507 | 22.0203 | 22.4983 | 21.9729 | 22.3347 | 0.6914 | 0.5783 |
| 2026-07 | 31 | 0.791 | 16.4223 | 16.2836 | 16.4670 | 16.5431 | 16.1793 | 16.3121 | 15.6356 | 15.5205 | 15.6588 | 15.5507 | 15.4219 | 15.3617 | 15.3078 | 15.1262 | 15.3033 | 15.0239 | 15.0776 | 14.8788 | 0.5268 | 0.3278 |
| 2026-08 | 31 | 0.876 | 16.6624 | 16.5656 | 16.6895 | 17.6983 | 16.6045 | 17.6369 | 15.6188 | 15.5597 | 15.6373 | 16.3437 | 15.5949 | 16.3807 | 15.1718 | 15.0681 | 15.1778 | 15.5481 | 15.1340 | 15.4607 | 0.7956 | 0.4471 |
| 2026-09 | 30 | 0.960 | 23.9224 | 23.9452 | 23.9318 | 25.8446 | 23.8179 | 25.6201 | 22.7462 | 22.7890 | 22.7529 | 23.9548 | 22.6767 | 23.7255 | 22.3490 | 22.3857 | 22.3348 | 23.3569 | 22.2716 | 23.1095 | 0.5979 | 0.3973 |

## 8. Kupiec and Christoffersen per quarter-hour slot across days (supplement), Holm over the 96 slots per family, level and file

Rejections at 5% after Holm, all files: Kupiec 1243 of 3456, Christoffersen (cc) 1463 of 3456. Per file and level (of 96 slots; slot detail in results/kupiec.csv):

| setting | layer | level | slots | kupiec_rej | cc_rej |
|---|---|---|---|---|---|
| hourly | raw | 80 | 96 | 92 | 92 |
| hourly | raw | 60 | 96 | 96 | 96 |
| hourly | split | 80 | 96 | 46 | 64 |
| hourly | split | 60 | 96 | 33 | 41 |
| hourly | rolling | 80 | 96 | 37 | 53 |
| hourly | rolling | 60 | 96 | 37 | 45 |
| hourly | acifrozen | 80 | 96 | 28 | 38 |
| hourly | acifrozen | 60 | 96 | 38 | 40 |
| hourly | aci | 80 | 96 | 29 | 44 |
| hourly | aci | 60 | 96 | 36 | 46 |
| hourly | acipooled | 80 | 96 | 38 | 46 |
| hourly | acipooled | 60 | 96 | 41 | 45 |
| interp | raw | 80 | 96 | 95 | 93 |
| interp | raw | 60 | 96 | 96 | 96 |
| interp | split | 80 | 96 | 38 | 42 |
| interp | split | 60 | 96 | 26 | 37 |
| interp | rolling | 80 | 96 | 18 | 36 |
| interp | rolling | 60 | 96 | 20 | 22 |
| interp | acifrozen | 80 | 96 | 17 | 26 |
| interp | acifrozen | 60 | 96 | 20 | 22 |
| interp | aci | 80 | 96 | 18 | 30 |
| interp | aci | 60 | 96 | 16 | 25 |
| interp | acipooled | 80 | 96 | 29 | 39 |
| interp | acipooled | 60 | 96 | 28 | 33 |
| native | raw | 80 | 96 | 84 | 81 |
| native | raw | 60 | 96 | 96 | 92 |
| native | split | 80 | 96 | 15 | 18 |
| native | split | 60 | 96 | 11 | 13 |
| native | rolling | 80 | 96 | 5 | 15 |
| native | rolling | 60 | 96 | 5 | 10 |
| native | acifrozen | 80 | 96 | 8 | 8 |
| native | acifrozen | 60 | 96 | 8 | 9 |
| native | aci | 80 | 96 | 10 | 19 |
| native | aci | 60 | 96 | 4 | 10 |
| native | acipooled | 80 | 96 | 16 | 20 |
| native | acipooled | 60 | 96 | 9 | 17 |

## 8b. Share ablation (step 5b, results/ablation.csv, written by esann27.ablation)

Native LEAR, rule (iii), on the ablation days; gap_crps = CRPS(level) - CRPS(regular), DM of the level against the regular run on the daily CRPS (Newey-West lag 7), pred_gap = Proposition A (LEAR burn-in inputs, pooled) CRPS_A(pi) - CRPS_A(pi_bar).

| level | pi | m | n_quarters | CRPS | MCB | DSC | UNC | PICP80 | width80 | gap_crps | gap_dsc_regular_minus | dm_p | pred_gap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| regular | 0.9629 | 350.5000 | 2688 | 22.9273 | 2.5436 | 26.1866 | 46.5702 | 72.69 | 88.9731 | 0.0000 | 0.0000 |  | 0.0000 |
| 0 | 0.0000 | 0.0000 | 2688 | 23.4818 | 2.8763 | 25.9646 | 46.5702 | 70.20 | 89.1876 | 0.5546 | 0.2219 | 0.002625 | 0.0529 |
| 0.25 | 0.2500 | 91.0000 | 2688 | 23.1701 | 2.5041 | 25.9043 | 46.5702 | 73.36 | 90.6317 | 0.2428 | 0.2823 | 2.623e-05 | 0.0335 |
| 0.5 | 0.5000 | 182.0000 | 2688 | 23.1287 | 2.4142 | 25.8557 | 46.5702 | 73.07 | 89.7744 | 0.2015 | 0.3308 | 5.587e-12 | 0.0124 |
| 0.75 | 0.7500 | 273.0000 | 2688 | 23.0836 | 2.4848 | 25.9714 | 46.5702 | 73.51 | 90.5048 | 0.1564 | 0.2151 | 0.02154 | 0.0037 |

## 9. Macros written to results/numbers.tex by this run

| macro | value |
|---|---|
| \dmPAciXGB | \ensuremath{<}0.001 |
| \dmPRawXGB | \ensuremath{<}0.001 |
| \gapCrpsAciXGB | \ensuremath{-0.37} |
| \gapCrpsAciXGBHi | \ensuremath{-0.28} |
| \gapCrpsAciXGBLo | \ensuremath{-0.47} |
| \gapCrpsRawXGB | \ensuremath{-0.50} |
| \gapCrpsRawXGBHi | \ensuremath{-0.40} |
| \gapCrpsRawXGBLo | \ensuremath{-0.62} |
| \gapDscAciXGB | 0.36 |
| \gapDscAciXGBHi | 0.49 |
| \gapDscAciXGBLo | 0.28 |
| \gapDscRawXGB | 0.38 |
| \gapDscRawXGBHi | 0.50 |
| \gapDscRawXGBLo | 0.28 |
| \gapMcbAciXGB | \ensuremath{-0.01} |
| \gapMcbAciXGBHi | 0.06 |
| \gapMcbAciXGBLo | \ensuremath{-0.03} |
| \gapMcbRawXGB | \ensuremath{-0.13} |
| \gapMcbRawXGBHi | \ensuremath{-0.05} |
| \gapMcbRawXGBLo | \ensuremath{-0.17} |
| \hlGapRemovedXGB | 25.4 |
| \hlPicpAci | 80.0 |
| \hlPicpSplit | 76.2 |
| \naiveMAEhourly | 34.53 |
| \naiveMAEinterp | 34.04 |
| \naiveMAEnative | 33.99 |
| \regCrpsAci | 15.05 |
| \regCrpsAciFrozen | 14.94 |
| \regCrpsRolling | 14.98 |
| \regCrpsSplit | 15.23 |
| \regPicpAciFrozen | 80.1 |
| \regPicpRolling | 78.7 |
| \regUnboundedAci | 0.1 |
| \regWidthAci | 60.58 |
| \regWidthAciFrozen | 57.93 |
| \slopeAciXGB | 0.45 |
| \slopeAciXGBP | 0.011 |
| \slopeAciXGBSe | 0.18 |
| \slopeRawXGB | 0.69 |
| \slopeRawXGBP | 0.002 |
| \slopeRawXGBSe | 0.22 |

## 10. Wall time

| stage | seconds |
|---|---|
| Proposition A | 1.5 |
| metrics + CORP | 5.3 |
| DM + regression + monthly | 0.1 |
| bootstrap | 123.7 |
| Kupiec + Christoffersen | 0.4 |
| tables + figures | 0.7 |
| total | 576.3 |

## 11. Step 5c: ACI-CQR with 24 groups (aciblock)

ACI-CQR with 24 Mondrian groups g = 4 (h // 4) + p (h = clock hour, p = quarter position), handover 8.1 row ACI-block, step 5c specification. Files conformal/<setting>_xgb_aciblock.parquet, evaluation period as above. `aci` = ACI-CQR with the four position groups; `- aci` = aciblock minus ACI-CQR in the same setting. XGB.

Metrics per setting (CRPS, MCB, DSC, UNC on the CRPS scale; PICP and width at the issued bounds):

| setting | CRPS | MCB | DSC | UNC | PICP80 | PICP60 | width80 | CRPS aci | MCB aci | CRPS - aci | MCB - aci |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hourly | 15.2356 | 0.3830 | 19.7637 | 34.6163 | 80.76 | 61.22 | 73.8275 | 15.8220 | 0.4077 | -0.5864 | -0.0248 |
| interp | 14.8972 | 0.3997 | 20.1188 | 34.6163 | 80.33 | 60.57 | 70.1933 | 15.0467 | 0.3435 | -0.1494 | 0.0561 |
| native | 14.7839 | 0.3897 | 20.2221 | 34.6163 | 80.39 | 60.57 | 70.6064 | 14.6727 | 0.3333 | 0.1112 | 0.0564 |

PICP-80 (%) by group, minimum and maximum over the 24 groups:

| setting | groups | PICP80 min | group min | PICP80 max | group max |
|---|---|---|---|---|---|
| hourly | 24 | 79.97 | 12 | 81.60 | 4 |
| interp | 24 | 79.75 | 17 | 80.93 | 23 |
| native | 24 | 79.81 | 3 | 81.23 | 23 |

Shares (%) of quarter-days with a formal unbounded side, a formal empty side, a clipped index: at the 80% pair (k = 1) and over all four pairs (share of quarter-day x pair combinations):

| setting | unbounded 80 (%) | empty 80 (%) | clipped 80 (%) | unbounded all pairs (%) | empty all pairs (%) | clipped all pairs (%) |
|---|---|---|---|---|---|---|
| hourly | 2.300 | 0.000 | 2.300 | 0.575 | 0.000 | 0.575 |
| interp | 1.855 | 0.000 | 1.855 | 0.464 | 0.000 | 0.464 |
| native | 2.115 | 0.000 | 2.115 | 0.529 | 0.000 | 0.529 |

Prop. 4.1 per side: max over the 24 groups and both sides of |mean over the 337 days of err - 0.1k| (err of the formal side), per setting, next to the bound (max(0.1k, 1 - 0.1k) + gamma) / (gamma 337):

| k | target | bound | hourly | interp | native | within bound |
|---|---|---|---|---|---|---|
| 1 | 0.1 | 0.05638 | 0.00460 | 0.00386 | 0.00386 | yes |
| 2 | 0.2 | 0.05045 | 0.00490 | 0.00623 | 0.00549 | yes |
| 3 | 0.3 | 0.04451 | 0.00786 | 0.00786 | 0.00564 | yes |
| 4 | 0.4 | 0.03858 | 0.00653 | 0.00460 | 0.00460 | yes |

Diebold-Mariano on the daily CRPS difference native - interp (Newey-West, Bartlett lag 7, normal p), ACI-CQR beside ACI-block:

| regime | mean | se | stat | p | T |
|---|---|---|---|---|---|
| aci | -0.3740 | 0.0592 | -6.3216 | 2.589e-10 | 337 |
| aciblock | -0.1133 | 0.0517 | -2.1906 | 0.02848 | 337 |

Bootstrap 90% intervals of native - interp (the same 1,000 draws as raw and ACI-CQR, circular blocks of 7 days, default_rng(2027)), ACI-CQR beside ACI-block:

| regime | quantity | native | interp | diff_native_minus_interp | lo5 | hi95 | boot_mean | boot_sd | n_draws |
|---|---|---|---|---|---|---|---|---|---|
| aci | CRPS | 14.6727 | 15.0467 | -0.3740 | -0.4708 | -0.2847 | -0.3745 | 0.0579 | 1000 |
| aciblock | CRPS | 14.7839 | 14.8972 | -0.1133 | -0.2050 | -0.0328 | -0.1139 | 0.0524 | 1000 |
| aci | MCB | 0.3333 | 0.3435 | -0.0102 | -0.0303 | 0.0614 | 0.0057 | 0.0277 | 1000 |
| aciblock | MCB | 0.3897 | 0.3997 | -0.0100 | -0.0259 | 0.0338 | 0.0012 | 0.0191 | 1000 |
| aci | DSC | 20.2769 | 19.9132 | 0.3637 | 0.2798 | 0.4857 | 0.3802 | 0.0665 | 1000 |
| aciblock | DSC | 20.2221 | 20.1188 | 0.1034 | 0.0294 | 0.2060 | 0.1151 | 0.0548 | 1000 |

Kupiec and Christoffersen (cc) per quarter-hour slot, rejections at 5% after Holm over the 96 slots, of 96 (descriptive only):

| setting | level | kupiec_rej aci | kupiec_rej aciblock | cc_rej aci | cc_rej aciblock |
|---|---|---|---|---|---|
| interp | 80 | 18 | 2 | 30 | 2 |
| interp | 60 | 16 | 2 | 25 | 3 |
| native | 80 | 10 | 0 | 19 | 0 |
| native | 60 | 4 | 1 | 10 | 2 |

Decision rule of 8.1 (row ACI-block), inputs: DM p = 0.028 (0.02848), CRPS gap lo5..hi95 = -0.2050 .. -0.0328.

rule: gap holds = DM p < 0.05 and lo5..hi95 excludes 0 -> yes

Macros written to results/numbers.tex by this run for step 5c (not listed in Section 9):

| macro | value |
|---|---|
| \dCrpsBlockInterpXGB | \ensuremath{-0.15} |
| \dCrpsBlockNativeXGB | 0.11 |
| \dmPBlockXGB | 0.028 |
| \gapCrpsBlockXGB | \ensuremath{-0.11} |
| \gapCrpsBlockXGBHi | \ensuremath{-0.03} |
| \gapCrpsBlockXGBLo | \ensuremath{-0.20} |
| \gapDscBlockXGB | 0.10 |
| \gapDscBlockXGBHi | 0.21 |
| \gapDscBlockXGBLo | 0.03 |
| \gapMcbBlockXGB | \ensuremath{-0.01} |
| \gapMcbBlockXGBHi | 0.03 |
| \gapMcbBlockXGBLo | \ensuremath{-0.03} |

Wall time of the step 5c work (not included in the stage rows of Section 10):

| stage | seconds |
|---|---|
| load + metrics + CORP | 1.0 |
| DM | 0.0 |
| bootstrap | 61.8 |
| Kupiec + Christoffersen | 0.1 |
| shares + groups + macros | 0.1 |

## 12. Step 5d: XGB with monthly re-tuning (xgbm)

XGB with its hyperparameters re-tuned on the first day of each month (handover Section 8 row "XGB monthly re-tuning", step 5d specification): model `xgbm`, settings interp and native, forecasts in forecasts_retune/, conformal/<setting>_xgbm_{aci,aciblock}.parquet. `xgb` = the frozen design (main analysis). Evaluation period as above.

Hyperparameters per re-tune date and setting (forecasts_retune/retune_log.csv, results/retune_xgb_hyperparameters.csv). Frozen set (hyperparameters_15min.json): point depth 3, eta 0.05, rounds 86; quantile eta 0.05 / 0.1: 7 / 2, rounds 106-426. differs = any of the 10 models (point, q10..q90) differs from the frozen set in depth, eta or rounds; point differs = the point model does.

| setting | retune_date | point depth | point eta | point rounds | q eta 0.05 | q eta 0.1 | q rounds min | q rounds max | models differing | differs | point differs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| interp | 2025-10-01 | 3 | 0.05 | 86 | 7 | 2 | 106 | 426 | 0 | no | no |
| interp | 2025-11-01 | 3 | 0.05 | 262 | 7 | 2 | 148 | 998 | 10 | yes | yes |
| interp | 2025-12-01 | 3 | 0.10 | 100 | 7 | 2 | 164 | 724 | 10 | yes | yes |
| interp | 2026-01-01 | 7 | 0.10 | 23 | 6 | 3 | 38 | 313 | 10 | yes | yes |
| interp | 2026-02-01 | 7 | 0.05 | 35 | 7 | 2 | 29 | 162 | 10 | yes | yes |
| interp | 2026-03-01 | 3 | 0.05 | 72 | 3 | 6 | 58 | 355 | 10 | yes | yes |
| interp | 2026-04-01 | 3 | 0.05 | 231 | 5 | 4 | 82 | 915 | 10 | yes | yes |
| interp | 2026-05-01 | 3 | 0.10 | 101 | 3 | 6 | 225 | 999 | 10 | yes | yes |
| interp | 2026-06-01 | 3 | 0.10 | 574 | 6 | 3 | 115 | 998 | 10 | yes | yes |
| interp | 2026-07-01 | 5 | 0.05 | 136 | 3 | 6 | 101 | 1000 | 10 | yes | yes |
| interp | 2026-08-01 | 3 | 0.05 | 135 | 3 | 6 | 104 | 494 | 10 | yes | yes |
| interp | 2026-09-01 | 5 | 0.05 | 90 | 4 | 5 | 84 | 331 | 10 | yes | yes |
| native | 2025-10-01 | 3 | 0.05 | 86 | 7 | 2 | 106 | 426 | 0 | no | no |
| native | 2025-11-01 | 3 | 0.05 | 262 | 7 | 2 | 148 | 998 | 10 | yes | yes |
| native | 2025-12-01 | 5 | 0.05 | 148 | 7 | 2 | 63 | 903 | 10 | yes | yes |
| native | 2026-01-01 | 7 | 0.05 | 54 | 5 | 4 | 48 | 560 | 10 | yes | yes |
| native | 2026-02-01 | 7 | 0.05 | 34 | 3 | 6 | 32 | 245 | 10 | yes | yes |
| native | 2026-03-01 | 3 | 0.05 | 70 | 7 | 2 | 69 | 308 | 10 | yes | yes |
| native | 2026-04-01 | 3 | 0.10 | 51 | 4 | 5 | 90 | 778 | 10 | yes | yes |
| native | 2026-05-01 | 3 | 0.05 | 179 | 2 | 7 | 107 | 992 | 10 | yes | yes |
| native | 2026-06-01 | 3 | 0.05 | 475 | 6 | 3 | 263 | 993 | 10 | yes | yes |
| native | 2026-07-01 | 5 | 0.05 | 126 | 2 | 7 | 63 | 1000 | 10 | yes | yes |
| native | 2026-08-01 | 3 | 0.05 | 117 | 3 | 6 | 105 | 642 | 10 | yes | yes |
| native | 2026-09-01 | 3 | 0.10 | 39 | 5 | 4 | 100 | 365 | 10 | yes | yes |

Setting-dates differing from the frozen set: 22 of 24 (point model: 22 of 24).

Selected configurations, interp against native per re-tune date (10 models). The 1 Nov 2025 re-tune (validation 6 Sep - 31 Oct 2025, 31 native days in the native file) is different (6 of 10 models) across the settings.

| retune_date | interp = native | models differing |
|---|---|---|
| 2025-10-01 | yes | 0 |
| 2025-11-01 | no | 6 |
| 2025-12-01 | no | 10 |
| 2026-01-01 | no | 10 |
| 2026-02-01 | no | 9 |
| 2026-03-01 | no | 10 |
| 2026-04-01 | no | 9 |
| 2026-05-01 | no | 10 |
| 2026-06-01 | no | 10 |
| 2026-07-01 | no | 10 |
| 2026-08-01 | no | 9 |
| 2026-09-01 | no | 10 |

Metrics per setting and layer (CRPS, MCB, DSC, UNC on the CRPS scale; PICP and width at the issued bounds), xgbm next to xgb and the difference:

| setting | layer | model | CRPS | MCB | DSC | UNC | PICP80 | PICP60 | width80 |
|---|---|---|---|---|---|---|---|---|---|
| interp | raw | xgbm | 15.3135 | 0.7715 | 20.0743 | 34.6163 | 63.08 | 41.26 | 42.6908 |
| interp | raw | xgb | 15.3336 | 0.6980 | 19.9808 | 34.6163 | 64.72 | 42.71 | 43.4070 |
| interp | raw | xgbm - xgb | -0.0201 | 0.0734 | 0.0936 | 0.0000 | -1.65 | -1.45 | -0.7161 |
| interp | aci | xgbm | 14.9918 | 0.3515 | 19.9760 | 34.6163 | 80.00 | 59.95 | 62.0270 |
| interp | aci | xgb | 15.0467 | 0.3435 | 19.9132 | 34.6163 | 80.00 | 59.90 | 60.5849 |
| interp | aci | xgbm - xgb | -0.0549 | 0.0079 | 0.0628 | 0.0000 | 0.00 | 0.05 | 1.4421 |
| interp | aciblock | xgbm | 14.8602 | 0.4395 | 20.1956 | 34.6163 | 80.44 | 60.56 | 71.4334 |
| interp | aciblock | xgb | 14.8972 | 0.3997 | 20.1188 | 34.6163 | 80.33 | 60.57 | 70.1933 |
| interp | aciblock | xgbm - xgb | -0.0371 | 0.0398 | 0.0768 | 0.0000 | 0.11 | -0.01 | 1.2401 |
| native | raw | xgbm | 14.8011 | 0.6190 | 20.4342 | 34.6163 | 67.31 | 45.04 | 45.0654 |
| native | raw | xgb | 14.8324 | 0.5726 | 20.3565 | 34.6163 | 69.41 | 46.70 | 46.9334 |
| native | raw | xgbm - xgb | -0.0312 | 0.0464 | 0.0776 | 0.0000 | -2.10 | -1.66 | -1.8680 |
| native | aci | xgbm | 14.6144 | 0.3317 | 20.3336 | 34.6163 | 80.08 | 60.12 | 60.7084 |
| native | aci | xgb | 14.6727 | 0.3333 | 20.2769 | 34.6163 | 80.09 | 59.90 | 59.9978 |
| native | aci | xgbm - xgb | -0.0583 | -0.0016 | 0.0567 | 0.0000 | -0.01 | 0.22 | 0.7106 |
| native | aciblock | xgbm | 14.7437 | 0.4171 | 20.2898 | 34.6163 | 80.51 | 60.69 | 70.9318 |
| native | aciblock | xgb | 14.7839 | 0.3897 | 20.2221 | 34.6163 | 80.39 | 60.57 | 70.6064 |
| native | aciblock | xgbm - xgb | -0.0402 | 0.0274 | 0.0676 | 0.0000 | 0.12 | 0.12 | 0.3254 |

Diebold-Mariano on the daily CRPS difference native - interp (Newey-West, Bartlett lag 7, normal p), xgb beside xgbm:

| layer | model | mean | se | stat | p | T |
|---|---|---|---|---|---|---|
| raw | xgb | -0.5013 | 0.0721 | -6.9490 | 3.678e-12 | 337 |
| raw | xgbm | -0.5123 | 0.0821 | -6.2382 | 4.427e-10 | 337 |
| aci | xgb | -0.3740 | 0.0592 | -6.3216 | 2.589e-10 | 337 |
| aci | xgbm | -0.3774 | 0.0659 | -5.7234 | 1.044e-08 | 337 |
| aciblock | xgb | -0.1133 | 0.0517 | -2.1906 | 0.02848 | 337 |
| aciblock | xgbm | -0.1164 | 0.0601 | -1.9358 | 0.05289 | 337 |

Bootstrap 90% intervals of native - interp (the same 1,000 draws as every other regime, circular blocks of 7 days, default_rng(2027)), xgb beside xgbm:

| regime | quantity | model | native | interp | diff_native_minus_interp | lo5 | hi95 | boot_mean | boot_sd | n_draws |
|---|---|---|---|---|---|---|---|---|---|---|
| raw | CRPS | xgb | 14.8324 | 15.3336 | -0.5013 | -0.6184 | -0.3957 | -0.5015 | 0.0698 | 1000 |
| raw | CRPS | xgbm | 14.8011 | 15.3135 | -0.5123 | -0.6508 | -0.3882 | -0.5123 | 0.0791 | 1000 |
| aci | CRPS | xgb | 14.6727 | 15.0467 | -0.3740 | -0.4708 | -0.2847 | -0.3745 | 0.0579 | 1000 |
| aci | CRPS | xgbm | 14.6144 | 14.9918 | -0.3774 | -0.4827 | -0.2796 | -0.3777 | 0.0644 | 1000 |
| aciblock | CRPS | xgb | 14.7839 | 14.8972 | -0.1133 | -0.2050 | -0.0328 | -0.1139 | 0.0524 | 1000 |
| aciblock | CRPS | xgbm | 14.7437 | 14.8602 | -0.1165 | -0.2165 | -0.0246 | -0.1168 | 0.0602 | 1000 |
| raw | MCB | xgb | 0.5726 | 0.6980 | -0.1255 | -0.1658 | -0.0529 | -0.1149 | 0.0355 | 1000 |
| raw | MCB | xgbm | 0.6190 | 0.7715 | -0.1525 | -0.2061 | -0.0913 | -0.1462 | 0.0354 | 1000 |
| aci | MCB | xgb | 0.3333 | 0.3435 | -0.0102 | -0.0303 | 0.0614 | 0.0057 | 0.0277 | 1000 |
| aci | MCB | xgbm | 0.3317 | 0.3515 | -0.0198 | -0.0416 | 0.0249 | -0.0101 | 0.0211 | 1000 |
| aciblock | MCB | xgb | 0.3897 | 0.3997 | -0.0100 | -0.0259 | 0.0338 | 0.0012 | 0.0191 | 1000 |
| aciblock | MCB | xgbm | 0.4171 | 0.4395 | -0.0223 | -0.0321 | 0.0264 | -0.0055 | 0.0183 | 1000 |
| raw | DSC | xgb | 20.3565 | 19.9808 | 0.3758 | 0.2768 | 0.5036 | 0.3865 | 0.0728 | 1000 |
| raw | DSC | xgbm | 20.4342 | 20.0743 | 0.3598 | 0.2498 | 0.4942 | 0.3661 | 0.0759 | 1000 |
| aci | DSC | xgb | 20.2769 | 19.9132 | 0.3637 | 0.2798 | 0.4857 | 0.3802 | 0.0665 | 1000 |
| aci | DSC | xgbm | 20.3336 | 19.9760 | 0.3576 | 0.2573 | 0.4852 | 0.3675 | 0.0707 | 1000 |
| aciblock | DSC | xgb | 20.2221 | 20.1188 | 0.1034 | 0.0294 | 0.2060 | 0.1151 | 0.0548 | 1000 |
| aciblock | DSC | xgbm | 20.2898 | 20.1956 | 0.0942 | 0.0178 | 0.2116 | 0.1113 | 0.0611 | 1000 |

Monthly raw CRPS (results/retune_xgb_monthly.csv, LEAR from results/monthly.csv):

| month | n_days | mean_pi | interp LEAR | interp xgb | interp xgbm | interp xgbm - xgb | native LEAR | native xgb | native xgbm | native xgbm - xgb |
|---|---|---|---|---|---|---|---|---|---|---|
| 2025-10 | 3 | 0.080 | 9.9044 | 12.1588 | 12.1588 | 0.0000 | 10.0919 | 11.5747 | 11.5747 | 0.0000 |
| 2025-11 | 30 | 0.125 | 10.8257 | 11.2359 | 11.1388 | -0.0971 | 10.5483 | 11.0154 | 10.8487 | -0.1668 |
| 2025-12 | 31 | 0.209 | 9.7763 | 8.4351 | 8.3948 | -0.0403 | 9.5964 | 8.2390 | 7.9954 | -0.2436 |
| 2026-01 | 31 | 0.294 | 11.1242 | 10.8901 | 11.2676 | 0.3775 | 10.9075 | 10.7144 | 11.0306 | 0.3161 |
| 2026-02 | 28 | 0.375 | 8.1740 | 8.8892 | 8.4783 | -0.4109 | 8.0639 | 8.6983 | 8.3038 | -0.3946 |
| 2026-03 | 31 | 0.456 | 16.2579 | 16.5331 | 17.0505 | 0.5175 | 15.8779 | 16.0201 | 16.7375 | 0.7175 |
| 2026-04 | 30 | 0.540 | 18.3245 | 17.4105 | 17.1888 | -0.2217 | 17.6517 | 16.5560 | 16.4248 | -0.1312 |
| 2026-05 | 31 | 0.624 | 16.0905 | 16.3885 | 16.3667 | -0.0218 | 15.5076 | 15.6635 | 15.7220 | 0.0585 |
| 2026-06 | 30 | 0.707 | 22.3611 | 23.1898 | 22.6611 | -0.5287 | 21.5258 | 22.4983 | 21.5781 | -0.9203 |
| 2026-07 | 31 | 0.791 | 14.6414 | 15.5507 | 15.6498 | 0.0991 | 14.5934 | 15.0239 | 15.3321 | 0.3082 |
| 2026-08 | 31 | 0.876 | 15.7296 | 16.3437 | 16.1225 | -0.2212 | 15.0048 | 15.5481 | 15.3235 | -0.2246 |
| 2026-09 | 30 | 0.960 | 22.8710 | 23.9548 | 24.2259 | 0.2711 | 22.6161 | 23.3569 | 23.6306 | 0.2738 |

Decision rule of Section 8 (row "XGB monthly re-tuning"), inputs: raw DM p = <0.001 (4.427e-10), CRPS gap lo5..hi95 = -0.6508 .. -0.3882; ACI-CQR DM p = <0.001 (1.044e-08), CRPS gap lo5..hi95 = -0.4827 .. -0.2796.

rule: robust = DM p < 0.05 and lo5..hi95 excludes 0, raw and ACI-CQR -> yes

Macros written to results/numbers.tex by this run for step 5d (not listed in Section 9):

| macro | value |
|---|---|
| \dCrpsRetuneInterpXGB | \ensuremath{-0.02} |
| \dCrpsRetuneNativeXGB | \ensuremath{-0.03} |
| \dmPAciRetuneXGB | \ensuremath{<}0.001 |
| \dmPBlockRetuneXGB | 0.053 |
| \dmPRawRetuneXGB | \ensuremath{<}0.001 |
| \gapCrpsAciRetuneXGB | \ensuremath{-0.38} |
| \gapCrpsAciRetuneXGBHi | \ensuremath{-0.28} |
| \gapCrpsAciRetuneXGBLo | \ensuremath{-0.48} |
| \gapCrpsBlockRetuneXGB | \ensuremath{-0.12} |
| \gapCrpsBlockRetuneXGBHi | \ensuremath{-0.02} |
| \gapCrpsBlockRetuneXGBLo | \ensuremath{-0.22} |
| \gapCrpsRawRetuneXGB | \ensuremath{-0.51} |
| \gapCrpsRawRetuneXGBHi | \ensuremath{-0.39} |
| \gapCrpsRawRetuneXGBLo | \ensuremath{-0.65} |

Wall time of step 5d: re-tuning and forecasts from forecasts_retune/ (per item = per re-tune, per day), conformal from forecasts_retune/conformal_wall.csv (per item = per evaluation day), evaluation = this run (not included in the stage rows of Section 10):

| stage | seconds | per item (s) |
|---|---|---|
| re-tuning interp (12 dates) | 227.0 | 18.920 |
| re-tuning native (12 dates) | 229.2 | 19.103 |
| forecasts interp (365 days) | 685.1 | 1.877 |
| forecasts native (365 days) | 715.9 | 1.961 |
| conformal interp aci | 0.6 | 0.002 |
| conformal interp aciblock | 0.9 | 0.003 |
| conformal native aci | 0.5 | 0.002 |
| conformal native aciblock | 0.9 | 0.003 |
| evaluation: load + metrics + CORP | 1.9 |  |
| evaluation: DM | 0.0 |  |
| evaluation: bootstrap | 187.5 |  |
| evaluation: monthly + hyperparameters | 0.0 |  |
| evaluation: tables + macros | 0.0 |  |

## 13. Step 5e: diagnostic layers (symblock, locblock, locslot), defined after the 24-group result

Diagnostic layers, defined after the 24-group result of step 5c and run with its workspace values known (handover 8.1 rows Sym-block, Loc-block, Loc-slot, step 5e specification). Diagnostics, not tests. Files conformal/<setting>_xgb_{symblock,locblock,locslot}.parquet, evaluation period as above. `symblock` = ACI-CQR with the 24 groups of aciblock and a symmetric score (width only); `locblock` / `locslot` = recentering by the rolling 28-day median of y - q50 per block group (24) / quarter-hour slot (96), then the four-group ACI-CQR (location only). `aci` and `aciblock` repeated beside them. XGB.

Metrics per setting and regime (CRPS, MCB, DSC, UNC on the CRPS scale; PICP and width at the issued bounds):

| setting | regime | CRPS | MCB | DSC | UNC | PICP80 | PICP60 | width80 |
|---|---|---|---|---|---|---|---|---|
| hourly | aci | 15.8220 | 0.4077 | 19.2021 | 34.6163 | 80.14 | 60.06 | 65.1911 |
| hourly | aciblock | 15.2356 | 0.3830 | 19.7637 | 34.6163 | 80.76 | 61.22 | 73.8275 |
| hourly | symblock | 15.8883 | 0.4636 | 19.1916 | 34.6163 | 80.05 | 59.98 | 68.4469 |
| hourly | locblock | 15.1683 | 0.3984 | 19.8464 | 34.6163 | 80.20 | 60.05 | 62.5140 |
| hourly | locslot | 15.0080 | 0.3901 | 19.9984 | 34.6163 | 80.35 | 60.04 | 62.0705 |
| interp | aci | 15.0467 | 0.3435 | 19.9132 | 34.6163 | 80.00 | 59.90 | 60.5849 |
| interp | aciblock | 14.8972 | 0.3997 | 20.1188 | 34.6163 | 80.33 | 60.57 | 70.1933 |
| interp | symblock | 15.0756 | 0.3630 | 19.9037 | 34.6163 | 80.02 | 59.84 | 62.7626 |
| interp | locblock | 14.8144 | 0.3772 | 20.1791 | 34.6163 | 80.03 | 59.84 | 60.3330 |
| interp | locslot | 14.7270 | 0.3878 | 20.2772 | 34.6163 | 80.09 | 59.82 | 59.5108 |
| native | aci | 14.6727 | 0.3333 | 20.2769 | 34.6163 | 80.09 | 59.90 | 59.9978 |
| native | aciblock | 14.7839 | 0.3897 | 20.2221 | 34.6163 | 80.39 | 60.57 | 70.6064 |
| native | symblock | 14.7237 | 0.3629 | 20.2556 | 34.6163 | 80.11 | 59.82 | 61.9313 |
| native | locblock | 14.6859 | 0.3787 | 20.3091 | 34.6163 | 80.07 | 59.88 | 60.3107 |
| native | locslot | 14.7014 | 0.3930 | 20.3079 | 34.6163 | 80.13 | 59.94 | 60.1108 |

Diebold-Mariano on the daily CRPS difference native - interp (Newey-West, Bartlett lag 7, normal p):

| regime | mean | se | stat | p | T |
|---|---|---|---|---|---|
| aci | -0.3740 | 0.0592 | -6.3216 | 2.589e-10 | 337 |
| aciblock | -0.1133 | 0.0517 | -2.1906 | 0.02848 | 337 |
| symblock | -0.3519 | 0.0616 | -5.7138 | 1.105e-08 | 337 |
| locblock | -0.1286 | 0.0543 | -2.3687 | 0.01785 | 337 |
| locslot | -0.0256 | 0.0509 | -0.5036 | 0.6145 | 337 |

Bootstrap 90% intervals of native - interp (the same 1,000 draws as every other regime, circular blocks of 7 days, default_rng(2027)):

| regime | quantity | native | interp | diff_native_minus_interp | lo5 | hi95 | boot_mean | boot_sd | n_draws |
|---|---|---|---|---|---|---|---|---|---|
| aci | CRPS | 14.6727 | 15.0467 | -0.3740 | -0.4708 | -0.2847 | -0.3745 | 0.0579 | 1000 |
| aciblock | CRPS | 14.7839 | 14.8972 | -0.1133 | -0.2050 | -0.0328 | -0.1139 | 0.0524 | 1000 |
| symblock | CRPS | 14.7237 | 15.0756 | -0.3519 | -0.4530 | -0.2586 | -0.3523 | 0.0602 | 1000 |
| locblock | CRPS | 14.6859 | 14.8144 | -0.1285 | -0.2185 | -0.0414 | -0.1294 | 0.0540 | 1000 |
| locslot | CRPS | 14.7014 | 14.7270 | -0.0255 | -0.1105 | 0.0571 | -0.0262 | 0.0510 | 1000 |
| aci | MCB | 0.3333 | 0.3435 | -0.0102 | -0.0303 | 0.0614 | 0.0057 | 0.0277 | 1000 |
| aciblock | MCB | 0.3897 | 0.3997 | -0.0100 | -0.0259 | 0.0338 | 0.0012 | 0.0191 | 1000 |
| symblock | MCB | 0.3629 | 0.3630 | -0.0000 | -0.0286 | 0.0779 | 0.0144 | 0.0327 | 1000 |
| locblock | MCB | 0.3787 | 0.3772 | 0.0015 | -0.0184 | 0.0542 | 0.0128 | 0.0221 | 1000 |
| locslot | MCB | 0.3930 | 0.3878 | 0.0052 | -0.0147 | 0.0616 | 0.0175 | 0.0242 | 1000 |
| aci | DSC | 20.2769 | 19.9132 | 0.3637 | 0.2798 | 0.4857 | 0.3802 | 0.0665 | 1000 |
| aciblock | DSC | 20.2221 | 20.1188 | 0.1034 | 0.0294 | 0.2060 | 0.1151 | 0.0548 | 1000 |
| symblock | DSC | 20.2556 | 19.9037 | 0.3519 | 0.2581 | 0.4791 | 0.3666 | 0.0716 | 1000 |
| locblock | DSC | 20.3091 | 20.1791 | 0.1300 | 0.0483 | 0.2416 | 0.1421 | 0.0590 | 1000 |
| locslot | DSC | 20.3079 | 20.2772 | 0.0307 | -0.0513 | 0.1409 | 0.0437 | 0.0582 | 1000 |

Shares W and L (handover Section 8 row "Paper decisions of 10 Nov and step 5e scope"): G = |CRPS gap native - interp| under aci (G4), aciblock (G24), symblock (GS), locblock (GL); W = (G4 - GS) / (G4 - G24), L = (G4 - GL) / (G4 - G24); recentering if W <= 1/3 and L >= 2/3 in both models, width if W >= 2/3 and L <= 1/3 in both, mixed otherwise. The other model is read from its metrics file (pending until its step 5e rows exist).

| model | G4 | G24 | GS | GL | W | L |
|---|---|---|---|---|---|---|
| LEAR | 0.3076 | 0.0682 | 0.3087 | 0.0641 | -0.004 | 1.017 |
| XGB | 0.3740 | 0.1133 | 0.3519 | 0.1285 | 0.085 | 0.942 |

reading: W = -0.004 / 0.085, L = 1.017 / 0.942 (LEAR / XGB) -> recentering

Residual profile (results/profile_xgb.csv): mean and unscaled MAD of the raw residual y - q50 per block group g = 4 (h // 4) + p over the evaluation period, interp and native:

| group | block | position | mean interp | mad interp | mean native | mad native | n |
|---|---|---|---|---|---|---|---|
| 0 | 00-03 | 0 | 5.55 | 5.90 | 4.98 | 6.02 | 1347 |
| 1 | 00-03 | 1 | 1.84 | 4.92 | 1.77 | 5.25 | 1347 |
| 2 | 00-03 | 2 | 0.18 | 5.03 | 0.84 | 5.11 | 1347 |
| 3 | 00-03 | 3 | -1.70 | 5.39 | 0.10 | 5.22 | 1347 |
| 4 | 04-07 | 0 | -0.40 | 7.97 | -0.07 | 7.75 | 1348 |
| 5 | 04-07 | 1 | 1.45 | 7.72 | 1.79 | 7.68 | 1348 |
| 6 | 04-07 | 2 | 2.72 | 7.92 | 2.79 | 8.31 | 1348 |
| 7 | 04-07 | 3 | 2.81 | 10.01 | 2.94 | 9.59 | 1348 |
| 8 | 08-11 | 0 | 12.34 | 18.98 | 8.28 | 17.49 | 1348 |
| 9 | 08-11 | 1 | 3.15 | 15.80 | 2.11 | 15.85 | 1348 |
| 10 | 08-11 | 2 | -2.40 | 14.95 | -1.21 | 15.55 | 1348 |
| 11 | 08-11 | 3 | -9.34 | 15.32 | -4.60 | 16.15 | 1348 |
| 12 | 12-15 | 0 | -2.61 | 15.54 | -1.83 | 15.20 | 1348 |
| 13 | 12-15 | 1 | -1.63 | 14.88 | -1.25 | 15.34 | 1348 |
| 14 | 12-15 | 2 | 0.39 | 15.93 | 0.09 | 15.86 | 1348 |
| 15 | 12-15 | 3 | 2.76 | 16.05 | 1.30 | 15.99 | 1348 |
| 16 | 16-19 | 0 | -8.04 | 15.15 | -3.24 | 14.55 | 1348 |
| 17 | 16-19 | 1 | -0.46 | 13.47 | 1.38 | 13.62 | 1348 |
| 18 | 16-19 | 2 | 6.48 | 14.44 | 5.14 | 14.43 | 1348 |
| 19 | 16-19 | 3 | 15.86 | 18.14 | 9.96 | 16.05 | 1348 |
| 20 | 20-23 | 0 | 14.52 | 11.14 | 12.46 | 10.72 | 1348 |
| 21 | 20-23 | 1 | 8.37 | 9.94 | 7.45 | 9.57 | 1348 |
| 22 | 20-23 | 2 | 5.24 | 10.04 | 5.21 | 9.59 | 1348 |
| 23 | 20-23 | 3 | -0.10 | 9.47 | 2.91 | 9.38 | 1348 |

Sym-block, two-sided Prop. 4.1: max over the 24 groups of |mean over the 337 days of err - 0.2k| (err of the formal two-sided set), per setting, next to the bound (max(0.2k, 1 - 0.2k) + gamma) / (gamma 337):

| k | target | bound | hourly | interp | native | within bound |
|---|---|---|---|---|---|---|
| 1 | 0.2 | 0.05045 | 0.00549 | 0.00475 | 0.00475 | yes |
| 2 | 0.4 | 0.03858 | 0.00504 | 0.00727 | 0.00653 | yes |
| 3 | 0.6 | 0.03858 | 0.00460 | 0.00608 | 0.00460 | yes |
| 4 | 0.8 | 0.05045 | 0.00401 | 0.00475 | 0.00564 | yes |

Shares (%) of quarter-days with a formal unbounded side, a formal empty side, a clipped index: at the 80% pair (k = 1) and over all four pairs:

| regime | setting | unbounded 80 (%) | empty 80 (%) | clipped 80 (%) | unbounded all pairs (%) | empty all pairs (%) | clipped all pairs (%) |
|---|---|---|---|---|---|---|---|
| symblock | hourly | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| symblock | interp | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| symblock | native | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| locblock | hourly | 0.148 | 0.000 | 0.148 | 0.037 | 0.000 | 0.037 |
| locblock | interp | 0.594 | 0.000 | 0.594 | 0.148 | 0.000 | 0.148 |
| locblock | native | 0.371 | 0.000 | 0.371 | 0.093 | 0.000 | 0.093 |
| locslot | hourly | 0.297 | 0.000 | 0.297 | 0.074 | 0.000 | 0.074 |
| locslot | interp | 0.519 | 0.000 | 0.519 | 0.130 | 0.000 | 0.130 |
| locslot | native | 0.371 | 0.000 | 0.371 | 0.093 | 0.000 | 0.093 |

Kupiec per quarter-hour slot, rejections at 5% after Holm over the 96 slots, of 96 (descriptive only):

| setting | level | aci | aciblock | symblock | locblock | locslot |
|---|---|---|---|---|---|---|
| hourly | 80 | 29 | 5 | 10 | 27 | 26 |
| hourly | 60 | 36 | 3 | 7 | 30 | 29 |
| interp | 80 | 18 | 2 | 4 | 13 | 13 |
| interp | 60 | 16 | 2 | 5 | 9 | 10 |
| native | 80 | 10 | 0 | 1 | 5 | 3 |
| native | 60 | 4 | 1 | 1 | 3 | 5 |

Christoffersen (cc) per quarter-hour slot, rejections at 5% after Holm over the 96 slots, of 96:

| setting | level | aci | aciblock | symblock | locblock | locslot |
|---|---|---|---|---|---|---|
| hourly | 80 | 44 | 6 | 13 | 36 | 33 |
| hourly | 60 | 46 | 10 | 10 | 39 | 41 |
| interp | 80 | 30 | 2 | 6 | 20 | 16 |
| interp | 60 | 25 | 3 | 7 | 15 | 16 |
| native | 80 | 19 | 0 | 2 | 10 | 11 |
| native | 60 | 10 | 2 | 1 | 5 | 7 |

Macros written to results/numbers.tex by this run for step 5e (not listed in Section 9):

| macro | value |
|---|---|
| \dmPLocBlockXGB | 0.018 |
| \dmPSymBlockXGB | \ensuremath{<}0.001 |
| \gapCrpsLocBlockXGB | \ensuremath{-0.13} |
| \gapCrpsLocBlockXGBHi | \ensuremath{-0.04} |
| \gapCrpsLocBlockXGBLo | \ensuremath{-0.22} |
| \gapCrpsSymBlockXGB | \ensuremath{-0.35} |
| \gapCrpsSymBlockXGBHi | \ensuremath{-0.26} |
| \gapCrpsSymBlockXGBLo | \ensuremath{-0.45} |
| \gapDscLocBlockXGB | 0.13 |
| \gapDscSymBlockXGB | 0.35 |
| \profEveQthreeInterpXGB | +15.9 |
| \profEveQzeroInterpXGB | \ensuremath{-}8.0 |
| \profMornQthreeInterpXGB | \ensuremath{-}9.3 |
| \profMornQzeroInterpXGB | +12.3 |
| \widthLocBlockInterpXGB | 60.33 |
| \widthLocBlockNativeXGB | 60.31 |

Wall time of the step 5e work (not included in the stage rows of Section 10):

| stage | seconds |
|---|---|
| load + metrics + CORP | 2.9 |
| DM | 0.0 |
| bootstrap | 188.6 |
| Kupiec + Christoffersen | 0.2 |
| shares + profile + macros | 0.2 |

