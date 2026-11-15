# Step 5.2 evaluation summary: LEAR

`python -m esann27.evaluate --model lear`, git 17792b2, run 2026-11-10 21:58. Evaluation period 2025-10-29 .. 2026-09-30 (337 days, 32,348 quarters), target = native price. CRPS = 2 x mean pinball over 9 levels. MCB / DSC / UNC on the CRPS scale (2 x mean over levels). Newey-West: Bartlett, lag 7, no small-sample correction, normal p. Bootstrap: circular blocks of 7 days, 49 blocks truncated to 337 days, 1000 draws, default_rng(2027), the same draws for both settings and both regimes.

## 1. Proposition A (burn-in 1-28 Oct 2025 only, computed and written before any evaluation metric)

| position | sigma_q native | interp sd | kappa_q | s_I | s_N |
|---|---|---|---|---|---|
| :00 | 22.0664 | 7.2719 | 0.3295 | 26.6508 | 33.8277 |
| :15 | 9.1120 | 2.8322 | 0.3108 | 25.7955 | 27.2106 |
| :30 | 8.9016 | 2.8450 | 0.3196 | 25.7969 | 27.1408 |
| :45 | 21.9797 | 7.2568 | 0.3302 | 26.6467 | 33.7712 |

s_H = 25.6395 EUR/MWh (std, ddof 1, of the 672 hourly residuals y - yhat of the hourly LEAR point forecast against the hourly mean native price; residual mean 5.5435). F = ECDF of the residuals / s_H (not centered). Per-hour s_H (information only, not used): min 8.40 (h0), max 53.48 (h19).

Predicted raw PICP-80 (%) and CRPS of the native setting at pi = 0, 0.5, 1. gap = CRPS(pi) - CRPS(1) (item (i)), gap_interp_minus_native = CRPS(0) - CRPS(pi) (used for Fig. 1 and \predCrpsGapHalf).

| pi | position | picp80 | crps | gap | gap_interp_minus_native |
|---|---|---|---|---|---|
| 0.0000 | 0 | 74.4048 | 17.6411 | 0.1028 | 0.0000 |
| 0.5000 | 0 | 76.4881 | 17.5618 | 0.0235 | 0.0793 |
| 1.0000 | 0 | 79.9107 | 17.5383 | 0.0000 | 0.1028 |
| 0.0000 | 1 | 79.1667 | 14.1120 | 0.0043 | 0.0000 |
| 0.5000 | 1 | 79.3155 | 14.1095 | 0.0019 | 0.0025 |
| 1.0000 | 1 | 79.9107 | 14.1076 | 0.0000 | 0.0043 |
| 0.0000 | 2 | 79.1667 | 14.0754 | 0.0040 | 0.0000 |
| 0.5000 | 2 | 79.4643 | 14.0732 | 0.0018 | 0.0022 |
| 1.0000 | 2 | 79.9107 | 14.0715 | 0.0000 | 0.0040 |
| 0.0000 | 3 | 74.4048 | 17.6105 | 0.1015 | 0.0000 |
| 0.5000 | 3 | 76.4881 | 17.5322 | 0.0232 | 0.0783 |
| 1.0000 | 3 | 79.9107 | 17.5090 | 0.0000 | 0.1015 |
| 0.0000 | pooled | 76.7857 | 15.8598 | 0.0531 | 0.0000 |
| 0.5000 | pooled | 77.9390 | 15.8192 | 0.0126 | 0.0406 |
| 1.0000 | pooled | 79.9107 | 15.8066 | 0.0000 | 0.0531 |

Predicted raw PICP-80 by position over the evaluation days (Fig. 2(b) dashes): interp at pi = 0, native = mean over the 337 days of the prediction at pi_D.

| kind | position | pi | picp80 | crps |
|---|---|---|---|---|
| eval_native | 0 | 0.5385 | 76.8471 | 17.5697 |
| eval_interp | 0 | 0.0000 | 74.4048 | 17.6411 |
| eval_native | 1 | 0.5385 | 79.4122 | 14.1093 |
| eval_interp | 1 | 0.0000 | 79.1667 | 14.1120 |
| eval_native | 2 | 0.5385 | 79.4519 | 14.0730 |
| eval_interp | 2 | 0.0000 | 79.1667 | 14.0754 |
| eval_native | 3 | 0.5385 | 76.8590 | 17.5400 |
| eval_interp | 3 | 0.0000 | 74.4048 | 17.6105 |

## 2. Metrics per file (evaluation period)

| model | setting | layer | MAE | RMSE | rMAE | CRPS | MCB | DSC | UNC | PICP80 | width80 | PICP60 | width60 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| naive | Hourly | raw | 34.5251 | 54.9515 | 1.0000 | 28.7666 |  |  |  | 76.84 | 98.6328 | 54.27 | 50.7747 |
| naive | Interpolated | raw | 34.0425 | 54.4415 | 1.0000 | 28.3784 |  |  |  | 77.69 | 99.5431 | 55.24 | 51.1107 |
| naive | Native | raw | 33.9859 | 54.8241 | 1.0000 | 28.3856 |  |  |  | 77.96 | 100.6974 | 55.95 | 51.7408 |
| lear | Hourly | raw | 19.8373 | 33.0582 | 0.5746 | 16.1432 | 0.6270 | 19.1002 | 34.6163 | 71.14 | 52.8564 | 51.72 | 32.3506 |
| lear | Hourly | split | 19.8373 | 33.0582 | 0.5746 | 16.1490 | 0.5875 | 19.0548 | 34.6163 | 76.30 | 59.6594 | 58.40 | 37.5103 |
| lear | Hourly | rolling | 19.8373 | 33.0582 | 0.5746 | 15.9826 | 0.4849 | 19.1187 | 34.6163 | 78.97 | 62.1420 | 59.44 | 37.5302 |
| lear | Hourly | acifrozen | 19.8373 | 33.0582 | 0.5746 | 15.9324 | 0.4834 | 19.1673 | 34.6163 | 80.14 | 63.2833 | 60.47 | 38.0978 |
| lear | Hourly | aci | 19.8373 | 33.0582 | 0.5746 | 16.0310 | 0.4855 | 19.0708 | 34.6163 | 80.07 | 65.1705 | 60.32 | 38.9240 |
| lear | Hourly | acipooled | 19.8373 | 33.0582 | 0.5746 | 16.0690 | 0.5391 | 19.0865 | 34.6163 | 80.01 | 64.7961 | 60.20 | 38.8344 |
| lear | Interpolated | raw | 18.9174 | 31.9181 | 0.5557 | 15.0808 | 0.5544 | 20.0899 | 34.6163 | 71.88 | 51.6520 | 52.46 | 31.6882 |
| lear | Interpolated | split | 18.9174 | 31.9181 | 0.5557 | 15.0440 | 0.5079 | 20.0802 | 34.6163 | 78.10 | 57.6547 | 59.45 | 36.0173 |
| lear | Interpolated | rolling | 18.9174 | 31.9181 | 0.5557 | 15.0354 | 0.4978 | 20.0787 | 34.6163 | 78.93 | 59.1905 | 59.34 | 35.8682 |
| lear | Interpolated | acifrozen | 18.9174 | 31.9181 | 0.5557 | 14.9720 | 0.4978 | 20.1422 | 34.6163 | 80.12 | 60.0854 | 60.34 | 36.0444 |
| lear | Interpolated | aci | 18.9174 | 31.9181 | 0.5557 | 15.0827 | 0.5135 | 20.0471 | 34.6163 | 80.06 | 62.1310 | 60.21 | 36.9741 |
| lear | Interpolated | acipooled | 18.9174 | 31.9181 | 0.5557 | 15.1029 | 0.5189 | 20.0323 | 34.6163 | 80.00 | 61.7592 | 60.12 | 36.8850 |
| lear | Native | raw | 18.5957 | 31.3455 | 0.5472 | 14.6955 | 0.5306 | 20.4514 | 34.6163 | 75.54 | 54.7308 | 56.65 | 33.7597 |
| lear | Native | split | 18.5957 | 31.3455 | 0.5472 | 14.6948 | 0.5067 | 20.4282 | 34.6163 | 78.67 | 57.9463 | 59.26 | 35.3580 |
| lear | Native | rolling | 18.5957 | 31.3455 | 0.5472 | 14.7357 | 0.5330 | 20.4137 | 34.6163 | 78.67 | 58.7144 | 58.81 | 35.6808 |
| lear | Native | acifrozen | 18.5957 | 31.3455 | 0.5472 | 14.6719 | 0.5274 | 20.4719 | 34.6163 | 79.87 | 59.5265 | 60.01 | 35.8685 |
| lear | Native | aci | 18.5957 | 31.3455 | 0.5472 | 14.7751 | 0.5597 | 20.4010 | 34.6163 | 80.01 | 61.4066 | 60.13 | 37.1567 |
| lear | Native | acipooled | 18.5957 | 31.3455 | 0.5472 | 14.7769 | 0.5592 | 20.3987 | 34.6163 | 79.96 | 61.2072 | 60.07 | 37.0701 |

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

## 4. Table 2 (results/table2_lear.tex), interpolated LEAR

| regime | CRPS | PICP80 | width80 | unbounded |
|---|---|---|---|---|
| split | 15.0440 | 78.0976 | 57.6547 | 0.0000 |
| rolling | 15.0354 | 78.9291 | 59.1905 | 0.0000 |
| acifrozen | 14.9720 | 80.1224 | 60.0854 | 0.0000 |
| aci | 15.0827 | 80.0575 | 62.1310 | 0.0742 |

```latex
\begin{tabular}{|l|c|c|c|c|}
\hline
Regime & CRPS & PICP-80 & Width-80 & Unbounded \\
\hline
Split-CQR & 15.04 & 78.1 & 57.65 & 0.0 \\
Rolling-CQR & 15.04 & 78.9 & 59.19 & 0.0 \\
ACI-frozen & 14.97 & 80.1 & 60.09 & 0.0 \\
ACI-CQR & 15.08 & 80.1 & 62.13 & 0.1 \\
\hline
\end{tabular}
```

Unbounded = share (%) of quarter-days with a formal unbounded side at the 80% pair. All regimes:

| setting | regime | unbounded_pct |
|---|---|---|
| hourly | split | 0.000 |
| hourly | rolling | 0.000 |
| hourly | acifrozen | 0.000 |
| hourly | aci | 0.000 |
| hourly | acipooled | 0.000 |
| interp | split | 0.000 |
| interp | rolling | 0.000 |
| interp | acifrozen | 0.000 |
| interp | aci | 0.074 |
| interp | acipooled | 0.000 |
| native | split | 0.000 |
| native | rolling | 0.000 |
| native | acifrozen | 0.000 |
| native | aci | 0.000 |
| native | acipooled | 0.000 |

## 5. Diebold-Mariano (daily CRPS native - interp) and pi-regression (daily gap interp - native on pi_D)

| regime | dm_mean_native_minus_interp | dm_se | dm_stat | dm_p | intercept | intercept_se | slope | slope_se | slope_z | slope_p | T |
|---|---|---|---|---|---|---|---|---|---|---|---|
| raw | -0.3852 | 0.0664 | -5.8040 | 6.474e-09 | 0.1806 | 0.1031 | 0.3799 | 0.2417 | 1.5719 | 0.116 | 337 |
| aci | -0.3075 | 0.0547 | -5.6176 | 1.936e-08 | 0.1689 | 0.0892 | 0.2576 | 0.1944 | 1.3246 | 0.1853 | 337 |

## 6. Bootstrap 90% intervals of native - interp

| regime | quantity | native | interp | diff_native_minus_interp | lo5 | hi95 | boot_mean | boot_sd | n_draws |
|---|---|---|---|---|---|---|---|---|---|
| raw | CRPS | 14.6955 | 15.0808 | -0.3852 | -0.4958 | -0.2831 | -0.3865 | 0.0650 | 1000 |
| raw | MCB | 0.5306 | 0.5544 | -0.0237 | -0.0883 | 0.0253 | -0.0293 | 0.0345 | 1000 |
| raw | DSC | 20.4514 | 20.0899 | 0.3615 | 0.2781 | 0.4454 | 0.3573 | 0.0496 | 1000 |
| aci | CRPS | 14.7751 | 15.0827 | -0.3076 | -0.3993 | -0.2258 | -0.3077 | 0.0541 | 1000 |
| aci | MCB | 0.5597 | 0.5135 | 0.0462 | -0.0049 | 0.0795 | 0.0407 | 0.0257 | 1000 |
| aci | DSC | 20.4010 | 20.0471 | 0.3539 | 0.2683 | 0.4300 | 0.3485 | 0.0491 | 1000 |

## 7. Monthly CRPS (results/monthly.csv)

| month | n_days | mean_pi | hourly_aci | hourly_acifrozen | hourly_acipooled | hourly_raw | hourly_rolling | hourly_split | interp_aci | interp_acifrozen | interp_acipooled | interp_raw | interp_rolling | interp_split | native_aci | native_acifrozen | native_acipooled | native_raw | native_rolling | native_split | gap_raw (interp-native) | gap_aci (interp-native) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2025-10 | 3 | 0.080 | 10.5718 | 10.5368 | 10.7173 | 10.7283 | 10.5394 | 10.5143 | 9.8661 | 9.8467 | 10.0002 | 9.9044 | 9.8524 | 9.8394 | 10.1875 | 10.1421 | 10.2671 | 10.0919 | 10.1436 | 10.1025 | -0.1875 | -0.3214 |
| 2025-11 | 30 | 0.125 | 11.2624 | 11.2561 | 11.3068 | 11.3292 | 11.2731 | 11.2949 | 10.7252 | 10.7289 | 10.7735 | 10.8257 | 10.7597 | 10.7710 | 10.5221 | 10.5056 | 10.5434 | 10.5483 | 10.5159 | 10.4966 | 0.2774 | 0.2031 |
| 2025-12 | 31 | 0.209 | 8.9685 | 8.9917 | 8.9866 | 9.6171 | 9.0386 | 9.3359 | 9.2704 | 9.2947 | 9.2953 | 9.7763 | 9.3284 | 9.6210 | 9.1220 | 9.2470 | 9.1333 | 9.5964 | 9.2553 | 9.9312 | 0.1799 | 0.1485 |
| 2026-01 | 31 | 0.294 | 11.7583 | 11.3803 | 11.7967 | 11.0837 | 11.6745 | 11.3229 | 11.8466 | 11.3410 | 11.8491 | 11.1242 | 11.6794 | 11.1462 | 11.6102 | 11.1673 | 11.6039 | 10.9075 | 11.5090 | 10.7425 | 0.2167 | 0.2364 |
| 2026-02 | 28 | 0.375 | 8.4651 | 8.4300 | 8.4962 | 8.5465 | 8.5179 | 8.4801 | 8.1949 | 8.1466 | 8.2203 | 8.1740 | 8.2026 | 8.1708 | 8.1650 | 8.1075 | 8.1734 | 8.0639 | 8.1422 | 8.1821 | 0.1101 | 0.0299 |
| 2026-03 | 31 | 0.456 | 17.5644 | 17.4521 | 17.6292 | 17.6727 | 17.4558 | 17.5452 | 16.2295 | 16.1454 | 16.2580 | 16.2579 | 16.0968 | 16.0816 | 15.9058 | 15.8324 | 15.9076 | 15.8779 | 15.7830 | 15.7413 | 0.3800 | 0.3237 |
| 2026-04 | 30 | 0.540 | 19.8176 | 19.7304 | 19.8704 | 19.9430 | 19.7175 | 19.8598 | 18.2977 | 18.2297 | 18.3183 | 18.3245 | 18.2128 | 18.2196 | 17.7253 | 17.6439 | 17.7270 | 17.6517 | 17.6280 | 17.5842 | 0.6727 | 0.5724 |
| 2026-05 | 31 | 0.624 | 17.2635 | 17.2606 | 17.3265 | 17.5470 | 17.2696 | 17.6517 | 15.9790 | 15.9791 | 16.0014 | 16.0905 | 16.0009 | 16.1239 | 15.4000 | 15.3878 | 15.3990 | 15.5076 | 15.4550 | 15.4537 | 0.5828 | 0.5791 |
| 2026-06 | 30 | 0.707 | 23.9126 | 23.7578 | 23.9420 | 24.2223 | 23.8914 | 24.2569 | 22.1653 | 22.0433 | 22.1804 | 22.3611 | 22.1850 | 22.3166 | 21.4987 | 21.3213 | 21.4870 | 21.5258 | 21.4928 | 21.4007 | 0.8354 | 0.6666 |
| 2026-07 | 31 | 0.791 | 16.0886 | 15.9463 | 16.1133 | 15.7363 | 15.8937 | 15.7661 | 15.0364 | 14.8703 | 15.0367 | 14.6414 | 14.8704 | 14.6759 | 14.9317 | 14.8414 | 14.9185 | 14.5934 | 14.8153 | 14.8117 | 0.0480 | 0.1048 |
| 2026-08 | 31 | 0.876 | 17.3824 | 17.2253 | 17.4232 | 17.6905 | 17.3701 | 17.8758 | 15.6893 | 15.4867 | 15.7114 | 15.7296 | 15.6865 | 15.7874 | 15.2697 | 15.0055 | 15.2688 | 15.0048 | 15.2618 | 14.9909 | 0.7248 | 0.4197 |
| 2026-09 | 30 | 0.960 | 24.1396 | 24.1124 | 24.1384 | 24.4740 | 23.9945 | 24.5501 | 22.7543 | 22.7087 | 22.7562 | 22.8710 | 22.6495 | 22.8613 | 22.6088 | 22.5664 | 22.6104 | 22.6161 | 22.4661 | 22.5505 | 0.2549 | 0.1455 |

## 8. Kupiec and Christoffersen per quarter-hour slot across days (supplement), Holm over the 96 slots per family, level and file

Rejections at 5% after Holm, all files: Kupiec 741 of 3456, Christoffersen (cc) 1035 of 3456. Per file and level (of 96 slots; slot detail in results/kupiec.csv):

| setting | layer | level | slots | kupiec_rej | cc_rej |
|---|---|---|---|---|---|
| hourly | raw | 80 | 96 | 50 | 60 |
| hourly | raw | 60 | 96 | 37 | 44 |
| hourly | split | 80 | 96 | 45 | 59 |
| hourly | split | 60 | 96 | 32 | 54 |
| hourly | rolling | 80 | 96 | 26 | 34 |
| hourly | rolling | 60 | 96 | 26 | 42 |
| hourly | acifrozen | 80 | 96 | 24 | 33 |
| hourly | acifrozen | 60 | 96 | 27 | 42 |
| hourly | aci | 80 | 96 | 23 | 32 |
| hourly | aci | 60 | 96 | 25 | 42 |
| hourly | acipooled | 80 | 96 | 32 | 48 |
| hourly | acipooled | 60 | 96 | 34 | 48 |
| interp | raw | 80 | 96 | 60 | 62 |
| interp | raw | 60 | 96 | 35 | 40 |
| interp | split | 80 | 96 | 28 | 34 |
| interp | split | 60 | 96 | 20 | 30 |
| interp | rolling | 80 | 96 | 17 | 29 |
| interp | rolling | 60 | 96 | 20 | 24 |
| interp | acifrozen | 80 | 96 | 16 | 23 |
| interp | acifrozen | 60 | 96 | 18 | 27 |
| interp | aci | 80 | 96 | 15 | 26 |
| interp | aci | 60 | 96 | 16 | 25 |
| interp | acipooled | 80 | 96 | 25 | 36 |
| interp | acipooled | 60 | 96 | 27 | 33 |
| native | raw | 80 | 96 | 16 | 19 |
| native | raw | 60 | 96 | 4 | 8 |
| native | split | 80 | 96 | 17 | 17 |
| native | split | 60 | 96 | 13 | 15 |
| native | rolling | 80 | 96 | 1 | 7 |
| native | rolling | 60 | 96 | 1 | 4 |
| native | acifrozen | 80 | 96 | 3 | 9 |
| native | acifrozen | 60 | 96 | 2 | 6 |
| native | aci | 80 | 96 | 2 | 9 |
| native | aci | 60 | 96 | 0 | 3 |
| native | acipooled | 80 | 96 | 4 | 10 |
| native | acipooled | 60 | 96 | 0 | 1 |

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
| \dmPAciLEAR | \ensuremath{<}0.001 |
| \dmPRawLEAR | \ensuremath{<}0.001 |
| \gapCrpsAciLEAR | \ensuremath{-0.31} |
| \gapCrpsAciLEARHi | \ensuremath{-0.23} |
| \gapCrpsAciLEARLo | \ensuremath{-0.40} |
| \gapCrpsRawLEAR | \ensuremath{-0.39} |
| \gapCrpsRawLEARHi | \ensuremath{-0.28} |
| \gapCrpsRawLEARLo | \ensuremath{-0.50} |
| \gapDscAciLEAR | 0.35 |
| \gapDscAciLEARHi | 0.43 |
| \gapDscAciLEARLo | 0.27 |
| \gapDscRawLEAR | 0.36 |
| \gapDscRawLEARHi | 0.45 |
| \gapDscRawLEARLo | 0.28 |
| \gapMcbAciLEAR | 0.05 |
| \gapMcbAciLEARHi | 0.08 |
| \gapMcbAciLEARLo | 0.00 |
| \gapMcbRawLEAR | \ensuremath{-0.02} |
| \gapMcbRawLEARHi | 0.03 |
| \gapMcbRawLEARLo | \ensuremath{-0.09} |
| \hlGapRemovedLEAR | 20.1 |
| \hlPicpAciLEAR | 80.1 |
| \hlPicpSplitLEAR | 78.1 |
| \naiveMAEhourly | 34.53 |
| \naiveMAEinterp | 34.04 |
| \naiveMAEnative | 33.99 |
| \predCrpsGapHalf | 0.04 |
| \predPicpHalf | 77.9 |
| \predPicpZero | 76.8 |
| \slopeAciLEAR | 0.26 |
| \slopeAciLEARP | 0.185 |
| \slopeAciLEARSe | 0.19 |
| \slopeRawLEAR | 0.38 |
| \slopeRawLEARP | 0.116 |
| \slopeRawLEARSe | 0.24 |

## 10. Wall time

| stage | seconds |
|---|---|
| Proposition A | 1.5 |
| metrics + CORP | 5.4 |
| DM + regression + monthly | 0.1 |
| bootstrap | 102.0 |
| Kupiec + Christoffersen | 0.4 |
| tables + figures | 0.6 |
| total | 322.3 |

## 11. Step 5c: ACI-CQR with 24 groups (aciblock)

ACI-CQR with 24 Mondrian groups g = 4 (h // 4) + p (h = clock hour, p = quarter position), handover 8.1 row ACI-block, step 5c specification. Files conformal/<setting>_lear_aciblock.parquet, evaluation period as above. `aci` = ACI-CQR with the four position groups; `- aci` = aciblock minus ACI-CQR in the same setting. LEAR.

Metrics per setting (CRPS, MCB, DSC, UNC on the CRPS scale; PICP and width at the issued bounds):

| setting | CRPS | MCB | DSC | UNC | PICP80 | PICP60 | width80 | CRPS aci | MCB aci | CRPS - aci | MCB - aci |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hourly | 15.4462 | 0.4883 | 19.6585 | 34.6163 | 80.66 | 61.52 | 71.1854 | 16.0310 | 0.4855 | -0.5848 | 0.0029 |
| interp | 14.9742 | 0.5434 | 20.1856 | 34.6163 | 80.41 | 60.93 | 69.6405 | 15.0827 | 0.5135 | -0.1086 | 0.0299 |
| native | 14.9059 | 0.5849 | 20.2953 | 34.6163 | 80.28 | 60.84 | 69.2274 | 14.7751 | 0.5597 | 0.1309 | 0.0252 |

PICP-80 (%) by group, minimum and maximum over the 24 groups:

| setting | groups | PICP80 min | group min | PICP80 max | group max |
|---|---|---|---|---|---|
| hourly | 24 | 79.81 | 3 | 82.12 | 23 |
| interp | 24 | 79.51 | 3 | 81.45 | 23 |
| native | 24 | 79.44 | 3 | 81.01 | 8 |

Shares (%) of quarter-days with a formal unbounded side, a formal empty side, a clipped index: at the 80% pair (k = 1) and over all four pairs (share of quarter-day x pair combinations):

| setting | unbounded 80 (%) | empty 80 (%) | clipped 80 (%) | unbounded all pairs (%) | empty all pairs (%) | clipped all pairs (%) |
|---|---|---|---|---|---|---|
| hourly | 1.805 | 0.000 | 1.805 | 0.451 | 0.000 | 0.451 |
| interp | 1.768 | 0.000 | 1.768 | 0.445 | 0.000 | 0.445 |
| native | 2.003 | 0.000 | 2.003 | 0.516 | 0.000 | 0.516 |

Prop. 4.1 per side: max over the 24 groups and both sides of |mean over the 337 days of err - 0.1k| (err of the formal side), per setting, next to the bound (max(0.1k, 1 - 0.1k) + gamma) / (gamma 337):

| k | target | bound | hourly | interp | native | within bound |
|---|---|---|---|---|---|---|
| 1 | 0.1 | 0.05638 | 0.00386 | 0.00386 | 0.00386 | yes |
| 2 | 0.2 | 0.05045 | 0.00623 | 0.00475 | 0.00475 | yes |
| 3 | 0.3 | 0.04451 | 0.00786 | 0.00712 | 0.00786 | yes |
| 4 | 0.4 | 0.03858 | 0.00801 | 0.00801 | 0.00653 | yes |

Diebold-Mariano on the daily CRPS difference native - interp (Newey-West, Bartlett lag 7, normal p), ACI-CQR beside ACI-block:

| regime | mean | se | stat | p | T |
|---|---|---|---|---|---|
| aci | -0.3075 | 0.0547 | -5.6176 | 1.936e-08 | 337 |
| aciblock | -0.0681 | 0.0433 | -1.5734 | 0.1156 | 337 |

Bootstrap 90% intervals of native - interp (the same 1,000 draws as raw and ACI-CQR, circular blocks of 7 days, default_rng(2027)), ACI-CQR beside ACI-block:

| regime | quantity | native | interp | diff_native_minus_interp | lo5 | hi95 | boot_mean | boot_sd | n_draws |
|---|---|---|---|---|---|---|---|---|---|
| aci | CRPS | 14.7751 | 15.0827 | -0.3076 | -0.3993 | -0.2258 | -0.3077 | 0.0541 | 1000 |
| aciblock | CRPS | 14.9059 | 14.9742 | -0.0682 | -0.1397 | 0.0044 | -0.0684 | 0.0437 | 1000 |
| aci | MCB | 0.5597 | 0.5135 | 0.0462 | -0.0049 | 0.0795 | 0.0407 | 0.0257 | 1000 |
| aciblock | MCB | 0.5849 | 0.5434 | 0.0414 | -0.0002 | 0.0695 | 0.0372 | 0.0210 | 1000 |
| aci | DSC | 20.4010 | 20.0471 | 0.3539 | 0.2683 | 0.4300 | 0.3485 | 0.0491 | 1000 |
| aciblock | DSC | 20.2953 | 20.1856 | 0.1097 | 0.0362 | 0.1780 | 0.1056 | 0.0442 | 1000 |

Kupiec and Christoffersen (cc) per quarter-hour slot, rejections at 5% after Holm over the 96 slots, of 96 (descriptive only):

| setting | level | kupiec_rej aci | kupiec_rej aciblock | cc_rej aci | cc_rej aciblock |
|---|---|---|---|---|---|
| interp | 80 | 15 | 3 | 26 | 5 |
| interp | 60 | 16 | 6 | 25 | 7 |
| native | 80 | 2 | 0 | 9 | 0 |
| native | 60 | 0 | 0 | 3 | 2 |

Decision rule of 8.1 (row ACI-block), inputs: DM p = 0.116 (0.1156), CRPS gap lo5..hi95 = -0.1397 .. 0.0044.

rule: gap holds = DM p < 0.05 and lo5..hi95 excludes 0 -> no

Macros written to results/numbers.tex by this run for step 5c (not listed in Section 9):

| macro | value |
|---|---|
| \dCrpsBlockInterpLEAR | \ensuremath{-0.11} |
| \dCrpsBlockNativeLEAR | 0.13 |
| \dmPBlockLEAR | 0.116 |
| \gapCrpsBlockLEAR | \ensuremath{-0.07} |
| \gapCrpsBlockLEARHi | 0.00 |
| \gapCrpsBlockLEARLo | \ensuremath{-0.14} |
| \gapDscBlockLEAR | 0.11 |
| \gapDscBlockLEARHi | 0.18 |
| \gapDscBlockLEARLo | 0.04 |
| \gapMcbBlockLEAR | 0.04 |
| \gapMcbBlockLEARHi | 0.07 |
| \gapMcbBlockLEARLo | 0.00 |

Wall time of the step 5c work (not included in the stage rows of Section 10):

| stage | seconds |
|---|---|
| load + metrics + CORP | 1.0 |
| DM | 0.0 |
| bootstrap | 51.5 |
| Kupiec + Christoffersen | 0.1 |
| shares + groups + macros | 0.1 |

## 12. Step 5e: diagnostic layers (symblock, locblock, locslot), defined after the 24-group result

Diagnostic layers, defined after the 24-group result of step 5c and run with its workspace values known (handover 8.1 rows Sym-block, Loc-block, Loc-slot, step 5e specification). Diagnostics, not tests. Files conformal/<setting>_lear_{symblock,locblock,locslot}.parquet, evaluation period as above. `symblock` = ACI-CQR with the 24 groups of aciblock and a symmetric score (width only); `locblock` / `locslot` = recentering by the rolling 28-day median of y - q50 per block group (24) / quarter-hour slot (96), then the four-group ACI-CQR (location only). `aci` and `aciblock` repeated beside them. LEAR.

Metrics per setting and regime (CRPS, MCB, DSC, UNC on the CRPS scale; PICP and width at the issued bounds):

| setting | regime | CRPS | MCB | DSC | UNC | PICP80 | PICP60 | width80 |
|---|---|---|---|---|---|---|---|---|
| hourly | aci | 16.0310 | 0.4855 | 19.0708 | 34.6163 | 80.07 | 60.32 | 65.1705 |
| hourly | aciblock | 15.4462 | 0.4883 | 19.6585 | 34.6163 | 80.66 | 61.52 | 71.1854 |
| hourly | symblock | 16.0266 | 0.4606 | 19.0504 | 34.6163 | 80.14 | 60.12 | 67.6466 |
| hourly | locblock | 15.4308 | 0.4571 | 19.6426 | 34.6163 | 80.05 | 60.20 | 63.2941 |
| hourly | locslot | 15.2733 | 0.4723 | 19.8154 | 34.6163 | 80.08 | 60.20 | 62.5918 |
| interp | aci | 15.0827 | 0.5135 | 20.0471 | 34.6163 | 80.06 | 60.21 | 62.1310 |
| interp | aciblock | 14.9742 | 0.5434 | 20.1856 | 34.6163 | 80.41 | 60.93 | 69.6405 |
| interp | symblock | 15.0665 | 0.4774 | 20.0273 | 34.6163 | 80.10 | 60.11 | 64.3452 |
| interp | locblock | 14.9328 | 0.5142 | 20.1977 | 34.6163 | 79.98 | 60.11 | 62.3029 |
| interp | locslot | 14.8952 | 0.5420 | 20.2631 | 34.6163 | 79.99 | 60.10 | 61.9241 |
| native | aci | 14.7751 | 0.5597 | 20.4010 | 34.6163 | 80.01 | 60.13 | 61.4066 |
| native | aciblock | 14.9059 | 0.5849 | 20.2953 | 34.6163 | 80.28 | 60.84 | 69.2274 |
| native | symblock | 14.7578 | 0.5055 | 20.3640 | 34.6163 | 80.11 | 60.03 | 63.7737 |
| native | locblock | 14.8687 | 0.5470 | 20.2947 | 34.6163 | 79.97 | 60.10 | 62.3947 |
| native | locslot | 14.9253 | 0.5621 | 20.2532 | 34.6163 | 80.02 | 60.21 | 62.7760 |

Diebold-Mariano on the daily CRPS difference native - interp (Newey-West, Bartlett lag 7, normal p):

| regime | mean | se | stat | p | T |
|---|---|---|---|---|---|
| aci | -0.3075 | 0.0547 | -5.6176 | 1.936e-08 | 337 |
| aciblock | -0.0681 | 0.0433 | -1.5734 | 0.1156 | 337 |
| symblock | -0.3086 | 0.0591 | -5.2249 | 1.742e-07 | 337 |
| locblock | -0.0640 | 0.0507 | -1.2643 | 0.2061 | 337 |
| locslot | 0.0302 | 0.0509 | 0.5931 | 0.5531 | 337 |

Bootstrap 90% intervals of native - interp (the same 1,000 draws as every other regime, circular blocks of 7 days, default_rng(2027)):

| regime | quantity | native | interp | diff_native_minus_interp | lo5 | hi95 | boot_mean | boot_sd | n_draws |
|---|---|---|---|---|---|---|---|---|---|
| aci | CRPS | 14.7751 | 15.0827 | -0.3076 | -0.3993 | -0.2258 | -0.3077 | 0.0541 | 1000 |
| aciblock | CRPS | 14.9059 | 14.9742 | -0.0682 | -0.1397 | 0.0044 | -0.0684 | 0.0437 | 1000 |
| symblock | CRPS | 14.7578 | 15.0665 | -0.3087 | -0.4106 | -0.2220 | -0.3097 | 0.0582 | 1000 |
| locblock | CRPS | 14.8687 | 14.9328 | -0.0641 | -0.1490 | 0.0132 | -0.0649 | 0.0503 | 1000 |
| locslot | CRPS | 14.9253 | 14.8952 | 0.0301 | -0.0569 | 0.1077 | 0.0293 | 0.0507 | 1000 |
| aci | MCB | 0.5597 | 0.5135 | 0.0462 | -0.0049 | 0.0795 | 0.0407 | 0.0257 | 1000 |
| aciblock | MCB | 0.5849 | 0.5434 | 0.0414 | -0.0002 | 0.0695 | 0.0372 | 0.0210 | 1000 |
| symblock | MCB | 0.5055 | 0.4774 | 0.0280 | -0.0244 | 0.0723 | 0.0265 | 0.0295 | 1000 |
| locblock | MCB | 0.5470 | 0.5142 | 0.0329 | -0.0026 | 0.0667 | 0.0333 | 0.0221 | 1000 |
| locslot | MCB | 0.5621 | 0.5420 | 0.0202 | -0.0101 | 0.0558 | 0.0234 | 0.0201 | 1000 |
| aci | DSC | 20.4010 | 20.0471 | 0.3539 | 0.2683 | 0.4300 | 0.3485 | 0.0491 | 1000 |
| aciblock | DSC | 20.2953 | 20.1856 | 0.1097 | 0.0362 | 0.1780 | 0.1056 | 0.0442 | 1000 |
| symblock | DSC | 20.3640 | 20.0273 | 0.3367 | 0.2596 | 0.4213 | 0.3361 | 0.0492 | 1000 |
| locblock | DSC | 20.2947 | 20.1977 | 0.0970 | 0.0228 | 0.1744 | 0.0983 | 0.0473 | 1000 |
| locslot | DSC | 20.2532 | 20.2631 | -0.0100 | -0.0817 | 0.0726 | -0.0059 | 0.0473 | 1000 |

Shares W and L (handover Section 8 row "Paper decisions of 10 Nov and step 5e scope"): G = |CRPS gap native - interp| under aci (G4), aciblock (G24), symblock (GS), locblock (GL); W = (G4 - GS) / (G4 - G24), L = (G4 - GL) / (G4 - G24); recentering if W <= 1/3 and L >= 2/3 in both models, width if W >= 2/3 and L <= 1/3 in both, mixed otherwise. The other model is read from its metrics file (pending until its step 5e rows exist).

| model | G4 | G24 | GS | GL | W | L |
|---|---|---|---|---|---|---|
| LEAR | 0.3076 | 0.0682 | 0.3087 | 0.0641 | -0.004 | 1.017 |
| XGB | 0.3740 | 0.1133 | 0.3519 | 0.1285 | 0.085 | 0.942 |

reading: W = -0.004 / 0.085, L = 1.017 / 0.942 (LEAR / XGB) -> recentering

Residual profile (results/profile_lear.csv): mean and unscaled MAD of the raw residual y - q50 per block group g = 4 (h // 4) + p over the evaluation period, interp and native:

| group | block | position | mean interp | mad interp | mean native | mad native | n |
|---|---|---|---|---|---|---|---|
| 0 | 00-03 | 0 | 4.69 | 5.43 | 3.17 | 5.52 | 1347 |
| 1 | 00-03 | 1 | 0.99 | 4.77 | 0.09 | 5.22 | 1347 |
| 2 | 00-03 | 2 | -1.00 | 4.78 | -0.64 | 5.13 | 1347 |
| 3 | 00-03 | 3 | -3.32 | 5.43 | -1.43 | 5.40 | 1347 |
| 4 | 04-07 | 0 | -1.54 | 7.98 | -0.91 | 7.75 | 1348 |
| 5 | 04-07 | 1 | 0.24 | 7.45 | 0.39 | 7.41 | 1348 |
| 6 | 04-07 | 2 | 0.81 | 7.59 | 0.47 | 8.03 | 1348 |
| 7 | 04-07 | 3 | 0.54 | 9.14 | 0.15 | 9.46 | 1348 |
| 8 | 08-11 | 0 | 9.34 | 18.51 | 4.04 | 17.52 | 1348 |
| 9 | 08-11 | 1 | 0.79 | 16.67 | -0.33 | 16.34 | 1348 |
| 10 | 08-11 | 2 | -4.18 | 17.07 | -2.80 | 16.92 | 1348 |
| 11 | 08-11 | 3 | -10.49 | 17.79 | -5.58 | 17.72 | 1348 |
| 12 | 12-15 | 0 | -2.05 | 17.94 | -0.46 | 17.69 | 1348 |
| 13 | 12-15 | 1 | -1.10 | 17.66 | -0.20 | 17.65 | 1348 |
| 14 | 12-15 | 2 | 0.71 | 17.66 | 0.78 | 17.63 | 1348 |
| 15 | 12-15 | 3 | 3.04 | 18.82 | 2.09 | 18.75 | 1348 |
| 16 | 16-19 | 0 | -8.82 | 14.59 | -3.44 | 13.92 | 1348 |
| 17 | 16-19 | 1 | -2.04 | 13.16 | -0.15 | 12.98 | 1348 |
| 18 | 16-19 | 2 | 4.46 | 14.52 | 2.55 | 13.95 | 1348 |
| 19 | 16-19 | 3 | 13.54 | 18.83 | 6.79 | 16.70 | 1348 |
| 20 | 20-23 | 0 | 13.08 | 11.02 | 9.24 | 10.73 | 1348 |
| 21 | 20-23 | 1 | 8.10 | 9.69 | 6.14 | 9.43 | 1348 |
| 22 | 20-23 | 2 | 5.69 | 9.44 | 4.97 | 9.62 | 1348 |
| 23 | 20-23 | 3 | 0.52 | 9.61 | 3.56 | 9.51 | 1348 |

Sym-block, two-sided Prop. 4.1: max over the 24 groups of |mean over the 337 days of err - 0.2k| (err of the formal two-sided set), per setting, next to the bound (max(0.2k, 1 - 0.2k) + gamma) / (gamma 337):

| k | target | bound | hourly | interp | native | within bound |
|---|---|---|---|---|---|---|
| 1 | 0.2 | 0.05045 | 0.00252 | 0.00490 | 0.00401 | yes |
| 2 | 0.4 | 0.03858 | 0.00653 | 0.00608 | 0.00682 | yes |
| 3 | 0.6 | 0.03858 | 0.00579 | 0.00579 | 0.00504 | yes |
| 4 | 0.8 | 0.05045 | 0.00415 | 0.00475 | 0.00549 | yes |

Shares (%) of quarter-days with a formal unbounded side, a formal empty side, a clipped index: at the 80% pair (k = 1) and over all four pairs:

| regime | setting | unbounded 80 (%) | empty 80 (%) | clipped 80 (%) | unbounded all pairs (%) | empty all pairs (%) | clipped all pairs (%) |
|---|---|---|---|---|---|---|---|
| symblock | hourly | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| symblock | interp | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| symblock | native | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| locblock | hourly | 0.074 | 0.000 | 0.074 | 0.019 | 0.000 | 0.019 |
| locblock | interp | 0.074 | 0.000 | 0.074 | 0.019 | 0.000 | 0.019 |
| locblock | native | 0.074 | 0.000 | 0.074 | 0.019 | 0.000 | 0.019 |
| locslot | hourly | 0.148 | 0.000 | 0.148 | 0.037 | 0.000 | 0.037 |
| locslot | interp | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| locslot | native | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

Kupiec per quarter-hour slot, rejections at 5% after Holm over the 96 slots, of 96 (descriptive only):

| setting | level | aci | aciblock | symblock | locblock | locslot |
|---|---|---|---|---|---|---|
| hourly | 80 | 23 | 7 | 7 | 14 | 17 |
| hourly | 60 | 25 | 8 | 8 | 19 | 18 |
| interp | 80 | 15 | 3 | 4 | 10 | 13 |
| interp | 60 | 16 | 6 | 6 | 8 | 8 |
| native | 80 | 2 | 0 | 0 | 1 | 1 |
| native | 60 | 0 | 0 | 0 | 0 | 0 |

Christoffersen (cc) per quarter-hour slot, rejections at 5% after Holm over the 96 slots, of 96:

| setting | level | aci | aciblock | symblock | locblock | locslot |
|---|---|---|---|---|---|---|
| hourly | 80 | 32 | 11 | 13 | 35 | 32 |
| hourly | 60 | 42 | 13 | 14 | 32 | 39 |
| interp | 80 | 26 | 5 | 6 | 17 | 21 |
| interp | 60 | 25 | 7 | 7 | 16 | 20 |
| native | 80 | 9 | 0 | 3 | 2 | 6 |
| native | 60 | 3 | 2 | 1 | 4 | 3 |

Macros written to results/numbers.tex by this run for step 5e (not listed in Section 9):

| macro | value |
|---|---|
| \dmPLocBlockLEAR | 0.206 |
| \dmPSymBlockLEAR | \ensuremath{<}0.001 |
| \gapCrpsLocBlockLEAR | \ensuremath{-0.06} |
| \gapCrpsLocBlockLEARHi | 0.01 |
| \gapCrpsLocBlockLEARLo | \ensuremath{-0.15} |
| \gapCrpsSymBlockLEAR | \ensuremath{-0.31} |
| \gapCrpsSymBlockLEARHi | \ensuremath{-0.22} |
| \gapCrpsSymBlockLEARLo | \ensuremath{-0.41} |
| \gapDscLocBlockLEAR | 0.10 |
| \gapDscSymBlockLEAR | 0.34 |
| \profEveQthreeInterpLEAR | +13.5 |
| \profEveQzeroInterpLEAR | \ensuremath{-}8.8 |
| \profMornQthreeInterpLEAR | \ensuremath{-}10.5 |
| \profMornQzeroInterpLEAR | +9.3 |
| \widthLocBlockInterpLEAR | 62.30 |
| \widthLocBlockNativeLEAR | 62.39 |

Wall time of the step 5e work (not included in the stage rows of Section 10):

| stage | seconds |
|---|---|
| load + metrics + CORP | 2.9 |
| DM | 0.0 |
| bootstrap | 156.2 |
| Kupiec + Christoffersen | 0.3 |
| shares + profile + macros | 0.2 |

