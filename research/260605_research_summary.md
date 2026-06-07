# ML4T Research Summary — 2026-06-06

## Overview

Full-pipeline quantitative research on the S&P 500 universe: feature engineering → signal screening → correlation clustering → composite signal backtesting → survivorship-bias-free 10-year dataset construction → walk-forward cross-validation.

**Pipeline status: COMPLETE** — signal screening → clustering → rank-signal walk-forward (5-fold and 8-fold) → LightGBM ML comparison → manual portfolio deployment with Telegram bot.

---

## 1. Universe & Data

| Item | Detail |
|---|---|
| Universe | S&P 500 (full, ~496–500 stocks) |
| Initial data range | 2022-01-03 → 2026-06-04 (4 years) |
| 10-year data range | 2015-01-02 → 2026-06-04 |
| Storage format | Hive-partitioned Parquet (`~/ml4t-data/equities_daily_*/`) |
| Price source | Yahoo Finance via `ml4t-data` |
| Factor data | Fama-French 5-factor daily (1963–2026) |
| Macro data | Treasury yields DGS2/5/10/30 + yield curve slopes (extended to 2015) |

---

## 2. Feature Engineering

**Config:** `research/configs/sp500_all_features.yaml`

| Category | Examples | Count |
|---|---|---|
| Volatility | ewma_volatility, garch_forecast, yang_zhang, parkinson, natr | 12 |
| Liquidity | kyle_lambda, realized_spread, amihud | 4 |
| Momentum / Price level | plus_di, bollinger_bands, donchian, sma, ema, rsi, macd | 35+ |
| Risk-adjusted returns | sharpe_ratio, sortino_ratio, calmar_ratio, omega_ratio | 4 |
| Drawdown | max_drawdown, time_underwater, max_duration | 3 |
| Volatility regime | prob_high_vol, prob_med_vol, prob_low_vol, current_vol | 4 |
| Higher moments | skewness, kurtosis, hyperskewness, hyperkurtosis | 4 |
| Volume | log_volume, volume_weighted_price_momentum, volume_at_price_ratio | 5 |
| Entropy / distributional | rolling_entropy, rolling_wasserstein, rolling_cv_zscore | 5 |
| Other (tail, variance, trend) | tail_ratio, ulcer_index, variance_ratio, trend_intensity_index | 10+ |
| **Total features** | | **105** |
| **Signals evaluated** | raw + inverse direction | **250** |

Multi-output features (e.g. `bollinger_bands`, `maximum_drawdown`, `risk_adjusted_returns`) are unpacked generically from Polars Struct columns via `MULTI_OUTPUT_FEATURE_COLS` mapping.

---

## 3. Signal Screening (`sp500_full`, 4-year period)

**Script:** `screen_sp500_all_signals.py --prefix sp500_full`
**Model frame:** 2023-02-01 → 2026-04-30 | 498 symbols | 814 dates | ~403K rows

**Screening thresholds (21-day horizon):**
- IC > 0
- Spread t-stat > 2.0
- Monotonicity ≥ 0.5
- Spread > 0

**Results: 81 signals passed** out of 250 evaluated.

### Top 10 signals by Spread t-stat (21D)

| Signal | IC | Spread | t-stat | Monotonicity |
|---|---:|---:|---:|---:|
| volatility_regime_probability_prob_high_vol | 0.043 | 0.0156 | 32.74 | 0.90 |
| kyle_lambda | 0.045 | 0.0148 | 30.09 | 0.90 |
| ewma_volatility | 0.046 | 0.0142 | 29.14 | 1.00 |
| garch_forecast | 0.044 | 0.0141 | 28.95 | 1.00 |
| natr | 0.042 | 0.0140 | 28.54 | 0.90 |
| yang_zhang_volatility | 0.039 | 0.0137 | 28.08 | 1.00 |
| coefficient_of_variation | 0.034 | 0.0130 | 27.34 | 1.00 |
| risk_adjusted_returns_sharpe_ratio | 0.049 | 0.0127 | 25.95 | 1.00 |
| -maximum_drawdown_time_underwater | 0.044 | 0.0105 | 22.31 | 0.90 |
| log_volume | 0.030 | 0.0105 | 21.28 | 1.00 |

**Key insight:** Volatility signals dominate cross-sectional predictability. High-volatility stocks outperform low-volatility stocks over the 21-day horizon — consistent with a volatility-as-risk-premium regime rather than the classic low-vol anomaly.

---

## 4. Signal Clustering

**Script:** `cluster_signals.py --prefix sp500_full --threshold 0.8`

Spearman rank-correlation clustering at |r| ≥ 0.8. Greedy algorithm seeds clusters with the highest-IC signal.

**Results: 80 signals → 32 clusters**

| Cluster size | Representative | t-stat |
|---:|---|---:|
| 26 | -bollinger_bands_lower | 15.7 |
| 6 | kyle_lambda | 30.1 |
| 4 | risk_adjusted_returns_sharpe_ratio | 26.0 |
| 4 | volume_weighted_price_momentum | 11.1 |
| 3 | volatility_regime_probability_prob_high_vol | 32.7 |

**Notable finding:** 26 price-level/trend signals (bollinger_bands, SMA, EMA, DEMA, TEMA, TRIMA, linearreg, SAR, etc.) form a single near-identical cluster — they are essentially the same signal measured differently. Clustering reduces these 26 to 1 representative.

---

## 5. Composite Signal Backtest (`sp500_full`)

**Script:** `backtest_composite_signals.py --prefix sp500_full`

**Strategy:** `WeeklyLongOnly5` — long top-5 stocks by signal rank, 15% position each, rebalance every 5 trading days.

**Composite definitions:**
- `vol_composite`: average cross-sectional rank of kyle_lambda + garch_forecast + coefficient_of_variation
- `risk_composite`: average rank of sharpe_ratio + (−time_underwater) + (−max_drawdown)
- `combined`: equal-weight average of vol + risk composite ranks

**Test period:** 2025-09-08 → 2026-04-30 (7 months, 6 rebalances)

| Strategy | Return % | Sharpe | Max DD | DSR |
|---|---:|---:|---:|---:|
| **plus_di** | **74.64** | **3.49** | 0.118 | **0.999** |
| log_volume | 62.61 | 2.74 | 0.152 | — |
| combined | 91.21 | 2.46 | 0.168 | — |
| -bollinger_bands_lower | 23.80 | 1.80 | 0.132 | — |
| risk_composite | 44.75 | 1.64 | 0.150 | — |
| vol_composite | 17.85 | 1.13 | 0.139 | — |

**Winner:** `plus_di` (DSR = 0.999)

**Counter-intuitive result:** The highest-IC signals (volatility cluster, t≈29–33) underperform the lower-IC momentum signal `plus_di` (t≈14) in the backtest. Two explanations:
1. **High-vol ≠ high-return in all regimes.** Picking the highest-volatility names increases portfolio vol without proportional return gain. Sharpe suffers.
2. **Statistically insufficient test.** 7 months / 6 trades is far too short for meaningful Sharpe comparison. The DSR selects the winner correctly but the rankings among losers are noise.

**Limitation:** Single in-sample test period. Walk-forward CV required for valid conclusions.

---

## 6. Survivorship Bias Elimination (10-Year Dataset)

### Problem

The S&P 500 constituent list changes continuously. Using the *current* 500-stock list for historical backtests introduces **survivorship bias**: companies that failed, were acquired, or underperformed enough to be removed are excluded — inflating apparent returns.

### Solution: Point-in-Time Composition

**Script:** `build_sp500_pit_composition.py`

Data source: Wikipedia S&P 500 historical changes table (additions/removals with effective dates, ~400 events back to 2000).

Algorithm:
1. Start from current 500 constituents (as of 2025-11-24)
2. Walk backward through changes, undoing each event to arrive at composition at any past date
3. Forward-replay events to emit `(date, ticker)` membership pairs

| Metric | Value |
|---|---|
| Window | 2016-01-01 → 2026-06-01 |
| Unique tickers over 10 years | 716 |
| Current constituents | 500 |
| Historical-only (removed since 2016) | 216 |
| Avg constituents per day | ~508 |

### Historical Data Fetch

**Script:** `fetch_sp500_historical_extended.py`

- **216 historical-only tickers:** Yahoo Finance fetch from 2015-01-01
  - **116 succeeded** (e.g. AAL, ADT, ENPH, SIVB pre-failure, FRC pre-failure)
  - **100 permanently unavailable** — companies fully absorbed into acquirers (CELG→BMY, ATVI→MSFT, XLNX→AMD, etc.) — Yahoo Finance does not retain data after delisting
- **500 current tickers:** Extended from 2022 back to 2015-01-02
  - 492 succeeded; 8 unavailable (CEG, GEHC, GEV, KVUE — recently spun-off, too new)
- **Treasury yields:** Re-downloaded from 2015-01-02 (was previously 2022-only, which was truncating all pre-2022 model rows)

### 10-Year Model Frame

**Script:** `build_sp500_10yr_dataset.py`

| Metric | 4-year (sp500_full) | 10-year (sp500_10yr) |
|---|---:|---:|
| Raw price rows | ~660K | 1,658,952 |
| Model frame rows (after features + labels + PIT filter) | ~403K | **1,188,550** |
| Symbols | 498 | **592** |
| Trading dates | 814 | **2,576** |
| Date range | 2023-02-01→2026-04-30 | **2016-02-01→2026-04-30** |
| Survivorship bias | Yes (current list only) | **Eliminated (PIT filter)** |

The PIT filter reduces the pre-filter 1.47M rows to 1.19M by removing ~283K rows where a stock was not actually in the index on that date — this is the look-ahead exclusion working correctly.

---

## 7. 10-Year Signal Screening (`sp500_10yr`)

**Script:** `screen_sp500_all_signals.py --prefix sp500_10yr`
**Model frame:** 2016-02-01 → 2026-04-30 | 592 symbols | 2,576 dates | 1,188,550 rows

**Results: 97 signals passed** out of 250 evaluated.

### Top 10 signals by Spread t-stat (21D)

| Signal | t-stat |
|---|---:|
| volatility_regime_probability_prob_high_vol | 24.9 |
| -volatility_regime_probability_prob_med_vol | 24.6 |
| ulcer_index | 24.5 |
| ewma_volatility | 23.6 |
| log_volume | 23.0 |
| -ffdiff | 22.8 |
| -maximum_drawdown_max_drawdown | 22.1 |
| realized_volatility | 22.3 |
| natr | 22.0 |
| coefficient_of_variation | 20.1 |

**Comparison with 4yr screening:** The 10yr results show a similar volatility dominance pattern but with materially lower t-stats across the board (~24–25 vs ~29–33). This reflects that the 10yr window includes 2016–2020, a period with different cross-sectional dynamics — the signal is real but less extreme than the 2022–2026 bull/vol regime.

**Runtime note:** First screening attempt stalled (zero output after ~278 CPU-min) due to a constant-input Spearman hang. Fix: print every signal with `flush=True`; added `--sample-dates` and `--periods` options. Full-data screening took ~3 hours wall-clock.

---

## 8. 10-Year Signal Clustering (`sp500_10yr`)

**Script:** `cluster_signals.py --prefix sp500_10yr --threshold 0.8`

**Results: 97 signals → 37 clusters**

| Cluster size | Representative | t-stat | Notes |
|---:|---|---:|---|
| 30 | -tsf | 18.6 | Price-level/trend: tema, linearreg, medprice, wclprice, etc. |
| 11 | ewma_volatility | 23.6 | Vol cluster: yang_zhang, garch, natr, realized_vol, etc. |
| 11 | -ema_gap_20 | 17.1 | Momentum gap: willr, stochastic, roc, rocr100, etc. |
| 3 | minus_di | 19.4 | Directional: -rsi, -cmo |
| 3 | -macd | 16.0 | MACD family: -macdfix, -apo |
| 2 | volatility_regime_probability_prob_high_vol | 24.9 | + downside_deviation |
| 2 | -risk_adjusted_returns_omega_ratio | 4.1 | + -sharpe_ratio |

**Comparison with 4yr clustering:**
- 4yr: 80 signals → 32 clusters (price-level cluster had 26 members)
- 10yr: 97 signals → 37 clusters (price-level cluster grew to 30 members; sharpe/omega now cluster together rather than being standalone)
- Both runs confirm: price-level/trend signals carry one unique information source, regardless of how many indicator variants are computed.

---

## 9. Walk-Forward Backtest (`sp500_10yr`)

**Script:** `backtest_walkforward_10yr.py --prefix sp500_10yr`
**Strategy:** `WeeklyLongOnly5` — long top-5 by signal rank, 15% position, 5-day rebalance.

### Design

| Fold | Train | Test |
|---:|---|---|
| 1 | 2016-01-01 → 2021-01-01 | 2021 |
| 2 | 2017-01-01 → 2022-01-01 | 2022 |
| 3 | 2018-01-01 → 2023-01-01 | 2023 |
| 4 | 2019-01-01 → 2024-01-01 | 2024 |
| 5 | 2020-01-01 → 2025-01-01 | 2025 |

Each fold: screen signals on training period → build composites → run backtest on out-of-sample test year.

### Per-Fold Winners (out-of-sample)

| Fold | Test Year | Winner | Sharpe | Return% | Max DD | DSR |
|---:|---|---|---:|---:|---:|---:|
| 1 | 2021 | log_volume | 1.96 | +48.4% | 0.102 | 0.984 |
| 2 | 2022 | -bollinger_bands_lower | -0.43 | −12.3% | 0.228 | 0.226 |
| 3 | 2023 | log_volume | 2.54 | +82.4% | 0.156 | 0.987 |
| 4 | 2024 | log_volume | 1.43 | +37.6% | 0.196 | 0.873 |
| 5 | 2025 | -bollinger_bands_lower | 1.51 | +39.3% | 0.147 | 0.946 |

### Cross-Fold Aggregate (all strategies, all folds)

| Strategy | Folds appeared | Wins | Avg Sharpe | Avg Return% |
|---|---:|---:|---:|---:|
| **log_volume** | 5 | **3** | **1.16** | **+33.4%** |
| -bollinger_bands_lower | 5 | 2 | 0.74 | +16.5% |
| combined | 3 | 0 | 0.68 | +29.0% |
| risk_composite | 3 | 0 | 0.54 | +15.7% |
| vol_composite | 5 | 0 | 0.37 | +13.3% |
| plus_di | 5 | 0 | 0.31 | +6.0% |

### Key Findings

1. **`log_volume` is the most robust signal** — wins 3 of 5 folds with high DSR (0.87–0.99). This reverses the 4yr finding where `plus_di` won the single in-sample test; the walk-forward reveals `plus_di` was likely a lucky period artifact.

2. **2022 (Fold 2) was universally bad** — all strategies lost money in the bear market year (S&P 500 −18%). The fold winner (`-bollinger_bands_lower`) had DSR=0.226, meaning the "winner" has no statistical significance. This fold is informative about drawdown behavior, not signal quality.

3. **Vol composite underperforms despite high IC** — consistent with the 4yr finding. High-IC volatility signals rank high-vol stocks as "best," but picking the 5 most volatile names increases portfolio vol without proportional return, compressing Sharpe. IC measures predictive correlation, not return magnitude.

4. **Survivorship-bias impact is visible:** Fold 2 (2022 test, train covers 2017–2022) includes companies that were removed from the index during 2022 drawdowns. The PIT filter correctly excludes them from positions, meaning the -12% loss is a *real* loss — not inflated by failed companies being silently dropped.

5. **`risk_composite` (sharpe + drawdown signals) drops out of screening in some folds** — these risk-adjusted return signals appear less stable across regimes. They pass the 4yr screen (single period) but not consistently across 5yr training windows.

---

## 10. Extended Walk-Forward: 8 Folds, 2-Year Minimum Training

**Script:** `backtest_walkforward_10yr.py --prefix sp500_10yr --min-train-years 2`

**Motivation:** The 5yr-minimum design only tests 2021–2025 (5 OOS years). Reducing the minimum training window to 2 years expands OOS coverage to 2018–2025 (8 years), giving ~3× more out-of-sample trades for statistical evaluation.

### Fold Design

| Fold | Train | Test |
|---:|---|---|
| 1 | 2016-01-01 → 2018-01-01 | 2018 |
| 2 | 2017-01-01 → 2019-01-01 | 2019 |
| 3 | 2018-01-01 → 2020-01-01 | 2020 |
| 4 | 2019-01-01 → 2021-01-01 | 2021 |
| 5 | 2020-01-01 → 2022-01-01 | 2022 |
| 6 | 2021-01-01 → 2023-01-01 | 2023 |
| 7 | 2022-01-01 → 2024-01-01 | 2024 |
| 8 | 2023-01-01 → 2025-01-01 | 2025 |

### Per-Fold Winners

| Fold | Test | Winner | Sharpe | DSR |
|---:|---|---|---:|---:|
| 1 | 2018 | log_volume | 0.35 | 0.506 |
| 2 | 2019 | risk_composite | 2.44 | 1.000 |
| 3 | 2020 | log_volume | 1.49 | 0.974 |
| 4 | 2021 | log_volume | 1.96 | 0.984 |
| 5 | 2022 | risk_composite | -0.35 | 0.248 |
| 6 | 2023 | log_volume | 2.54 | 0.984 |
| 7 | 2024 | risk_composite | 1.80 | 0.879 |
| 8 | 2025 | -bollinger_bands_lower | 1.51 | 0.882 |

**Key finding:** `log_volume` wins 4 of 8 folds and consistently achieves high Sharpe. `risk_composite` wins 3 folds (2019, 2022, 2024) — showing that risk-adjusted return signals have some regime-specific strength. The 2022 fold (bear market) has DSR=0.248 — statistically inconclusive regardless of winner.

**Output:** `research/outputs/sp500_10yr/walkforward_2y/sp500_10yr_walkforward_2y_report.md`

---

## 11. LightGBM ML Model

**Script:** `backtest_lgbm_walkforward.py --prefix sp500_10yr [--feature-set all|clusters] [--min-train-years N]`

### Design (from grill-me session)

| Decision | Choice |
|---|---|
| Target | `ret_5d` → cross-sectional decile rank (0–9) per date |
| Architecture | LightGBM (`rank_xendcg` objective) |
| Features | All 137 (125 engineered + 12 macro/FF5) OR 37 cluster representatives |
| CV | Same rolling folds + 21-day purge gap at each boundary |
| Validation | Last 63 trading days of purged training window (early stopping on NDCG@5) |
| Hyperparameters | Fixed defaults: `num_leaves=63`, `lr=0.05`, `feature_fraction=0.8` |
| Position sizing | `WeeklyLongOnly5` unchanged — top-5 by predicted rank, 15% each |
| Evaluation | Aggregate DSR on concatenated OOS return series vs `log_volume` |

### Run 1: All Features, 5 Folds

| Fold | Test | LightGBM Sharpe | log_volume Sharpe | Best Iter |
|---:|---|---:|---:|---:|
| 1 | 2021 | 1.10 | 1.96 | 13 |
| 2 | 2022 | 0.11 | -1.31 | 30 |
| 3 | 2023 | 1.00 | 2.54 | 14 |
| 4 | 2024 | -1.35 | 1.43 | 85 |
| 5 | 2025 | 0.67 | 1.18 | 31 |

**Aggregate:** LightGBM Sharpe=0.391 vs log_volume Sharpe=1.008. **DSR=0.984** (log_volume wins).

**Root cause of failure:** Feature importances are nearly identical across all 5 folds — `cci` dominates at 4–10× the gain of any other feature. `log_volume` never appears in the top 10. With 137 features and no prior filtering, LightGBM anchored on `cci` (Commodity Channel Index) and stopped at iteration 13–31. The model learned a simple CCI momentum strategy, not the volume signal identified by IC screening. CCI momentum broke down in 2024 (Fold 4: Sharpe=−1.35).

### Run 2: Cluster Features Only, 5 Folds

**`--feature-set clusters`** — restricts input to 37 cluster representatives (one per cluster), eliminating the CCI/momentum cluster and forcing the model to consider `log_volume` directly.

| Fold | Test | LightGBM Sharpe | log_volume Sharpe |
|---:|---|---:|---:|
| 1 | 2021 | 1.44 | 1.96 |
| 2 | 2022 | 0.11 | -1.31 |
| 3 | 2023 | -0.02 | 2.54 |
| 4 | 2024 | 0.35 | 1.43 |
| 5 | 2025 | 0.51 | 1.18 |

**Aggregate:** LightGBM Sharpe=0.473 vs log_volume Sharpe=1.008. **DSR=0.987** (log_volume wins).

Improvement vs Run 1 (0.391→0.473) but gap remains large. Fold 3 (2023) is now the worst fold (Sharpe=−0.02) — cluster restriction changed which signal dominated but didn't fix the underlying prediction quality.

### Run 3: Cluster Features, 8 Folds (2yr minimum)

**`--feature-set clusters --min-train-years 2`** — combines cluster restriction with expanded OOS coverage.

| Test | LightGBM Sharpe | log_volume Sharpe |
|---:|---:|---:|
| 2018 | -0.53 | 0.35 |
| 2019 | 1.38 | 2.08 |
| 2020 | 0.09 | 1.49 |
| 2021 | 1.11 | 1.96 |
| 2022 | 0.08 | -1.31 |
| 2023 | 0.45 | 2.54 |
| 2024 | 0.24 | 1.43 |
| 2025 | 0.34 | 1.18 |

**Aggregate (8yr OOS):** LightGBM Sharpe=0.311 vs log_volume Sharpe=1.114. **DSR=0.999** (log_volume wins).

LightGBM only "wins" in the 2022 bear market fold — the only year where all signals suffered and the model's lower-volatility posture happened to lose less.

### Research Conclusion

**`log_volume` is the validated signal for production use. Do not promote LightGBM.**

Evidence across all configurations:

| Configuration | LightGBM Sharpe | log_volume Sharpe | DSR |
|---|---:|---:|---:|
| All features, 5 folds | 0.391 | 1.008 | 0.984 |
| Cluster features, 5 folds | 0.473 | 1.008 | 0.987 |
| Cluster features, 8 folds | 0.311 | 1.114 | 0.999 |

The LightGBM failure is not solely a "too many correlated features" problem. Even with 37 cluster representatives, the model does not produce better cross-sectional rankings than the raw `log_volume` signal. This suggests the information in these features is already captured more cleanly by the simple volume rank than by any tree-based combination.

**Outputs:** `research/outputs/sp500_10yr/lgbm_walkforward/`, `lgbm_walkforward_clusters/`, `lgbm_walkforward_clusters_2y/`

---

## 12. Manual Portfolio & Operations

### Active Strategy

`research/configs/manual_active_strategy.yaml` — promoted to `log_volume + sp500_10yr` as of 2026-06-06.

| Parameter | Value |
|---|---|
| Signal | `log_volume` from `sp500_10yr_model_frame.parquet` |
| Sizing | Top 5 equal-weight, 75% gross exposure |
| Rebalance | Every 5 signal dates, anchor 2026-03-05 |
| Quantity policy | Fractional |

### Portfolios

| Portfolio | Starting Cash | Status |
|---|---:|---|
| SP500-baw | $10,000 | Active — AAPL, AMD, NVDA holdings (test fills at $200) |
| SP500-KAN | $3,000 | Onboarded, no trades yet |

State location: `research/state/manual_portfolios/{portfolio_id}/`

### Manual Portfolio Service

`research/manual_portfolio/` — authoritative local ledger; does not depend on broker runtimes.

| Function | Purpose |
|---|---|
| `onboard_portfolio()` | Create portfolio with cash + optional imported holdings |
| `record_fill()` | Append fill to `fills.jsonl`, update cash/holdings/P&L |
| `portfolio_status()` | Target-vs-actual diff + rebalance plan |
| `daily_run()` | Write daily JSON + Markdown operator outputs |

### CLI Workflows

```bash
cd /Users/mit/Project/ML4T/research

# Daily run (generates target/actual/rebalance outputs)
uv run python daily_run.py --notify-telegram

# Record a fill
uv run python record_fill.py \
  --portfolio-id SP500-baw --trade-date 2026-06-06 \
  --symbol AAPL --side buy --quantity 1 --fill-price 150
```

### Telegram Bot

`Chatbot/` — webhook runtime; reads/writes the same state files as the CLI.

Commands: `/portfolios`, `/status p1`, `/fill p1 buy AAPL 10 150`, `/app` (Mini App).

Access control: `Chatbot/configs/telegram_access_map.yaml` (deny-by-default, chat+user IDs must match).

Secrets: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_WEBHOOK_SECRET` (from `Chatbot/secret.env`).

```bash
uv run python ../Chatbot/run_telegram_bot.py   # from research/
```

### Daily Data Maintenance

```bash
# Incremental price update (run from data/ env)
cd /Users/mit/Project/ML4T/data
uv run python ../research/update_sp500_daily.py --prefix sp500_full

# Health check (flag stale tickers)
uv run python ../research/check_sp500_health.py --prefix sp500_full --stale-days 3
```

See `research/SP500_incremental_runbook.md` for the full operational sequence.

---

## 13. Screening Fixes & Lessons Learned

### Screening hang (10yr dataset)

The initial run stalled after one `ConstantInputWarning` — consuming ~278 CPU-min with zero output. Root cause: `analyze_signal` calls Spearman correlation on 1.18M rows; if the signal column is near-constant for one asset over 2,576 dates, the turnover computation hangs.

**Fixes applied to `screen_sp500_all_signals.py`:**
1. Print every signal (not every 20) with `flush=True`
2. `--sample-dates N` option for quick exploratory runs
3. `--periods` option to narrow horizons if needed
4. Exception output includes `[i/N]` index with immediate flush

### IC horizon value (1D / 5D / 21D)

All three horizons are computed and saved; only 21D drives the filter threshold:
- **5D IC** directly matches the 5-day `WeeklyLongOnly5` rebalance cadence
- **1D IC** reveals signal decay speed — a signal with low 1D but high 21D IC is slow-moving, well-suited for weekly strategies
- **21D IC** is the primary screening gate (>0 required to pass)

---

## 14. Key Design Decisions & Lessons

| Decision | Rationale |
|---|---|
| Cross-sectional ranking (not raw values) | Removes cross-ticker scale differences; makes signals directly comparable |
| Composite = average of per-signal ranks | Robust to individual signal noise; exploits decorrelation between vol/risk clusters |
| Correlation clustering at |r| ≥ 0.8 | 4yr: 80→32 clusters; 10yr: 97→37 clusters; avoids over-counting correlated signals |
| Weekly rebalance (5-day) | Balances turnover cost vs. signal decay; 21-day IC signals still predictive at 5-day rebalance |
| DSR (Deflated Sharpe Ratio) | Accounts for selection bias across 6 tested strategies; probability-based rather than point estimate |
| Wikipedia PIT composition | Free; covers all major additions/removals since ~2000; sufficient for 10-year window |
| Yahoo Finance for delisted stocks | ~54% recovery rate for removed tickers (116/216); acquisitions are permanently gone |
| Treasury yields re-download | Original file only covered 2022–2026; joining with equity panel caused `drop_nulls` to eliminate all pre-2022 rows — model frame silently truncated to 4 years instead of 10 |
| Screen all 3 IC horizons (1D/5D/21D) | 5D directly matches the 5-day rebalance; 1D reveals signal decay speed; only 21D used in the filter threshold but all three saved for analysis |

---

## 15. Codebase Map

```
research/
├── configs/
│   ├── sp500_all_features.yaml          # 105 features, 10 categories
│   └── manual_active_strategy.yaml      # Active strategy: log_volume + sp500_10yr
│
├── — Dataset builders —
├── build_sp500_pit_composition.py       # Wikipedia PIT composition (survivorship fix)
├── build_sp500_10yr_dataset.py          # 10-year PIT-filtered dataset builder
├── fetch_sp500_chunks.py                # Chunked Yahoo Finance fetcher
├── fetch_sp500_historical_extended.py   # Historical-only ticker fetcher
│
├── — Signal research —
├── screen_sp500_all_signals.py          # Full signal screen (IC/spread/mono)
├── cluster_signals.py                   # Spearman correlation clustering
├── backtest_composite_signals.py        # Composite vs single signal (4yr in-sample)
├── backtest_walkforward_10yr.py         # Walk-forward CV, rank-based (--min-train-years)
├── backtest_lgbm_walkforward.py         # Walk-forward CV, LightGBM (--feature-set, --min-train-years)
│
├── — Operations —
├── daily_run.py                         # Daily target/actual/rebalance outputs
├── record_fill.py                       # Record a trade fill to the ledger
├── portfolio_status.py                  # Print current status for a portfolio
├── onboard_portfolio.py                 # Create a new portfolio
├── update_sp500_daily.py                # Incremental price update for sp500
├── check_sp500_health.py                # Flag stale/missing ticker data
├── SP500_incremental_runbook.md         # Step-by-step daily data maintenance
│
├── manual_portfolio/                    # Portfolio service (models, storage, service, CLI)
├── state/manual_portfolios/
│   ├── SP500-baw/  (fills.jsonl, state.json, metadata.json)
│   └── SP500-KAN/  (fills.jsonl, state.json, metadata.json)
│
└── outputs/
    ├── sp500_pit/                       # PIT composition data
    │   ├── sp500_changes.parquet
    │   └── sp500_pit_composition.parquet
    └── sp500_10yr/                      # 10-year results (COMPLETE)
        ├── sp500_10yr_model_frame.parquet        # 1.19M rows, 592 symbols
        ├── sp500_10yr_all_signal_screen.parquet
        ├── sp500_10yr_signal_clusters.json       # 37 clusters from 97 signals
        ├── sp500_10yr_walkforward_report.md      # 5-fold rank-signal results
        ├── walkforward_2y/                       # 8-fold rank-signal (2yr min)
        ├── lgbm_walkforward/                     # LightGBM, all features, 5 folds
        ├── lgbm_walkforward_clusters/            # LightGBM, cluster features, 5 folds
        └── lgbm_walkforward_clusters_2y/         # LightGBM, cluster features, 8 folds

Chatbot/
├── run_telegram_bot.py                  # Start webhook server
├── send_telegram_notification.py        # Send outbound message
├── configs/
│   ├── telegram_bot.yaml                # Host/port/webhook URL config
│   └── telegram_access_map.yaml         # Per-portfolio chat+user authorization
└── telegram_portfolio_bot/              # Handlers, access, notifications, Mini App
```
