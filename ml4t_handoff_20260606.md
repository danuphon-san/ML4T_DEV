# ML4T Project Handoff — 2026-06-06

This document enables a fresh agent to understand the full history and current state of the ML4T quantitative research pipeline, and to continue work from exactly where the previous sessions left off.

---

## 1. Project Overview

**Repo:** `/Users/mit/Project/ML4T`
**Goal:** Full-pipeline quantitative research on the S&P 500 universe — feature engineering → signal screening → clustering → backtesting → survivorship-bias-free evaluation → ML model comparison.
**Stack:** Python monorepo with 7 packages (`data`, `engineer`, `models`, `diagnostic`, `backtest`, `live`, `specs`) + Rust ITCH parser. All managed with `uv`. See `CLAUDE.md` for full architecture.

---

## 2. Existing Documentation to Read First

Before doing anything, read these files in order:

| File | What it contains |
|---|---|
| `/Users/mit/Project/ML4T/CLAUDE.md` | Repo architecture, all package commands, test baselines |
| `/Users/mit/Project/ML4T/research/260605_research_summary.md` | Complete English research summary — all findings through signal screening, clustering, walk-forward backtest, and survivorship bias elimination (Sections 1–12) |
| `/Users/mit/Project/ML4T/instructions_th.md` | Thai-language pipeline guide — rationale for every script from data fetch to backtesting (comprehensive, for human reference) |

---

## 3. What Has Been Completed (Two Sessions)

### Session 1 (2026-06-05): Full Pipeline Build

All steps below are **fully complete and outputs exist on disk**.

**3.1 Universe & Data**
- S&P 500 full universe, 2022–2026 (4yr) and 2015–2026 (10yr)
- Storage: Hive-partitioned Parquet at `~/ml4t-data/equities_daily_*/`
- Treasury yields re-downloaded from 2015-01-01 (original was 2022-only, causing silent truncation via `drop_nulls`)

**3.2 Survivorship Bias Elimination**
- `research/build_sp500_pit_composition.py`: scrapes Wikipedia S&P 500 changes table, reconstructs point-in-time (PIT) composition via backward-unwind + forward-replay
- 716 unique tickers identified over 2016–2026 (500 current + 216 historical-only)
- `research/fetch_sp500_historical_extended.py`: fetched 116/216 historical tickers; 100 permanently unavailable (acquisitions)
- Output: `research/outputs/sp500_pit/sp500_pit_composition.parquet`

**3.3 10-Year Model Frame**
- Script: `research/build_sp500_10yr_dataset.py`
- PIT filter applied: 1,188,550 rows, 592 symbols, 2,576 dates (2016-02-01 → 2026-04-30)
- Output: `research/outputs/sp500_10yr/sp500_10yr_model_frame.parquet` (970 MB)
- 105+ engineered features + 12 macro/FF5 context columns + labels

**3.4 Signal Screening (4yr and 10yr)**
- Script: `research/screen_sp500_all_signals.py --prefix sp500_10yr`
- 10yr: 97/250 signals passed (IC>0, spread_t>2.0, monotonicity≥0.5)
- Top signals: `volatility_regime_probability_prob_high_vol` (t=24.9), `ulcer_index` (t=24.5), `ewma_volatility` (t=23.6), `log_volume` (t=23.0)
- Outputs: `research/outputs/sp500_10yr/sp500_10yr_all_signal_screen.parquet`, `sp500_10yr_all_top_signals.json`

**3.5 Signal Clustering (10yr)**
- Script: `research/cluster_signals.py --prefix sp500_10yr --threshold 0.8`
- 97 signals → 37 clusters (Spearman |r| ≥ 0.8)
- Key clusters: price-level/trend (30 members, rep=`-tsf`), volatility (11 members, rep=`ewma_volatility`), momentum gap (11 members, rep=`-ema_gap_20`)
- Output: `research/outputs/sp500_10yr/sp500_10yr_signal_clusters.json`

**3.6 Walk-Forward Backtest (rank-based signals, 5 folds)**
- Script: `research/backtest_walkforward_10yr.py --prefix sp500_10yr`
- Strategy: `WeeklyLongOnly5` — long top-5 by signal rank, 15% each, 5-day rebalance
- **`log_volume` wins 3/5 folds** (avg Sharpe=1.16 across 5 folds)
- Key finding: volatility signals have highest IC but underperform in backtests; `log_volume` is the most robust cross-sectional signal
- Full results in `research/260605_research_summary.md` Section 9
- Output: `research/outputs/sp500_10yr/sp500_10yr_walkforward_report.md`

---

### Session 2 (2026-06-06): LightGBM ML Model

**3.7 Design Decisions (grill-me session)**

A full design interview was conducted. All decisions are final:

| Decision | Choice | Rationale |
|---|---|---|
| ML goal | Replace rank-based signals with ML model | Keep `WeeklyLongOnly5` strategy intact; change only signal generation |
| Target variable | `ret_5d` → cross-sectional decile rank (0–9) per date | Regression carries more info than triple-barrier labels; `ret_5d` already in model frame |
| Architecture | LightGBM | Tabular cross-sectional data; fast; feature importances; no need for LSTM's sequential memory given pre-engineered features |
| Feature set | All 105+ engineered features + 12 macro/FF5 context | No pre-screening (avoids look-ahead bias); macro context addresses 2022 bear market regime |
| CV structure | Same 5 folds + **21-day purge gap** at each boundary | `ret_5d` label for last training dates leaks 5 days into test period; 21-day buffer eliminates contamination |
| Loss function | `rank_xendcg` | Optimises NDCG; immune to return outliers; directly optimises what `WeeklyLongOnly5` needs (ordering, not magnitude) |
| Label encoding | Decile rank of `ret_5d` within each date (0–9) | Quintile (0–4) too coarse for top-5 out of 590 stocks; decile gives better gradient signal |
| Hyperparameters | Fixed defaults + early stopping on NDCG@5 (validation = last 63 trading days of purged training) | Avoids Optuna per-fold search; early stopping handles `num_boost_round` |
| Position sizing | `WeeklyLongOnly5` unchanged | Isolates signal quality change from position sizing change |
| Evaluation | Aggregate DSR on concatenated 5-year OOS series (LightGBM vs `log_volume`) | Statistically rigorous; accounts for selection bias; ~1,260 OOS trading days |

**3.8 LightGBM Script**
- New file: `research/backtest_lgbm_walkforward.py`
- LightGBM added to research dependencies: `lightgbm>=4.6.0` in `research/pyproject.toml`
- Outputs: `research/outputs/sp500_10yr/lgbm_walkforward/`

**3.9 LightGBM Walk-Forward Results (COMPLETE)**

All 5 folds finished. `log_volume` wins decisively.

| Fold | Test | LightGBM Sharpe | log_volume Sharpe |
|---:|---|---:|---:|
| 1 | 2021 | 1.10 | 1.96 |
| 2 | 2022 | 0.11 | -1.31 |
| 3 | 2023 | 1.00 | 2.54 |
| 4 | 2024 | **-1.35** | 1.43 |
| 5 | 2025 | 0.67 | 1.18 |

**Aggregate 5-year OOS:** LightGBM Sharpe=0.391 vs log_volume Sharpe=1.008. **DSR=0.984** (log_volume wins at 98.4% confidence).

Full report: `research/outputs/sp500_10yr/lgbm_walkforward/sp500_10yr_lgbm_walkforward_report.md`

**3.10 LightGBM Failure Diagnosis**

Feature importances (consistent across all 5 folds):

| Rank | Feature | Gain (fold 1) |
|---:|---|---:|
| 1 | `cci` | 4,235 |
| 2 | `ema_gap_20` | 966 |
| 3 | `stochf` | 882 |
| 4 | `Mkt-RF` | 253 |
| ... | ... | ... |
| — | `log_volume` | **not in top 10** |

**Root cause:** With 137 features and no prior filtering, LightGBM anchors on `cci` (Commodity Channel Index) — an easily-exploitable overbought/oversold pattern — and stops at iteration 13–31 (early stopping kicks in immediately). The model learns a CCI momentum strategy, ignoring `log_volume`. CCI momentum failed in 2024 (Fold 4: Sharpe=-1.35) — a mega-cap AI-dominated market where simple cross-sectional momentum broke down.

---

## 4. Open Questions / Next Steps

The following questions were raised but not yet resolved:

### 4.1 Coverage gap: only 5 of 10 years are tested OOS
The current walk-forward design uses a 5-year minimum training window, so the earliest test period is 2021. Years 2016–2020 are never tested out-of-sample. To test 2018–2025, use a 2-year minimum rolling window (8 folds instead of 5).

**Status:** Identified but not implemented. Agreed this is worth doing.

### 4.2 Training window sensitivity analysis
Should we run scenarios with 1yr, 2yr, 3yr, 4yr, 5yr minimum training windows to check if conclusions are robust?

**Status:** Discussed. Agreed this is valid as a **robustness check only** (not for hyperparameter selection — that would be overfitting). No implementation yet.

### 4.3 LightGBM fix: constrain feature set
The most targeted fix for the CCI-domination problem: constrain LightGBM inputs to the **37 cluster representatives** (one per cluster, from `sp500_10yr_signal_clusters.json`). This eliminates the `cci`/`stochf` momentum cluster and forces the model to consider `log_volume`.

**Status:** Identified, not yet implemented. This is the recommended next step for the LightGBM work.

### 4.4 Live trading
The `live/` package has Alpaca/IB integration. `log_volume` is the current best signal for `WeeklyLongOnly5`. No live trading work has been started.

---

## 5. Key Technical Gotchas

| Issue | What happened | Fix |
|---|---|---|
| Treasury yields truncation | `treasury_yields.parquet` only covered 2022+; `drop_nulls` silently removed all pre-2022 rows from model frame | Re-downloaded yields from 2015-01-01 via yfinance proxy tickers |
| PIT composition algorithm | Backward pass must undo ALL events `d >= start` to arrive at correct composition at window start | Fixed in `build_sp500_pit_composition.py` |
| Screening hang (10yr) | First signal hit constant-input Spearman correlation; script only printed every 20 signals | Print every signal with `flush=True`; added `--sample-dates` and `--periods` options |
| `build_sp500_10yr_dataset.py` wrong venv | Running `uv run python /path/...` from wrong directory picks up wrong venv | Always run as: `sh -c 'cd /path/to/research && uv run python script.py'` |
| LightGBM early stopping at 13 iterations | 137 features, CCI dominates; val NDCG doesn't improve after ~15 trees | Next fix: restrict to 37 cluster representatives |

---

## 6. Codebase Map (research scripts)

```
research/
├── configs/sp500_all_features.yaml          # 105 features, 10 categories
├── build_research_dataset.py                # 4-year sp500_full dataset builder
├── build_sp500_pit_composition.py           # Wikipedia PIT composition (survivorship fix)
├── build_sp500_10yr_dataset.py              # 10-year PIT-filtered dataset builder
├── fetch_sp500_chunks.py                    # Chunked Yahoo Finance fetcher
├── fetch_sp500_historical_extended.py       # Historical-only ticker fetcher
├── screen_sp500_all_signals.py              # Full signal screen (IC/spread/mono)
├── cluster_signals.py                       # Spearman correlation clustering
├── backtest_composite_signals.py            # Composite vs single signal (4yr, in-sample)
├── backtest_walkforward_10yr.py             # Walk-forward CV, rank-based signals (5 folds)
├── backtest_lgbm_walkforward.py             # Walk-forward CV, LightGBM model (5 folds) ← NEW
├── research_universe.py                     # UniverseSpec helper (prefix → paths)
└── outputs/
    ├── sp500_full/                          # 4-year results (complete)
    ├── sp500_pit/                           # PIT composition data
    └── sp500_10yr/                          # 10-year results (complete)
        ├── sp500_10yr_model_frame.parquet   # 1.19M rows, 592 symbols, 2016–2026
        ├── sp500_10yr_all_signal_screen.parquet
        ├── sp500_10yr_all_top_signals.json
        ├── sp500_10yr_signal_clusters.json  # 37 clusters from 97 signals
        ├── sp500_10yr_walkforward_report.md # Rank-based 5-fold results
        └── lgbm_walkforward/               # LightGBM 5-fold results
            ├── sp500_10yr_lgbm_walkforward_report.md
            ├── fold_01/ … fold_05/         # Per-fold: feature_importance.json, daily_returns
```

---

## 7. Environment Setup

```bash
# Run any research script
cd /Users/mit/Project/ML4T/research
uv run python <script.py> [args]

# Dependencies include: polars, numpy, pandas, lightgbm>=4.6.0, pyarrow,
#   ml4t-engineer[ta,store], ml4t-models, ml4t-diagnostic (local packages)

# macOS: diagnostic package requires libomp
brew install libomp
```

---

## 8. Suggested Next Steps (in priority order)

1. **Fix LightGBM: restrict features to 37 cluster representatives**
   - Read cluster reps from `research/outputs/sp500_10yr/sp500_10yr_signal_clusters.json`
   - Add `--feature-set [all|clusters]` argument to `backtest_lgbm_walkforward.py`
   - Re-run all 5 folds and compare DSR vs current result

2. **Extend walk-forward to test 2018–2025 (8 folds, 2yr min training)**
   - Update `FOLDS` in both `backtest_walkforward_10yr.py` and `backtest_lgbm_walkforward.py`
   - Re-run both and compare aggregate DSR with more OOS years

3. **Training window sensitivity analysis (robustness check only)**
   - Run scenarios for 2yr, 3yr, 4yr, 5yr min training windows
   - Report all results; do NOT select best window for live use (overfitting risk)
   - Purpose: validate that `log_volume` conclusion holds across all window lengths

4. **Live trading design**
   - `log_volume` + `WeeklyLongOnly5` is the current best validated strategy
   - `live/` package has Alpaca/IB integration
   - Need: paper trading setup, position sizing review, transaction cost validation

---

## 9. Suggested Skills for Next Session

- `/grill-me` — Use before implementing the LightGBM cluster-restricted feature set to stress-test the design (e.g., should we use cluster representatives from the full 10yr screen or re-cluster within each fold's training window?)
- `/diagnose` — If LightGBM with cluster features still fails, use to systematically diagnose whether the problem is the feature set, the objective, or the training window
- `/to-issues` — Once research conclusions are stable, use to break the live trading work into independently-implementable GitHub issues

---

## 10. Continuation Update — 2026-06-06

This section was appended after the original handoff above. It records the work completed later on 2026-06-06 and supersedes the earlier "not yet implemented" notes for cluster-restricted LightGBM and 2-year walk-forward coverage.

### 10.1 LightGBM cluster feature set implemented

`research/backtest_lgbm_walkforward.py` now supports:

```bash
uv run python backtest_lgbm_walkforward.py --prefix sp500_10yr --feature-set clusters
uv run python backtest_lgbm_walkforward.py --prefix sp500_10yr --feature-set clusters --min-train-years 2
```

The cluster feature set reads representatives from:

`research/outputs/sp500_10yr/sp500_10yr_signal_clusters.json`

It resolves signal representatives into 37 base engineered feature columns. Macro/FF5 columns are not included in the cluster run, so the cluster model uses 37 engineered features and 0 macro/FF5 context columns.

### 10.2 Cluster LightGBM 5-fold result

Output report:

`research/outputs/sp500_10yr/lgbm_walkforward_clusters/sp500_10yr_lgbm_walkforward_clusters_report.md`

Result:

| Strategy | Aggregate 5-year OOS Sharpe |
|---|---:|
| LightGBM, cluster features | 0.473 |
| log_volume baseline | 1.008 |

DSR probability, best of 2: **0.987**.

Interpretation: constraining LightGBM to 37 cluster representatives improved LightGBM only slightly versus the all-feature run (0.473 vs 0.391 Sharpe), but did not close the gap. `log_volume` still wins with statistically significant confidence.

### 10.3 2-year minimum walk-forward implemented

`research/backtest_walkforward_10yr.py` now supports:

```bash
uv run python backtest_walkforward_10yr.py --prefix sp500_10yr --min-train-years 2
```

This expands OOS coverage from 5 folds (2021-2025) to 8 folds (2018-2025):

| Fold | Train | Test |
|---:|---|---|
| 1 | 2016-01-01 to 2018-01-01 | 2018 |
| 2 | 2017-01-01 to 2019-01-01 | 2019 |
| 3 | 2018-01-01 to 2020-01-01 | 2020 |
| 4 | 2019-01-01 to 2021-01-01 | 2021 |
| 5 | 2020-01-01 to 2022-01-01 | 2022 |
| 6 | 2021-01-01 to 2023-01-01 | 2023 |
| 7 | 2022-01-01 to 2024-01-01 | 2024 |
| 8 | 2023-01-01 to 2025-01-01 | 2025 |

Rank-signal output report:

`research/outputs/sp500_10yr/walkforward_2y/sp500_10yr_walkforward_2y_report.md`

Main finding: `log_volume` remains very strong across the expanded OOS window. It is not the top strategy in every single year, but it has consistently high Sharpe in most folds and remains the strongest simple baseline to beat.

### 10.4 Cluster LightGBM 2-year walk-forward result

Output report:

`research/outputs/sp500_10yr/lgbm_walkforward_clusters_2y/sp500_10yr_lgbm_walkforward_clusters_2y_report.md`

Result:

| Strategy | Aggregate OOS Sharpe |
|---|---:|
| LightGBM, cluster features, 2-year rolling train | 0.311 |
| log_volume baseline, 2-year rolling train | 1.114 |

DSR probability, best of 2: **0.999**.

Per-fold result summary:

| Test year | LightGBM Sharpe | log_volume Sharpe |
|---:|---:|---:|
| 2018 | -0.53 | 0.35 |
| 2019 | 1.38 | 2.08 |
| 2020 | 0.09 | 1.49 |
| 2021 | 1.11 | 1.96 |
| 2022 | 0.08 | -1.31 |
| 2023 | 0.45 | 2.54 |
| 2024 | 0.24 | 1.43 |
| 2025 | 0.34 | 1.18 |

Interpretation: LightGBM only beats `log_volume` in the 2022 bear-market fold. Across the full 2018-2025 OOS period, `log_volume` is decisively superior.

### 10.5 Current research conclusion

The strongest current conclusion is:

**Use `log_volume` as the validated baseline signal for `WeeklyLongOnly5`; do not promote LightGBM yet.**

Evidence:

- Original all-feature LightGBM lost to `log_volume`: Sharpe 0.391 vs 1.008, DSR 0.984.
- Cluster-restricted 5-fold LightGBM still lost: Sharpe 0.473 vs 1.008, DSR 0.987.
- Cluster-restricted 8-fold / 2-year LightGBM lost more clearly: Sharpe 0.311 vs 1.114, DSR 0.999.
- `log_volume` remained robust when OOS coverage expanded from 2021-2025 to 2018-2025.

The LightGBM failure is no longer just a "too many correlated features / CCI domination" issue. Even after reducing to one representative per cluster, the model does not produce a better rank ordering than the simple `log_volume` baseline.

### 10.6 Remaining next steps

Recommended next work:

1. Update `research/260605_research_summary.md` with the cluster LightGBM and 2-year walk-forward results above. The file currently ends at the first 5-fold walk-forward story and does not yet include the later LightGBM robustness results.
2. Run a formal training-window sensitivity table for rank-signal baselines and LightGBM: 1-year, 2-year, 3-year, 4-year, 5-year minimum training windows. Treat this strictly as robustness analysis, not model selection.
3. If more ML is attempted, change the modeling question instead of only changing features. Options: pairwise/listwise ranking diagnostics, regime-specific models, sector-neutral ranking, turnover-aware labels, or portfolio-level objective evaluation.
4. Start live/paper-trading design around `log_volume + WeeklyLongOnly5`, including transaction costs, liquidity caps, position limits, and broker integration through `live/`.

### 10.7 Suggested skills for the next agent

- `/diagnose` — Use if continuing ML work; the next diagnostic question is why LightGBM fails even with cluster representatives.
- `/grill-me` — Use before changing the ML objective or adding regime/sector constraints.
- `/to-issues` — Use when converting the validated `log_volume` live-trading path into implementation tickets.

---

## 11. Continuation Update — Telegram + Manual Portfolio Workflow

This section records work that was not included in the first continuation append. It covers the Telegram implementation and the manual portfolio operator workflow.

### 11.1 Manual portfolio service implemented

Core implementation lives in:

- `research/manual_portfolio/models.py`
- `research/manual_portfolio/storage.py`
- `research/manual_portfolio/registry.py`
- `research/manual_portfolio/service.py`
- `research/manual_portfolio/cli.py`

The manual portfolio service is the authoritative local ledger and target-vs-actual workflow. It intentionally does not depend on broker runtimes or `live/` accounting state.

State location:

`research/state/manual_portfolios/`

Current observed state includes:

- `research/state/manual_portfolios/SP500-baw/state.json`
- `research/state/manual_portfolios/SP500-baw/metadata.json`
- `research/state/manual_portfolios/SP500-baw/fills.jsonl`
- `research/state/manual_portfolios/.telegram/processed_updates.jsonl`

Main service functions:

- `onboard_portfolio(...)`: creates metadata, initial cash, optional imported holdings, state file, and fill journal.
- `record_fill(...)`: appends a fill to `fills.jsonl`, updates cash, holdings, average cost, and realized P&L.
- `portfolio_status(...)`: compares actual holdings against promoted strategy targets and builds a rebalance plan.
- `daily_run(...)`: writes daily target, actual, rebalance, and Markdown operator outputs per portfolio.

Accounting semantics:

- Buy: decreases cash by quantity * fill price + commission + slippage; blends average cost including trade costs.
- Sell: validates available quantity, increases cash by proceeds net of costs, updates realized P&L, and closes/remains holdings.
- Duplicate `fill_id` values are rejected to protect the append-only journal.

### 11.2 Promotion registry / active portfolio strategy

Active strategy config:

`research/configs/manual_active_strategy.yaml`

Current config observed:

- `active_strategy_id: obv`
- `source_prefix: sp100_seed`
- rebalance cadence: every 5 signal dates, anchor `2026-03-05`
- sizing: top 3 equal-weight positions, 60% gross exposure
- quantity policy: fractional
- signal artifact: `../outputs/sp100_seed/backtests_long_only/obv/predictions.parquet`
- price artifact: `../outputs/sp100_seed/sp100_seed_signal_prices.parquet`

Important: this manual workflow currently points to the older `sp100_seed` / `obv` promotion registry, not the newer `sp500_10yr` / `log_volume` research conclusion. If the next work is to align operations with the latest research, update `manual_active_strategy.yaml` carefully and regenerate/verify the referenced signal and price artifacts first.

### 11.3 CLI workflow

Run from the research environment:

```bash
cd /Users/mit/Project/ML4T/research
```

Daily run:

```bash
uv run python daily_run.py
uv run python daily_run.py --portfolio-id SP500-baw
uv run python daily_run.py --notify-telegram
```

Record a fill:

```bash
uv run python record_fill.py \
  --portfolio-id SP500-baw \
  --trade-date 2026-06-06 \
  --symbol AAPL \
  --side buy \
  --quantity 1 \
  --fill-price 100 \
  --notify-telegram
```

Other console entrypoints are exposed through `research/manual_portfolio/cli.py` and the `research/pyproject.toml` scripts.

Daily outputs are written under each portfolio's state directory by date, including:

- `daily_run.json`
- `target_snapshot.json`
- `actual_snapshot.json`
- `rebalance_plan.json`
- `daily_summary.md`

### 11.4 Telegram bot implementation

Telegram transport lives in:

`Chatbot/`

Key files:

- `Chatbot/README.md`
- `Chatbot/run_telegram_bot.py`
- `Chatbot/send_telegram_notification.py`
- `Chatbot/configs/telegram_bot.yaml`
- `Chatbot/configs/telegram_access_map.yaml`
- `Chatbot/telegram_portfolio_bot/telegram_bot.py`
- `Chatbot/telegram_portfolio_bot/telegram_access.py`
- `Chatbot/telegram_portfolio_bot/telegram_notifications.py`
- `Chatbot/telegram_portfolio_bot/telegram_mini_app.py`
- `Chatbot/telegram_portfolio_bot/webapp/`

The Telegram bot does not own portfolio state. It calls `research/manual_portfolio` service functions and writes to the same JSON/JSONL files as the CLI.

Runtime:

```bash
cd /Users/mit/Project/ML4T/research
uv run python ../Chatbot/run_telegram_bot.py
```

Secrets:

- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_WEBHOOK_SECRET`

Secrets may be sourced from `Chatbot/secret.env`, but do not commit real secrets. `Chatbot/secret.env.example` is the safe template.

Access control:

- Configured in `Chatbot/configs/telegram_access_map.yaml`.
- Deny-by-default.
- When both chat ID and user ID are present, both must match the same portfolio entry.
- Delivery chat is configured per portfolio with `delivery_chat_id`.

Webhook / app routes:

- `GET /health`
- `POST /telegram/webhook`
- `GET /mini-app`
- `GET /mini-app/api/me`
- `GET /mini-app/api/portfolios/<portfolio_id>/overview`
- `GET /mini-app/api/portfolios/<portfolio_id>/holdings`
- `GET /mini-app/api/portfolios/<portfolio_id>/rebalance`
- `GET /mini-app/api/portfolios/<portfolio_id>/activity`
- `POST /mini-app/api/portfolios/<portfolio_id>/fills`

The webhook handler uses Telegram's secret-token header and a dedupe journal at:

`research/state/manual_portfolios/.telegram/processed_updates.jsonl`

### 11.5 Telegram commands and Mini App

Supported bot commands documented in `Chatbot/README.md`:

```text
/start
/app
/portfolios
/status p1
/fill p1 buy AAPL 10 100
/fill p1 sell AAPL 5 110 1.25 0.50 partial take profit
/help
```

Implementation notes:

- `/status` calls `portfolio_status(...)` and formats the target-vs-actual/rebalance view.
- `/fill` calls `record_fill(...)`.
- Telegram fill IDs are deterministic from portfolio ID, chat ID, message ID, and message date to prevent duplicate processing.
- `/start` and `/app` return a Telegram Web App button for the Mini App dashboard.
- Mini App API requests authenticate Telegram `initData`; portfolio data is not exposed through unauthenticated public endpoints.
- The Mini App can view overview, holdings, rebalance plan, activity, and submit fills through the same manual ledger path.

Outbound notifications:

```bash
cd /Users/mit/Project/ML4T/research
uv run python ../Chatbot/send_telegram_notification.py \
  --portfolio-id SP500-baw \
  --text "test message"
```

Notification hooks:

- `daily_run.py --notify-telegram`
- `record_fill.py --notify-telegram`

Notification failures are reported in command output and do not roll back local portfolio state or daily artifacts.

### 11.6 Tests

Relevant tests:

- `research/tests/test_manual_portfolio.py`
- `Chatbot/tests/test_telegram_bot.py`

Coverage observed from the test files includes:

- cash-only and imported-holdings onboarding
- buy, partial sell, full close, average cost, realized P&L
- rebalance day vs non-rebalance day behavior
- target/actual drift and rebalance instruction generation
- Telegram joint chat/user authorization
- unauthorized status/fill denial
- Telegram fill writes to the same fill journal with deterministic IDs
- webhook secret enforcement and update dedupe
- Mini App auth and portfolio endpoints
- notification success/failure handling

Recommended verification after changing this workflow:

```bash
cd /Users/mit/Project/ML4T/research
uv run pytest tests/test_manual_portfolio.py ../Chatbot/tests/test_telegram_bot.py -q
```

### 11.7 Next steps for portfolio workflow

1. Decide whether to promote the latest validated research signal (`log_volume + WeeklyLongOnly5`) into `research/configs/manual_active_strategy.yaml`, replacing the current `sp100_seed` / `obv` setup.
2. If promoting, create or point to production-ready signal and price artifacts for the chosen universe, then run `portfolio_status` and `daily_run` locally before enabling Telegram notification.
3. Review the real `SP500-baw` state and fill journal before making any operational recommendation; this handoff records file locations but does not validate current economic exposure.
4. For live/paper trading, keep this manual ledger separate from broker state until reconciliation semantics are explicitly designed.
