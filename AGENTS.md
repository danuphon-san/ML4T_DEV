# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Overview

ML4T is a monorepo of seven interconnected Python packages for quantitative finance ML, plus a Rust ITCH parser and research scripts. The canonical data pipeline is:

```
data → engineer → models → diagnostic → backtest → live
```

All Python packages use `uv` for dependency management and `hatchling` as the build backend. Packages live under the `ml4t.<package>` namespace.

## Package Summary

| Package | Purpose | Python |
|---------|---------|--------|
| `data/` | Fetch and store market data (Yahoo, Databento, OANDA, crypto, macro) | >=3.11 |
| `engineer/` | Features (120+ indicators), labeling (triple-barrier, etc.), non-time bars | >=3.12 |
| `models/` | Latent-factor models (PCA/RPCA/IPCA/CAE), SAE, LSTM portfolio | >=3.12 |
| `diagnostic/` | Signal validation, CPCV/walk-forward CV, feature diagnostics, reporting | >=3.12, <3.15 |
| `backtest/` | Event-driven backtesting engine; same `Strategy` class reused in `live` | >=3.12 |
| `live/` | Live trading (Alpaca, IB, CCXT); async engine with shadow mode | >=3.12 |
| `specs/` | Shared contract types (`FeedSpec`, `MarketDataSpec`, artifact types) | >=3.12 |
| `itch-parser/` | Rust parser for NASDAQ TotalView-ITCH 5.0 (400M+ msgs/min) | Rust |

## Commands

Each package is developed independently. Run all commands from the package subdirectory.

### Setup

```bash
# data
cd data && uv sync --dev

# engineer (pin to 3.12; ta and store extras needed for full test coverage)
cd engineer && uv sync --python 3.12 --extra ta --extra store

# models
cd models && uv sync

# diagnostic (also requires: brew install libomp on macOS)
cd diagnostic && uv sync

# backtest / live / specs
cd <package> && uv sync --dev
```

### Run tests

```bash
uv run pytest tests/ -q                       # all packages
uv run pytest tests/ -q -o addopts=''         # engineer (disable default addopts)
uv run pytest tests/path/test_file.py -q      # single file
uv run pytest tests/ -q -k "test_name"        # single test by name
```

### Lint / format

```bash
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

### Type check

```bash
uv run ty check src/        # primary (Astral ty)
uv run mypy src/             # secondary (data/engineer)
```

### itch-parser (Rust)

```bash
cd itch-parser
cargo build --release
cargo test
```

## macOS Requirements

- **`diagnostic`**: requires `brew install libomp` for LightGBM/XGBoost OpenMP runtime.
- **`engineer`**: pin to Python 3.12 (`uv sync --python 3.12`). Python 3.14 had intermittent JIT-related crashes on arm64 (non-reproducible, treat as stateful).

## Verified Test Baselines (macOS, 2026-06-03)

| Package | Result |
|---------|--------|
| data | 2835 passed, 14 skipped (39s) |
| engineer | 3201 passed, 1 skipped (26s) |
| models | 56 passed (15s) |
| diagnostic | 5226 passed, 21 skipped (36s) |

## Architecture Notes

### Cross-package contracts
`specs/` defines shared types (`FeedSpec`, `MarketDataSpec`, `ArtifactStorage`, `ArtifactKind`) that `backtest`, `diagnostic`, `models`, and `engineer` import. Changes to `specs` can cascade across all packages.

### Backtest ↔ Live parity
`backtest/strategy.py` defines the `Strategy` base class. `live/engine.py` imports and runs the same class, so a strategy written for backtesting works in production without modification.

### Storage layer
Primary format is Parquet (Apache Arrow) with Hive partitioning by date/ticker in `data`. `engineer` optionally uses DuckDB as a feature store backend (enabled via `--extra store`). Live trading uses JSONL for crash-safe execution journaling.

### Streaming vs eager
`data` and `engineer` use Polars `LazyFrame` for query optimization; materialize only when needed. `diagnostic` and `backtest` may use Pandas DataFrames at their boundaries—check each module's public API.

### Test markers
- `data`: `slow`, `paid_tier`, `integration`, `requires_api_key`, `optional_dependency`
- `engineer`: `perf` (excluded by default addopts), `validation`, `benchmark`, `property`
- `diagnostic`: `slow`, `integration`, `property` (Hypothesis)

Skip paid/slow markers in local runs: `-m "not slow and not paid_tier and not requires_api_key"`.

### Local patches already applied
- **`data` tests**: Path normalization fixes for macOS `/private/tmp` resolution in several test files.
- **`engineer`**: `src/ml4t/engineer/labeling/horizon_labels.py` converts `NaN` t-values to nulls in `trend_scanning_labels`.

## Research Pipeline

`research/` contains standalone scripts that exercise the full pipeline on an SP500 seed universe:

```bash
cd research
# main pipeline: fetch → engineer → label → diagnose
uv run python build_sp20_research_dataset.py
```

Outputs land in `research/outputs/` as Parquet panels and JSON reports. Configs are in `research/configs/`.
