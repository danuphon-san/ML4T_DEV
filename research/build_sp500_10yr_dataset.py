"""Build a 10-year point-in-time SP500 research dataset.

Uses:
  - research/outputs/sp500_pit/sp500_pit_composition.parquet  (which tickers on which dates)
  - ~/ml4t-data/equities_daily_*/  (price history back to 2015)

Produces (in research/outputs/sp500_10yr/):
  sp500_10yr_model_frame.parquet  — full panel, 2016-2026
  sp500_10yr_summary.json         — coverage stats

Key difference from sp500_full:
  On each date, only include tickers that were ACTUALLY in the S&P500 at that time,
  eliminating look-ahead / survivorship bias in the cross-sectional signal calculation.

Usage:
    uv run python build_sp500_10yr_dataset.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
for repo in ("engineer", "models"):
    src = REPO_ROOT / repo / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

from ml4t.engineer import compute_features
from ml4t.engineer.config import LabelingConfig
from ml4t.engineer.labeling import triple_barrier_labels

from research_universe import FACTOR_FILE, MACRO_FILE, STORAGE_ROOT

SP500_ALL_FEATURES = REPO_ROOT / "research" / "configs" / "sp500_all_features.yaml"
from build_research_dataset import (
    CONTEXT_COLS,
    load_feature_names,
    resolve_characteristic_cols,
    compute_symbol_features,
    enrich_returns,
    apply_labels,
    build_model_frame,
)

PIT_FILE = REPO_ROOT / "research" / "outputs" / "sp500_pit" / "sp500_pit_composition.parquet"
OUTPUT_DIR = REPO_ROOT / "research" / "outputs" / "sp500_10yr"
PREFIX = "sp500_10yr"


def load_symbol_history(symbol: str) -> pl.DataFrame | None:
    patterns = [
        f"equities_daily_{symbol}",
        f"{symbol}",
        f"yahoo_daily_{symbol}",
    ]
    paths = []
    for pat in patterns:
        paths.extend(sorted(STORAGE_ROOT.glob(f"{pat}/year=*/month=*/data.parquet")))
    if not paths:
        return None
    frame = pl.concat([pl.read_parquet(p) for p in paths], how="vertical")
    return (
        frame.unique(subset=["timestamp", "symbol"], maintain_order=True)
        .sort("timestamp")
        .with_columns(pl.col("timestamp").cast(pl.Datetime("us")))
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-config", default=str(SP500_ALL_FEATURES))
    parser.add_argument("--start", default="2015-01-01", help="Raw data fetch start (for warmup)")
    parser.add_argument("--model-start", default="2016-01-01", help="Model frame start date")
    parser.add_argument("--model-end", default="2026-06-01", help="Model frame end date")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    feature_config = Path(args.feature_config)
    feature_names = load_feature_names(feature_config)
    characteristic_cols = resolve_characteristic_cols(feature_names)

    # Load point-in-time composition
    if not PIT_FILE.exists():
        print("ERROR: Run build_sp500_pit_composition.py first.")
        sys.exit(1)
    pit = pl.read_parquet(PIT_FILE).with_columns(
        pl.col("date").cast(pl.Date)
    )
    all_tickers = sorted(pit["ticker"].unique().to_list())
    print(f"Point-in-time composition: {len(all_tickers)} unique tickers over history")

    # Load price history for each ticker (skip unavailable)
    frames: list[pl.DataFrame] = []
    missing: list[str] = []
    for ticker in all_tickers:
        df = load_symbol_history(ticker)
        if df is None:
            missing.append(ticker)
        else:
            frames.append(df)

    print(f"Available tickers: {len(frames)} / {len(all_tickers)}")
    print(f"Missing (no data in storage): {len(missing)}")
    if missing:
        print(f"  First 20 missing: {missing[:20]}")

    raw_panel = pl.concat(frames, how="vertical").sort(["symbol", "timestamp"])
    print(f"Raw panel: {raw_panel.height:,} rows, {raw_panel['symbol'].n_unique()} symbols")
    print(f"  Date range: {raw_panel['timestamp'].min()} → {raw_panel['timestamp'].max()}")

    # Compute features (per-symbol)
    print("\nComputing features...")
    featured_panel = compute_symbol_features(raw_panel, feature_config)
    print(f"Feature panel: {featured_panel.height:,} rows")

    # Enrich with derived columns
    print("Enriching returns...")
    enriched = enrich_returns(featured_panel)

    # Merge factor/macro context
    print("Merging context...")
    factor_frame = pl.read_parquet(FACTOR_FILE).with_columns(pl.col("timestamp").cast(pl.Datetime("us"))).sort("timestamp")
    macro_frame = pl.read_parquet(MACRO_FILE).with_columns(pl.col("timestamp").cast(pl.Datetime("us"))).sort("timestamp")
    merged = enriched.join(factor_frame, on="timestamp", how="left").join(macro_frame, on="timestamp", how="left").sort(["symbol", "timestamp"])

    # Apply triple-barrier labels
    print("Applying labels...")
    labeled = apply_labels(merged)

    # Build model frame
    print("Building model frame...")
    model_frame_full = build_model_frame(labeled, characteristic_cols)

    # Apply date filter for the model window
    model_start = pl.lit(args.model_start).str.to_datetime("%Y-%m-%d").cast(pl.Datetime("us"))
    model_end = pl.lit(args.model_end).str.to_datetime("%Y-%m-%d").cast(pl.Datetime("us"))
    model_frame_full = model_frame_full.filter(
        (pl.col("timestamp") >= model_start) & (pl.col("timestamp") <= model_end)
    )

    # Apply point-in-time composition filter
    # For each (date, symbol), keep only rows where the symbol was in the S&P500
    print("Applying point-in-time composition filter...")
    pit_pl = pit.with_columns(
        pl.col("date").cast(pl.Date)
    )
    model_dates = model_frame_full.with_columns(
        pl.col("timestamp").cast(pl.Date).alias("date")
    )
    # Anti-join approach: mark which rows are in the PIT composition
    model_with_pit = model_dates.join(
        pit_pl.rename({"ticker": "symbol"}),
        on=["date", "symbol"],
        how="inner",
    ).drop("date")

    print(f"\nModel frame stats:")
    print(f"  Before PIT filter: {model_frame_full.height:,} rows")
    print(f"  After PIT filter:  {model_with_pit.height:,} rows")
    print(f"  Date range: {model_with_pit['timestamp'].min()} → {model_with_pit['timestamp'].max()}")
    print(f"  Symbols: {model_with_pit['symbol'].n_unique()}")
    print(f"  Dates: {model_with_pit['timestamp'].n_unique()}")

    # Save
    model_with_pit.write_parquet(OUTPUT_DIR / f"{PREFIX}_model_frame.parquet")
    print(f"\nSaved → {OUTPUT_DIR}/{PREFIX}_model_frame.parquet")

    summary = {
        "prefix": PREFIX,
        "model_start": args.model_start,
        "model_end": args.model_end,
        "all_pit_tickers": len(all_tickers),
        "available_tickers": len(frames),
        "missing_tickers": missing,
        "raw_rows": raw_panel.height,
        "model_rows": model_with_pit.height,
        "model_date_min": str(model_with_pit["timestamp"].min()),
        "model_date_max": str(model_with_pit["timestamp"].max()),
        "model_symbols": model_with_pit["symbol"].n_unique(),
        "model_dates": model_with_pit["timestamp"].n_unique(),
        "feature_names": feature_names,
        "characteristic_cols": characteristic_cols,
    }
    (OUTPUT_DIR / f"{PREFIX}_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved → {OUTPUT_DIR}/{PREFIX}_summary.json")
    print("\nDone!")


if __name__ == "__main__":
    main()
