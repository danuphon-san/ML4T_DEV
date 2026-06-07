from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
for repo in ("engineer", "models"):
    src_path = REPO_ROOT / repo / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

from ml4t.engineer import compute_features, create_dataset_builder
from ml4t.engineer.config import LabelingConfig
from ml4t.engineer.labeling import triple_barrier_labels
from ml4t.models import cross_section_batch_from_long_frame


STORAGE_ROOT = Path.home() / "ml4t-data"
SYMBOL_FILE = REPO_ROOT / "data" / "examples" / "symbols" / "sp20_seed_2026-06-03.txt"
FEATURE_CONFIG = REPO_ROOT / "research" / "configs" / "sp20_core_features.yaml"
FACTOR_FILE = REPO_ROOT / "data" / "data" / "factors" / "fama-french" / "ff5_daily.parquet"
MACRO_FILE = STORAGE_ROOT / "treasury_yields.parquet"
OUTPUT_DIR = REPO_ROOT / "research" / "outputs"
DEFAULT_DIAGNOSTIC_FACTOR = "-atr"

PRICE_COLS = ["open", "high", "low", "close", "volume"]
CHARACTERISTIC_COLS = [
    "rsi",
    "macd",
    "atr",
    "obv",
    "sma_gap_20",
    "ema_gap_20",
    "log_volume",
    "ret_5d",
]
CONTEXT_COLS = [
    "Mkt-RF",
    "SMB",
    "HML",
    "RMW",
    "CMA",
    "RF",
    "DGS2",
    "DGS5",
    "DGS10",
    "DGS30",
    "YIELD_CURVE_SLOPE",
    "YIELD_CURVE_5_10",
]


def read_symbols(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def load_symbol_history(symbol: str, storage_root: Path) -> pl.DataFrame:
    parquet_paths = sorted(storage_root.glob(f"yahoo_daily_{symbol}/year=*/month=*/data.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found for {symbol} under {storage_root}")
    frame = pl.concat([pl.read_parquet(path) for path in parquet_paths], how="vertical")
    return (
        frame.unique(subset=["timestamp", "symbol"], maintain_order=True)
        .sort("timestamp")
        .with_columns(pl.col("timestamp").cast(pl.Datetime("us")))
    )


def load_equity_panel(symbols: list[str], storage_root: Path) -> pl.DataFrame:
    frames = [load_symbol_history(symbol, storage_root) for symbol in symbols]
    return pl.concat(frames, how="vertical").sort(["symbol", "timestamp"])


def load_factor_frame(path: Path) -> pl.DataFrame:
    return (
        pl.read_parquet(path)
        .with_columns(pl.col("timestamp").cast(pl.Datetime("us")))
        .sort("timestamp")
    )


def load_macro_frame(path: Path) -> pl.DataFrame:
    return (
        pl.read_parquet(path)
        .with_columns(pl.col("timestamp").cast(pl.Datetime("us")))
        .sort("timestamp")
    )


def compute_symbol_features(panel: pl.DataFrame, feature_config: Path) -> pl.DataFrame:
    featured_frames: list[pl.DataFrame] = []
    for symbol_frame in panel.partition_by("symbol", maintain_order=True):
        symbol_frame = symbol_frame.sort("timestamp")
        featured_frames.append(compute_features(symbol_frame, feature_config))
    return pl.concat(featured_frames, how="vertical").sort(["symbol", "timestamp"])


def enrich_returns(panel: pl.DataFrame) -> pl.DataFrame:
    return panel.with_columns(
        [
            (pl.col("close").pct_change().over("symbol")).alias("ret_1d"),
            (pl.col("close").pct_change(5).over("symbol")).alias("ret_5d"),
            ((pl.col("close").shift(-1).over("symbol") / pl.col("close")) - 1.0).alias("ret_1d_fwd"),
            ((pl.col("close") / pl.col("sma")) - 1.0).alias("sma_gap_20"),
            ((pl.col("close") / pl.col("ema")) - 1.0).alias("ema_gap_20"),
            pl.col("volume").log().alias("log_volume"),
        ]
    )


def merge_context(panel: pl.DataFrame, factor_frame: pl.DataFrame, macro_frame: pl.DataFrame) -> pl.DataFrame:
    return (
        panel.join(factor_frame, on="timestamp", how="left")
        .join(macro_frame, on="timestamp", how="left")
        .sort(["symbol", "timestamp"])
    )


def apply_labels(panel: pl.DataFrame) -> pl.DataFrame:
    config = LabelingConfig.triple_barrier(
        upper_barrier=0.03,
        lower_barrier=0.02,
        max_holding_period=10,
    )
    return triple_barrier_labels(
        panel,
        config=config,
        price_col="close",
        high_col="high",
        low_col="low",
        timestamp_col="timestamp",
        group_col="symbol",
    ).sort(["symbol", "timestamp"])


def build_model_frame(labeled_panel: pl.DataFrame) -> pl.DataFrame:
    required_cols = CHARACTERISTIC_COLS + CONTEXT_COLS + ["ret_1d_fwd", "label"]
    selected = (
        labeled_panel.drop_nulls(subset=required_cols)
        .sort(["timestamp", "symbol"])
        .select(
            [
                "timestamp",
                "symbol",
                "close",
                "ret_1d",
                "ret_1d_fwd",
                "label",
                "label_return",
                "label_bars",
                "label_duration",
                "barrier_hit",
                *CHARACTERISTIC_COLS,
                *CONTEXT_COLS,
            ]
        )
    )
    finite_cols = [
        "close",
        "ret_1d",
        "ret_1d_fwd",
        "label_return",
        *CHARACTERISTIC_COLS,
        *CONTEXT_COLS,
    ]
    return selected.filter(pl.all_horizontal([pl.col(col).is_finite() for col in finite_cols]))


def build_signal_inputs(model_frame: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    factor = model_frame.select(
        [
            pl.col("timestamp").alias("date"),
            pl.col("symbol").alias("asset"),
            (-pl.col("atr")).alias("factor"),
        ]
    )
    prices = model_frame.select(
        [
            pl.col("timestamp").alias("date"),
            pl.col("symbol").alias("asset"),
            pl.col("close").alias("price"),
        ]
    )
    return factor, prices


def run_diagnostic_report(
    factor_path: Path,
    prices_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    result_json_path = output_dir / "sp20_seed_signal_analysis.json"
    report_path = output_dir / "sp20_seed_diagnostic_report.md"
    code = f"""
import json
from pathlib import Path
import polars as pl
from ml4t.diagnostic import analyze_signal

factor = pl.read_parquet({str(factor_path)!r})
prices = pl.read_parquet({str(prices_path)!r})
result = analyze_signal(factor=factor, prices=prices, periods=(1, 5, 21), quantiles=5, min_assets=10)
result.to_json({str(result_json_path)!r})

period_keys = ("1D", "5D", "21D")
metrics = {{
    period: {{
        "ic": result.ic[period],
        "ic_t_stat": result.ic_t_stat[period],
        "ic_p_value": result.ic_p_value[period],
        "spread": result.spread[period],
        "spread_t_stat": result.spread_t_stat[period],
        "spread_p_value": result.spread_p_value[period],
        "monotonicity": result.monotonicity[period],
        "turnover": result.turnover[period],
    }}
    for period in period_keys
}}

report_lines = [
    "# SP20 Seed Diagnostic Report",
    "",
    f"Default factor: `{DEFAULT_DIAGNOSTIC_FACTOR}`",
    f"Assets: `{{result.n_assets}}`",
    f"Dates: `{{result.n_dates}}`",
    f"Date range: `{{result.date_range[0]}}` to `{{result.date_range[1]}}`",
    "",
    "## Summary",
    "",
    "```text",
    result.summary().strip(),
    "```",
    "",
    "## Metrics",
    "",
    "| Period | IC | IC t-stat | IC p-value | Spread | Spread t-stat | Spread p-value | Monotonicity | Turnover |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]
for period in period_keys:
    metric = metrics[period]
    report_lines.append(
        f"| {{period}} | {{metric['ic']:.4f}} | {{metric['ic_t_stat']:.2f}} | "
        f"{{metric['ic_p_value']:.3g}} | {{metric['spread']:.4f}} | "
        f"{{metric['spread_t_stat']:.2f}} | {{metric['spread_p_value']:.3g}} | "
        f"{{metric['monotonicity']:.3f}} | {{metric['turnover']:.3f}} |"
    )
Path({str(report_path)!r}).write_text("\\n".join(report_lines) + "\\n")
print(json.dumps({{
    "factor_name": "{DEFAULT_DIAGNOSTIC_FACTOR}",
    "json_path": {str(result_json_path)!r},
    "report_path": {str(report_path)!r},
    "metrics": metrics,
    "summary": result.summary(),
}}, indent=2))
"""
    completed = subprocess.run(
        ["uv", "run", "python", "-c", code],
        cwd=REPO_ROOT / "diagnostic",
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def save_cross_section_batch(model_frame: pl.DataFrame, output_dir: Path) -> dict[str, object]:
    batch = cross_section_batch_from_long_frame(
        model_frame,
        feature_cols=CHARACTERISTIC_COLS,
        return_col="ret_1d_fwd",
        context_cols=CONTEXT_COLS,
        timestamp_col="timestamp",
        entity_col="symbol",
        metadata={"universe": "sp20_seed"},
    )
    np.savez(
        output_dir / "sp20_seed_batch.npz",
        characteristics=batch.characteristics,
        returns=batch.returns,
        context_features=batch.context_features,
        mask=batch.mask,
    )
    metadata = {
        "timestamps": [str(timestamp) for timestamp in batch.timestamps],
        "asset_ids": list(batch.asset_ids),
        "feature_cols": CHARACTERISTIC_COLS,
        "context_cols": CONTEXT_COLS,
        "n_periods": batch.n_periods,
        "n_slots": batch.n_assets,
    }
    (output_dir / "sp20_seed_batch_metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def save_summary(output_dir: Path, **summary: object) -> None:
    (output_dir / "sp20_seed_summary.json").write_text(json.dumps(summary, indent=2))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    symbols = read_symbols(SYMBOL_FILE)
    raw_panel = load_equity_panel(symbols, STORAGE_ROOT)
    raw_panel.write_parquet(OUTPUT_DIR / "sp20_seed_panel.parquet")

    featured_panel = compute_symbol_features(raw_panel, FEATURE_CONFIG)
    featured_panel.write_parquet(OUTPUT_DIR / "sp20_seed_feature_panel.parquet")

    factor_frame = load_factor_frame(FACTOR_FILE)
    macro_frame = load_macro_frame(MACRO_FILE)
    merged_panel = merge_context(enrich_returns(featured_panel), factor_frame, macro_frame)

    labeled_panel = apply_labels(merged_panel)
    labeled_panel.write_parquet(OUTPUT_DIR / "sp20_seed_labeled_panel.parquet")

    model_frame = build_model_frame(labeled_panel)
    model_frame.write_parquet(OUTPUT_DIR / "sp20_seed_model_frame.parquet")

    factor_input, prices_input = build_signal_inputs(model_frame)
    factor_path = OUTPUT_DIR / "sp20_seed_signal_factor.parquet"
    prices_path = OUTPUT_DIR / "sp20_seed_signal_prices.parquet"
    factor_input.write_parquet(factor_path)
    prices_input.write_parquet(prices_path)
    diagnostic_summary = run_diagnostic_report(factor_path, prices_path, OUTPUT_DIR)

    builder = create_dataset_builder(
        features=model_frame.select(CHARACTERISTIC_COLS + CONTEXT_COLS),
        labels=model_frame["label"],
        dates=model_frame["timestamp"],
        scaler="robust",
    )
    X_train, X_test, y_train, y_test = builder.train_test_split(train_size=0.8, shuffle=False)

    batch_metadata = save_cross_section_batch(model_frame, OUTPUT_DIR)

    summary = {
        "symbols": symbols,
        "raw_rows": raw_panel.height,
        "feature_rows": featured_panel.height,
        "labeled_rows": labeled_panel.height,
        "model_rows": model_frame.height,
        "date_min": str(model_frame["timestamp"].min()),
        "date_max": str(model_frame["timestamp"].max()),
        "characteristic_cols": CHARACTERISTIC_COLS,
        "context_cols": CONTEXT_COLS,
        "label_counts": model_frame.group_by("label").len().sort("label").to_dicts(),
        "train_rows": X_train.height,
        "test_rows": X_test.height,
        "train_label_counts": y_train.value_counts().sort("label").to_dicts(),
        "test_label_counts": y_test.value_counts().sort("label").to_dicts(),
        "batch": batch_metadata,
        "default_diagnostic_factor": DEFAULT_DIAGNOSTIC_FACTOR,
        "diagnostic": diagnostic_summary,
        "outputs": {
            "panel": str(OUTPUT_DIR / "sp20_seed_panel.parquet"),
            "feature_panel": str(OUTPUT_DIR / "sp20_seed_feature_panel.parquet"),
            "labeled_panel": str(OUTPUT_DIR / "sp20_seed_labeled_panel.parquet"),
            "model_frame": str(OUTPUT_DIR / "sp20_seed_model_frame.parquet"),
            "signal_factor": str(OUTPUT_DIR / "sp20_seed_signal_factor.parquet"),
            "signal_prices": str(OUTPUT_DIR / "sp20_seed_signal_prices.parquet"),
            "signal_analysis_json": str(OUTPUT_DIR / "sp20_seed_signal_analysis.json"),
            "diagnostic_report": str(OUTPUT_DIR / "sp20_seed_diagnostic_report.md"),
            "batch_npz": str(OUTPUT_DIR / "sp20_seed_batch.npz"),
            "batch_metadata": str(OUTPUT_DIR / "sp20_seed_batch_metadata.json"),
        },
    }
    save_summary(OUTPUT_DIR, **summary)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
