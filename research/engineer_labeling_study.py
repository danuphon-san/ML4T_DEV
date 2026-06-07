from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from ml4t.engineer.config import LabelingConfig
from ml4t.engineer.labeling import (
    atr_triple_barrier_labels,
    fixed_time_horizon_labels,
    rolling_percentile_binary_labels,
    trend_scanning_labels,
    triple_barrier_labels,
)

from research_universe import universe_spec


REPO_ROOT = Path(__file__).resolve().parents[1]


def read_symbols(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def label_distribution(df: pl.DataFrame, col: str) -> dict[str, int]:
    if col not in df.columns:
        return {}
    counts = (
        df.filter(pl.col(col).is_not_null())
        .group_by(col)
        .len("count")
        .sort(col)
    )
    return {str(row[col]): int(row["count"]) for row in counts.to_dicts()}


def barrier_distribution(df: pl.DataFrame) -> dict[str, int]:
    if "barrier_hit" not in df.columns:
        return {}
    counts = (
        df.filter(pl.col("barrier_hit").is_not_null())
        .group_by("barrier_hit")
        .len("count")
        .sort("barrier_hit")
    )
    return {str(row["barrier_hit"]): int(row["count"]) for row in counts.to_dicts()}


def summarize_method(name: str, df: pl.DataFrame, label_col: str) -> dict[str, object]:
    labeled = df.filter(pl.col(label_col).is_not_null()) if label_col in df.columns else pl.DataFrame()
    summary = {
        "method": name,
        "rows_with_labels": labeled.height,
        "label_column": label_col,
        "distribution": label_distribution(df, label_col),
    }
    if "barrier_hit" in df.columns:
        summary["barrier_distribution"] = barrier_distribution(df)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="sp500_full")
    parser.add_argument("--sample-symbols", type=int, default=20)
    args = parser.parse_args()

    spec = universe_spec(args.prefix)
    symbols = read_symbols(spec.symbol_file)[: args.sample_symbols]
    panel = pl.read_parquet(spec.output_dir / f"{spec.prefix}_panel.parquet")
    panel = panel.filter(pl.col("symbol").is_in(symbols)).sort(["symbol", "timestamp"])

    triple = triple_barrier_labels(
        panel,
        config=LabelingConfig.triple_barrier(
            upper_barrier=0.03,
            lower_barrier=0.02,
            max_holding_period=10,
        ),
        price_col="close",
        high_col="high",
        low_col="low",
        timestamp_col="timestamp",
        group_col="symbol",
    )

    atr = atr_triple_barrier_labels(
        panel,
        config=LabelingConfig.atr_barrier(
            atr_tp_multiple=2.0,
            atr_sl_multiple=1.0,
            atr_period=14,
            max_holding_period=10,
        ),
        price_col="close",
        timestamp_col="timestamp",
        group_col="symbol",
    )

    fixed = fixed_time_horizon_labels(
        panel,
        horizon=10,
        method="binary",
        price_col="close",
        group_col="symbol",
        timestamp_col="timestamp",
    )

    percentile = rolling_percentile_binary_labels(
        panel,
        horizon=10,
        percentile=95,
        direction="long",
        lookback_window=126,
        price_col="close",
        group_col="symbol",
        timestamp_col="timestamp",
    )

    trend = trend_scanning_labels(
        panel,
        min_window=5,
        max_window=20,
        price_col="close",
        group_col="symbol",
        timestamp_col="timestamp",
    )

    summaries = [
        summarize_method("triple_barrier", triple, "label"),
        summarize_method("atr_triple_barrier", atr, "label"),
        summarize_method("fixed_time_horizon_binary", fixed, "label_direction_10p"),
        summarize_method("rolling_percentile_long_p95_h10", percentile, "label_long_p95_h10"),
        summarize_method("trend_scanning", trend, "label"),
    ]

    overview = {
        "prefix": spec.prefix,
        "sample_symbols": symbols,
        "sample_rows": panel.height,
        "methods": summaries,
    }

    output_dir = spec.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{spec.prefix}_engineer_labeling_study.json"
    report_path = output_dir / f"{spec.prefix}_engineer_labeling_study.md"
    summary_path.write_text(json.dumps(overview, indent=2))

    lines = [
        f"# {spec.prefix} Engineer Labeling Study",
        "",
        f"- Sample symbols: `{len(symbols)}`",
        f"- Sample rows: `{panel.height}`",
        "",
        "## Methods",
        "",
    ]
    for method in summaries:
        lines.append(
            f"- `{method['method']}`: `{method['rows_with_labels']}` rows, distribution `{method['distribution']}`"
        )
        if "barrier_distribution" in method:
            lines.append(f"  barrier hits: `{method['barrier_distribution']}`")
    report_path.write_text("\n".join(lines) + "\n")

    print(json.dumps({"summary_path": str(summary_path), "report_path": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
