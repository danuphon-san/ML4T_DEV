"""Screen every numeric feature column in the model_frame as a cross-sectional signal.

For each feature, tests both raw and inverse direction (e.g. rsi and -rsi),
then selects signals that pass IC, spread, and monotonicity thresholds.

Usage:
    uv run python screen_sp500_all_signals.py --prefix sp500_full
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass

import polars as pl

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
diagnostic_src = REPO_ROOT / "diagnostic" / "src"
if str(diagnostic_src) not in sys.path:
    sys.path.insert(0, str(diagnostic_src))
from ml4t.diagnostic import analyze_signal

from research_universe import universe_spec


# Columns that are not features and should be excluded from screening
_NON_FEATURE_COLS = frozenset(
    [
        "timestamp", "symbol", "date", "asset",
        "open", "high", "low", "close", "volume",
        "ret_1d", "ret_1d_fwd", "ret_5d",
        "label", "label_return", "label_bars", "label_duration", "barrier_hit",
        # Fama-French + macro context
        "Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF",
        "DGS2", "DGS5", "DGS10", "DGS30",
        "YIELD_CURVE_SLOPE", "YIELD_CURVE_5_10",
    ]
)

IC_THRESHOLD = 0.0
SPREAD_T_THRESHOLD = 2.0
MONOTONICITY_THRESHOLD = 0.5


@dataclass(frozen=True)
class SignalSpec:
    name: str
    expression: pl.Expr


def discover_feature_cols(model_frame: pl.DataFrame) -> list[str]:
    """Return all numeric columns that are candidate features."""
    return [
        col
        for col in model_frame.columns
        if col not in _NON_FEATURE_COLS
        and model_frame[col].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)
    ]


def build_signal_specs(feature_cols: list[str]) -> list[SignalSpec]:
    specs: list[SignalSpec] = []
    for col in feature_cols:
        specs.append(SignalSpec(col, pl.col(col)))
        specs.append(SignalSpec(f"-{col}", -pl.col(col)))
    return specs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="sp500_full")
    parser.add_argument(
        "--sample-dates", type=int, default=0,
        help="Randomly sample N dates for screening (0 = use all). "
             "Recommended: 800 for 10yr datasets to keep runtime manageable."
    )
    parser.add_argument(
        "--periods", default="1,5,21",
        help="Comma-separated list of forward-return horizons in days (default: 1,5,21)"
    )
    args = parser.parse_args()

    periods = tuple(int(p) for p in args.periods.split(","))

    spec = universe_spec(args.prefix)
    model_frame = pl.read_parquet(spec.output_dir / f"{spec.prefix}_model_frame.parquet")

    # Optional date sampling — reduces rows while preserving cross-sectional structure
    if args.sample_dates > 0:
        all_dates = model_frame["timestamp"].unique().sort().to_list()
        if len(all_dates) > args.sample_dates:
            import random
            random.seed(42)
            sampled = sorted(random.sample(all_dates, args.sample_dates))
            model_frame = model_frame.filter(pl.col("timestamp").is_in(sampled))
            print(f"Date sampling: {len(all_dates)} → {len(sampled)} dates  ({model_frame.height:,} rows)")

    prices = model_frame.select(
        [pl.col("timestamp").alias("date"), pl.col("symbol").alias("asset"), pl.col("close").alias("price")]
    )

    feature_cols = discover_feature_cols(model_frame)
    print(f"Features discovered: {len(feature_cols)}")
    print(f"Signal specs to evaluate: {len(feature_cols) * 2}")
    print(f"Periods: {periods}\n")

    signal_specs = build_signal_specs(feature_cols)
    rows: list[dict] = []
    for i, sig in enumerate(signal_specs, 1):
        factor = model_frame.select(
            [pl.col("timestamp").alias("date"), pl.col("symbol").alias("asset"), sig.expression.alias("factor")]
        ).drop_nulls()
        if factor.height == 0:
            continue
        try:
            result = analyze_signal(factor=factor, prices=prices, periods=periods, quantiles=5, min_assets=10)
            def _get(d: dict, key: str, default: float = float("nan")) -> float:
                return float(d.get(key, default) or default)

            # Use the longest available period as primary metric
            period_keys = [f"{p}D" for p in sorted(periods)]
            primary = period_keys[-1]  # e.g. "21D"
            mid = period_keys[len(period_keys) // 2] if len(period_keys) > 1 else primary
            short = period_keys[0]

            rows.append(
                {
                    "signal": sig.name,
                    "ic_1d":  _get(result.ic, short),
                    "ic_5d":  _get(result.ic, mid),
                    "ic_21d": _get(result.ic, primary),
                    "ic_t_21d": _get(result.ic_t_stat, primary),
                    "spread_1d":  _get(result.spread, short),
                    "spread_5d":  _get(result.spread, mid),
                    "spread_21d": _get(result.spread, primary),
                    "spread_t_5d":  _get(result.spread_t_stat, mid),
                    "spread_t_21d": _get(result.spread_t_stat, primary),
                    "monotonicity_21d": _get(result.monotonicity, primary),
                    "turnover_21d": _get(result.turnover, primary),
                }
            )
            print(f"  [{i}/{len(signal_specs)}] {sig.name}: IC={_get(result.ic, primary):.4f} spread_t={_get(result.spread_t_stat, primary):.2f}", flush=True)
        except Exception as e:
            print(f"  [{i}/{len(signal_specs)}] SKIP {sig.name}: {e}", flush=True)

    results = pl.DataFrame(rows).sort(
        by=["spread_t_21d", "ic_21d", "monotonicity_21d"], descending=[True, True, True]
    )
    selected = results.filter(
        (pl.col("spread_t_21d") > SPREAD_T_THRESHOLD)
        & (pl.col("spread_21d") > IC_THRESHOLD)
        & (pl.col("ic_21d") > IC_THRESHOLD)
        & (pl.col("monotonicity_21d") >= MONOTONICITY_THRESHOLD)
    )

    results.write_parquet(spec.output_dir / f"{spec.prefix}_all_signal_screen.parquet")
    top = selected.to_dicts()
    (spec.output_dir / f"{spec.prefix}_all_top_signals.json").write_text(json.dumps(top, indent=2))

    # Best signal factor for downstream use
    if top:
        best_name = top[0]["signal"]
        best_expr = next(s.expression for s in signal_specs if s.name == best_name)
        model_frame.select(
            [pl.col("timestamp").alias("date"), pl.col("symbol").alias("asset"), best_expr.alias("factor")]
        ).write_parquet(spec.output_dir / f"{spec.prefix}_all_signal_factor_top1.parquet")

    # Report
    report_lines = [
        f"# {spec.prefix} Full Signal Screen",
        "",
        f"Features discovered: `{len(feature_cols)}`",
        f"Signals evaluated: `{results.height}` (raw + inverse)",
        f"Signals selected: `{selected.height}`",
        "",
        "## Selected Signals",
        "",
        "| Signal | IC 21D | Spread 21D | Spread t-stat 21D | Monotonicity 21D | Turnover 21D |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in top:
        report_lines.append(
            f"| {row['signal']} | {row['ic_21d']:.4f} | {row['spread_21d']:.4f}"
            f" | {row['spread_t_21d']:.2f} | {row['monotonicity_21d']:.3f} | {row['turnover_21d']:.3f} |"
        )
    (spec.output_dir / f"{spec.prefix}_all_signal_screen_report.md").write_text("\n".join(report_lines) + "\n")

    print(f"\nDone. Tested {results.height} signals, selected {selected.height}.")
    print(json.dumps(top[:10], indent=2))


if __name__ == "__main__":
    main()
