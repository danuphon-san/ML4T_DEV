"""Backtest 3 composite signals built from cluster representatives.

Composites:
  vol_composite  — top volatility cluster signals (kyle_lambda, garch_forecast, coef_of_var)
  risk_composite — risk-adjusted / drawdown signals (sharpe, -max_drawdown, -time_underwater)
  combined       — equal-weight average of vol + risk composite

Also includes the single best signal per cluster as individual benchmarks.

Usage:
    uv run python backtest_composite_signals.py --prefix sp500_full
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
for repo in ("backtest", "diagnostic"):
    src = REPO_ROOT / repo / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

from ml4t.backtest import BacktestConfig, DataFeed, Engine
from ml4t.backtest.config import ShareType
from ml4t.backtest.strategies.templates import LongShortStrategy
from ml4t.diagnostic.evaluation.stats import deflated_sharpe_ratio

from research_universe import universe_spec


# ── Composite definitions ─────────────────────────────────────────────────────

VOL_SIGNALS = [
    "kyle_lambda",          # t=30.1  volatility/liquidity cluster rep
    "garch_forecast",       # t=28.9  forward-looking vol cluster rep
    "coefficient_of_variation",  # t=27.3  standalone
]

RISK_SIGNALS = [
    "risk_adjusted_returns_sharpe_ratio",   # t=26.0
    "-maximum_drawdown_time_underwater",    # t=22.3
    "-maximum_drawdown_max_drawdown",       # t=20.6
]

# Individual benchmarks (best single signal from other clusters)
SINGLE_BENCHMARKS = [
    "-bollinger_bands_lower",   # t=15.7  price-level cluster rep
    "plus_di",                  # t=14.2  momentum cluster rep
    "log_volume",               # t=21.3  standalone volume
]


# ── Strategy ──────────────────────────────────────────────────────────────────

class WeeklyLongOnly5(LongShortStrategy):
    signal_column = "signal"
    long_count = 5
    short_count = 0
    position_size = 0.15
    rebalance_frequency = 5


# ── Signal builders ───────────────────────────────────────────────────────────

def _resolve_expr(name: str) -> pl.Expr:
    """Return polars expression for raw or negated column."""
    if name.startswith("-"):
        return -pl.col(name[1:])
    return pl.col(name)


def build_composite(frame: pl.DataFrame, signal_names: list[str]) -> pl.Series:
    """Cross-sectional rank each signal per date, then average ranks."""
    rank_exprs = [
        _resolve_expr(s).rank(method="average").over("timestamp").alias(s)
        for s in signal_names
    ]
    ranks = frame.select(rank_exprs)
    return ranks.select(pl.mean_horizontal(pl.all())).to_series().alias("signal")


def build_signal_df(frame: pl.DataFrame, signal_col: pl.Series) -> pl.DataFrame:
    return frame.select(["timestamp", pl.col("symbol").alias("asset")]).with_columns(
        signal_col.alias("signal")
    )


def build_single_signal_df(frame: pl.DataFrame, name: str) -> pl.DataFrame:
    return frame.select(
        [
            "timestamp",
            pl.col("symbol").alias("asset"),
            _resolve_expr(name).alias("signal"),
        ]
    )


# ── Backtest runner ───────────────────────────────────────────────────────────

def run_backtest(
    name: str,
    prices: pl.DataFrame,
    signals: pl.DataFrame,
    out_dir: Path,
) -> tuple[object, list[float]]:
    strategy = WeeklyLongOnly5()
    config = BacktestConfig.from_preset("realistic")
    config.share_type = ShareType.FRACTIONAL
    result = Engine(DataFeed(prices_df=prices, signals_df=signals), strategy, config).run()
    safe = name.replace("-", "neg_")
    (out_dir / safe).mkdir(parents=True, exist_ok=True)
    result.to_parquet(out_dir / safe)
    daily_returns = result.to_daily_returns(calendar="NYSE")
    pl.DataFrame({"daily_returns": daily_returns}).write_parquet(out_dir / safe / "daily_returns.parquet")
    return result, daily_returns.to_numpy().tolist()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="sp500_full")
    args = parser.parse_args()

    spec = universe_spec(args.prefix)
    out_dir = spec.output_dir / "backtests_composite"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_frame = pl.read_parquet(spec.output_dir / f"{spec.prefix}_model_frame.parquet")

    # Train/test split (80/20 by date)
    dates = model_frame["timestamp"].unique().sort().to_list()
    n_train = int(len(dates) * 0.8)
    test_dates = set(dates[n_train:])
    test_frame = model_frame.filter(pl.col("timestamp").is_in(test_dates)).sort(["timestamp", "symbol"])

    prices = test_frame.select(
        ["timestamp", pl.col("symbol").alias("asset"), "open", "high", "low", "close", "volume"]
    )

    print(f"Test period: {min(test_dates)} → {max(test_dates)}  ({len(test_dates)} dates, {test_frame['symbol'].n_unique()} assets)\n")

    # Build all candidate signals
    candidates: dict[str, pl.DataFrame] = {}

    vol_composite = build_composite(test_frame, VOL_SIGNALS)
    candidates["vol_composite"] = build_signal_df(test_frame, vol_composite)

    risk_composite = build_composite(test_frame, RISK_SIGNALS)
    candidates["risk_composite"] = build_signal_df(test_frame, risk_composite)

    # Combined = average of the two composite rank series
    combined = pl.Series(
        "signal",
        [(v + r) / 2 for v, r in zip(vol_composite.to_list(), risk_composite.to_list())],
    )
    candidates["combined"] = build_signal_df(test_frame, combined)

    for name in SINGLE_BENCHMARKS:
        candidates[name] = build_single_signal_df(test_frame, name)

    # Run backtests
    rows: list[dict] = []
    returns_matrix: list[list[float]] = []
    names: list[str] = []

    for name, signals in candidates.items():
        print(f"Running: {name} ...")
        result, returns = run_backtest(name, prices, signals, out_dir)
        names.append(name)
        returns_matrix.append(returns)
        rows.append(
            {
                "name": name,
                "total_return_pct": float(result.metrics.get("total_return_pct", 0.0)),
                "sharpe": float(result.metrics.get("sharpe", 0.0)),
                "max_drawdown": float(result.metrics.get("max_drawdown", 0.0)),
                "final_value": float(result.metrics.get("final_value", 0.0)),
                "n_trades": len(result.trades),
            }
        )
        print(f"  → Sharpe={rows[-1]['sharpe']:.2f}  Return={rows[-1]['total_return_pct']:.1f}%  DD={rows[-1]['max_drawdown']:.3f}")

    # DSR
    dsr = deflated_sharpe_ratio(
        returns_matrix, frequency="daily", correlation_method="effective_rank", min_k_eff=2.0
    )
    rows = sorted(rows, key=lambda x: x["sharpe"], reverse=True)
    winner = rows[0]["name"]

    for row in rows:
        row["dsr_probability"] = float(dsr.probability) if row["name"] == winner else float("nan")

    summary = {
        "test_start": str(min(test_dates)),
        "test_end": str(max(test_dates)),
        "strategies": names,
        "results": rows,
        "dsr_probability": float(dsr.probability),
        "winner": winner,
    }
    (spec.output_dir / f"{spec.prefix}_backtest_composite_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    # Report
    report = [
        "# Composite Signal Backtest",
        "",
        f"Test period: `{summary['test_start']}` → `{summary['test_end']}`",
        "",
        "**Composite definitions:**",
        f"- `vol_composite`: {', '.join(VOL_SIGNALS)}",
        f"- `risk_composite`: {', '.join(RISK_SIGNALS)}",
        "- `combined`: equal-weight average of vol + risk composite ranks",
        "",
        "| Strategy | Return % | Sharpe | Max DD | Trades | DSR |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        dsr_str = f"{row['dsr_probability']:.3f}" if not __import__("math").isnan(row["dsr_probability"]) else "—"
        report.append(
            f"| {row['name']} | {row['total_return_pct']:.2f} | {row['sharpe']:.2f}"
            f" | {row['max_drawdown']:.3f} | {row['n_trades']} | {dsr_str} |"
        )
    report += ["", f"**Winner:** `{winner}` (DSR probability: {float(dsr.probability):.3f})"]
    (spec.output_dir / f"{spec.prefix}_backtest_composite_report.md").write_text(
        "\n".join(report) + "\n"
    )
    print("\n" + "\n".join(report))


if __name__ == "__main__":
    main()
