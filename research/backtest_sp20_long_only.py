from __future__ import annotations

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

from compare_sp20_ipca_vs_signal_support import build_ipca_prediction_factor, build_top_signal_factor


OUTPUT_DIR = REPO_ROOT / "research" / "outputs"
BACKTEST_DIR = OUTPUT_DIR / "backtests_long_only"
MODEL_FRAME_PATH = OUTPUT_DIR / "sp20_seed_model_frame.parquet"
TOP_SIGNALS_PATH = OUTPUT_DIR / "sp20_seed_top_signals.json"
SUMMARY_PATH = OUTPUT_DIR / "sp20_seed_backtest_long_only_summary.json"
REPORT_PATH = OUTPUT_DIR / "sp20_seed_backtest_long_only_report.md"


class WeeklyLongOnly3(LongShortStrategy):
    signal_column = "signal"
    long_count = 3
    short_count = 0
    position_size = 0.20
    rebalance_frequency = 5


def build_test_frame(model_frame: pl.DataFrame) -> pl.DataFrame:
    dates = model_frame.select("timestamp").unique().sort("timestamp")["timestamp"].to_list()
    n_train = int(len(dates) * 0.8)
    test_dates = set(dates[n_train:])
    return model_frame.filter(pl.col("timestamp").is_in(test_dates)).sort(["timestamp", "symbol"])


def build_prices(test_frame: pl.DataFrame) -> pl.DataFrame:
    return test_frame.select(
        [
            "timestamp",
            pl.col("symbol").alias("asset"),
            pl.col("close").alias("open"),
            pl.col("close").alias("high"),
            pl.col("close").alias("low"),
            "close",
            pl.lit(0.0).alias("volume"),
        ]
    )


def candidate_factors(model_frame: pl.DataFrame, benchmark_signal: str) -> dict[str, pl.DataFrame]:
    return {
        benchmark_signal: build_top_signal_factor(model_frame, benchmark_signal).rename(
            {"date": "timestamp", "factor": "signal"}
        ),
        "-atr": build_top_signal_factor(model_frame, "-atr").rename(
            {"date": "timestamp", "factor": "signal"}
        ),
        "log_volume": build_top_signal_factor(model_frame, "log_volume").rename(
            {"date": "timestamp", "factor": "signal"}
        ),
        "ipca": build_ipca_prediction_factor(model_frame).rename(
            {"date": "timestamp", "factor": "signal"}
        ),
    }


def run_backtest(name: str, prices: pl.DataFrame, signals: pl.DataFrame):
    strategy = WeeklyLongOnly3()
    config = BacktestConfig.from_preset("realistic")
    config.share_type = ShareType.FRACTIONAL
    feed = DataFeed(prices_df=prices, signals_df=signals)
    result = Engine(feed, strategy, config).run()
    out_dir = BACKTEST_DIR / name
    result.to_parquet(out_dir)
    daily_returns = result.to_daily_returns(calendar="NYSE")
    pl.DataFrame({"daily_returns": daily_returns}).write_parquet(out_dir / "daily_returns.parquet")
    returns = daily_returns.to_numpy()
    return result, returns, out_dir


def write_report(summary: dict[str, object]) -> None:
    lines = [
        "# SP20 Long-Only Backtest Comparison",
        "",
        f"Tested strategies: `{', '.join(summary['strategies'])}`",
        "",
        "| Strategy | Total Return % | Sharpe | Max Drawdown | Final Value | Trades | DSR/PSR Probability |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["results"]:
        lines.append(
            f"| {row['name']} | {row['total_return_pct']:.2f} | {row['sharpe']:.2f} | "
            f"{row['max_drawdown']:.4f} | {row['final_value']:.2f} | {row['n_trades']} | {row['dsr_probability']:.3f} |"
        )
    lines.extend(["", "## Winner", "", summary["winner_text"]])
    REPORT_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    model_frame = pl.read_parquet(MODEL_FRAME_PATH)
    test_frame = build_test_frame(model_frame)
    prices = build_prices(test_frame)

    top_signals = json.loads(TOP_SIGNALS_PATH.read_text())
    benchmark_signal = top_signals[0]["signal"]
    factors = candidate_factors(model_frame, benchmark_signal)

    results_rows: list[dict[str, object]] = []
    returns_matrix: list = []
    names: list[str] = []

    for name, factor in factors.items():
        result, returns, out_dir = run_backtest(name.replace("-", "neg_"), prices, factor)
        names.append(name)
        returns_matrix.append(returns)
        results_rows.append(
            {
                "name": name,
                "total_return_pct": float(result.metrics.get("total_return_pct", 0.0)),
                "sharpe": float(result.metrics.get("sharpe", 0.0)),
                "max_drawdown": float(result.metrics.get("max_drawdown", 0.0)),
                "final_value": float(result.metrics.get("final_value", 0.0)),
                "n_trades": len(result.trades),
                "artifacts": str(out_dir),
            }
        )

    dsr = deflated_sharpe_ratio(
        returns_matrix,
        frequency="daily",
        correlation_method="effective_rank",
        min_k_eff=2.0,
    )
    leader_name = max(results_rows, key=lambda x: x["sharpe"])["name"]

    for row in results_rows:
        row["dsr_probability"] = float(dsr.probability) if row["name"] == leader_name else float("nan")

    sorted_rows = sorted(results_rows, key=lambda x: (x["sharpe"], x["total_return_pct"]), reverse=True)
    winner = sorted_rows[0]
    winner_text = (
        f"`{winner['name']}` is the strongest long-only backtest by Sharpe/return in this first pass. "
        f"DSR leader: `{leader_name}` with probability `{float(dsr.probability):.3f}`."
    )

    summary = {
        "strategies": names,
        "results": sorted_rows,
        "dsr_probability": float(dsr.probability),
        "winner_text": winner_text,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    write_report(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
