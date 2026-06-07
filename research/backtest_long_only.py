from __future__ import annotations

import argparse
import json
import sys

import polars as pl

from research_universe import universe_spec

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
for repo in ("backtest", "diagnostic"):
    src = REPO_ROOT / repo / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
from ml4t.backtest import BacktestConfig, DataFeed, Engine
from ml4t.backtest.config import ShareType
from ml4t.backtest.strategies.templates import LongShortStrategy
from ml4t.diagnostic.evaluation.stats import deflated_sharpe_ratio
from compare_sp20_ipca_vs_signal_support import (
    build_ipca_prediction_factor,
    build_rppca_prediction_factor,
    build_top_signal_factor,
)


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
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    )


def run_backtest(name: str, prices: pl.DataFrame, signals: pl.DataFrame, out_dir):
    strategy = WeeklyLongOnly3()
    config = BacktestConfig.from_preset("realistic")
    config.share_type = ShareType.FRACTIONAL
    result = Engine(DataFeed(prices_df=prices, signals_df=signals), strategy, config).run()
    out_dir.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out_dir)
    daily_returns = result.to_daily_returns(calendar="NYSE")
    pl.DataFrame({"daily_returns": daily_returns}).write_parquet(out_dir / "daily_returns.parquet")
    return result, daily_returns.to_numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="sp20_seed")
    args = parser.parse_args()
    spec = universe_spec(args.prefix)
    backtest_dir = spec.output_dir / "backtests_long_only"
    backtest_dir.mkdir(parents=True, exist_ok=True)
    model_frame = pl.read_parquet(spec.output_dir / f"{spec.prefix}_model_frame.parquet")
    test_frame = build_test_frame(model_frame)
    prices = build_prices(test_frame)
    top = json.loads((spec.output_dir / f"{spec.prefix}_top_signals.json").read_text())
    benchmark_signal = top[0]["signal"]
    candidates = {
        benchmark_signal: build_top_signal_factor(model_frame, benchmark_signal).rename({"date": "timestamp", "factor": "signal"}),
    }
    for signal_name in [row["signal"] for row in top[1:3]]:
        if signal_name not in candidates:
            candidates[signal_name] = build_top_signal_factor(model_frame, signal_name).rename({"date": "timestamp", "factor": "signal"})
    for signal_name in ["-atr", "log_volume"]:
        if signal_name not in candidates:
            candidates[signal_name] = build_top_signal_factor(model_frame, signal_name).rename({"date": "timestamp", "factor": "signal"})
    candidates["ipca"] = build_ipca_prediction_factor(model_frame, spec.prefix).rename({"date": "timestamp", "factor": "signal"})
    candidates["rp_pca"] = build_rppca_prediction_factor(model_frame, spec.prefix).rename({"date": "timestamp", "factor": "signal"})
    rows=[]; returns_matrix=[]; names=[]
    for name, sig in candidates.items():
        result, returns = run_backtest(name.replace("-", "neg_"), prices, sig, backtest_dir / name.replace("-", "neg_"))
        names.append(name); returns_matrix.append(returns)
        rows.append({"name": name, "total_return_pct": float(result.metrics.get("total_return_pct", 0.0)), "sharpe": float(result.metrics.get("sharpe", 0.0)), "max_drawdown": float(result.metrics.get("max_drawdown", 0.0)), "final_value": float(result.metrics.get("final_value", 0.0)), "n_trades": len(result.trades), "artifacts": str(backtest_dir / name.replace("-", "neg_"))})
    dsr = deflated_sharpe_ratio(returns_matrix, frequency="daily", correlation_method="effective_rank", min_k_eff=2.0)
    leader = max(rows, key=lambda x: x["sharpe"])["name"]
    for row in rows:
        row["dsr_probability"] = float(dsr.probability) if row["name"] == leader else float("nan")
    rows = sorted(rows, key=lambda x: (x["sharpe"], x["total_return_pct"]), reverse=True)
    winner = rows[0]
    summary = {"strategies": names, "results": rows, "dsr_probability": float(dsr.probability), "winner_text": f"`{winner['name']}` is the strongest long-only backtest by Sharpe/return in this first pass. DSR leader: `{leader}` with probability `{float(dsr.probability):.3f}`."}
    (spec.output_dir / f"{spec.prefix}_backtest_long_only_summary.json").write_text(json.dumps(summary, indent=2))
    report = ["# Long-Only Backtest Comparison", "", f"Tested strategies: `{', '.join(names)}`", "", "| Strategy | Total Return % | Sharpe | Max Drawdown | Final Value | Trades | DSR/PSR Probability |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        report.append(f"| {row['name']} | {row['total_return_pct']:.2f} | {row['sharpe']:.2f} | {row['max_drawdown']:.4f} | {row['final_value']:.2f} | {row['n_trades']} | {row['dsr_probability']:.3f} |")
    report.extend(["", "## Winner", "", summary["winner_text"]])
    (spec.output_dir / f"{spec.prefix}_backtest_long_only_report.md").write_text("\n".join(report) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
