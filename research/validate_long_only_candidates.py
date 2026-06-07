from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass

import numpy as np
import polars as pl

from research_universe import universe_spec

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
for repo in ("backtest", "diagnostic"):
    src = REPO_ROOT / repo / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

from ml4t.backtest import BacktestConfig, DataFeed, Engine
from ml4t.backtest.config import CommissionType, ShareType, SlippageType
from ml4t.backtest.strategies.templates import LongShortStrategy
from ml4t.diagnostic.evaluation.stats import deflated_sharpe_ratio

from compare_sp20_ipca_vs_signal_support import (
    build_ipca_prediction_factor,
    build_rppca_prediction_factor,
    build_top_signal_factor,
)


@dataclass(frozen=True)
class ValidationScenario:
    long_count: int
    rebalance_frequency: int
    cost_profile: str
    commission_rate: float
    slippage_rate: float
    stop_slippage_rate: float


SCENARIOS = [
    ValidationScenario(3, 5, "realistic", 0.002, 0.002, 0.001),
    ValidationScenario(3, 10, "realistic", 0.002, 0.002, 0.001),
    ValidationScenario(5, 5, "realistic", 0.002, 0.002, 0.001),
    ValidationScenario(5, 10, "realistic", 0.002, 0.002, 0.001),
    ValidationScenario(3, 5, "stress", 0.003, 0.003, 0.0015),
    ValidationScenario(3, 10, "stress", 0.003, 0.003, 0.0015),
    ValidationScenario(5, 5, "stress", 0.003, 0.003, 0.0015),
    ValidationScenario(5, 10, "stress", 0.003, 0.003, 0.0015),
]


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


def make_strategy(long_count: int, rebalance_frequency: int) -> LongShortStrategy:
    strategy_type = type(
        f"LongOnly{long_count}Every{rebalance_frequency}",
        (LongShortStrategy,),
        {
            "signal_column": "signal",
            "long_count": long_count,
            "short_count": 0,
            "position_size": min(0.20, 0.95 / max(long_count, 1)),
            "rebalance_frequency": rebalance_frequency,
        },
    )
    return strategy_type()


def build_config(scenario: ValidationScenario) -> BacktestConfig:
    config = BacktestConfig.from_preset("realistic")
    config.share_type = ShareType.FRACTIONAL
    config.commission_type = CommissionType.PERCENTAGE
    config.commission_rate = scenario.commission_rate
    config.slippage_type = SlippageType.PERCENTAGE
    config.slippage_rate = scenario.slippage_rate
    config.stop_slippage_rate = scenario.stop_slippage_rate
    return config


def subperiod_metrics(returns: np.ndarray) -> dict[str, float]:
    if returns.size == 0:
        return {"return": float("nan"), "sharpe": float("nan")}
    equity = np.cumprod(1.0 + returns)
    total_return = float(equity[-1] - 1.0)
    std = float(np.std(returns, ddof=1)) if returns.size > 1 else 0.0
    sharpe = float(np.mean(returns) / std * np.sqrt(252.0)) if std > 0 else float("nan")
    return {"return": total_return, "sharpe": sharpe}


def max_drawdown(returns: np.ndarray) -> float:
    if returns.size == 0:
        return float("nan")
    equity = np.cumprod(1.0 + returns)
    running_peak = np.maximum.accumulate(equity)
    drawdowns = equity / running_peak - 1.0
    return float(abs(drawdowns.min()))


def run_variant(prices: pl.DataFrame, signals: pl.DataFrame, scenario: ValidationScenario) -> tuple[object, np.ndarray]:
    strategy = make_strategy(scenario.long_count, scenario.rebalance_frequency)
    config = build_config(scenario)
    result = Engine(DataFeed(prices_df=prices, signals_df=signals), strategy, config).run()
    daily_returns = result.to_daily_returns(calendar="NYSE").to_numpy()
    return result, daily_returns


def aggregate_candidate(rows: list[dict[str, object]]) -> dict[str, object]:
    sharpe_values = [float(row["sharpe"]) for row in rows]
    return_values = [float(row["total_return_pct"]) for row in rows]
    first_half_positive = sum(float(row["first_half_return_pct"]) > 0.0 for row in rows)
    second_half_positive = sum(float(row["second_half_return_pct"]) > 0.0 for row in rows)
    consistent_runs = sum(
        (float(row["first_half_return_pct"]) > 0.0) and (float(row["second_half_return_pct"]) > 0.0)
        for row in rows
    )
    return {
        "candidate": rows[0]["candidate"],
        "n_runs": len(rows),
        "mean_sharpe": float(np.mean(sharpe_values)),
        "median_sharpe": float(np.median(sharpe_values)),
        "min_sharpe": float(np.min(sharpe_values)),
        "mean_return_pct": float(np.mean(return_values)),
        "min_return_pct": float(np.min(return_values)),
        "max_return_pct": float(np.max(return_values)),
        "first_half_positive_runs": int(first_half_positive),
        "second_half_positive_runs": int(second_half_positive),
        "consistent_positive_runs": int(consistent_runs),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="sp100_seed")
    args = parser.parse_args()

    spec = universe_spec(args.prefix)
    model_frame = pl.read_parquet(spec.output_dir / f"{spec.prefix}_model_frame.parquet")
    test_frame = build_test_frame(model_frame)
    prices = build_prices(test_frame)
    top_signals = json.loads((spec.output_dir / f"{spec.prefix}_top_signals.json").read_text())
    benchmark_signal = top_signals[0]["signal"]
    candidates = {
        benchmark_signal: build_top_signal_factor(model_frame, benchmark_signal).rename({"date": "timestamp", "factor": "signal"}),
    }
    for signal_name in [row["signal"] for row in top_signals[1:3]]:
        if signal_name not in candidates:
            candidates[signal_name] = build_top_signal_factor(model_frame, signal_name).rename({"date": "timestamp", "factor": "signal"})
    if "log_volume" not in candidates:
        candidates["log_volume"] = build_top_signal_factor(model_frame, "log_volume").rename({"date": "timestamp", "factor": "signal"})
    candidates["ipca"] = build_ipca_prediction_factor(model_frame, spec.prefix).rename({"date": "timestamp", "factor": "signal"})
    candidates["rp_pca"] = build_rppca_prediction_factor(model_frame, spec.prefix).rename({"date": "timestamp", "factor": "signal"})

    detailed_rows: list[dict[str, object]] = []
    candidate_returns: dict[str, list[np.ndarray]] = {name: [] for name in candidates}
    scenario_winners: list[dict[str, object]] = []

    for scenario in SCENARIOS:
        scenario_rows: list[dict[str, object]] = []
        for name, signals in candidates.items():
            result, returns = run_variant(prices, signals, scenario)
            split = returns.size // 2
            first_half = subperiod_metrics(returns[:split])
            second_half = subperiod_metrics(returns[split:])
            row = {
                "candidate": name,
                "long_count": scenario.long_count,
                "rebalance_frequency": scenario.rebalance_frequency,
                "cost_profile": scenario.cost_profile,
                "commission_rate": scenario.commission_rate,
                "slippage_rate": scenario.slippage_rate,
                "stop_slippage_rate": scenario.stop_slippage_rate,
                "total_return_pct": float(result.metrics.get("total_return_pct", 0.0)),
                "sharpe": float(result.metrics.get("sharpe", 0.0)),
                "max_drawdown_pct": float(result.metrics.get("max_drawdown_pct", 0.0)),
                "num_fills": int(result.metrics.get("num_fills", 0)),
                "num_rebalance_events": int(result.metrics.get("num_rebalance_events", 0)),
                "total_commission": float(result.metrics.get("total_commission", 0.0)),
                "total_slippage": float(result.metrics.get("total_slippage", 0.0)),
                "first_half_return_pct": first_half["return"] * 100.0,
                "first_half_sharpe": first_half["sharpe"],
                "second_half_return_pct": second_half["return"] * 100.0,
                "second_half_sharpe": second_half["sharpe"],
                "full_window_max_drawdown_pct": max_drawdown(returns) * 100.0,
            }
            detailed_rows.append(row)
            scenario_rows.append(row)
            candidate_returns[name].append(returns)

        winner = max(scenario_rows, key=lambda row: (float(row["sharpe"]), float(row["total_return_pct"])))
        scenario_winners.append(
            {
                "long_count": scenario.long_count,
                "rebalance_frequency": scenario.rebalance_frequency,
                "cost_profile": scenario.cost_profile,
                "winner": winner["candidate"],
                "winner_sharpe": winner["sharpe"],
                "winner_return_pct": winner["total_return_pct"],
            }
        )

    detailed = pl.DataFrame(detailed_rows).sort(
        by=["cost_profile", "long_count", "rebalance_frequency", "sharpe"],
        descending=[False, False, False, True],
    )
    detailed.write_parquet(spec.output_dir / f"{spec.prefix}_validation_sweep.parquet")

    aggregate_rows = [
        aggregate_candidate([row for row in detailed_rows if row["candidate"] == candidate])
        for candidate in candidates
    ]
    aggregate_rows.sort(key=lambda row: (row["median_sharpe"], row["mean_return_pct"]), reverse=True)

    returns_matrix = [series for series_list in candidate_returns.values() for series in series_list]
    dsr = deflated_sharpe_ratio(
        returns_matrix,
        frequency="daily",
        correlation_method="effective_rank",
        min_k_eff=2.0,
    )

    winner_counts: dict[str, int] = {}
    for row in scenario_winners:
        winner_counts[row["winner"]] = winner_counts.get(row["winner"], 0) + 1

    winner = aggregate_rows[0]
    summary = {
        "prefix": spec.prefix,
        "benchmark_signal": benchmark_signal,
        "scenarios_tested": len(SCENARIOS),
        "candidate_order": list(candidates.keys()),
        "scenario_winner_counts": winner_counts,
        "aggregate": aggregate_rows,
        "scenario_winners": scenario_winners,
        "overall_dsr_probability": float(dsr.probability),
        "recommended_benchmark": winner["candidate"],
        "winner_text": (
            f"`{winner['candidate']}` is the recommended benchmark after the sweep. "
            f"It won `{winner_counts.get(winner['candidate'], 0)}/{len(SCENARIOS)}` scenarios, "
            f"with median Sharpe `{winner['median_sharpe']:.2f}` and mean return `{winner['mean_return_pct']:.2f}%`."
        ),
    }
    (spec.output_dir / f"{spec.prefix}_validation_summary.json").write_text(json.dumps(summary, indent=2))

    report_lines = [
        "# Candidate Validation Sweep",
        "",
        f"Universe: `{spec.prefix}`",
        f"Benchmark signal from screen: `{benchmark_signal}`",
        "Candidates: `"
        + ", ".join(candidates.keys())
        + "`",
        "",
        "Scenarios:",
        "- `realistic`: 20 bps commission + 20 bps slippage, 10 bps stop slippage",
        "- `stress`: 30 bps commission + 30 bps slippage, 15 bps stop slippage",
        "- holdings: `3` and `5`",
        "- rebalance: every `5` and `10` bars",
        "",
        "## Aggregate Ranking",
        "",
        "| Candidate | Runs | Mean Sharpe | Median Sharpe | Min Sharpe | Mean Return % | First-Half Positive | Second-Half Positive | Consistent Positive |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregate_rows:
        report_lines.append(
            f"| {row['candidate']} | {row['n_runs']} | {row['mean_sharpe']:.2f} | {row['median_sharpe']:.2f} | "
            f"{row['min_sharpe']:.2f} | {row['mean_return_pct']:.2f} | {row['first_half_positive_runs']} | "
            f"{row['second_half_positive_runs']} | {row['consistent_positive_runs']} |"
        )

    report_lines.extend(
        [
            "",
            "## Scenario Winners",
            "",
            "| Cost | Long Count | Rebalance | Winner | Sharpe | Return % |",
            "| --- | ---: | ---: | --- | ---: | ---: |",
        ]
    )
    for row in scenario_winners:
        report_lines.append(
            f"| {row['cost_profile']} | {row['long_count']} | {row['rebalance_frequency']} | {row['winner']} | "
            f"{row['winner_sharpe']:.2f} | {row['winner_return_pct']:.2f} |"
        )

    top_variants = detailed.sort("sharpe", descending=True).head(10).to_dicts()
    report_lines.extend(
        [
            "",
            "## Top Variants",
            "",
            "| Candidate | Cost | Long Count | Rebalance | Sharpe | Return % | Max DD % | First Half % | Second Half % |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in top_variants:
        report_lines.append(
            f"| {row['candidate']} | {row['cost_profile']} | {row['long_count']} | {row['rebalance_frequency']} | "
            f"{row['sharpe']:.2f} | {row['total_return_pct']:.2f} | {row['max_drawdown_pct']:.2f} | "
            f"{row['first_half_return_pct']:.2f} | {row['second_half_return_pct']:.2f} |"
        )

    report_lines.extend(
        [
            "",
            "## Recommendation",
            "",
            summary["winner_text"],
            "",
            f"Overall DSR across validation variants: `{summary['overall_dsr_probability']:.3f}`",
        ]
    )
    (spec.output_dir / f"{spec.prefix}_validation_report.md").write_text("\n".join(report_lines) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
