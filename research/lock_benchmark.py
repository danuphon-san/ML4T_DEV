from __future__ import annotations

import argparse
import json

from research_universe import universe_spec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="sp500_full")
    args = parser.parse_args()

    spec = universe_spec(args.prefix)
    validation_summary = json.loads(
        (spec.output_dir / f"{spec.prefix}_validation_summary.json").read_text()
    )
    backtest_summary = json.loads(
        (spec.output_dir / f"{spec.prefix}_backtest_long_only_summary.json").read_text()
    )

    benchmark_name = validation_summary["recommended_benchmark"]
    benchmark_row = next(
        row for row in validation_summary["aggregate"] if row["candidate"] == benchmark_name
    )
    first_pass_row = next(
        row for row in backtest_summary["results"] if row["name"] == benchmark_name
    )

    lock = {
        "prefix": spec.prefix,
        "locked_benchmark": benchmark_name,
        "validation_basis": {
            "scenarios_tested": validation_summary["scenarios_tested"],
            "wins": validation_summary["scenario_winner_counts"].get(benchmark_name, 0),
            "median_sharpe": benchmark_row["median_sharpe"],
            "mean_return_pct": benchmark_row["mean_return_pct"],
            "overall_dsr_probability": validation_summary["overall_dsr_probability"],
        },
        "first_pass_backtest": {
            "total_return_pct": first_pass_row["total_return_pct"],
            "sharpe": first_pass_row["sharpe"],
            "max_drawdown": first_pass_row["max_drawdown"],
            "n_trades": first_pass_row["n_trades"],
        },
        "notes": [
            "Locked after corrected long-only backtest with real OHLC inputs.",
            "Use as the benchmark strategy for future model comparisons on this universe.",
        ],
    }

    (spec.output_dir / f"{spec.prefix}_benchmark_lock.json").write_text(
        json.dumps(lock, indent=2)
    )
    report = [
        f"# {spec.prefix} Benchmark Lock",
        "",
        f"Locked benchmark: `{benchmark_name}`",
        "",
        "## Validation Basis",
        "",
        f"- Scenarios tested: `{lock['validation_basis']['scenarios_tested']}`",
        f"- Scenario wins: `{lock['validation_basis']['wins']}`",
        f"- Median Sharpe: `{lock['validation_basis']['median_sharpe']:.2f}`",
        f"- Mean return: `{lock['validation_basis']['mean_return_pct']:.2f}%`",
        f"- Overall DSR: `{lock['validation_basis']['overall_dsr_probability']:.3f}`",
        "",
        "## First-Pass Backtest",
        "",
        f"- Return: `{lock['first_pass_backtest']['total_return_pct']:.2f}%`",
        f"- Sharpe: `{lock['first_pass_backtest']['sharpe']:.2f}`",
        f"- Max drawdown: `{lock['first_pass_backtest']['max_drawdown']:.4f}`",
        f"- Trades: `{lock['first_pass_backtest']['n_trades']}`",
    ]
    (spec.output_dir / f"{spec.prefix}_benchmark_lock.md").write_text(
        "\n".join(report) + "\n"
    )
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()
