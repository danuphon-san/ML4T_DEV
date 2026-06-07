from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
for repo in ("diagnostic", "models"):
    src = REPO_ROOT / repo / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

from ml4t.diagnostic import analyze_signal

from compare_sp20_ipca_vs_signal_support import build_ipca_prediction_factor, build_top_signal_factor


OUTPUT_DIR = REPO_ROOT / "research" / "outputs"
MODEL_FRAME_PATH = OUTPUT_DIR / "sp20_seed_model_frame.parquet"
TOP_SIGNALS_PATH = OUTPUT_DIR / "sp20_seed_top_signals.json"
SUMMARY_PATH = OUTPUT_DIR / "sp20_seed_ipca_vs_signal_summary.json"
REPORT_PATH = OUTPUT_DIR / "sp20_seed_ipca_vs_signal_report.md"


def build_test_prices(model_frame: pl.DataFrame) -> pl.DataFrame:
    dates = model_frame.select("timestamp").unique().sort("timestamp")["timestamp"].to_list()
    n_train = int(len(dates) * 0.8)
    test_dates = set(dates[n_train:])
    test_frame = model_frame.filter(pl.col("timestamp").is_in(test_dates)).sort(["timestamp", "symbol"])
    prices = test_frame.select(
        [
            pl.col("timestamp").alias("date"),
            pl.col("symbol").alias("asset"),
            pl.col("close").alias("price"),
        ]
    )
    return prices


def summarize(result: object) -> dict[str, float]:
    return {
        "ic_1d": float(result.ic["1D"]),
        "ic_5d": float(result.ic["5D"]),
        "ic_21d": float(result.ic["21D"]),
        "spread_1d": float(result.spread["1D"]),
        "spread_5d": float(result.spread["5D"]),
        "spread_21d": float(result.spread["21D"]),
        "spread_t_21d": float(result.spread_t_stat["21D"]),
        "monotonicity_21d": float(result.monotonicity["21D"]),
        "turnover_21d": float(result.turnover["21D"]),
    }


def write_report(summary: dict[str, object]) -> None:
    sig = summary["signal"]
    ipca = summary["ipca"]
    lines = [
        "# SP20 IPCA vs Signal",
        "",
        f"Benchmark signal: `{summary['benchmark_signal']}`",
        f"Test dates: `{summary['n_test_dates']}`",
        f"Test assets per date: approximately `{summary['n_assets']}`",
        "",
        "## Comparison",
        "",
        "| Metric | Benchmark Signal | IPCA |",
        "| --- | ---: | ---: |",
        f"| IC 5D | {sig['ic_5d']:.4f} | {ipca['ic_5d']:.4f} |",
        f"| IC 21D | {sig['ic_21d']:.4f} | {ipca['ic_21d']:.4f} |",
        f"| Spread 5D | {sig['spread_5d']:.4f} | {ipca['spread_5d']:.4f} |",
        f"| Spread 21D | {sig['spread_21d']:.4f} | {ipca['spread_21d']:.4f} |",
        f"| Spread t-stat 21D | {sig['spread_t_21d']:.2f} | {ipca['spread_t_21d']:.2f} |",
        f"| Monotonicity 21D | {sig['monotonicity_21d']:.3f} | {ipca['monotonicity_21d']:.3f} |",
        f"| Turnover 21D | {sig['turnover_21d']:.3f} | {ipca['turnover_21d']:.3f} |",
        "",
        "## Winner",
        "",
        summary["winner_text"],
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    model_frame = pl.read_parquet(MODEL_FRAME_PATH)
    top_signals = json.loads(TOP_SIGNALS_PATH.read_text())
    benchmark_signal = top_signals[0]["signal"]

    prices = build_test_prices(model_frame)
    signal_factor = build_top_signal_factor(model_frame, benchmark_signal)
    ipca_factor = build_ipca_prediction_factor(model_frame)

    signal_result = analyze_signal(signal_factor, prices, periods=(1, 5, 21), quantiles=5, min_assets=10)
    ipca_result = analyze_signal(ipca_factor, prices, periods=(1, 5, 21), quantiles=5, min_assets=10)

    signal_summary = summarize(signal_result)
    ipca_summary = summarize(ipca_result)

    winner = "Benchmark signal remains stronger on 21D spread t-stat." \
        if signal_summary["spread_t_21d"] >= ipca_summary["spread_t_21d"] \
        else "IPCA improves on the benchmark signal on 21D spread t-stat."

    summary = {
        "benchmark_signal": benchmark_signal,
        "n_test_dates": prices.select("date").n_unique(),
        "n_assets": prices.select("asset").n_unique(),
        "signal": signal_summary,
        "ipca": ipca_summary,
        "winner_text": winner,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    write_report(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
