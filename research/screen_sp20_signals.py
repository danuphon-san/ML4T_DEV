from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
diagnostic_src = REPO_ROOT / "diagnostic" / "src"
if str(diagnostic_src) not in sys.path:
    sys.path.insert(0, str(diagnostic_src))

from ml4t.diagnostic import analyze_signal


OUTPUT_DIR = REPO_ROOT / "research" / "outputs"
MODEL_FRAME_PATH = OUTPUT_DIR / "sp20_seed_model_frame.parquet"
SCREEN_RESULTS_PATH = OUTPUT_DIR / "sp20_seed_signal_screen.parquet"
SCREEN_REPORT_PATH = OUTPUT_DIR / "sp20_seed_signal_screen_report.md"
TOP_SIGNALS_PATH = OUTPUT_DIR / "sp20_seed_top_signals.json"
TOP_FACTOR_PATH = OUTPUT_DIR / "sp20_seed_signal_factor_top1.parquet"
PRICES_PATH = OUTPUT_DIR / "sp20_seed_signal_prices.parquet"


@dataclass(frozen=True)
class SignalSpec:
    name: str
    expression: pl.Expr


def build_signal_specs() -> list[SignalSpec]:
    rank_cols = {
        "rsi_rank": pl.col("rsi").rank(method="average").over("timestamp"),
        "macd_rank": pl.col("macd").rank(method="average").over("timestamp"),
        "atr_rank": pl.col("atr").rank(method="average").over("timestamp"),
        "sma_gap_rank": pl.col("sma_gap_20").rank(method="average").over("timestamp"),
        "ema_gap_rank": pl.col("ema_gap_20").rank(method="average").over("timestamp"),
        "ret_5d_rank": pl.col("ret_5d").rank(method="average").over("timestamp"),
        "log_volume_rank": pl.col("log_volume").rank(method="average").over("timestamp"),
    }
    return [
        SignalSpec("rsi", pl.col("rsi")),
        SignalSpec("-rsi", -pl.col("rsi")),
        SignalSpec("macd", pl.col("macd")),
        SignalSpec("-macd", -pl.col("macd")),
        SignalSpec("atr", pl.col("atr")),
        SignalSpec("-atr", -pl.col("atr")),
        SignalSpec("sma_gap_20", pl.col("sma_gap_20")),
        SignalSpec("-sma_gap_20", -pl.col("sma_gap_20")),
        SignalSpec("ema_gap_20", pl.col("ema_gap_20")),
        SignalSpec("-ema_gap_20", -pl.col("ema_gap_20")),
        SignalSpec("ret_5d", pl.col("ret_5d")),
        SignalSpec("-ret_5d", -pl.col("ret_5d")),
        SignalSpec("ret_1d", pl.col("ret_1d")),
        SignalSpec("-ret_1d", -pl.col("ret_1d")),
        SignalSpec("log_volume", pl.col("log_volume")),
        SignalSpec("-log_volume", -pl.col("log_volume")),
        SignalSpec("obv", pl.col("obv")),
        SignalSpec("-obv", -pl.col("obv")),
        SignalSpec("rsi_rank", rank_cols["rsi_rank"]),
        SignalSpec("macd_rank", rank_cols["macd_rank"]),
        SignalSpec("-atr_rank", -rank_cols["atr_rank"]),
        SignalSpec("-sma_gap_rank", -rank_cols["sma_gap_rank"]),
        SignalSpec("-ema_gap_rank", -rank_cols["ema_gap_rank"]),
        SignalSpec("-ret_5d_rank", -rank_cols["ret_5d_rank"]),
        SignalSpec("-log_volume_rank", -rank_cols["log_volume_rank"]),
    ]


def build_prices(model_frame: pl.DataFrame) -> pl.DataFrame:
    return model_frame.select(
        [
            pl.col("timestamp").alias("date"),
            pl.col("symbol").alias("asset"),
            pl.col("close").alias("price"),
        ]
    )


def evaluate_signals(model_frame: pl.DataFrame, specs: list[SignalSpec]) -> pl.DataFrame:
    prices = build_prices(model_frame)
    rows: list[dict[str, float | str]] = []
    for spec in specs:
        factor = model_frame.select(
            [
                pl.col("timestamp").alias("date"),
                pl.col("symbol").alias("asset"),
                spec.expression.alias("factor"),
            ]
        )
        result = analyze_signal(
            factor=factor,
            prices=prices,
            periods=(1, 5, 21),
            quantiles=5,
            min_assets=10,
        )
        rows.append(
            {
                "signal": spec.name,
                "ic_1d": float(result.ic["1D"]),
                "ic_5d": float(result.ic["5D"]),
                "ic_21d": float(result.ic["21D"]),
                "ic_t_21d": float(result.ic_t_stat["21D"]),
                "spread_1d": float(result.spread["1D"]),
                "spread_5d": float(result.spread["5D"]),
                "spread_21d": float(result.spread["21D"]),
                "spread_t_5d": float(result.spread_t_stat["5D"]),
                "spread_t_21d": float(result.spread_t_stat["21D"]),
                "monotonicity_21d": float(result.monotonicity["21D"]),
                "turnover_21d": float(result.turnover["21D"]),
            }
        )
    return pl.DataFrame(rows).sort(
        by=["spread_t_21d", "ic_21d", "monotonicity_21d"],
        descending=[True, True, True],
    )


def select_top_signals(results: pl.DataFrame, top_n: int = 5) -> pl.DataFrame:
    filtered = results.filter(
        (pl.col("spread_t_21d") > 2.0)
        & (pl.col("spread_21d") > 0.0)
        & (pl.col("ic_21d") > 0.0)
        & (pl.col("monotonicity_21d") >= 0.5)
    )
    if filtered.is_empty():
        filtered = results.head(top_n)
    return filtered.head(top_n)


def write_report(results: pl.DataFrame, selected: pl.DataFrame, output_path: Path) -> None:
    lines = [
        "# SP20 Signal Screen",
        "",
        f"Signals tested: `{results.height}`",
        f"Signals selected: `{selected.height}`",
        "",
        "## Selected Signals",
        "",
        "| Signal | IC 21D | Spread 21D | Spread t-stat 21D | Monotonicity 21D | Turnover 21D |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in selected.to_dicts():
        lines.append(
            f"| {row['signal']} | {row['ic_21d']:.4f} | {row['spread_21d']:.4f} | "
            f"{row['spread_t_21d']:.2f} | {row['monotonicity_21d']:.3f} | {row['turnover_21d']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Top 10 by 21D Spread t-stat",
            "",
            "```text",
            results.head(10).select(
                ["signal", "ic_21d", "spread_21d", "spread_t_21d", "monotonicity_21d", "turnover_21d"]
            ).__str__(),
            "```",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    model_frame = pl.read_parquet(MODEL_FRAME_PATH)
    prices = build_prices(model_frame)
    prices.write_parquet(PRICES_PATH)

    specs = build_signal_specs()
    results = evaluate_signals(model_frame, specs)
    selected = select_top_signals(results, top_n=5)

    results.write_parquet(SCREEN_RESULTS_PATH)
    write_report(results, selected, SCREEN_REPORT_PATH)

    top = selected.to_dicts()
    TOP_SIGNALS_PATH.write_text(json.dumps(top, indent=2))

    best_signal_name = top[0]["signal"]
    best_spec = next(spec for spec in specs if spec.name == best_signal_name)
    top_factor = model_frame.select(
        [
            pl.col("timestamp").alias("date"),
            pl.col("symbol").alias("asset"),
            best_spec.expression.alias("factor"),
        ]
    )
    top_factor.write_parquet(TOP_FACTOR_PATH)

    summary = {
        "tested": results.height,
        "selected": top,
        "top_factor_path": str(TOP_FACTOR_PATH),
        "screen_results_path": str(SCREEN_RESULTS_PATH),
        "screen_report_path": str(SCREEN_REPORT_PATH),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
