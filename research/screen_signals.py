from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass

import polars as pl

from research_universe import universe_spec

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
diagnostic_src = REPO_ROOT / "diagnostic" / "src"
if str(diagnostic_src) not in sys.path:
    sys.path.insert(0, str(diagnostic_src))
from ml4t.diagnostic import analyze_signal


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
        SignalSpec("rsi", pl.col("rsi")), SignalSpec("-rsi", -pl.col("rsi")),
        SignalSpec("macd", pl.col("macd")), SignalSpec("-macd", -pl.col("macd")),
        SignalSpec("atr", pl.col("atr")), SignalSpec("-atr", -pl.col("atr")),
        SignalSpec("sma_gap_20", pl.col("sma_gap_20")), SignalSpec("-sma_gap_20", -pl.col("sma_gap_20")),
        SignalSpec("ema_gap_20", pl.col("ema_gap_20")), SignalSpec("-ema_gap_20", -pl.col("ema_gap_20")),
        SignalSpec("ret_5d", pl.col("ret_5d")), SignalSpec("-ret_5d", -pl.col("ret_5d")),
        SignalSpec("ret_1d", pl.col("ret_1d")), SignalSpec("-ret_1d", -pl.col("ret_1d")),
        SignalSpec("log_volume", pl.col("log_volume")), SignalSpec("-log_volume", -pl.col("log_volume")),
        SignalSpec("obv", pl.col("obv")), SignalSpec("-obv", -pl.col("obv")),
        SignalSpec("rsi_rank", rank_cols["rsi_rank"]), SignalSpec("macd_rank", rank_cols["macd_rank"]),
        SignalSpec("-atr_rank", -rank_cols["atr_rank"]), SignalSpec("-sma_gap_rank", -rank_cols["sma_gap_rank"]),
        SignalSpec("-ema_gap_rank", -rank_cols["ema_gap_rank"]), SignalSpec("-ret_5d_rank", -rank_cols["ret_5d_rank"]),
        SignalSpec("-log_volume_rank", -rank_cols["log_volume_rank"]),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="sp20_seed")
    args = parser.parse_args()
    spec = universe_spec(args.prefix)
    model_frame = pl.read_parquet(spec.output_dir / f"{spec.prefix}_model_frame.parquet")
    prices = model_frame.select([pl.col("timestamp").alias("date"), pl.col("symbol").alias("asset"), pl.col("close").alias("price")])
    rows = []
    specs = build_signal_specs()
    for sig in specs:
        factor = model_frame.select([pl.col("timestamp").alias("date"), pl.col("symbol").alias("asset"), sig.expression.alias("factor")])
        result = analyze_signal(factor=factor, prices=prices, periods=(1, 5, 21), quantiles=5, min_assets=10)
        rows.append({
            "signal": sig.name,
            "ic_1d": float(result.ic["1D"]), "ic_5d": float(result.ic["5D"]), "ic_21d": float(result.ic["21D"]),
            "ic_t_21d": float(result.ic_t_stat["21D"]), "spread_1d": float(result.spread["1D"]),
            "spread_5d": float(result.spread["5D"]), "spread_21d": float(result.spread["21D"]),
            "spread_t_5d": float(result.spread_t_stat["5D"]), "spread_t_21d": float(result.spread_t_stat["21D"]),
            "monotonicity_21d": float(result.monotonicity["21D"]), "turnover_21d": float(result.turnover["21D"]),
        })
    results = pl.DataFrame(rows).sort(by=["spread_t_21d", "ic_21d", "monotonicity_21d"], descending=[True, True, True])
    selected = results.filter((pl.col("spread_t_21d") > 2.0) & (pl.col("spread_21d") > 0.0) & (pl.col("ic_21d") > 0.0) & (pl.col("monotonicity_21d") >= 0.5)).head(5)
    if selected.is_empty():
        selected = results.head(5)
    results.write_parquet(spec.output_dir / f"{spec.prefix}_signal_screen.parquet")
    top = selected.to_dicts()
    (spec.output_dir / f"{spec.prefix}_top_signals.json").write_text(json.dumps(top, indent=2))
    best = next(s for s in specs if s.name == top[0]["signal"])
    model_frame.select([pl.col("timestamp").alias("date"), pl.col("symbol").alias("asset"), best.expression.alias("factor")]).write_parquet(spec.output_dir / f"{spec.prefix}_signal_factor_top1.parquet")
    report = [
        f"# {spec.prefix} Signal Screen", "",
        f"Signals tested: `{results.height}`", f"Signals selected: `{selected.height}`", "",
        "## Selected Signals", "",
        "| Signal | IC 21D | Spread 21D | Spread t-stat 21D | Monotonicity 21D | Turnover 21D |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in top:
        report.append(f"| {row['signal']} | {row['ic_21d']:.4f} | {row['spread_21d']:.4f} | {row['spread_t_21d']:.2f} | {row['monotonicity_21d']:.3f} | {row['turnover_21d']:.3f} |")
    (spec.output_dir / f"{spec.prefix}_signal_screen_report.md").write_text("\n".join(report) + "\n")
    print(json.dumps({"tested": results.height, "selected": top}, indent=2))


if __name__ == "__main__":
    main()
