"""Walk-forward backtest on the 10-year SP500 point-in-time dataset.

Design:
  Rolling 1-year test windows built from a configurable minimum training window.
  Default is the original 5-year train / 1-year test design.

Signal selection: For each fold, run cross-sectional screening on the training period
  to select the best composite (vol + risk + combined) and best single signals.
Then backtest on the test period using those signals.

Usage:
    uv run python backtest_walkforward_10yr.py --prefix sp500_10yr
    uv run python backtest_walkforward_10yr.py --prefix sp500_10yr --fold 3
    uv run python backtest_walkforward_10yr.py --prefix sp500_10yr --min-train-years 2
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
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

# ── Fold definitions ─────────────────────────────────────────────────────────

@dataclass
class Fold:
    idx: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str


DATA_START_YEAR = 2016
TEST_END_YEAR = 2026


def generate_folds(min_train_years: int) -> list[Fold]:
    if min_train_years < 1:
        raise ValueError("min_train_years must be >= 1")

    folds: list[Fold] = []
    first_test_year = DATA_START_YEAR + min_train_years
    idx = 1
    for test_year in range(first_test_year, TEST_END_YEAR):
        train_start_year = test_year - min_train_years
        folds.append(
            Fold(
                idx=idx,
                train_start=f"{train_start_year:04d}-01-01",
                train_end=f"{test_year:04d}-01-01",
                test_start=f"{test_year:04d}-01-01",
                test_end=f"{test_year + 1:04d}-01-01",
            )
        )
        idx += 1
    return folds


def window_label(min_train_years: int) -> str:
    return f"{min_train_years}y"


# ── Signal groups (same as composite backtest) ───────────────────────────────

VOL_SIGNALS = [
    "kyle_lambda",
    "garch_forecast",
    "coefficient_of_variation",
]
RISK_SIGNALS = [
    "risk_adjusted_returns_sharpe_ratio",
    "-maximum_drawdown_time_underwater",
    "-maximum_drawdown_max_drawdown",
]
SINGLE_BENCHMARKS = [
    "-bollinger_bands_lower",
    "plus_di",
    "log_volume",
]

IC_THRESHOLD = 0.0
SPREAD_T_THRESHOLD = 2.0
MONOTONICITY_THRESHOLD = 0.5

OOS_PERIODS = (5, 21)


# ── Strategy ─────────────────────────────────────────────────────────────────

class WeeklyLongOnly5(LongShortStrategy):
    signal_column = "signal"
    long_count = 5
    short_count = 0
    position_size = 0.15
    rebalance_frequency = 5


# ── Signal utilities ──────────────────────────────────────────────────────────

def _expr(name: str) -> pl.Expr:
    return -pl.col(name[1:]) if name.startswith("-") else pl.col(name)


def available_signals(frame: pl.DataFrame, signals: list[str]) -> list[str]:
    cols = set(frame.columns)
    return [s for s in signals if s.lstrip("-") in cols]


def screen_signals_on_training(
    train_frame: pl.DataFrame,
    signals: list[str],
) -> dict[str, float]:
    """Return {signal_name: spread_t_21d} for signals that pass screening thresholds."""
    from ml4t.diagnostic import analyze_signal

    results: dict[str, float] = {}
    dates = train_frame["timestamp"].unique().sort()
    for name in signals:
        base = name.lstrip("-")
        if base not in train_frame.columns:
            continue
        factor = train_frame.select([
            pl.col("timestamp").alias("date"),
            pl.col("symbol").alias("asset"),
            _expr(name).alias("factor"),
        ])
        prices = train_frame.select([
            pl.col("timestamp").alias("date"),
            pl.col("symbol").alias("asset"),
            pl.col("close").alias("price"),
        ])
        try:
            result = analyze_signal(factor=factor, prices=prices, periods=(21,), quantiles=5, min_assets=10)
            ic = result.ic.get("21D", 0.0) or 0.0
            spread = result.spread.get("21D", 0.0) or 0.0
            spread_t = result.spread_t_stat.get("21D", 0.0) or 0.0
            mono = result.monotonicity.get("21D", 0.0) or 0.0
            if ic > IC_THRESHOLD and spread > 0 and spread_t > SPREAD_T_THRESHOLD and mono >= MONOTONICITY_THRESHOLD:
                results[name] = spread_t
        except Exception:
            pass
    return results


def evaluate_oos_ic(
    prices_test: pl.DataFrame,
    signals_df: pl.DataFrame,
) -> dict[str, float]:
    """Compute OOS IC (Pearson), Rank IC (Spearman), spread, and monotonicity."""
    from ml4t.diagnostic import analyze_signal

    factor = signals_df.select([
        pl.col("timestamp").alias("date"),
        pl.col("asset"),
        pl.col("signal").alias("factor"),
    ])
    prices = prices_test.select([
        pl.col("timestamp").alias("date"),
        pl.col("asset"),
        pl.col("close").alias("price"),
    ])

    out: dict[str, float] = {}
    try:
        r_pearson = analyze_signal(
            factor=factor,
            prices=prices,
            periods=OOS_PERIODS,
            quantiles=5,
            min_assets=10,
            ic_method="pearson",
        )
        r_spearman = analyze_signal(
            factor=factor,
            prices=prices,
            periods=OOS_PERIODS,
            quantiles=5,
            min_assets=10,
            ic_method="spearman",
        )
        for p in OOS_PERIODS:
            k = f"{p}D"
            out[f"oos_ic_{p}d"] = r_pearson.ic.get(k, float("nan"))
            out[f"oos_ic_{p}d_tstat"] = r_pearson.ic_t_stat.get(k, float("nan"))
            out[f"oos_rank_ic_{p}d"] = r_spearman.ic.get(k, float("nan"))
            out[f"oos_rank_ic_{p}d_tstat"] = r_spearman.ic_t_stat.get(k, float("nan"))
            out[f"oos_ic_ir_{p}d"] = r_spearman.ic_ir.get(k, float("nan"))
            out[f"oos_spread_{p}d"] = r_spearman.spread.get(k, float("nan"))
            out[f"oos_spread_tstat_{p}d"] = r_spearman.spread_t_stat.get(k, float("nan"))
            out[f"oos_monotonicity_{p}d"] = r_spearman.monotonicity.get(k, float("nan"))
    except Exception as exc:
        print(f"    OOS IC eval error: {exc}")

    return out


def build_composite(frame: pl.DataFrame, signal_names: list[str]) -> pl.Series:
    rank_exprs = [
        _expr(s).rank(method="average").over("timestamp").alias(s)
        for s in signal_names
    ]
    ranks = frame.select(rank_exprs)
    return ranks.select(pl.mean_horizontal(pl.all())).to_series().alias("signal")


def make_signals_df(frame: pl.DataFrame, sig: pl.Series) -> pl.DataFrame:
    return frame.select(["timestamp", pl.col("symbol").alias("asset")]).with_columns(
        sig.alias("signal")
    )


def make_single_signals_df(frame: pl.DataFrame, name: str) -> pl.DataFrame:
    return frame.select([
        "timestamp", pl.col("symbol").alias("asset"),
        _expr(name).alias("signal"),
    ])


# ── Backtest runner ───────────────────────────────────────────────────────────

def run_one(name: str, prices: pl.DataFrame, signals: pl.DataFrame, out_dir: Path):
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


# ── Per-fold logic ────────────────────────────────────────────────────────────

def run_fold(fold: Fold, model_frame: pl.DataFrame, min_train_years: int, out_dir: Path) -> dict[str, Any]:
    print(f"\n{'='*60}")
    print(
        f"Fold {fold.idx} [{window_label(min_train_years)}]: "
        f"Train {fold.train_start}→{fold.train_end}, Test {fold.test_start}→{fold.test_end}"
    )

    def dt(s: str) -> datetime:
        return datetime.strptime(s, "%Y-%m-%d")

    train = model_frame.filter(
        (pl.col("timestamp") >= dt(fold.train_start)) & (pl.col("timestamp") < dt(fold.train_end))
    )
    test = model_frame.filter(
        (pl.col("timestamp") >= dt(fold.test_start)) & (pl.col("timestamp") < dt(fold.test_end))
    )
    print(f"  Train rows: {train.height:,}  Test rows: {test.height:,}")

    if test.height == 0:
        print(f"  SKIP: no test data for fold {fold.idx}")
        return {}

    # Screen signals on training period
    all_signals = VOL_SIGNALS + RISK_SIGNALS + SINGLE_BENCHMARKS
    print("  Screening signals on training data...")
    passing = screen_signals_on_training(train, all_signals)
    print(f"  Passing signals: {list(passing.keys())}")

    # Build signal candidates for test period
    prices_test = test.select([
        "timestamp", pl.col("symbol").alias("asset"), "open", "high", "low", "close", "volume"
    ])

    candidates: dict[str, pl.DataFrame] = {}

    # Composites (use signals that pass screening in training)
    vol_avail = [s for s in VOL_SIGNALS if s in passing]
    risk_avail = [s for s in RISK_SIGNALS if s in passing]

    if vol_avail:
        vol_sig = build_composite(test, vol_avail)
        candidates["vol_composite"] = make_signals_df(test, vol_sig)

    if risk_avail:
        risk_sig = build_composite(test, risk_avail)
        candidates["risk_composite"] = make_signals_df(test, risk_sig)

    if vol_avail and risk_avail:
        vol_sig2 = build_composite(test, vol_avail)
        risk_sig2 = build_composite(test, risk_avail)
        combined = pl.Series("signal", [(v + r) / 2 for v, r in zip(vol_sig2.to_list(), risk_sig2.to_list())])
        candidates["combined"] = make_signals_df(test, combined)

    for name in SINGLE_BENCHMARKS:
        if name in passing or name.lstrip("-") in test.columns:
            candidates[name] = make_single_signals_df(test, name)

    if not candidates:
        print("  SKIP: no valid candidates for this fold")
        return {}

    fold_dir = out_dir / f"fold_{fold.idx:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    # ── OOS IC evaluation (primary: signal validity) ──────────────────────────
    print("  Evaluating OOS IC on test period...")
    oos_ic_map: dict[str, dict[str, float]] = {}
    for name, sdf in candidates.items():
        oos = evaluate_oos_ic(prices_test, sdf)
        oos_ic_map[name] = oos
        ric5 = oos.get("oos_rank_ic_5d", float("nan"))
        ric21 = oos.get("oos_rank_ic_21d", float("nan"))
        t21 = oos.get("oos_rank_ic_21d_tstat", float("nan"))
        print(f"    {name}: RankIC_5d={ric5:+.4f}  RankIC_21d={ric21:+.4f} (t={t21:.2f})")

    # ── Backtest (supplementary) ──────────────────────────────────────────────
    rows: list[dict] = []
    returns_matrix: list[list[float]] = []
    names: list[str] = []

    for name, sdf in candidates.items():
        oos = oos_ic_map.get(name, {})
        row: dict = {
            "name": name,
            "fold": fold.idx,
            "min_train_years": min_train_years,
            **oos,
            "total_return_pct": float("nan"),
            "sharpe": float("nan"),
            "max_drawdown": float("nan"),
            "n_trades": 0,
            "dsr_probability": float("nan"),
        }
        try:
            result, daily_ret = run_one(name, prices_test, sdf, fold_dir)
            names.append(name)
            returns_matrix.append(daily_ret)
            row.update({
                "total_return_pct": float(result.metrics.get("total_return_pct", 0.0)),
                "sharpe": float(result.metrics.get("sharpe", 0.0)),
                "max_drawdown": float(result.metrics.get("max_drawdown", 0.0)),
                "n_trades": len(result.trades),
            })
            ric21 = oos.get("oos_rank_ic_21d", float("nan"))
            print(f"    {name}: RankIC_21d={ric21:+.4f}  Sharpe={row['sharpe']:.2f} (supplementary)")
        except Exception as exc:
            print(f"    {name}: backtest FAIL {exc}")
        rows.append(row)

    if rows:
        rows.sort(key=lambda x: x.get("oos_rank_ic_21d", float("-inf")), reverse=True)
        winner = rows[0]["name"]
        try:
            if len(returns_matrix) >= 2:
                dsr = deflated_sharpe_ratio(returns_matrix, frequency="daily", correlation_method="effective_rank", min_k_eff=2.0)
                for row in rows:
                    row["dsr_probability"] = float(dsr.probability) if row["name"] == winner else float("nan")
                dsr_prob = float(dsr.probability)
            else:
                dsr_prob = float("nan")
        except Exception:
            dsr_prob = float("nan")
            for row in rows:
                row["dsr_probability"] = float("nan")

        ric21 = rows[0].get("oos_rank_ic_21d", float("nan"))
        print(f"  Winner (by RankIC_21d): {winner}  RankIC_21d={ric21:+.4f}  DSR={dsr_prob:.3f}")
    else:
        dsr_prob = float("nan")

    fold_result = {
        "fold": fold.idx,
        "min_train_years": min_train_years,
        "train_start": fold.train_start,
        "train_end": fold.train_end,
        "test_start": fold.test_start,
        "test_end": fold.test_end,
        "passing_signals": passing,
        "results": rows,
        "dsr_probability": dsr_prob,
    }
    (fold_dir / "fold_result.json").write_text(json.dumps(fold_result, indent=2))
    return fold_result


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="sp500_10yr")
    parser.add_argument("--fold", type=int, default=0, help="Run only this fold (1-5), 0 = all")
    parser.add_argument(
        "--min-train-years",
        type=int,
        default=5,
        help="Minimum rolling training window in years.",
    )
    args = parser.parse_args()

    model_path = REPO_ROOT / "research" / "outputs" / args.prefix / f"{args.prefix}_model_frame.parquet"
    if not model_path.exists():
        print(f"ERROR: {model_path} not found. Run build_sp500_10yr_dataset.py first.")
        sys.exit(1)

    model_frame = pl.read_parquet(model_path)
    print(f"Model frame: {model_frame.height:,} rows, {model_frame['symbol'].n_unique()} symbols")
    print(f"Date range: {model_frame['timestamp'].min()} → {model_frame['timestamp'].max()}")

    folds = generate_folds(args.min_train_years)
    print(f"Min train years: {args.min_train_years}")
    print(f"Generated folds: {len(folds)}")

    label = window_label(args.min_train_years)
    out_dir = REPO_ROOT / "research" / "outputs" / args.prefix / f"walkforward_{label}"
    out_dir.mkdir(parents=True, exist_ok=True)

    folds_to_run = [f for f in folds if args.fold == 0 or f.idx == args.fold]
    all_results: list[dict] = []

    for fold in folds_to_run:
        result = run_fold(fold, model_frame, args.min_train_years, out_dir)
        if result:
            all_results.append(result)

    if not all_results:
        print("No folds completed.")
        return

    # Aggregate: collect all test-period returns across folds
    summary_rows: list[dict] = []

    for fr in all_results:
        for row in fr.get("results", []):
            summary_rows.append(row)

    def _fmt(v: float, fmt: str = ".4f") -> str:
        return "—" if math.isnan(v) else format(v, fmt)

    # ── Per-signal verdict (pre-committed before inspecting results) ──────────
    # Primary gate : oos_spread_tstat_21d > 2.0 in >= 3 of 5 folds
    # Secondary gate: mean oos_rank_ic_21d > 0.02 AND mean oos_ic_ir_21d > 0.3
    # Verdict       : PASS (both gates), WEAK (primary only), FAIL (primary fails)
    signal_names_ordered = list(dict.fromkeys(row["name"] for row in summary_rows))
    verdicts: dict[str, dict] = {}
    for sig in signal_names_ordered:
        sig_rows = [r for r in summary_rows if r["name"] == sig]
        n_primary_pass = sum(
            1 for r in sig_rows
            if not math.isnan(r.get("oos_spread_tstat_21d", float("nan")))
            and r.get("oos_spread_tstat_21d", 0.0) > 2.0
        )
        primary_passes = n_primary_pass >= 3
        rank_ic_vals = [
            r.get("oos_rank_ic_21d", float("nan")) for r in sig_rows
            if not math.isnan(r.get("oos_rank_ic_21d", float("nan")))
        ]
        ic_ir_vals = [
            r.get("oos_ic_ir_21d", float("nan")) for r in sig_rows
            if not math.isnan(r.get("oos_ic_ir_21d", float("nan")))
        ]
        mean_rank_ic = sum(rank_ic_vals) / len(rank_ic_vals) if rank_ic_vals else float("nan")
        mean_ic_ir = sum(ic_ir_vals) / len(ic_ir_vals) if ic_ir_vals else float("nan")
        secondary_passes = (
            not math.isnan(mean_rank_ic) and mean_rank_ic > 0.02
            and not math.isnan(mean_ic_ir) and mean_ic_ir > 0.3
        )
        if primary_passes and secondary_passes:
            verdict = "PASS"
        elif primary_passes:
            verdict = "WEAK"
        else:
            verdict = "FAIL"
        verdicts[sig] = {
            "signal": sig,
            "n_folds": len(sig_rows),
            "n_primary_pass": n_primary_pass,
            "mean_rank_ic_21d": mean_rank_ic,
            "mean_ic_ir_21d": mean_ic_ir,
            "primary_passes": primary_passes,
            "secondary_passes": secondary_passes,
            "verdict": verdict,
        }

    # Build aggregate report
    report = [
        "# Walk-Forward Backtest — 10-Year SP500 (Point-in-Time)",
        "",
        f"Min train years: {args.min_train_years}",
        f"Folds completed: {len(all_results)}",
        "",
        "| Fold | Train Start | Train End | Test Start | Test End |",
        "| --- | --- | --- | --- | --- |",
    ]
    for fold in folds_to_run:
        report.append(
            f"| {fold.idx} | {fold.train_start} | {fold.train_end} | {fold.test_start} | {fold.test_end} |"
        )

    # Signal validity verdict summary
    report += [
        "",
        "## Signal Validity Verdict",
        "",
        "Primary gate  : `oos_spread_tstat_21d > 2.0` in >= 3 of 5 folds",
        "Secondary gate: `mean RankIC_21d > 0.02` AND `mean IC-IR_21d > 0.3`",
        "",
        "| Signal | Pass Folds (Primary) | Mean RankIC 21d | Mean IC-IR 21d | Verdict |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for v in verdicts.values():
        report.append(
            f"| {v['signal']}"
            f" | {v['n_primary_pass']}/{v['n_folds']}"
            f" | {_fmt(v['mean_rank_ic_21d'])}"
            f" | {_fmt(v['mean_ic_ir_21d'], '.3f')}"
            f" | {v['verdict']} |"
        )

    # Primary: OOS IC metrics (signal validity)
    report += [
        "",
        "## OOS Signal Validity (Primary)",
        "",
        "| Fold | Signal | RankIC 5d | t(5d) | RankIC 21d | t(21d) | IC 21d | t(21d) | Spread T 21d | Mono 21d |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        report.append(
            f"| {row['fold']} | {row['name']}"
            f" | {_fmt(row.get('oos_rank_ic_5d', float('nan')))}"
            f" | {_fmt(row.get('oos_rank_ic_5d_tstat', float('nan')), '.2f')}"
            f" | {_fmt(row.get('oos_rank_ic_21d', float('nan')))}"
            f" | {_fmt(row.get('oos_rank_ic_21d_tstat', float('nan')), '.2f')}"
            f" | {_fmt(row.get('oos_ic_21d', float('nan')))}"
            f" | {_fmt(row.get('oos_ic_21d_tstat', float('nan')), '.2f')}"
            f" | {_fmt(row.get('oos_spread_tstat_21d', float('nan')), '.2f')}"
            f" | {_fmt(row.get('oos_monotonicity_21d', float('nan')), '.3f')}"
            f" |"
        )

    # Supplementary: backtest metrics
    report += [
        "",
        "## Backtest Metrics (Supplementary)",
        "",
        "| Fold | Signal | Return % | Sharpe | Max DD | Trades | DSR |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        report.append(
            f"| {row['fold']} | {row['name']}"
            f" | {_fmt(row.get('total_return_pct', float('nan')), '.2f')}"
            f" | {_fmt(row.get('sharpe', float('nan')), '.2f')}"
            f" | {_fmt(row.get('max_drawdown', float('nan')), '.3f')}"
            f" | {row.get('n_trades', 0)}"
            f" | {_fmt(row.get('dsr_probability', float('nan')), '.3f')}"
            f" |"
        )

    report_text = "\n".join(report) + "\n"
    report_path = out_dir / f"{args.prefix}_walkforward_{label}_report.md"
    report_path.write_text(report_text)
    print("\n" + report_text)

    summary = {
        "min_train_years": args.min_train_years,
        "generated_folds": [fold.__dict__ for fold in folds],
        "executed_folds": [fold.__dict__ for fold in folds_to_run],
        "folds": all_results,
        "rows": summary_rows,
        "verdicts": list(verdicts.values()),
    }
    (out_dir / f"{args.prefix}_walkforward_{label}_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
