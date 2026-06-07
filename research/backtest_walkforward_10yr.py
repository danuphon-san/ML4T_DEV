"""Walk-forward backtest on the 10-year SP500 point-in-time dataset.

Two independent experiments:

  Phase 1: Global-universe N-sweep (absolute top-N)
    Long-only top-N from the global universe at each rebalance.
    N_LIST = [5, 10, 15, 20, 25, 30], equal-weight (position_size = 1/N).

  Phase 2: Sector-neutral K-sweep (within FF-12)
    Cross-sectional within-sector z-scoring; long top-K per FF-12 industry
    with "take all available" fallback when a sector has < K stocks.
    K_LIST = [1, 2, 3].

Primary metric for both phases is OOS IC (signal validity). The backtest
metrics (Sharpe, Return, Max DD, DSR) are supplementary.

Pre-committed verdict thresholds:
  Phase 1: oos_spread_tstat_21d > 2.0 in >= 3/5 folds
           AND mean(oos_rank_ic_21d) > 0.02 AND mean(oos_ic_ir_21d) > 0.3
  Phase 2: oos_spread_tstat_21d > 2.0 in >= 3/5 folds
           AND mean(oos_rank_ic_21d) > 0.01 AND mean(oos_ic_ir_21d) > 0.3

Usage:
    uv run python backtest_walkforward_10yr.py --prefix sp500_10yr
    uv run python backtest_walkforward_10yr.py --prefix sp500_10yr --mode phase1
    uv run python backtest_walkforward_10yr.py --prefix sp500_10yr --mode phase2
    uv run python backtest_walkforward_10yr.py --prefix sp500_10yr --fold 1
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

# ── Phase 1 config ──────────────────────────────────────────────────────────
N_LIST = [5, 10, 15, 20, 25, 30]
P1_PRIMARY_FOLDS_REQUIRED = 3
P1_PRIMARY_TSTAT_THRESHOLD = 2.0
P1_SECONDARY_MEAN_RANK_IC = 0.02
P1_SECONDARY_MEAN_IC_IR = 0.3

# ── Phase 2 config ──────────────────────────────────────────────────────────
K_LIST = [1, 2, 3]
P2_PRIMARY_FOLDS_REQUIRED = 3
P2_PRIMARY_TSTAT_THRESHOLD = 2.0
P2_SECONDARY_MEAN_RANK_IC = 0.01  # lowered: within-sector IC is naturally smaller
P2_SECONDARY_MEAN_IC_IR = 0.3

SECTORS_FILE = REPO_ROOT / "research" / "outputs" / "sp500_pit" / "sp500_ff12_sectors.parquet"
REBALANCE_FREQUENCY = 5  # weekly


# ── Strategies ──────────────────────────────────────────────────────────────

def make_long_only_strategy(n: int) -> LongShortStrategy:
    """Build a fully-invested equal-weight long-only top-N strategy."""
    cls = type(
        f"LongOnlyN{n}",
        (LongShortStrategy,),
        {
            "signal_column": "signal",
            "long_count": n,
            "short_count": 0,
            "position_size": 1.0 / n,
            "rebalance_frequency": REBALANCE_FREQUENCY,
        },
    )
    return cls()


def make_stratified_strategy(k: int, sector_map: dict[str, str]) -> LongShortStrategy:
    """Build a sector-stratified top-K-per-sector long-only strategy.

    Position size = 1 / (K * n_active_sectors). Sectors with <K stocks contribute
    fewer positions, producing minor cash drag (acceptable per design).
    """
    active_sectors = {s for s in sector_map.values() if s and s != "Other"}
    n_sectors = len(active_sectors)
    target_n = k * n_sectors
    pos_size = 1.0 / target_n if target_n > 0 else 0.0

    class StratifiedSectorLongOnly(LongShortStrategy):
        signal_column = "signal"
        long_count = 999  # unused; rank_assets is overridden
        short_count = 0
        position_size = pos_size
        rebalance_frequency = REBALANCE_FREQUENCY

        def rank_assets(self, data: dict[str, dict]) -> tuple[list[str], list[str]]:
            by_sector: dict[str, list[tuple[str, float]]] = {}
            for asset, bar in data.items():
                bar_signals = bar.get("signals", {})
                signal = bar_signals.get(self.signal_column) if bar_signals else None
                if signal is None:
                    continue
                if isinstance(signal, float) and math.isnan(signal):
                    continue
                sector = sector_map.get(asset)
                if not sector or sector == "Other":
                    continue
                by_sector.setdefault(sector, []).append((asset, signal))

            long_assets: list[str] = []
            for sigs in by_sector.values():
                sigs.sort(key=lambda x: x[1], reverse=True)
                long_assets.extend([s[0] for s in sigs[:k]])
            return long_assets, []

    return StratifiedSectorLongOnly()


# ── Signal utilities ─────────────────────────────────────────────────────────

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
            factor=factor, prices=prices, periods=OOS_PERIODS,
            quantiles=5, min_assets=10, ic_method="pearson",
        )
        r_spearman = analyze_signal(
            factor=factor, prices=prices, periods=OOS_PERIODS,
            quantiles=5, min_assets=10, ic_method="spearman",
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


# ── Phase 2 helpers ─────────────────────────────────────────────────────────

def load_sector_map() -> dict[str, str]:
    """Load ticker → FF-12 industry mapping (from build_sp500_ff12_sectors.py output)."""
    if not SECTORS_FILE.exists():
        raise FileNotFoundError(
            f"Sector mapping missing at {SECTORS_FILE}. "
            "Run: uv run python build_sp500_ff12_sectors.py first."
        )
    df = pl.read_parquet(SECTORS_FILE)
    return {row["ticker"]: row["ff12_industry"] for row in df.iter_rows(named=True)}


def apply_zscore_within_sector(
    signals_df: pl.DataFrame,
    sector_map: dict[str, str],
) -> pl.DataFrame:
    """Z-score the 'signal' column within each (timestamp, sector) group.

    Z_{i,s} = (signal_i - mean_s) / std_s
    Rows in 'Other'/unknown sectors are dropped (will not be selected anyway).
    """
    sector_pl = pl.DataFrame({
        "asset": list(sector_map.keys()),
        "sector": list(sector_map.values()),
    })
    with_sec = (
        signals_df.join(sector_pl, on="asset", how="left")
        .filter(pl.col("sector").is_not_null() & (pl.col("sector") != "Other"))
    )
    z = with_sec.with_columns([
        ((pl.col("signal") - pl.col("signal").mean().over(["timestamp", "sector"]))
         / pl.col("signal").std().over(["timestamp", "sector"]))
        .alias("signal_z"),
    ])
    return z.select([
        "timestamp", "asset",
        pl.col("signal_z").alias("signal"),
    ])


# ── Backtest runner ──────────────────────────────────────────────────────────

def run_one(
    name: str,
    strategy: LongShortStrategy,
    prices: pl.DataFrame,
    signals: pl.DataFrame,
    out_dir: Path,
) -> tuple[Any, list[float]]:
    config = BacktestConfig.from_preset("realistic")
    config.share_type = ShareType.FRACTIONAL
    result = Engine(DataFeed(prices_df=prices, signals_df=signals), strategy, config).run()
    safe = name.replace("-", "neg_").replace(" ", "_")
    (out_dir / safe).mkdir(parents=True, exist_ok=True)
    result.to_parquet(out_dir / safe)
    daily_returns = result.to_daily_returns(calendar="NYSE")
    pl.DataFrame({"daily_returns": daily_returns}).write_parquet(out_dir / safe / "daily_returns.parquet")
    return result, daily_returns.to_numpy().tolist()


# ── Build candidates for a fold (Phase 1: global signal) ────────────────────

def build_candidates_global(
    train: pl.DataFrame,
    test: pl.DataFrame,
    passing: dict[str, float],
) -> dict[str, pl.DataFrame]:
    """Build composite + single-signal candidates on the global universe."""
    candidates: dict[str, pl.DataFrame] = {}

    vol_avail = [s for s in VOL_SIGNALS if s in passing]
    risk_avail = [s for s in RISK_SIGNALS if s in passing]

    if vol_avail:
        candidates["vol_composite"] = make_signals_df(test, build_composite(test, vol_avail))
    if risk_avail:
        candidates["risk_composite"] = make_signals_df(test, build_composite(test, risk_avail))
    if vol_avail and risk_avail:
        vol_sig = build_composite(test, vol_avail)
        risk_sig = build_composite(test, risk_avail)
        combined = pl.Series("signal", [(v + r) / 2 for v, r in zip(vol_sig.to_list(), risk_sig.to_list())])
        candidates["combined"] = make_signals_df(test, combined)

    for name in SINGLE_BENCHMARKS:
        if name in passing or name.lstrip("-") in test.columns:
            candidates[name] = make_single_signals_df(test, name)

    return candidates


# ── Phase 1: N-sweep fold runner ─────────────────────────────────────────────

def run_phase1_fold(
    fold: Fold,
    model_frame: pl.DataFrame,
    min_train_years: int,
    out_dir: Path,
) -> dict[str, Any]:
    print(f"\n{'='*60}")
    print(f"[Phase 1] Fold {fold.idx} [{window_label(min_train_years)}]: "
          f"Train {fold.train_start}→{fold.train_end}, Test {fold.test_start}→{fold.test_end}")

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

    all_signals = VOL_SIGNALS + RISK_SIGNALS + SINGLE_BENCHMARKS
    print("  Screening signals on training data...")
    passing = screen_signals_on_training(train, all_signals)
    print(f"  Passing signals: {list(passing.keys())}")

    prices_test = test.select([
        "timestamp", pl.col("symbol").alias("asset"), "open", "high", "low", "close", "volume"
    ])
    candidates = build_candidates_global(train, test, passing)
    if not candidates:
        print("  SKIP: no valid candidates for this fold")
        return {}

    fold_dir = out_dir / f"fold_{fold.idx:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    # OOS IC (N-independent; computed once per candidate)
    print("  Evaluating OOS IC on test period...")
    oos_ic_map: dict[str, dict[str, float]] = {}
    for name, sdf in candidates.items():
        oos = evaluate_oos_ic(prices_test, sdf)
        oos_ic_map[name] = oos
        ric5 = oos.get("oos_rank_ic_5d", float("nan"))
        ric21 = oos.get("oos_rank_ic_21d", float("nan"))
        t21 = oos.get("oos_rank_ic_21d_tstat", float("nan"))
        print(f"    {name}: RankIC_5d={ric5:+.4f}  RankIC_21d={ric21:+.4f} (t={t21:.2f})")

    # Backtest: N-sweep per candidate (supplementary)
    rows: list[dict] = []
    print(f"  Backtest N-sweep over {N_LIST}...")
    for name, sdf in candidates.items():
        oos = oos_ic_map.get(name, {})
        n_metrics: dict[int, dict[str, float]] = {}
        n_returns: dict[int, list[float]] = {}
        for n in N_LIST:
            strategy = make_long_only_strategy(n)
            run_name = f"{name}_n{n}"
            try:
                result, daily_ret = run_one(run_name, strategy, prices_test, sdf, fold_dir)
                n_metrics[n] = {
                    "total_return_pct": float(result.metrics.get("total_return_pct", 0.0)),
                    "sharpe": float(result.metrics.get("sharpe", 0.0)),
                    "max_drawdown": float(result.metrics.get("max_drawdown", 0.0)),
                    "n_trades": len(result.trades),
                }
                n_returns[n] = daily_ret
            except Exception as exc:
                print(f"    {run_name}: backtest FAIL {exc}")
                n_metrics[n] = {
                    "total_return_pct": float("nan"),
                    "sharpe": float("nan"),
                    "max_drawdown": float("nan"),
                    "n_trades": 0,
                }
        # Print a one-liner Sharpe curve per candidate
        sharpe_line = "  ".join(f"N{n}={n_metrics[n]['sharpe']:.2f}" for n in N_LIST)
        ric21 = oos.get("oos_rank_ic_21d", float("nan"))
        print(f"    {name}: RankIC_21d={ric21:+.4f} | {sharpe_line}")

        row = {
            "name": name,
            "fold": fold.idx,
            "min_train_years": min_train_years,
            **oos,
        }
        for n, m in n_metrics.items():
            row[f"sharpe_n{n}"] = m["sharpe"]
            row[f"return_n{n}"] = m["total_return_pct"]
            row[f"maxdd_n{n}"] = m["max_drawdown"]
            row[f"trades_n{n}"] = m["n_trades"]
        rows.append(row)

    # DSR across N-sweeps for the winning signal (by RankIC_21d)
    rows.sort(key=lambda x: x.get("oos_rank_ic_21d", float("-inf")), reverse=True)
    winner = rows[0]["name"] if rows else None
    dsr_prob = float("nan")
    if winner is not None:
        winner_returns = [n_returns_dict for n_returns_dict in []]  # placeholder
        # We don't keep returns_matrix across signals here; DSR is per-signal across N.
        # Build a per-signal returns matrix for the winner from its own per-N runs.
        winner_n_returns: list[list[float]] = []
        for n in N_LIST:
            sub = fold_dir / f"{winner}_n{n}".replace("-", "neg_") / "daily_returns.parquet"
            if sub.exists():
                try:
                    arr = pl.read_parquet(sub)["daily_returns"].to_numpy().tolist()
                    winner_n_returns.append(arr)
                except Exception:
                    pass
        if len(winner_n_returns) >= 2:
            try:
                dsr = deflated_sharpe_ratio(
                    winner_n_returns, frequency="daily",
                    correlation_method="effective_rank", min_k_eff=2.0,
                )
                dsr_prob = float(dsr.probability)
            except Exception:
                dsr_prob = float("nan")
        rows[0]["dsr_probability"] = dsr_prob
        ric21 = rows[0].get("oos_rank_ic_21d", float("nan"))
        print(f"  Winner (by RankIC_21d): {winner}  RankIC_21d={ric21:+.4f}  DSR(N-sweep)={dsr_prob:.3f}")

    fold_result = {
        "phase": "phase1",
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


# ── Phase 2: sector-neutral K-sweep fold runner ─────────────────────────────

def run_phase2_fold(
    fold: Fold,
    model_frame: pl.DataFrame,
    sector_map: dict[str, str],
    min_train_years: int,
    out_dir: Path,
) -> dict[str, Any]:
    print(f"\n{'='*60}")
    print(f"[Phase 2] Fold {fold.idx} [{window_label(min_train_years)}]: "
          f"Train {fold.train_start}→{fold.train_end}, Test {fold.test_start}→{fold.test_end}")

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

    all_signals = VOL_SIGNALS + RISK_SIGNALS + SINGLE_BENCHMARKS
    print("  Screening signals on training data...")
    passing = screen_signals_on_training(train, all_signals)
    print(f"  Passing signals: {list(passing.keys())}")

    prices_test = test.select([
        "timestamp", pl.col("symbol").alias("asset"), "open", "high", "low", "close", "volume"
    ])
    raw_candidates = build_candidates_global(train, test, passing)
    if not raw_candidates:
        print("  SKIP: no valid candidates for this fold")
        return {}

    # Z-score each candidate signal within FF-12 sectors per timestamp
    print("  Applying within-sector Z-scoring (FF-12)...")
    candidates: dict[str, pl.DataFrame] = {}
    for name, sdf in raw_candidates.items():
        zsdf = apply_zscore_within_sector(sdf, sector_map)
        if zsdf.height == 0:
            print(f"    {name}: skipped (empty after sector filter)")
            continue
        candidates[name] = zsdf

    if not candidates:
        print("  SKIP: no candidates after Z-scoring")
        return {}

    fold_dir = out_dir / f"fold_{fold.idx:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    # OOS IC on the z-scored signal (within-sector predictive power)
    print("  Evaluating OOS IC on Z-scored test period...")
    oos_ic_map: dict[str, dict[str, float]] = {}
    for name, sdf in candidates.items():
        oos = evaluate_oos_ic(prices_test, sdf)
        oos_ic_map[name] = oos
        ric5 = oos.get("oos_rank_ic_5d", float("nan"))
        ric21 = oos.get("oos_rank_ic_21d", float("nan"))
        t21 = oos.get("oos_rank_ic_21d_tstat", float("nan"))
        print(f"    {name}: RankIC_5d={ric5:+.4f}  RankIC_21d={ric21:+.4f} (t={t21:.2f})")

    # Backtest K-sweep
    rows: list[dict] = []
    print(f"  Backtest K-sweep over {K_LIST} per FF-12 sector...")
    for name, sdf in candidates.items():
        oos = oos_ic_map.get(name, {})
        k_metrics: dict[int, dict[str, float]] = {}
        for k in K_LIST:
            strategy = make_stratified_strategy(k, sector_map)
            run_name = f"{name}_k{k}"
            try:
                result, _daily_ret = run_one(run_name, strategy, prices_test, sdf, fold_dir)
                k_metrics[k] = {
                    "total_return_pct": float(result.metrics.get("total_return_pct", 0.0)),
                    "sharpe": float(result.metrics.get("sharpe", 0.0)),
                    "max_drawdown": float(result.metrics.get("max_drawdown", 0.0)),
                    "n_trades": len(result.trades),
                }
            except Exception as exc:
                print(f"    {run_name}: backtest FAIL {exc}")
                k_metrics[k] = {
                    "total_return_pct": float("nan"),
                    "sharpe": float("nan"),
                    "max_drawdown": float("nan"),
                    "n_trades": 0,
                }
        sharpe_line = "  ".join(f"K{k}={k_metrics[k]['sharpe']:.2f}" for k in K_LIST)
        ric21 = oos.get("oos_rank_ic_21d", float("nan"))
        print(f"    {name}: RankIC_21d={ric21:+.4f} | {sharpe_line}")

        row = {
            "name": name,
            "fold": fold.idx,
            "min_train_years": min_train_years,
            **oos,
        }
        for k, m in k_metrics.items():
            row[f"sharpe_k{k}"] = m["sharpe"]
            row[f"return_k{k}"] = m["total_return_pct"]
            row[f"maxdd_k{k}"] = m["max_drawdown"]
            row[f"trades_k{k}"] = m["n_trades"]
        rows.append(row)

    rows.sort(key=lambda x: x.get("oos_rank_ic_21d", float("-inf")), reverse=True)
    winner = rows[0]["name"] if rows else None
    if winner is not None:
        ric21 = rows[0].get("oos_rank_ic_21d", float("nan"))
        print(f"  Winner (by RankIC_21d): {winner}  RankIC_21d={ric21:+.4f}")

    fold_result = {
        "phase": "phase2",
        "fold": fold.idx,
        "min_train_years": min_train_years,
        "train_start": fold.train_start,
        "train_end": fold.train_end,
        "test_start": fold.test_start,
        "test_end": fold.test_end,
        "passing_signals": passing,
        "results": rows,
    }
    (fold_dir / "fold_result.json").write_text(json.dumps(fold_result, indent=2))
    return fold_result


# ── Verdict computation ──────────────────────────────────────────────────────

def compute_verdicts(
    summary_rows: list[dict],
    primary_folds_required: int,
    primary_tstat: float,
    secondary_mean_rank_ic: float,
    secondary_mean_ic_ir: float,
) -> dict[str, dict]:
    """Apply two-gate pre-committed verdict logic across folds per signal."""
    signal_names = list(dict.fromkeys(row["name"] for row in summary_rows))
    out: dict[str, dict] = {}
    for sig in signal_names:
        sig_rows = [r for r in summary_rows if r["name"] == sig]
        n_primary_pass = sum(
            1 for r in sig_rows
            if not math.isnan(r.get("oos_spread_tstat_21d", float("nan")))
            and r.get("oos_spread_tstat_21d", 0.0) > primary_tstat
        )
        primary_passes = n_primary_pass >= primary_folds_required
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
            not math.isnan(mean_rank_ic) and mean_rank_ic > secondary_mean_rank_ic
            and not math.isnan(mean_ic_ir) and mean_ic_ir > secondary_mean_ic_ir
        )
        if primary_passes and secondary_passes:
            verdict = "PASS"
        elif primary_passes:
            verdict = "WEAK"
        else:
            verdict = "FAIL"
        out[sig] = {
            "signal": sig,
            "n_folds": len(sig_rows),
            "n_primary_pass": n_primary_pass,
            "mean_rank_ic_21d": mean_rank_ic,
            "mean_ic_ir_21d": mean_ic_ir,
            "primary_passes": primary_passes,
            "secondary_passes": secondary_passes,
            "verdict": verdict,
        }
    return out


# ── Report writing ───────────────────────────────────────────────────────────

def _fmt(v: float, fmt: str = ".4f") -> str:
    if v is None:
        return "—"
    try:
        if math.isnan(v):
            return "—"
    except (TypeError, ValueError):
        return str(v)
    return format(v, fmt)


def _ic_table(summary_rows: list[dict]) -> list[str]:
    lines = [
        "",
        "## OOS Signal Validity (Primary)",
        "",
        "| Fold | Signal | RankIC 5d | t(5d) | RankIC 21d | t(21d) | IC 21d | t(21d) | Spread T 21d | Mono 21d |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
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
    return lines


def _verdict_table(verdicts: dict[str, dict], secondary_threshold: float) -> list[str]:
    lines = [
        "",
        "## Signal Validity Verdict",
        "",
        f"Primary gate  : `oos_spread_tstat_21d > 2.0` in >= 3 of 5 folds",
        f"Secondary gate: `mean RankIC_21d > {secondary_threshold}` AND `mean IC-IR_21d > 0.3`",
        "",
        "| Signal | Pass Folds (Primary) | Mean RankIC 21d | Mean IC-IR 21d | Verdict |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for v in verdicts.values():
        lines.append(
            f"| {v['signal']}"
            f" | {v['n_primary_pass']}/{v['n_folds']}"
            f" | {_fmt(v['mean_rank_ic_21d'])}"
            f" | {_fmt(v['mean_ic_ir_21d'], '.3f')}"
            f" | {v['verdict']} |"
        )
    return lines


def _phase1_n_pivot(summary_rows: list[dict]) -> list[str]:
    lines = [
        "",
        "## Backtest N-Sensitivity (Supplementary)",
        "",
        "Per-signal mean Sharpe across folds (headline):",
        "",
        "| Signal | " + " | ".join(f"N={n}" for n in N_LIST) + " |",
        "| --- | " + " | ".join("---:" for _ in N_LIST) + " |",
    ]
    signals = list(dict.fromkeys(row["name"] for row in summary_rows))
    for sig in signals:
        sig_rows = [r for r in summary_rows if r["name"] == sig]
        cells = []
        for n in N_LIST:
            vals = [r.get(f"sharpe_n{n}", float("nan")) for r in sig_rows]
            vals = [v for v in vals if not math.isnan(v)]
            mean = sum(vals) / len(vals) if vals else float("nan")
            cells.append(_fmt(mean, ".2f"))
        lines.append(f"| {sig} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "Per-fold Sharpe matrix (drill-down):",
        "",
        "| Fold | Signal | " + " | ".join(f"N={n}" for n in N_LIST) + " |",
        "| --- | --- | " + " | ".join("---:" for _ in N_LIST) + " |",
    ]
    for row in summary_rows:
        cells = [_fmt(row.get(f"sharpe_n{n}", float("nan")), ".2f") for n in N_LIST]
        lines.append(f"| {row['fold']} | {row['name']} | " + " | ".join(cells) + " |")
    return lines


def _phase2_k_pivot(summary_rows: list[dict]) -> list[str]:
    lines = [
        "",
        "## Backtest K-Sensitivity (Supplementary)",
        "",
        "Per-signal mean Sharpe across folds (headline):",
        "",
        "| Signal | " + " | ".join(f"K={k}" for k in K_LIST) + " |",
        "| --- | " + " | ".join("---:" for _ in K_LIST) + " |",
    ]
    signals = list(dict.fromkeys(row["name"] for row in summary_rows))
    for sig in signals:
        sig_rows = [r for r in summary_rows if r["name"] == sig]
        cells = []
        for k in K_LIST:
            vals = [r.get(f"sharpe_k{k}", float("nan")) for r in sig_rows]
            vals = [v for v in vals if not math.isnan(v)]
            mean = sum(vals) / len(vals) if vals else float("nan")
            cells.append(_fmt(mean, ".2f"))
        lines.append(f"| {sig} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "Per-fold Sharpe matrix (drill-down):",
        "",
        "| Fold | Signal | " + " | ".join(f"K={k}" for k in K_LIST) + " |",
        "| --- | --- | " + " | ".join("---:" for _ in K_LIST) + " |",
    ]
    for row in summary_rows:
        cells = [_fmt(row.get(f"sharpe_k{k}", float("nan")), ".2f") for k in K_LIST]
        lines.append(f"| {row['fold']} | {row['name']} | " + " | ".join(cells) + " |")
    return lines


def write_phase_report(
    phase: str,
    out_dir: Path,
    fold_results: list[dict],
    args: argparse.Namespace,
    label: str,
    folds_to_run: list[Fold],
) -> tuple[str, dict[str, dict]]:
    """Write per-phase report and return (report_text, verdicts)."""
    summary_rows: list[dict] = [r for fr in fold_results for r in fr.get("results", [])]
    if not summary_rows:
        return "", {}

    if phase == "phase1":
        secondary_threshold = P1_SECONDARY_MEAN_RANK_IC
        title = "Walk-Forward Backtest — Phase 1 (Global N-Sweep)"
        pivot_section = _phase1_n_pivot(summary_rows)
    else:
        secondary_threshold = P2_SECONDARY_MEAN_RANK_IC
        title = "Walk-Forward Backtest — Phase 2 (Sector-Neutral K-Sweep, FF-12)"
        pivot_section = _phase2_k_pivot(summary_rows)

    verdicts = compute_verdicts(
        summary_rows,
        primary_folds_required=P1_PRIMARY_FOLDS_REQUIRED,
        primary_tstat=P1_PRIMARY_TSTAT_THRESHOLD,
        secondary_mean_rank_ic=secondary_threshold,
        secondary_mean_ic_ir=P1_SECONDARY_MEAN_IC_IR,
    )

    report: list[str] = [
        f"# {title}",
        "",
        f"Min train years: {args.min_train_years}",
        f"Folds completed: {len(fold_results)}",
        "",
        "| Fold | Train Start | Train End | Test Start | Test End |",
        "| --- | --- | --- | --- | --- |",
    ]
    for fold in folds_to_run:
        report.append(
            f"| {fold.idx} | {fold.train_start} | {fold.train_end} | {fold.test_start} | {fold.test_end} |"
        )
    report += _verdict_table(verdicts, secondary_threshold)
    report += _ic_table(summary_rows)
    report += pivot_section

    report_text = "\n".join(report) + "\n"
    suffix = "phase1" if phase == "phase1" else "phase2"
    report_path = out_dir / f"{args.prefix}_walkforward_{suffix}_{label}_report.md"
    report_path.write_text(report_text)

    summary = {
        "phase": phase,
        "min_train_years": args.min_train_years,
        "executed_folds": [fold.__dict__ for fold in folds_to_run],
        "folds": fold_results,
        "rows": summary_rows,
        "verdicts": list(verdicts.values()),
    }
    (out_dir / f"{args.prefix}_walkforward_{suffix}_{label}_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    return report_text, verdicts


def write_combined_report(
    out_dir: Path,
    args: argparse.Namespace,
    label: str,
    folds_to_run: list[Fold],
    phase1_verdicts: dict[str, dict],
    phase2_verdicts: dict[str, dict],
) -> None:
    all_signals = list(dict.fromkeys(list(phase1_verdicts) + list(phase2_verdicts)))
    lines = [
        "# Walk-Forward Backtest — Combined Phase 1 + Phase 2 Summary",
        "",
        f"Min train years: {args.min_train_years}",
        f"Folds: {len(folds_to_run)}",
        "",
        "## Verdict Comparison",
        "",
        "Phase 1: global universe, absolute top-N selection",
        "Phase 2: within-FF-12 z-scored signal, stratified top-K per sector",
        "",
        "| Signal | P1 RankIC 21d | P1 Pass Folds | P1 Verdict | P2 RankIC 21d | P2 Pass Folds | P2 Verdict |",
        "| --- | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for sig in all_signals:
        p1 = phase1_verdicts.get(sig, {})
        p2 = phase2_verdicts.get(sig, {})
        p1_pf = f"{p1.get('n_primary_pass', '—')}/{p1.get('n_folds', '—')}" if p1 else "—"
        p2_pf = f"{p2.get('n_primary_pass', '—')}/{p2.get('n_folds', '—')}" if p2 else "—"
        lines.append(
            f"| {sig}"
            f" | {_fmt(p1.get('mean_rank_ic_21d', float('nan')))} | {p1_pf} | {p1.get('verdict', '—')}"
            f" | {_fmt(p2.get('mean_rank_ic_21d', float('nan')))} | {p2_pf} | {p2.get('verdict', '—')}"
            f" |"
        )
    text = "\n".join(lines) + "\n"
    (out_dir / f"{args.prefix}_walkforward_combined_{label}_report.md").write_text(text)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="sp500_10yr")
    parser.add_argument("--fold", type=int, default=0, help="Run only this fold (1-5), 0 = all")
    parser.add_argument("--min-train-years", type=int, default=5)
    parser.add_argument(
        "--mode",
        choices=["phase1", "phase2", "both"],
        default="both",
        help="Which experiment(s) to run.",
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
    folds_to_run = [f for f in folds if args.fold == 0 or f.idx == args.fold]
    label = window_label(args.min_train_years)

    base_out = REPO_ROOT / "research" / "outputs" / args.prefix
    phase1_verdicts: dict[str, dict] = {}
    phase2_verdicts: dict[str, dict] = {}

    if args.mode in ("phase1", "both"):
        out_dir = base_out / f"walkforward_phase1_{label}"
        out_dir.mkdir(parents=True, exist_ok=True)
        p1_results: list[dict] = []
        for fold in folds_to_run:
            result = run_phase1_fold(fold, model_frame, args.min_train_years, out_dir)
            if result:
                p1_results.append(result)
        if p1_results:
            text, phase1_verdicts = write_phase_report("phase1", out_dir, p1_results, args, label, folds_to_run)
            print("\n" + text)

    if args.mode in ("phase2", "both"):
        sector_map = load_sector_map()
        active_sectors = {s for s in sector_map.values() if s and s != "Other"}
        print(f"\n[Phase 2] FF-12 sectors active: {len(active_sectors)} → {sorted(active_sectors)}")
        out_dir = base_out / f"walkforward_phase2_{label}"
        out_dir.mkdir(parents=True, exist_ok=True)
        p2_results: list[dict] = []
        for fold in folds_to_run:
            result = run_phase2_fold(fold, model_frame, sector_map, args.min_train_years, out_dir)
            if result:
                p2_results.append(result)
        if p2_results:
            text, phase2_verdicts = write_phase_report("phase2", out_dir, p2_results, args, label, folds_to_run)
            print("\n" + text)

    if args.mode == "both" and (phase1_verdicts or phase2_verdicts):
        combined_dir = base_out / f"walkforward_combined_{label}"
        combined_dir.mkdir(parents=True, exist_ok=True)
        write_combined_report(combined_dir, args, label, folds_to_run, phase1_verdicts, phase2_verdicts)
        print(f"\nCombined report → {combined_dir}")


if __name__ == "__main__":
    main()
