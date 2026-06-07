"""Walk-forward LightGBM backtest on the 10-year SP500 PIT dataset.

Trains a LightGBM ranking model per fold to predict the cross-sectional decile
rank of 5-day forward returns, then backtests with the same WeeklyLongOnly5
strategy used in backtest_walkforward_10yr.py. Runs log_volume as a baseline
in the same script for direct apples-to-apples comparison via aggregate DSR.

Design (from grill-me session 2026-06-06):
  - Target:      ret_5d → cross-sectional decile rank per date (0=bottom, 9=top)
  - Objective:   rank_xendcg  (optimises NDCG; immune to return outliers)
  - Features:    all 105+ engineered features + 12 macro/FF5 context columns
  - CV:          rolling 1-year test folds + 21-day purge at each train/test boundary
  - Val set:     last 63 trading days of purged training window (early stopping)
  - Params:      fixed defaults + early stopping on NDCG@5
  - Strategy:    WeeklyLongOnly5 (top-5 by predicted rank, 15% each, 5-day rebal)
  - Evaluation:  aggregate DSR on concatenated 5-year OOS return series

Usage:
    uv run python backtest_lgbm_walkforward.py --prefix sp500_10yr
    uv run python backtest_lgbm_walkforward.py --prefix sp500_10yr --fold 1
    uv run python backtest_lgbm_walkforward.py --prefix sp500_10yr --feature-set clusters
    uv run python backtest_lgbm_walkforward.py --prefix sp500_10yr --feature-set clusters --min-train-years 2
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
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

# ── Fold definitions (identical to backtest_walkforward_10yr.py) ─────────────

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

PURGE_DAYS = 21   # trading days dropped from end of training (ret_5d label leakage buffer)
VAL_DAYS   = 63   # trading days held out for early-stopping validation (~3 months)


# ── Feature sets ─────────────────────────────────────────────────────────────

_EXCLUDE = frozenset([
    "timestamp", "symbol", "open", "high", "low", "close", "volume",
    "ret_1d", "ret_1d_fwd", "ret_5d",
    "label", "label_return", "label_bars", "label_duration", "barrier_hit",
])

MACRO_COLS = [
    "Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF",
    "DGS2", "DGS5", "DGS10", "DGS30",
    "YIELD_CURVE_SLOPE", "YIELD_CURVE_5_10",
]


def get_feature_cols(df: pl.DataFrame) -> list[str]:
    """Engineered features + macro/FF5 context, excluding target and metadata."""
    numeric = {pl.Float64, pl.Float32, pl.Int64, pl.Int32}
    engineered = [
        c for c in df.columns
        if c not in _EXCLUDE and c not in MACRO_COLS and df[c].dtype in numeric
    ]
    macro = [c for c in MACRO_COLS if c in df.columns]
    return engineered + macro


def load_cluster_feature_cols(prefix: str, model_frame: pl.DataFrame) -> list[str]:
    """Resolve cluster representatives to unique base feature columns."""
    cluster_path = REPO_ROOT / "research" / "outputs" / prefix / f"{prefix}_signal_clusters.json"
    if not cluster_path.exists():
        raise FileNotFoundError(f"Cluster file not found: {cluster_path}")

    payload = json.loads(cluster_path.read_text())
    representatives = payload.get("representatives")
    if not isinstance(representatives, list) or not representatives:
        raise ValueError(f"No representatives found in {cluster_path}")

    feature_cols: list[str] = []
    seen: set[str] = set()
    missing: list[str] = []

    for name in representatives:
        if not isinstance(name, str):
            raise ValueError(f"Invalid representative entry in {cluster_path}: {name!r}")
        base = name.lstrip("-")
        if base in seen:
            continue
        if base not in model_frame.columns:
            missing.append(base)
            continue
        seen.add(base)
        feature_cols.append(base)

    if missing:
        raise ValueError(
            "Cluster representatives missing from model frame: "
            + ", ".join(sorted(missing))
        )

    return feature_cols


def resolve_feature_set(prefix: str, feature_set: str, model_frame: pl.DataFrame) -> tuple[list[str], str]:
    if feature_set == "all":
        return get_feature_cols(model_frame), "all"
    if feature_set == "clusters":
        return load_cluster_feature_cols(prefix, model_frame), "clusters"
    raise ValueError(f"Unsupported feature set: {feature_set}")


# ── LightGBM configuration ───────────────────────────────────────────────────

LGBM_PARAMS: dict[str, Any] = {
    "objective":      "rank_xendcg",
    "metric":         "ndcg",
    "ndcg_eval_at":   [5, 10],
    "num_leaves":     63,
    "learning_rate":  0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq":   5,
    "min_data_in_leaf": 20,
    "verbose":        -1,
    "n_jobs":         -1,
}
MAX_ROUNDS          = 1000
EARLY_STOP_ROUNDS   = 50


# ── Strategy ─────────────────────────────────────────────────────────────────

class WeeklyLongOnly5(LongShortStrategy):
    signal_column      = "signal"
    long_count         = 5
    short_count        = 0
    position_size      = 0.15
    rebalance_frequency = 5


# ── Data preparation ──────────────────────────────────────────────────────────

def _decile_per_date(group: pd.Series) -> pd.Series:
    try:
        return pd.qcut(group.rank(method="first"), 10, labels=False).astype(float)
    except Exception:
        return pd.Series(np.nan, index=group.index)


def make_lgb_arrays(
    df: pd.DataFrame, feature_cols: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, decile_labels, group_sizes) sorted by timestamp, NaN rows dropped."""
    df = df.sort_values("timestamp", kind="stable").copy()
    df["_decile"] = df.groupby("timestamp", sort=True)["ret_5d"].transform(_decile_per_date)
    df = df.dropna(subset=["ret_5d", "_decile"])

    X     = df[feature_cols].values.astype(np.float32)
    y     = df["_decile"].values.astype(np.int32)
    group = df.groupby("timestamp", sort=True).size().values
    return X, y, group


def split_train_val(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove last PURGE_DAYS dates, then split off last VAL_DAYS as validation."""
    dates = sorted(df["timestamp"].unique())

    if len(dates) <= PURGE_DAYS:
        return df, pd.DataFrame(columns=df.columns)

    purge_cutoff = dates[-(PURGE_DAYS + 1)]
    purged = df[df["timestamp"] <= purge_cutoff]

    purged_dates = sorted(purged["timestamp"].unique())
    if len(purged_dates) <= VAL_DAYS:
        return purged, pd.DataFrame(columns=df.columns)

    val_cutoff = purged_dates[-(VAL_DAYS + 1)]
    train = purged[purged["timestamp"] <= val_cutoff]
    val   = purged[purged["timestamp"] >  val_cutoff]
    return train, val


def train_lgbm(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
) -> lgb.Booster:
    X_tr, y_tr, g_tr = make_lgb_arrays(train_df, feature_cols)
    X_va, y_va, g_va = make_lgb_arrays(val_df, feature_cols)

    ds_train = lgb.Dataset(X_tr, label=y_tr, group=g_tr, feature_name=feature_cols)
    ds_val   = lgb.Dataset(X_va, label=y_va, group=g_va, reference=ds_train)

    model = lgb.train(
        LGBM_PARAMS,
        ds_train,
        num_boost_round=MAX_ROUNDS,
        valid_sets=[ds_val],
        callbacks=[
            lgb.early_stopping(EARLY_STOP_ROUNDS, verbose=False),
            lgb.log_evaluation(100),
        ],
    )
    return model


# ── Signal generation ─────────────────────────────────────────────────────────

def lgbm_signals(test: pl.DataFrame, model: lgb.Booster, feature_cols: list[str]) -> pl.DataFrame:
    X = test.select(feature_cols).to_pandas().values.astype(np.float32)
    scores = model.predict(X)
    return (
        test
        .with_columns(pl.Series("_score", scores))
        .select([
            "timestamp",
            pl.col("symbol").alias("asset"),
            pl.col("_score").rank(method="average").over("timestamp").alias("signal"),
        ])
    )


def logvol_signals(test: pl.DataFrame) -> pl.DataFrame:
    return test.select([
        "timestamp",
        pl.col("symbol").alias("asset"),
        pl.col("log_volume").rank(method="average").over("timestamp").alias("signal"),
    ])


# ── Backtest runner ───────────────────────────────────────────────────────────

def run_backtest(name: str, prices: pl.DataFrame, signals: pl.DataFrame, out_dir: Path):
    config = BacktestConfig.from_preset("realistic")
    config.share_type = ShareType.FRACTIONAL
    result = Engine(DataFeed(prices_df=prices, signals_df=signals), WeeklyLongOnly5(), config).run()
    safe = name.replace("-", "neg_")
    (out_dir / safe).mkdir(parents=True, exist_ok=True)
    result.to_parquet(out_dir / safe)
    daily_returns = result.to_daily_returns(calendar="NYSE")
    pl.DataFrame({"daily_returns": daily_returns}).write_parquet(out_dir / safe / "daily_returns.parquet")
    return result, daily_returns.to_numpy().tolist()


# ── Per-fold logic ─────────────────────────────────────────────────────────────

def run_fold(
    fold: Fold,
    model_frame: pl.DataFrame,
    feature_cols: list[str],
    feature_set: str,
    min_train_years: int,
    out_dir: Path,
) -> dict[str, Any]:
    print(f"\n{'='*60}")
    print(
        f"Fold {fold.idx} [{feature_set}, {window_label(min_train_years)}]: "
        f"Train {fold.train_start}→{fold.train_end}  Test {fold.test_start}→{fold.test_end}"
    )

    def dt(s: str) -> datetime:
        return datetime.strptime(s, "%Y-%m-%d")

    train_pl = model_frame.filter(
        (pl.col("timestamp") >= dt(fold.train_start)) & (pl.col("timestamp") < dt(fold.train_end))
    )
    test_pl = model_frame.filter(
        (pl.col("timestamp") >= dt(fold.test_start)) & (pl.col("timestamp") < dt(fold.test_end))
    )
    print(f"  Train rows: {train_pl.height:,}  Test rows: {test_pl.height:,}")

    if test_pl.height == 0:
        print(f"  SKIP: no test data for fold {fold.idx}")
        return {}

    # Split training data with purge + validation
    train_pd = train_pl.select(["timestamp", "ret_5d"] + feature_cols).to_pandas()
    train_df, val_df = split_train_val(train_pd)
    print(f"  LightGBM train: {len(train_df):,} rows  val: {len(val_df):,} rows  (purge={PURGE_DAYS}d, val={VAL_DAYS}d)")

    # Train
    print("  Training LightGBM (rank_xendcg)...")
    model = train_lgbm(train_df, val_df, feature_cols)
    print(f"  Best iteration: {model.best_iteration}  (of {MAX_ROUNDS} max)")

    fold_dir = out_dir / f"fold_{fold.idx:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    fi = dict(zip(feature_cols, model.feature_importance(importance_type="gain").tolist()))
    (fold_dir / "feature_importance.json").write_text(
        json.dumps(dict(sorted(fi.items(), key=lambda x: x[1], reverse=True)), indent=2)
    )

    prices_test = test_pl.select([
        "timestamp", pl.col("symbol").alias("asset"), "open", "high", "low", "close", "volume"
    ])

    candidates = {
        "lgbm":       lgbm_signals(test_pl, model, feature_cols),
        "log_volume": logvol_signals(test_pl),
    }

    rows: list[dict] = []
    returns_map: dict[str, list[float]] = {}

    for name, sdf in candidates.items():
        try:
            result, daily_ret = run_backtest(name, prices_test, sdf, fold_dir)
            returns_map[name] = daily_ret
            rows.append({
                "name": name,
                "fold": fold.idx,
                "feature_set": feature_set,
                "min_train_years": min_train_years,
                "total_return_pct": float(result.metrics.get("total_return_pct", 0.0)),
                "sharpe":           float(result.metrics.get("sharpe", 0.0)),
                "max_drawdown":     float(result.metrics.get("max_drawdown", 0.0)),
                "n_trades":         len(result.trades),
                "best_iteration":   model.best_iteration if name == "lgbm" else None,
            })
            print(f"    {name}: Sharpe={rows[-1]['sharpe']:.2f}  Return={rows[-1]['total_return_pct']:.1f}%")
        except Exception as exc:
            print(f"    {name}: FAIL {exc}")

    fold_result: dict[str, Any] = {
        "fold":       fold.idx,
        "feature_set": feature_set,
        "min_train_years": min_train_years,
        "train_start": fold.train_start,
        "train_end":   fold.train_end,
        "test_start":  fold.test_start,
        "test_end":    fold.test_end,
        "results":     rows,
        "returns_map": returns_map,
    }
    (fold_dir / "fold_result.json").write_text(
        json.dumps({k: v for k, v in fold_result.items() if k != "returns_map"}, indent=2)
    )
    return fold_result


# ── Aggregate evaluation ──────────────────────────────────────────────────────

def annualised_sharpe(returns: list[float]) -> float:
    arr = np.array(returns)
    if arr.std() == 0:
        return float("nan")
    return float(arr.mean() / arr.std() * np.sqrt(252))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="sp500_10yr")
    parser.add_argument("--fold", type=int, default=0, help="Run only this fold (1-5), 0=all")
    parser.add_argument(
        "--feature-set",
        choices=("all", "clusters"),
        default="all",
        help="Feature set for LightGBM training.",
    )
    parser.add_argument(
        "--min-train-years",
        type=int,
        default=5,
        help="Minimum rolling training window in years.",
    )
    args = parser.parse_args()

    model_path = REPO_ROOT / "research" / "outputs" / args.prefix / f"{args.prefix}_model_frame.parquet"
    if not model_path.exists():
        print(f"ERROR: {model_path} not found.")
        sys.exit(1)

    model_frame = pl.read_parquet(model_path)
    print(f"Model frame: {model_frame.height:,} rows, {model_frame['symbol'].n_unique()} symbols")
    print(f"Date range:  {model_frame['timestamp'].min()} → {model_frame['timestamp'].max()}")

    folds = generate_folds(args.min_train_years)
    feature_cols, feature_set = resolve_feature_set(args.prefix, args.feature_set, model_frame)
    n_macro = len([c for c in feature_cols if c in MACRO_COLS])
    print(f"Min train years: {args.min_train_years}")
    print(f"Generated folds: {len(folds)}")
    print(f"Feature set: {feature_set}")
    print(f"Features:    {len(feature_cols)} total ({len(feature_cols) - n_macro} engineered + {n_macro} macro/FF5)")

    label = window_label(args.min_train_years)
    out_name = "lgbm_walkforward" if feature_set == "all" else f"lgbm_walkforward_{feature_set}"
    out_name = f"{out_name}_{label}"
    out_dir = REPO_ROOT / "research" / "outputs" / args.prefix / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    folds_to_run = [f for f in folds if args.fold == 0 or f.idx == args.fold]
    all_results: list[dict] = []

    for fold in folds_to_run:
        result = run_fold(fold, model_frame, feature_cols, feature_set, args.min_train_years, out_dir)
        if result:
            all_results.append(result)

    if not all_results:
        print("No folds completed.")
        return

    # Aggregate returns across all folds per strategy
    strategy_returns: dict[str, list[float]] = {}
    summary_rows: list[dict] = []

    for fr in all_results:
        for row in fr.get("results", []):
            summary_rows.append(row)
        for name, rets in fr.get("returns_map", {}).items():
            strategy_returns.setdefault(name, []).extend(rets)

    # Aggregate DSR: LightGBM vs log_volume on concatenated 5-year OOS series
    dsr_prob = float("nan")
    lgbm_sharpe = annualised_sharpe(strategy_returns.get("lgbm", []))
    logvol_sharpe = annualised_sharpe(strategy_returns.get("log_volume", []))

    if len(strategy_returns) >= 2:
        returns_matrix = [strategy_returns[n] for n in ("lgbm", "log_volume") if n in strategy_returns]
        try:
            dsr = deflated_sharpe_ratio(
                returns_matrix,
                frequency="daily",
                correlation_method="effective_rank",
                min_k_eff=2.0,
            )
            dsr_prob = float(dsr.probability)
        except Exception as exc:
            print(f"  DSR computation failed: {exc}")

    print(f"\n{'='*60}")
    print(f"AGGREGATE OOS [{feature_set}, {label}] (all folds concatenated)")
    print(f"  LightGBM   ann. Sharpe: {lgbm_sharpe:.3f}")
    print(f"  log_volume ann. Sharpe: {logvol_sharpe:.3f}")
    print(f"  DSR probability:        {dsr_prob:.3f}")

    # Write report
    report_lines = [
        "# LightGBM Walk-Forward Backtest — 10-Year SP500 (Point-in-Time)",
        "",
        f"Min train years: {args.min_train_years}",
        f"Feature set: {feature_set}",
        f"Folds completed: {len(all_results)}",
        f"Features: {len(feature_cols)} ({len(feature_cols) - n_macro} engineered + {n_macro} macro/FF5)",
        "Objective: rank_xendcg | Strategy: WeeklyLongOnly5 (top-5, 15% each, 5d rebalance)",
        "",
        "## Fold Windows",
        "",
        "| Fold | Train Start | Train End | Test Start | Test End |",
        "| --- | --- | --- | --- | --- |",
    ]
    for fold in folds_to_run:
        report_lines.append(
            f"| {fold.idx} | {fold.train_start} | {fold.train_end} | {fold.test_start} | {fold.test_end} |"
        )

    report_lines += [
        "",
        "## Per-Fold Results",
        "",
        "| Fold | Strategy | Return % | Sharpe | Max DD | Trades | Best Iter |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        bi = row.get("best_iteration")
        report_lines.append(
            f"| {row['fold']} | {row['name']} | {row['total_return_pct']:.2f}"
            f" | {row['sharpe']:.2f} | {row['max_drawdown']:.3f}"
            f" | {row['n_trades']} | {bi if bi is not None else '—'} |"
        )

    report_lines += [
        "",
        "## Aggregate OOS",
        "",
        "| Strategy | Ann. Sharpe |",
        "| --- | ---: |",
        f"| lgbm       | {lgbm_sharpe:.3f} |",
        f"| log_volume | {logvol_sharpe:.3f} |",
        "",
        f"DSR probability (best of 2): **{dsr_prob:.3f}**",
        "",
        "_(DSR > 0.95 → statistically significant at 95% confidence)_",
    ]

    report_text = "\n".join(report_lines) + "\n"
    report_stem = f"{args.prefix}_{out_name}"
    report_path = out_dir / f"{report_stem}_report.md"
    report_path.write_text(report_text)
    print("\n" + report_text)

    summary = {
        "min_train_years": args.min_train_years,
        "generated_folds": [fold.__dict__ for fold in folds],
        "executed_folds": [fold.__dict__ for fold in folds_to_run],
        "folds": [{k: v for k, v in fr.items() if k != "returns_map"} for fr in all_results],
        "feature_set": feature_set,
        "feature_count": len(feature_cols),
        "feature_columns": feature_cols,
        "rows":  summary_rows,
        "aggregate": {
            "lgbm_sharpe":   lgbm_sharpe,
            "logvol_sharpe": logvol_sharpe,
            "dsr_probability": dsr_prob,
        },
    }
    summary_path = out_dir / f"{report_stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
