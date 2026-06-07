"""Cluster the selected signals by cross-sectional rank-correlation.

For each date, rank signals across assets. Then compute pairwise Spearman
correlations across all dates. Group signals with |r| > threshold into clusters
and pick the cluster representative with the highest spread_t_21d.

Usage:
    uv run python cluster_signals.py --prefix sp500_full --threshold 0.8
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]

NON_FEATURE_COLS = frozenset(
    [
        "timestamp", "symbol", "open", "high", "low", "close", "volume",
        "ret_1d", "ret_1d_fwd", "ret_5d", "log_volume",
        "label", "label_return", "label_bars", "label_duration", "barrier_hit",
        "Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF",
        "DGS2", "DGS5", "DGS10", "DGS30", "YIELD_CURVE_SLOPE", "YIELD_CURVE_5_10",
    ]
)


def build_signal_matrix(model_frame: pl.DataFrame, signal_names: list[str]) -> np.ndarray:
    """Build (dates × assets × signals) rank matrix, then flatten to (dates × signals)
    by computing cross-sectional ranks per date. Returns (n_dates × n_signals) array."""
    dates = model_frame["timestamp"].unique().sort()
    n_dates = len(dates)
    n_signals = len(signal_names)

    # For correlation we need a (n_obs, n_signals) matrix of cross-sectional ranks.
    # Stack all (date, symbol) rows — already one row per date×symbol.
    # Compute cross-sectional rank per date for each signal.
    ranked_cols = []
    for sig in signal_names:
        col = pl.col(sig) if not sig.startswith("-") else -pl.col(sig[1:])
        expr = col.rank(method="average").over("timestamp").alias(sig)
        ranked_cols.append(expr)

    ranked = model_frame.select(["timestamp", "symbol"] + ranked_cols)
    # Pivot to wide: rows=timestamps, cols=symbols; one matrix per signal
    # Instead, just use the long frame directly — (n_obs, n_signals) is fine for corr
    signal_mat = ranked.select(signal_names).to_numpy().astype(np.float32)
    return signal_mat


def spearman_corr_matrix(mat: np.ndarray) -> np.ndarray:
    """Fast Spearman correlation via Pearson on already-ranked data."""
    # mat is (n_obs, n_signals), already rank-transformed per date via polars
    # Standardise columns then compute Pearson
    m = mat - np.nanmean(mat, axis=0, keepdims=True)
    norms = np.sqrt(np.nansum(m ** 2, axis=0, keepdims=True))
    norms[norms == 0] = 1.0
    m = m / norms
    # Replace NaN with 0 before dot product
    m = np.nan_to_num(m)
    corr = (m.T @ m)
    return corr


def cluster_by_correlation(
    signal_names: list[str],
    corr: np.ndarray,
    quality: dict[str, float],
    threshold: float,
) -> list[dict]:
    """Greedy clustering: assign each signal to the first cluster whose
    representative it correlates with above threshold (absolute value)."""
    n = len(signal_names)
    assigned = [False] * n
    clusters = []

    # Sort by quality descending so the best signal seeds each cluster
    order = sorted(range(n), key=lambda i: quality.get(signal_names[i], 0), reverse=True)

    for seed_idx in order:
        if assigned[seed_idx]:
            continue
        cluster_members = [seed_idx]
        assigned[seed_idx] = True
        for i in order:
            if assigned[i]:
                continue
            if abs(corr[seed_idx, i]) >= threshold:
                cluster_members.append(i)
                assigned[i] = True
        rep = signal_names[seed_idx]
        clusters.append(
            {
                "representative": rep,
                "spread_t_21d": quality.get(rep, 0),
                "size": len(cluster_members),
                "members": [signal_names[i] for i in cluster_members],
            }
        )
    return clusters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="sp500_full")
    parser.add_argument("--threshold", type=float, default=0.8)
    args = parser.parse_args()

    output_dir = REPO_ROOT / "research" / "outputs" / args.prefix

    # Load screen results to get quality scores
    screen = pl.read_parquet(output_dir / f"{args.prefix}_all_signal_screen.parquet")
    selected = screen.filter(
        (pl.col("spread_t_21d") > 2.0)
        & (pl.col("spread_21d") > 0)
        & (pl.col("ic_21d") > 0)
        & (pl.col("monotonicity_21d") >= 0.5)
        & pl.col("spread_t_21d").is_not_nan()
        & pl.col("spread_21d").is_not_nan()
    ).sort("spread_t_21d", descending=True)

    signal_names = selected["signal"].to_list()
    quality = dict(zip(selected["signal"].to_list(), selected["spread_t_21d"].to_list()))
    print(f"Signals to cluster: {len(signal_names)}")

    # Load model_frame and resolve columns
    model_frame = pl.read_parquet(output_dir / f"{args.prefix}_model_frame.parquet")

    # Separate positive and negative signals; check base columns exist
    resolvable = []
    for sig in signal_names:
        base = sig.lstrip("-")
        if base in model_frame.columns:
            resolvable.append(sig)
        else:
            print(f"  SKIP {sig}: column '{base}' not in model_frame")
    print(f"Resolvable signals: {len(resolvable)}")

    # Build rank matrix
    print("Computing cross-sectional ranks...")
    signal_mat = build_signal_matrix(model_frame, resolvable)
    print(f"Signal matrix shape: {signal_mat.shape}")

    # Correlation matrix
    print("Computing correlation matrix...")
    corr = spearman_corr_matrix(signal_mat)

    # Cluster
    print(f"Clustering at |r| >= {args.threshold}...")
    clusters = cluster_by_correlation(resolvable, corr, quality, args.threshold)

    representatives = [c["representative"] for c in clusters]
    print(f"\nClusters: {len(clusters)}  →  {len(representatives)} representatives")
    print(f"Signals consolidated from {len(resolvable)} → {len(representatives)}\n")

    for c in clusters:
        if c["size"] > 1:
            print(f"  [{c['size']}] {c['representative']} (t={c['spread_t_21d']:.1f})")
            for m in c["members"]:
                if m != c["representative"]:
                    print(f"        ↳ {m}")
        else:
            print(f"  [1] {c['representative']} (t={c['spread_t_21d']:.1f})")

    # Save
    result = {
        "threshold": args.threshold,
        "n_input": len(resolvable),
        "n_clusters": len(clusters),
        "representatives": representatives,
        "clusters": clusters,
    }
    out_path = output_dir / f"{args.prefix}_signal_clusters.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
