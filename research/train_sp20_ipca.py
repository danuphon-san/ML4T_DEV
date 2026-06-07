from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
models_src = REPO_ROOT / "models" / "src"
if str(models_src) not in sys.path:
    sys.path.insert(0, str(models_src))

from ml4t.models import (
    BetaLambdaMapper,
    CrossSectionBatch,
    ExpandingMeanFactorForecaster,
    IPCAConfig,
    IPCAModel,
    LatentFactorForecastPipeline,
)


OUTPUT_DIR = REPO_ROOT / "research" / "outputs"
BATCH_PATH = OUTPUT_DIR / "sp20_seed_batch.npz"
METADATA_PATH = OUTPUT_DIR / "sp20_seed_batch_metadata.json"
TOP_SIGNALS_PATH = OUTPUT_DIR / "sp20_seed_top_signals.json"
IPCA_REPORT_PATH = OUTPUT_DIR / "sp20_seed_ipca_report.md"
IPCA_SUMMARY_PATH = OUTPUT_DIR / "sp20_seed_ipca_summary.json"
IPCA_CONFIG = IPCAConfig(
    n_factors=2,
    max_iter=200,
    tol=1e-8,
    factor_ridge=1e-2,
    gamma_ridge=1e-2,
)


def load_batch() -> tuple[CrossSectionBatch, dict[str, object]]:
    arrays = np.load(BATCH_PATH)
    metadata = json.loads(METADATA_PATH.read_text())
    batch = CrossSectionBatch(
        characteristics=arrays["characteristics"],
        returns=arrays["returns"],
        context_features=arrays["context_features"],
        timestamps=tuple(metadata["timestamps"]),
        asset_ids=tuple(metadata["asset_ids"]),
        mask=arrays["mask"],
    )
    return batch, metadata


def split_batch(batch: CrossSectionBatch, train_ratio: float = 0.8) -> tuple[CrossSectionBatch, CrossSectionBatch]:
    n_train = int(batch.n_periods * train_ratio)
    train_slice = slice(0, n_train)
    test_slice = slice(n_train, batch.n_periods)
    train = CrossSectionBatch(
        characteristics=batch.characteristics[train_slice],
        returns=None if batch.returns is None else batch.returns[train_slice],
        context_features=None if batch.context_features is None else batch.context_features[train_slice],
        timestamps=batch.timestamps[train_slice],
        asset_ids=batch.asset_ids,
        mask=None if batch.mask is None else batch.mask[train_slice],
        metadata={"split": "train"},
    )
    test = CrossSectionBatch(
        characteristics=batch.characteristics[test_slice],
        returns=None if batch.returns is None else batch.returns[test_slice],
        context_features=None if batch.context_features is None else batch.context_features[test_slice],
        timestamps=batch.timestamps[test_slice],
        asset_ids=batch.asset_ids,
        mask=None if batch.mask is None else batch.mask[test_slice],
        metadata={"split": "test"},
    )
    return train, test


def scale_batches(
    train_batch: CrossSectionBatch,
    test_batch: CrossSectionBatch,
) -> tuple[CrossSectionBatch, CrossSectionBatch, dict[str, list[float]]]:
    train_chars = np.array(train_batch.characteristics, copy=True)
    test_chars = np.array(test_batch.characteristics, copy=True)
    train_mask = train_batch.mask if train_batch.mask is not None else np.ones(train_chars.shape[:2], dtype=bool)
    test_mask = test_batch.mask if test_batch.mask is not None else np.ones(test_chars.shape[:2], dtype=bool)

    means: list[float] = []
    stds: list[float] = []
    for feature_idx in range(train_chars.shape[2]):
        values = train_chars[:, :, feature_idx][train_mask]
        values = values[np.isfinite(values)]
        mean = float(np.mean(values)) if values.size else 0.0
        std = float(np.std(values)) if values.size else 1.0
        if std == 0.0 or not np.isfinite(std):
            std = 1.0
        means.append(mean)
        stds.append(std)
        train_chars[:, :, feature_idx] = (train_chars[:, :, feature_idx] - mean) / std
        test_chars[:, :, feature_idx] = (test_chars[:, :, feature_idx] - mean) / std

    scaled_train = CrossSectionBatch(
        characteristics=train_chars,
        returns=train_batch.returns,
        context_features=train_batch.context_features,
        timestamps=train_batch.timestamps,
        asset_ids=train_batch.asset_ids,
        mask=train_batch.mask,
        metadata=dict(train_batch.metadata) | {"scaled": True},
    )
    scaled_test = CrossSectionBatch(
        characteristics=test_chars,
        returns=test_batch.returns,
        context_features=test_batch.context_features,
        timestamps=test_batch.timestamps,
        asset_ids=test_batch.asset_ids,
        mask=test_batch.mask,
        metadata=dict(test_batch.metadata) | {"scaled": True},
    )
    return scaled_train, scaled_test, {"feature_means": means, "feature_stds": stds}


def evaluate_prediction(
    predicted_returns: np.ndarray,
    realized_returns: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    valid = mask & np.isfinite(predicted_returns) & np.isfinite(realized_returns)
    pred = predicted_returns[valid]
    actual = realized_returns[valid]
    if pred.size == 0:
        raise ValueError("No valid predicted/realized return pairs for evaluation")

    mse = float(np.mean((pred - actual) ** 2))
    mae = float(np.mean(np.abs(pred - actual)))
    corr = float(np.corrcoef(pred, actual)[0, 1]) if pred.size > 1 else float("nan")

    long_short_returns: list[float] = []
    for t in range(predicted_returns.shape[0]):
        valid_t = mask[t] & np.isfinite(predicted_returns[t]) & np.isfinite(realized_returns[t])
        if valid_t.sum() < 4:
            continue
        pred_t = predicted_returns[t, valid_t]
        actual_t = realized_returns[t, valid_t]
        n_bucket = max(1, valid_t.sum() // 5)
        order = np.argsort(pred_t)
        bottom = actual_t[order[:n_bucket]]
        top = actual_t[order[-n_bucket:]]
        long_short_returns.append(float(np.mean(top) - np.mean(bottom)))

    spread_mean = float(np.mean(long_short_returns)) if long_short_returns else float("nan")
    spread_t = (
        float(np.mean(long_short_returns) / (np.std(long_short_returns, ddof=1) / np.sqrt(len(long_short_returns))))
        if len(long_short_returns) > 1 and np.std(long_short_returns, ddof=1) > 0
        else float("nan")
    )
    return {
        "mse": mse,
        "mae": mae,
        "corr": corr,
        "long_short_mean": spread_mean,
        "long_short_t_stat": spread_t,
        "n_valid_pairs": float(pred.size),
        "n_test_periods": float(predicted_returns.shape[0]),
    }


def write_report(summary: dict[str, object]) -> None:
    lines = [
        "# SP20 IPCA Report",
        "",
        f"Train periods: `{summary['train_periods']}`",
        f"Test periods: `{summary['test_periods']}`",
        f"Top benchmark signals: `{', '.join(summary['top_signal_names'])}`",
        "",
        "## Structural Fit",
        "",
        f"- Config: `n_factors={summary['config']['n_factors']}, factor_ridge={summary['config']['factor_ridge']}, gamma_ridge={summary['config']['gamma_ridge']}, tol={summary['config']['tol']}`",
        f"- Converged: `{summary['fit']['converged']}`",
        f"- Train MSE: `{summary['fit']['train_mse']:.6f}`",
        f"- Mean abs return: `{summary['fit']['mean_abs_return']:.6f}`",
        "",
        "## Test Prediction",
        "",
        f"- MSE: `{summary['test']['mse']:.6f}`",
        f"- MAE: `{summary['test']['mae']:.6f}`",
        f"- Correlation: `{summary['test']['corr']:.4f}`",
        f"- Long-short mean: `{summary['test']['long_short_mean']:.6f}`",
        f"- Long-short t-stat: `{summary['test']['long_short_t_stat']:.2f}`",
    ]
    IPCA_REPORT_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    batch, metadata = load_batch()
    train_batch, test_batch = split_batch(batch, train_ratio=0.8)
    train_batch, test_batch, scaling = scale_batches(train_batch, test_batch)
    top_signals = json.loads(TOP_SIGNALS_PATH.read_text()) if TOP_SIGNALS_PATH.exists() else []

    pipeline = LatentFactorForecastPipeline(
        model=IPCAModel(IPCA_CONFIG),
        forecaster=ExpandingMeanFactorForecaster(),
        mapper=BetaLambdaMapper(),
    )
    fit_result = pipeline.fit(train_batch)
    prediction = pipeline.predict(test_batch)

    if test_batch.returns is None or test_batch.mask is None:
        raise ValueError("Test batch must include returns and mask")

    test_metrics = evaluate_prediction(
        predicted_returns=prediction.asset_forecast.expected_returns,
        realized_returns=test_batch.returns,
        mask=test_batch.mask,
    )
    summary = {
        "train_periods": len(train_batch.timestamps),
        "test_periods": len(test_batch.timestamps),
        "feature_cols": metadata["feature_cols"],
        "context_cols": metadata["context_cols"],
        "top_signal_names": [row["signal"] for row in top_signals[:5]],
        "config": {
            "n_factors": IPCA_CONFIG.n_factors,
            "max_iter": IPCA_CONFIG.max_iter,
            "tol": IPCA_CONFIG.tol,
            "factor_ridge": IPCA_CONFIG.factor_ridge,
            "gamma_ridge": IPCA_CONFIG.gamma_ridge,
        },
        "scaling": scaling,
        "fit": {
            "converged": fit_result.structural_fit.converged,
            "train_mse": float(fit_result.structural_fit.train_metrics["train_mse"]),
            "mean_abs_return": float(fit_result.structural_fit.train_metrics["mean_abs_return"]),
        },
        "test": test_metrics,
    }
    IPCA_SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    write_report(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
