from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "models" / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "models" / "src"))

from ml4t.models import (
    BetaLambdaMapper,
    CrossSectionBatch,
    ExpandingMeanFactorForecaster,
    IPCAConfig,
    IPCAModel,
    LatentFactorForecastPipeline,
    PersistentPanelBatch,
    RPPCAConfig,
    RPPCAModel,
    SAEConfig,
    SAEModel,
)

from research_universe import universe_spec


IPCA_CONFIG = IPCAConfig(
    n_factors=2,
    max_iter=200,
    tol=1e-8,
    factor_ridge=1e-2,
    gamma_ridge=1e-2,
)
RPPCA_CONFIG = RPPCAConfig(
    n_factors=2,
    gamma=5.0,
    base_moment="covariance",
    scale_by_asset_volatility=True,
    normalize_loadings="unit_length",
    orthogonalize_factors=True,
)
SAE_CONFIG = SAEConfig(
    bottleneck_dim=16,
    aux_hidden_dim=16,
    main_hidden_units=(64, 32, 32, 16),
    dropout_rates=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    noise_std=0.0,
    alpha=1.0,
    aux_weight=0.25,
    n_epochs=20,
    batch_size=4096,
    checkpoint_interval=5,
    lr=5e-4,
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _test_frame(model_frame: pl.DataFrame) -> tuple[pl.DataFrame, list[object]]:
    dates = model_frame.select("timestamp").unique().sort("timestamp")["timestamp"].to_list()
    n_train = int(len(dates) * 0.8)
    test_dates = dates[n_train:]
    test_frame = model_frame.filter(pl.col("timestamp").is_in(test_dates)).sort(["timestamp", "symbol"])
    return test_frame, test_dates


def _load_batch(prefix: str) -> tuple[CrossSectionBatch, dict[str, object]]:
    spec = universe_spec(prefix)
    arrays = np.load(spec.output_dir / f"{spec.prefix}_batch.npz")
    metadata = json.loads((spec.output_dir / f"{spec.prefix}_batch_metadata.json").read_text())
    batch = CrossSectionBatch(
        characteristics=arrays["characteristics"],
        returns=arrays["returns"],
        context_features=arrays["context_features"],
        timestamps=tuple(metadata["timestamps"]),
        asset_ids=tuple(metadata["asset_ids"]),
        mask=arrays["mask"],
    )
    return batch, metadata


def _split_batch(batch: CrossSectionBatch, train_ratio: float = 0.8) -> tuple[CrossSectionBatch, CrossSectionBatch]:
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


def _scale_batches(
    train_batch: CrossSectionBatch,
    test_batch: CrossSectionBatch,
) -> tuple[CrossSectionBatch, CrossSectionBatch]:
    train_chars = np.array(train_batch.characteristics, copy=True)
    test_chars = np.array(test_batch.characteristics, copy=True)
    train_mask = train_batch.mask if train_batch.mask is not None else np.ones(train_chars.shape[:2], dtype=bool)
    for feature_idx in range(train_chars.shape[2]):
        values = train_chars[:, :, feature_idx][train_mask]
        values = values[np.isfinite(values)]
        mean = float(np.mean(values)) if values.size else 0.0
        std = float(np.std(values)) if values.size else 1.0
        if std == 0.0 or not np.isfinite(std):
            std = 1.0
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
    return scaled_train, scaled_test


def _persistent_panels_from_frame(
    model_frame: pl.DataFrame,
    train_ratio: float = 0.8,
) -> tuple[PersistentPanelBatch, PersistentPanelBatch]:
    dates = model_frame.select("timestamp").unique().sort("timestamp")["timestamp"].to_list()
    n_train = int(len(dates) * train_ratio)
    train_dates = dates[:n_train]
    test_dates = dates[n_train:]
    asset_ids = (
        model_frame.select("symbol")
        .unique()
        .sort("symbol")["symbol"]
        .to_list()
    )
    train_panel = (
        model_frame.filter(pl.col("timestamp").is_in(train_dates))
        .select(["timestamp", "symbol", "ret_1d_fwd"])
        .pivot(on="symbol", index="timestamp", values="ret_1d_fwd")
        .sort("timestamp")
    )
    missing_assets = [asset_id for asset_id in asset_ids if asset_id not in train_panel.columns]
    if missing_assets:
        train_panel = train_panel.with_columns(
            [pl.lit(None, dtype=pl.Float64).alias(asset_id) for asset_id in missing_assets]
        )
    train_returns = train_panel.select(asset_ids).to_numpy()
    train = PersistentPanelBatch(
        returns=train_returns,
        timestamps=tuple(train_dates),
        asset_ids=tuple(asset_ids),
    )
    future = PersistentPanelBatch(
        timestamps=tuple(test_dates),
        asset_ids=tuple(asset_ids),
    )
    return train, future


def _split_train_validation(
    batch: CrossSectionBatch,
    validation_ratio: float = 0.2,
) -> tuple[CrossSectionBatch, CrossSectionBatch]:
    n_val = max(1, int(batch.n_periods * validation_ratio))
    n_val = min(n_val, batch.n_periods - 1)
    train_slice = slice(0, batch.n_periods - n_val)
    val_slice = slice(batch.n_periods - n_val, batch.n_periods)
    train = CrossSectionBatch(
        characteristics=batch.characteristics[train_slice],
        returns=None if batch.returns is None else batch.returns[train_slice],
        context_features=None if batch.context_features is None else batch.context_features[train_slice],
        timestamps=batch.timestamps[train_slice],
        asset_ids=batch.asset_ids,
        mask=None if batch.mask is None else batch.mask[train_slice],
        metadata=dict(batch.metadata) | {"split": "train_fit"},
    )
    validation = CrossSectionBatch(
        characteristics=batch.characteristics[val_slice],
        returns=None if batch.returns is None else batch.returns[val_slice],
        context_features=None if batch.context_features is None else batch.context_features[val_slice],
        timestamps=batch.timestamps[val_slice],
        asset_ids=batch.asset_ids,
        mask=None if batch.mask is None else batch.mask[val_slice],
        metadata=dict(batch.metadata) | {"split": "train_validation"},
    )
    return train, validation


def build_top_signal_factor(model_frame: pl.DataFrame, signal_name: str) -> pl.DataFrame:
    screen_mod = _load_module("screen_mod_generic", REPO_ROOT / "research" / "screen_signals.py")
    specs = {spec.name: spec.expression for spec in screen_mod.build_signal_specs()}
    if signal_name not in specs:
        raise ValueError(f"Unknown signal spec: {signal_name}")
    test_frame, _ = _test_frame(model_frame)
    return test_frame.select(
        [
            pl.col("timestamp").alias("date"),
            pl.col("symbol").alias("asset"),
            specs[signal_name].alias("factor"),
        ]
    )


def build_ipca_prediction_factor(model_frame: pl.DataFrame, prefix: str) -> pl.DataFrame:
    batch, _ = _load_batch(prefix)
    train_batch, test_batch = _split_batch(batch, train_ratio=0.8)
    train_batch, test_batch = _scale_batches(train_batch, test_batch)

    pipeline = LatentFactorForecastPipeline(
        model=IPCAModel(IPCA_CONFIG),
        forecaster=ExpandingMeanFactorForecaster(),
        mapper=BetaLambdaMapper(),
    )
    pipeline.fit(train_batch)
    prediction = pipeline.predict(test_batch)

    test_frame, test_dates = _test_frame(model_frame)
    rows: list[dict[str, object]] = []
    grouped = test_frame.partition_by("timestamp", maintain_order=True)
    expected = prediction.asset_forecast.expected_returns
    mask = test_batch.mask

    if mask is None:
        raise ValueError("Test batch mask is required for IPCA prediction alignment")
    if len(grouped) != expected.shape[0]:
        raise ValueError("Test frame dates and IPCA prediction periods do not align")

    for t_idx, date in enumerate(test_dates):
        frame_t = grouped[t_idx].sort("symbol")
        preds_t = expected[t_idx]
        valid_positions = np.isfinite(preds_t)
        preds_valid = preds_t[valid_positions]
        if frame_t.height != len(preds_valid):
            raise ValueError("Asset count mismatch between test frame and IPCA predictions")
        symbols = frame_t["symbol"].to_list()
        for symbol, pred in zip(symbols, preds_valid, strict=True):
            rows.append({"date": date, "asset": symbol, "factor": float(pred)})

    return pl.DataFrame(rows).sort(["date", "asset"])


def build_rppca_prediction_factor(model_frame: pl.DataFrame, prefix: str) -> pl.DataFrame:
    train_panel, future_panel = _persistent_panels_from_frame(model_frame, train_ratio=0.8)

    pipeline = LatentFactorForecastPipeline(
        model=RPPCAModel(RPPCA_CONFIG),
        forecaster=ExpandingMeanFactorForecaster(),
        mapper=BetaLambdaMapper(),
    )
    pipeline.fit(train_panel)
    prediction = pipeline.predict(future_panel)

    test_frame, test_dates = _test_frame(model_frame)
    grouped = test_frame.partition_by("timestamp", maintain_order=True)
    expected = prediction.asset_forecast.expected_returns
    asset_index = {asset: idx for idx, asset in enumerate(future_panel.asset_ids)}

    if len(grouped) != expected.shape[0]:
        raise ValueError("Test frame dates and RP-PCA prediction periods do not align")

    rows: list[dict[str, object]] = []
    for t_idx, date in enumerate(test_dates):
        frame_t = grouped[t_idx].sort("symbol")
        symbols = frame_t["symbol"].to_list()
        for symbol in symbols:
            idx = asset_index[symbol]
            pred = expected[t_idx, idx]
            rows.append({"date": date, "asset": symbol, "factor": float(pred)})

    return pl.DataFrame(rows).sort(["date", "asset"])


def build_sae_prediction_factor(model_frame: pl.DataFrame, prefix: str) -> pl.DataFrame:
    batch, _ = _load_batch(prefix)
    train_batch, test_batch = _split_batch(batch, train_ratio=0.8)
    train_batch, test_batch = _scale_batches(train_batch, test_batch)
    fit_batch, validation_batch = _split_train_validation(train_batch, validation_ratio=0.2)

    model = SAEModel(SAE_CONFIG)
    fit_summary = model.fit(fit_batch, validation_batch=validation_batch)
    prediction = model.predict(
        test_batch,
        checkpoint=fit_summary.best_epoch,
    )

    test_frame, test_dates = _test_frame(model_frame)
    grouped = test_frame.partition_by("timestamp", maintain_order=True)
    expected = prediction.signal_values
    mask = test_batch.mask
    if mask is None:
        raise ValueError("Test batch mask is required for SAE prediction alignment")
    if len(grouped) != expected.shape[0]:
        raise ValueError("Test frame dates and SAE prediction periods do not align")

    rows: list[dict[str, object]] = []
    for t_idx, date in enumerate(test_dates):
        frame_t = grouped[t_idx].sort("symbol")
        preds_t = expected[t_idx]
        valid_positions = np.isfinite(preds_t)
        preds_valid = preds_t[valid_positions]
        if frame_t.height != len(preds_valid):
            raise ValueError("Asset count mismatch between test frame and SAE predictions")
        symbols = frame_t["symbol"].to_list()
        for symbol, pred in zip(symbols, preds_valid, strict=True):
            rows.append({"date": date, "asset": symbol, "factor": float(pred)})

    return pl.DataFrame(rows).sort(["date", "asset"])
