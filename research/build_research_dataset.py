from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
for repo in ("engineer", "models"):
    src_path = REPO_ROOT / repo / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

from ml4t.engineer import compute_features, create_dataset_builder
from ml4t.engineer.config import LabelingConfig
from ml4t.engineer.labeling import triple_barrier_labels
from ml4t.models import cross_section_batch_from_long_frame

from research_universe import FACTOR_FILE, FEATURE_CONFIG, MACRO_FILE, STORAGE_ROOT, universe_spec


BASE_DERIVED_CHARACTERISTICS = ["log_volume", "ret_5d"]

# Features that produce multiple output columns instead of a single column with the feature name.
MULTI_OUTPUT_FEATURE_COLS: dict[str, list[str]] = {
    "bollinger_bands": ["bollinger_bands_upper", "bollinger_bands_middle", "bollinger_bands_lower"],
    "aroon": ["aroon_down", "aroon_up"],
    "donchian_channels": ["donchian_channels_0", "donchian_channels_1", "donchian_channels_2"],
    "higher_moments": [
        "higher_moments_skewness",
        "higher_moments_kurtosis",
        "higher_moments_hyperskewness",
        "higher_moments_hyperkurtosis",
    ],
    "maximum_drawdown": [
        "maximum_drawdown_max_drawdown",
        "maximum_drawdown_max_duration",
        "maximum_drawdown_current_drawdown",
        "maximum_drawdown_time_underwater",
    ],
    "volatility_regime_probability": [
        "volatility_regime_probability_prob_low_vol",
        "volatility_regime_probability_prob_med_vol",
        "volatility_regime_probability_prob_high_vol",
        "volatility_regime_probability_current_vol",
    ],
    "risk_adjusted_returns": [
        "risk_adjusted_returns_sharpe_ratio",
        "risk_adjusted_returns_sortino_ratio",
        "risk_adjusted_returns_calmar_ratio",
        "risk_adjusted_returns_omega_ratio",
    ],
}
CONTEXT_COLS = [
    "Mkt-RF",
    "SMB",
    "HML",
    "RMW",
    "CMA",
    "RF",
    "DGS2",
    "DGS5",
    "DGS10",
    "DGS30",
    "YIELD_CURVE_SLOPE",
    "YIELD_CURVE_5_10",
]
DEFAULT_DIAGNOSTIC_FACTOR = "-atr"


def read_symbols(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def load_feature_names(path: Path) -> list[str]:
    config = yaml.safe_load(path.read_text())
    return [item["name"] for item in config.get("features", [])]


def resolve_characteristic_cols(feature_names: list[str]) -> list[str]:
    characteristic_cols: list[str] = []
    for feature_name in feature_names:
        if feature_name in MULTI_OUTPUT_FEATURE_COLS:
            characteristic_cols.extend(MULTI_OUTPUT_FEATURE_COLS[feature_name])
        else:
            characteristic_cols.append(feature_name)
    if "sma" in feature_names:
        characteristic_cols.append("sma_gap_20")
    if "ema" in feature_names:
        characteristic_cols.append("ema_gap_20")
    characteristic_cols.extend(BASE_DERIVED_CHARACTERISTICS)
    return characteristic_cols


def load_symbol_history(symbol: str, storage_root: Path) -> pl.DataFrame:
    parquet_paths = sorted(storage_root.glob(f"yahoo_daily_{symbol}/year=*/month=*/data.parquet"))
    parquet_paths.extend(sorted(storage_root.glob(f"{symbol}/year=*/month=*/data.parquet")))
    parquet_paths.extend(sorted(storage_root.glob(f"equities_daily_{symbol}/year=*/month=*/data.parquet")))
    parquet_paths = sorted(set(parquet_paths))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found for {symbol} under {storage_root}")
    frame = pl.concat([pl.read_parquet(path) for path in parquet_paths], how="vertical")
    return (
        frame.unique(subset=["timestamp", "symbol"], maintain_order=True)
        .sort("timestamp")
        .with_columns(pl.col("timestamp").cast(pl.Datetime("us")))
    )


def load_equity_panel(symbols: list[str], storage_root: Path) -> tuple[pl.DataFrame, list[str], list[str]]:
    frames: list[pl.DataFrame] = []
    available_symbols: list[str] = []
    missing_symbols: list[str] = []
    for symbol in symbols:
        try:
            frames.append(load_symbol_history(symbol, storage_root))
            available_symbols.append(symbol)
        except FileNotFoundError:
            missing_symbols.append(symbol)
    if not frames:
        raise FileNotFoundError(f"No parquet files found for any requested symbol under {storage_root}")
    panel = pl.concat(frames, how="vertical").sort(["symbol", "timestamp"])
    return panel, available_symbols, missing_symbols


def load_factor_frame(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path).with_columns(pl.col("timestamp").cast(pl.Datetime("us"))).sort("timestamp")


def load_macro_frame(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path).with_columns(pl.col("timestamp").cast(pl.Datetime("us"))).sort("timestamp")


def compute_symbol_features(panel: pl.DataFrame, feature_config: Path) -> pl.DataFrame:
    featured_frames: list[pl.DataFrame] = []
    for symbol_frame in panel.partition_by("symbol", maintain_order=True):
        prepared = symbol_frame.sort("timestamp").with_columns(
            pl.col("close").pct_change().alias("returns")
        )
        computed = compute_features(prepared, feature_config)
        # Unpack any struct columns generically (e.g. bollinger_bands, aroon)
        for col_name in list(computed.columns):
            col_dtype = computed[col_name].dtype
            if isinstance(col_dtype, pl.Struct):
                computed = computed.with_columns(
                    [pl.col(col_name).struct.field(f.name).alias(f"{col_name}_{f.name}") for f in col_dtype.fields]
                ).drop(col_name)
        featured_frames.append(computed)
    return pl.concat(featured_frames, how="vertical").sort(["symbol", "timestamp"])


def enrich_returns(panel: pl.DataFrame) -> pl.DataFrame:
    return panel.with_columns(
        [
            (pl.col("close").pct_change().over("symbol")).alias("ret_1d"),
            (pl.col("close").pct_change(5).over("symbol")).alias("ret_5d"),
            ((pl.col("close").shift(-1).over("symbol") / pl.col("close")) - 1.0).alias("ret_1d_fwd"),
            ((pl.col("close") / pl.col("sma")) - 1.0).alias("sma_gap_20"),
            ((pl.col("close") / pl.col("ema")) - 1.0).alias("ema_gap_20"),
            pl.col("volume").log().alias("log_volume"),
        ]
    )


def merge_context(panel: pl.DataFrame, factor_frame: pl.DataFrame, macro_frame: pl.DataFrame) -> pl.DataFrame:
    return panel.join(factor_frame, on="timestamp", how="left").join(macro_frame, on="timestamp", how="left").sort(["symbol", "timestamp"])


def apply_labels(panel: pl.DataFrame) -> pl.DataFrame:
    config = LabelingConfig.triple_barrier(upper_barrier=0.03, lower_barrier=0.02, max_holding_period=10)
    return triple_barrier_labels(
        panel,
        config=config,
        price_col="close",
        high_col="high",
        low_col="low",
        timestamp_col="timestamp",
        group_col="symbol",
    ).sort(["symbol", "timestamp"])


def build_model_frame(labeled_panel: pl.DataFrame, characteristic_cols: list[str]) -> pl.DataFrame:
    required_cols = characteristic_cols + CONTEXT_COLS + ["ret_1d_fwd", "label"]
    selected = (
        labeled_panel.drop_nulls(subset=required_cols)
        .sort(["timestamp", "symbol"])
        .select(
            [
                "timestamp",
                "symbol",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "ret_1d",
                "ret_1d_fwd",
                "label",
                "label_return",
                "label_bars",
                "label_duration",
                "barrier_hit",
                *characteristic_cols,
                *CONTEXT_COLS,
            ]
        )
    )
    finite_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "ret_1d",
        "ret_1d_fwd",
        "label_return",
        *characteristic_cols,
        *CONTEXT_COLS,
    ]
    return selected.filter(pl.all_horizontal([pl.col(col).is_finite() for col in finite_cols]))


def build_signal_inputs(model_frame: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    factor = model_frame.select(
        [
            pl.col("timestamp").alias("date"),
            pl.col("symbol").alias("asset"),
            (-pl.col("atr")).alias("factor"),
        ]
    )
    prices = model_frame.select(
        [
            pl.col("timestamp").alias("date"),
            pl.col("symbol").alias("asset"),
            pl.col("close").alias("price"),
        ]
    )
    return factor, prices


def run_diagnostic_report(factor_path: Path, prices_path: Path, output_dir: Path, prefix: str) -> dict[str, object]:
    result_json_path = output_dir / f"{prefix}_signal_analysis.json"
    report_path = output_dir / f"{prefix}_diagnostic_report.md"
    code = f"""
import json
from pathlib import Path
import polars as pl
from ml4t.diagnostic import analyze_signal
factor = pl.read_parquet({str(factor_path)!r})
prices = pl.read_parquet({str(prices_path)!r})
result = analyze_signal(factor=factor, prices=prices, periods=(1, 5, 21), quantiles=5, min_assets=10)
result.to_json({str(result_json_path)!r})
period_keys = ("1D", "5D", "21D")
metrics = {{
    period: {{
        "ic": result.ic[period],
        "ic_t_stat": result.ic_t_stat[period],
        "ic_p_value": result.ic_p_value[period],
        "spread": result.spread[period],
        "spread_t_stat": result.spread_t_stat[period],
        "spread_p_value": result.spread_p_value[period],
        "monotonicity": result.monotonicity[period],
        "turnover": result.turnover[period],
    }}
    for period in period_keys
}}
report_lines = [
    "# Diagnostic Report",
    "",
    f"Universe: `{prefix}`",
    f"Default factor: `{DEFAULT_DIAGNOSTIC_FACTOR}`",
    f"Assets: `{{result.n_assets}}`",
    f"Dates: `{{result.n_dates}}`",
    f"Date range: `{{result.date_range[0]}}` to `{{result.date_range[1]}}`",
    "",
    "## Summary",
    "",
    "```text",
    result.summary().strip(),
    "```",
]
Path({str(report_path)!r}).write_text("\\n".join(report_lines) + "\\n")
print(json.dumps({{"factor_name":"{DEFAULT_DIAGNOSTIC_FACTOR}","json_path":{str(result_json_path)!r},"report_path":{str(report_path)!r},"metrics":metrics,"summary":result.summary()}}, indent=2))
"""
    completed = subprocess.run(["uv", "run", "python", "-c", code], cwd=REPO_ROOT / "diagnostic", check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


def save_cross_section_batch(
    model_frame: pl.DataFrame,
    output_dir: Path,
    prefix: str,
    characteristic_cols: list[str],
) -> dict[str, object]:
    batch = cross_section_batch_from_long_frame(
        model_frame,
        feature_cols=characteristic_cols,
        return_col="ret_1d_fwd",
        context_cols=CONTEXT_COLS,
        timestamp_col="timestamp",
        entity_col="symbol",
        metadata={"universe": prefix},
    )
    np.savez(
        output_dir / f"{prefix}_batch.npz",
        characteristics=batch.characteristics,
        returns=batch.returns,
        context_features=batch.context_features,
        mask=batch.mask,
    )
    metadata = {
        "timestamps": [str(timestamp) for timestamp in batch.timestamps],
        "asset_ids": list(batch.asset_ids),
        "feature_cols": characteristic_cols,
        "context_cols": CONTEXT_COLS,
        "n_periods": batch.n_periods,
        "n_slots": batch.n_assets,
    }
    (output_dir / f"{prefix}_batch_metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="sp20_seed")
    parser.add_argument("--feature-config", default=str(FEATURE_CONFIG))
    args = parser.parse_args()
    spec = universe_spec(args.prefix)
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    feature_config = Path(args.feature_config)
    feature_names = load_feature_names(feature_config)
    characteristic_cols = resolve_characteristic_cols(feature_names)

    symbols = read_symbols(spec.symbol_file)
    raw_panel, available_symbols, missing_symbols = load_equity_panel(symbols, STORAGE_ROOT)
    raw_panel.write_parquet(spec.output_dir / f"{spec.prefix}_panel.parquet")
    featured_panel = compute_symbol_features(raw_panel, feature_config)
    featured_panel.write_parquet(spec.output_dir / f"{spec.prefix}_feature_panel.parquet")
    merged_panel = merge_context(enrich_returns(featured_panel), load_factor_frame(FACTOR_FILE), load_macro_frame(MACRO_FILE))
    labeled_panel = apply_labels(merged_panel)
    labeled_panel.write_parquet(spec.output_dir / f"{spec.prefix}_labeled_panel.parquet")
    model_frame = build_model_frame(labeled_panel, characteristic_cols)
    model_frame.write_parquet(spec.output_dir / f"{spec.prefix}_model_frame.parquet")

    factor_input, prices_input = build_signal_inputs(model_frame)
    factor_path = spec.output_dir / f"{spec.prefix}_signal_factor.parquet"
    prices_path = spec.output_dir / f"{spec.prefix}_signal_prices.parquet"
    factor_input.write_parquet(factor_path)
    prices_input.write_parquet(prices_path)
    diagnostic_summary = run_diagnostic_report(factor_path, prices_path, spec.output_dir, spec.prefix)

    builder = create_dataset_builder(
        features=model_frame.select(characteristic_cols + CONTEXT_COLS),
        labels=model_frame["label"],
        dates=model_frame["timestamp"],
        scaler="robust",
    )
    X_train, X_test, y_train, y_test = builder.train_test_split(train_size=0.8, shuffle=False)
    batch_metadata = save_cross_section_batch(model_frame, spec.output_dir, spec.prefix, characteristic_cols)
    summary = {
        "prefix": spec.prefix,
        "symbols": symbols,
        "available_symbols": available_symbols,
        "missing_symbols": missing_symbols,
        "raw_rows": raw_panel.height,
        "feature_rows": featured_panel.height,
        "labeled_rows": labeled_panel.height,
        "model_rows": model_frame.height,
        "date_min": str(model_frame["timestamp"].min()),
        "date_max": str(model_frame["timestamp"].max()),
        "feature_config": str(feature_config),
        "feature_names": feature_names,
        "characteristic_cols": characteristic_cols,
        "context_cols": CONTEXT_COLS,
        "label_counts": model_frame.group_by("label").len().sort("label").to_dicts(),
        "train_rows": X_train.height,
        "test_rows": X_test.height,
        "train_label_counts": y_train.value_counts().sort("label").to_dicts(),
        "test_label_counts": y_test.value_counts().sort("label").to_dicts(),
        "batch": batch_metadata,
        "default_diagnostic_factor": DEFAULT_DIAGNOSTIC_FACTOR,
        "diagnostic": diagnostic_summary,
    }
    (spec.output_dir / f"{spec.prefix}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
