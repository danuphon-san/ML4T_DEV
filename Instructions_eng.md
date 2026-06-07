# ML4T Instructions (English)

## 1. Goal

Use the ML4T repos as one pipeline:

```text
data -> engineer -> models -> diagnostic
```

Each stage has a clear role:

- `data`: fetch and store market data
- `engineer`: create features, labels, and training-ready datasets
- `models`: train forecasting, asset-prediction, or portfolio models
- `diagnostic`: validate signal quality, model robustness, and statistical significance

## 2. Verified Local Setup

Root folder:

```bash
cd /Users/mit/Project/ML4T
```

Verified repos:

- `data`
- `engineer`
- `models`
- `diagnostic`

Important macOS note:

```bash
brew install libomp
```

This is required for `diagnostic` because `lightgbm` and `xgboost` need OpenMP.

## 3. Recommended Environment Commands

### `data`

```bash
cd /Users/mit/Project/ML4T/data
uv sync --dev
uv run pytest tests/ -q
```

### `engineer`

Recommended local setup:

```bash
cd /Users/mit/Project/ML4T/engineer
uv sync --python 3.12 --extra ta --extra store
uv run pytest tests/ -q -o addopts=''
```

Why:

- `ta-lib` improves indicator parity/performance coverage
- `duckdb` is used by store-related tests
- `Python 3.12` was the most stable local configuration

### `models`

```bash
cd /Users/mit/Project/ML4T/models
uv sync
uv run pytest tests/ -q
```

### `diagnostic`

```bash
cd /Users/mit/Project/ML4T/diagnostic
uv sync
uv run pytest tests/ -q
```

## 4. Process Flow

## Step A: Data ingestion

Use `ml4t.data` to fetch and store raw market data.

Minimal example:

```python
from ml4t.data import DataManager

dm = DataManager()
prices = dm.fetch("AAPL", "2020-01-01", "2024-12-31", provider="yahoo")
```

CLI example:

```bash
ml4t-data fetch -s AAPL --provider yahoo --start 2020-01-01
```

Typical output from this phase:

- OHLCV price history
- stored local datasets
- validated provider output

Use this phase when you need:

- equities
- ETFs
- crypto
- futures
- macro series
- factor data

## Step B: Feature engineering and labeling

Use `ml4t.engineer` after raw data is available.

### B1. Compute features

```python
from ml4t.engineer import compute_features

features = compute_features(
    prices,
    [
        "rsi",
        "macd",
        "atr",
        "obv",
    ],
)
```

You can also use parameterized specs:

```python
features = compute_features(
    prices,
    [
        {"name": "rsi", "params": {"period": 14}},
        {"name": "sma", "params": {"period": 50}},
    ],
)
```

### B2. Explore available features

```python
from ml4t.engineer import feature_catalog

print(feature_catalog.categories())
print(feature_catalog.list(category="momentum"))
print(feature_catalog.describe("rsi"))
```

### B3. Create labels

```python
from ml4t.engineer.config import LabelingConfig
from ml4t.engineer.labeling import triple_barrier_labels

config = LabelingConfig.triple_barrier(
    upper_barrier=0.03,
    lower_barrier=0.02,
    max_holding_period=10,
)

labeled = triple_barrier_labels(features, config=config)
```

Other useful labeling functions:

- `atr_triple_barrier_labels`
- `fixed_time_horizon_labels`
- `trend_scanning_labels`
- `rolling_percentile_binary_labels`
- `meta_labels`

### B4. Build ML-ready datasets

```python
from ml4t.engineer import create_dataset_builder

builder = create_dataset_builder(
    features=features,
    labels=labeled["label"],
)
```

Typical output from this phase:

- engineered features
- target labels
- train/test-ready datasets

## Step C: Model training

Use `ml4t.models` after features and labels are ready.

There are three main families:

1. latent-factor models
2. direct asset-prediction models
3. portfolio models

### C1. Latent-factor forecasting pipeline

```python
import numpy as np

from ml4t.models import (
    BetaLambdaMapper,
    CrossSectionBatch,
    ExpandingMeanFactorForecaster,
    IPCAConfig,
    IPCAModel,
    LatentFactorForecastPipeline,
)

batch = CrossSectionBatch(
    characteristics=np.random.randn(36, 250, 12),
    returns=np.random.randn(36, 250),
    timestamps=tuple(range(36)),
)

pipeline = LatentFactorForecastPipeline(
    model=IPCAModel(IPCAConfig(n_factors=3)),
    forecaster=ExpandingMeanFactorForecaster(),
    mapper=BetaLambdaMapper(),
)

fit_result = pipeline.fit(batch)
prediction = pipeline.predict(batch)
```

Common structural models:

- `PCAModel`
- `RPPCAModel`
- `IPCAModel`
- `CAEModel`

### C2. Direct asset prediction

```python
import numpy as np

from ml4t.models import CrossSectionBatch, SAEConfig, SAEModel

batch = CrossSectionBatch(
    characteristics=np.random.randn(24, 200, 20),
    returns=np.random.randn(24, 200),
    timestamps=tuple(range(24)),
)

model = SAEModel(SAEConfig(n_epochs=20, checkpoint_interval=5))
fit_summary = model.fit(batch, validation_batch=batch)
signals = model.predict(batch)
```

### C3. Portfolio models

```python
import numpy as np

from ml4t.models import LSTMPortfolioConfig, LSTMPortfolioModel, PortfolioSequenceBatch

batch = PortfolioSequenceBatch(
    features=np.random.randn(8, 63, 30, 10),
    returns=np.random.randn(8, 63, 30),
    timestamps=tuple(range(63)),
    asset_ids=tuple(f"asset_{i}" for i in range(30)),
)

model = LSTMPortfolioModel(
    LSTMPortfolioConfig(max_iters=20, checkpoint_every=5, default_checkpoint=20)
)
model.fit(batch, validation_batch=batch)
portfolio_prediction = model.predict(batch)
```

### C4. Export model outputs for downstream usage

```python
from ml4t.models import predictions_frame_from_asset_forecast, write_backtest_frames

frame = predictions_frame_from_asset_forecast(prediction.asset_forecast)
written = write_backtest_frames("artifacts/run_001", predictions=frame)
```

Typical output from this phase:

- expected returns
- asset signals
- latent factors
- portfolio weights
- artifacts for backtest/diagnostic

## Step D: Diagnostics and validation

Use `ml4t.diagnostic` after you have predictions, signals, trades, or portfolio outputs.

### D1. Validated cross-validation

```python
from ml4t.diagnostic import ValidatedCrossValidation
from ml4t.diagnostic.config import ValidatedCrossValidationConfig

config = ValidatedCrossValidationConfig(
    n_groups=10,
    n_test_groups=2,
    embargo_pct=0.01,
    label_horizon=5,
)
vcv = ValidatedCrossValidation(config=config)
result = vcv.fit_evaluate(X, y, model, times=times)
```

Use this when you need:

- CPCV
- purge/embargo logic
- deflated Sharpe style validation

### D2. Signal analysis

```python
from ml4t.diagnostic import analyze_signal

signal_result = analyze_signal(
    factor=factor_df,
    prices=prices_df,
    periods=(1, 5, 21),
)
```

### D3. Feature diagnostics

```python
from ml4t.diagnostic.config import DiagnosticConfig
from ml4t.diagnostic.evaluation import FeatureDiagnostics

fd = FeatureDiagnostics(config=DiagnosticConfig())
result = fd.run_diagnostics(features_df["feature_1"], name="feature_1")
```

### D4. Trade diagnostics

```python
from ml4t.diagnostic.evaluation import TradeAnalysis

analyzer = TradeAnalysis(trade_records)
worst = analyzer.worst_trades(n=20)
```

Useful high-level tools:

- `analyze_signal`
- `ValidatedCrossValidation`
- `FeatureDiagnostics`
- `TradeAnalysis`
- `BarrierAnalysis`
- `WalkForwardCV`
- `CombinatorialCV`

## 5. Practical Start Order

For a normal research task, use this order:

1. fetch or update raw data in `data`
2. compute features and labels in `engineer`
3. convert data into model batches in `models`
4. fit model and export predictions
5. validate results in `diagnostic`

## 6. Suggested Working Pattern

### Initial setup

```bash
cd /Users/mit/Project/ML4T/data && uv sync --dev
cd /Users/mit/Project/ML4T/engineer && uv sync --python 3.12 --extra ta --extra store
cd /Users/mit/Project/ML4T/models && uv sync
cd /Users/mit/Project/ML4T/diagnostic && uv sync
```

### Smoke tests

```bash
cd /Users/mit/Project/ML4T/data && uv run pytest tests/ -q
cd /Users/mit/Project/ML4T/engineer && uv run pytest tests/ -q -o addopts=''
cd /Users/mit/Project/ML4T/models && uv run pytest tests/ -q
cd /Users/mit/Project/ML4T/diagnostic && uv run pytest tests/ -q
```

## 7. Important Caveats

- `diagnostic` needs `libomp` on macOS.
- `engineer` was most stable locally on `Python 3.12`.
- `engineer` had an earlier intermittent native crash on macOS arm64 during a previous investigation, but the latest full rerun passed cleanly.
- The verified workflow in this summary is macOS-only.

## 8. What To Do Next

If you continue the repo sequence after these four:

```text
backtest -> live
```
