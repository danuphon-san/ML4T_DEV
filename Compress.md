# ML4T Compressed Handoff

## Scope

This workspace contains:

- `data`
- `engineer`
- `models`
- `diagnostic`
- `backtest`
- `live`
- support repos: `specs`, `itch-parser`

This handoff covers verified setup and usage for:

1. `data`
2. `engineer`
3. `models`
4. `diagnostic`

## Verified macOS Status

Date: `2026-06-03`

- `data`: clean
  - result: `2835 passed, 14 skipped, 282 deselected, 1 xpassed`
  - wall time: `39.02s`
- `engineer`: clean
  - result: `3201 passed, 1 skipped`
  - wall time: `25.70s`
- `models`: clean
  - result: `56 passed`
  - wall time: `14.50s`
- `diagnostic`: clean after OpenMP runtime install
  - result: `5226 passed, 21 skipped`
  - wall time: `35.80s`

## Important Local Changes Already Made

### `data`

Patched tests for macOS path normalization in:

- `tests/futures/test_downloader.py`
- `tests/test_cot.py`
- `tests/test_crypto_downloader.py`
- `tests/test_etf_downloader.py`
- `tests/test_fama_french_provider.py`
- `tests/test_macro_downloader.py`

Reason:

- macOS resolves some temp/cache paths as `/private/tmp/...` or `/private/var/...`
- tests were comparing exact strings too strictly

### `engineer`

Changed:

- `src/ml4t/engineer/labeling/horizon_labels.py`
- `uv.lock`

Reason:

- local `Python 3.14` had unstable native behavior earlier in JIT-heavy tests
- `engineer` was stabilized on `Python 3.12.13`
- `trend_scanning_labels` was adjusted to convert `NaN` t-values to nulls

### `diagnostic`

No repo code changes were required.

System dependency installed on macOS:

- `brew install libomp`

Reason:

- `lightgbm` and `xgboost` needed `libomp.dylib`

## Current Working Environment Notes

- `data`
  - package requires `>=3.11`
  - local verified run used current local environment successfully
- `engineer`
  - package requires `>=3.12`
  - best known local working setup: `Python 3.12.13`
  - optional extras needed for fuller coverage:
    - `ta-lib`
    - `duckdb`
- `models`
  - package requires `>=3.12`
- `diagnostic`
  - package requires `>=3.12,<3.15`
  - local macOS also requires Homebrew `libomp`

## Canonical Flow

```text
data -> engineer -> models -> diagnostic
```

Meaning:

1. `data`
   - fetch, update, validate, and store market data
2. `engineer`
   - compute features, labels, and training datasets
3. `models`
   - fit predictive or portfolio models
4. `diagnostic`
   - validate signal quality, CV robustness, drift, and model behavior

## Minimal Commands

### `data`

```bash
cd /Users/mit/Project/ML4T/data
uv sync --dev
uv run pytest tests/ -q
```

### `engineer`

```bash
cd /Users/mit/Project/ML4T/engineer
uv sync --python 3.12 --extra ta --extra store
uv run pytest tests/ -q -o addopts=''
```

### `models`

```bash
cd /Users/mit/Project/ML4T/models
uv sync
uv run pytest tests/ -q
```

### `diagnostic`

```bash
brew install libomp
cd /Users/mit/Project/ML4T/diagnostic
uv sync
uv run pytest tests/ -q
```

## Minimal Usage Flow

### 1. Fetch data

```python
from ml4t.data import DataManager

dm = DataManager()
prices = dm.fetch("AAPL", "2020-01-01", "2024-12-31", provider="yahoo")
```

### 2. Engineer features and labels

```python
from ml4t.engineer import compute_features
from ml4t.engineer.config import LabelingConfig
from ml4t.engineer.labeling import triple_barrier_labels

features = compute_features(prices, ["rsi", "macd", "atr", "obv"])
config = LabelingConfig.triple_barrier(
    upper_barrier=0.03,
    lower_barrier=0.02,
    max_holding_period=10,
)
labels = triple_barrier_labels(features, config=config)
```

### 3. Fit model

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

pipeline.fit(batch)
prediction = pipeline.predict(batch)
```

### 4. Diagnose signal/model

```python
from ml4t.diagnostic import ValidatedCrossValidation, analyze_signal
from ml4t.diagnostic.config import ValidatedCrossValidationConfig

vcv = ValidatedCrossValidation(
    config=ValidatedCrossValidationConfig(
        n_groups=10,
        n_test_groups=2,
        embargo_pct=0.01,
        label_horizon=5,
    )
)

signal_result = analyze_signal(
    factor=factor_df,
    prices=prices_df,
    periods=(1, 5, 21),
)
```

## Known Caveats

- `engineer` had an intermittent earlier `133` native crash on local macOS arm64 with JIT-heavy paths.
- That crash did not reproduce in the latest full rerun, so treat it as intermittent/stateful, not resolved by proof.
- `diagnostic` requires `libomp` on macOS for `lightgbm` and `xgboost`.
- Raspberry Pi runs were useful for comparison, but current workflow is now macOS-only.

## Suggested Future Skill Name

If a dedicated Codex skill is created later, a reasonable name is:

- `ml4t-pipeline-operator`

Suggested scope:

- setup verification
- repo-by-repo test execution
- flow documentation
- data -> engineer -> models -> diagnostic orchestration

No `SKILL.md` was created yet because the current request only needs documentation.

## Next Logical Repo

If continuing the implementation path:

```text
backtest -> live
```
