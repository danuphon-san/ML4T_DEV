# ML4T Research Scaffold

This folder holds the first reusable research pipeline on top of the live `sp20_seed` universe.

## What It Builds

`build_sp20_research_dataset.py` loads the Yahoo daily equity partitions under `~/ml4t-data`,
merges daily Fama-French and macro context, computes a compact feature set with `ml4t.engineer`,
applies triple-barrier labels, runs a default `-atr` diagnostic signal check, and writes artifacts
for both `diagnostic` and `models`.

## Run

Use the `engineer` environment because it already has the feature-engineering dependencies:

```bash
cd /Users/mit/Project/ML4T/engineer
uv run python ../research/build_sp20_research_dataset.py
```

## Outputs

By default, outputs are written to `ML4T/research/outputs/`:

- `sp20_seed_panel.parquet`: merged raw equity panel
- `sp20_seed_feature_panel.parquet`: panel with technical features
- `sp20_seed_labeled_panel.parquet`: panel with labels and forward returns
- `sp20_seed_model_frame.parquet`: cleaned modeling frame
- `sp20_seed_signal_factor.parquet`: default factor-style input for `ml4t.diagnostic.analyze_signal()` using `-atr`
- `sp20_seed_signal_prices.parquet`: price input for `ml4t.diagnostic.analyze_signal()`
- `sp20_seed_signal_analysis.json`: machine-readable `SignalResult` export
- `sp20_seed_diagnostic_report.md`: human-readable diagnostic report
- `sp20_seed_batch.npz`: `CrossSectionBatch` arrays for `ml4t.models`
- `sp20_seed_batch_metadata.json`: feature/context metadata for the saved batch
- `sp20_seed_summary.json`: run summary with row counts and output paths

The default diagnostic factor is `-atr`, because it was the strongest first-pass standalone signal
in this seed universe.

## Next Step

Typical follow-on commands:

```bash
cat /Users/mit/Project/ML4T/research/outputs/sp20_seed_diagnostic_report.md
```

```bash
cd /Users/mit/Project/ML4T/models
uv run python - <<'PY'
import json
import numpy as np
from pathlib import Path
from ml4t.models import CrossSectionBatch, ExpandingMeanFactorForecaster, BetaLambdaMapper, IPCAConfig, IPCAModel, LatentFactorForecastPipeline

root = Path("/Users/mit/Project/ML4T/research/outputs")
arrays = np.load(root / "sp20_seed_batch.npz")
batch = CrossSectionBatch(
    characteristics=arrays["characteristics"],
    returns=arrays["returns"],
    context_features=arrays["context_features"],
    timestamps=tuple(json.loads((root / "sp20_seed_batch_metadata.json").read_text())["timestamps"]),
    asset_ids=tuple(json.loads((root / "sp20_seed_batch_metadata.json").read_text())["asset_ids"]),
    mask=arrays["mask"],
)

pipeline = LatentFactorForecastPipeline(
    model=IPCAModel(IPCAConfig(n_factors=3)),
    forecaster=ExpandingMeanFactorForecaster(),
    mapper=BetaLambdaMapper(),
)
pipeline.fit(batch)
prediction = pipeline.predict(batch)
print(prediction.asset_forecast.expected_returns.shape)
PY
```
