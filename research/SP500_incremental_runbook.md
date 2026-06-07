# SP500 Incremental Runbook

## Daily Update

Run the incremental update across the full `SP500` universe in chunks:

```bash
cd /Users/mit/Project/ML4T/data
uv run python ../research/update_sp500_daily.py --prefix sp500_full
```

This uses:

- [update_sp500_daily.py](/Users/mit/Project/ML4T/research/update_sp500_daily.py)
- [fetch_sp500_chunks.py](/Users/mit/Project/ML4T/research/fetch_sp500_chunks.py)

Behavior:

- updates all symbols with `ml4t-data update --strategy incremental`
- keeps chunking at `50` symbols by default
- writes chunk logs under:
  - [fetch_chunks](/Users/mit/Project/ML4T/research/outputs/sp500_full/fetch_chunks)

## Rebuild Research Dataset

If the daily update completes and you want to refresh the research frame:

```bash
cd /Users/mit/Project/ML4T/data
uv run python ../research/update_sp500_daily.py --prefix sp500_full --rebuild-dataset
```

## Health Check

Check metadata freshness for the `SP500` universe:

```bash
cd /Users/mit/Project/ML4T/data
uv run python ../research/check_sp500_health.py --prefix sp500_full --stale-days 3
```

Outputs:

- [sp500_full_health_summary.json](/Users/mit/Project/ML4T/research/outputs/sp500_full/sp500_full_health_summary.json)
- [sp500_full_health_records.json](/Users/mit/Project/ML4T/research/outputs/sp500_full/sp500_full_health_records.json)

## Optional Quality Review

The formal anomaly summary for the `SP500` panel is already available:

- [sp500_full_anomaly_report.md](/Users/mit/Project/ML4T/research/outputs/sp500_full/sp500_full_anomaly_report.md)

## Recommended Order

1. Run incremental update.
2. Run health check.
3. Rebuild research dataset only if update coverage is acceptable.
4. Rerun signal/backtest validation when you actually want to refresh research conclusions.
