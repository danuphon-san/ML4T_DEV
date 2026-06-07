"""Fetch historical price data for all S&P500 point-in-time constituents.

Two modes:
  historical-only  — fetch only tickers that were in the index historically but
                     are NOT in the current SP500 (the 216 removed/delisted tickers).
  extend-all       — also extend current SP500 back to START (we have from 2022;
                     this fetches 2015-2022 so a 10-year model frame is possible).

Requires: research/outputs/sp500_pit/sp500_pit_composition.parquet
          (built by build_sp500_pit_composition.py)

Usage:
    uv run python fetch_sp500_historical_extended.py --mode historical-only
    uv run python fetch_sp500_historical_extended.py --mode extend-all
    uv run python fetch_sp500_historical_extended.py --mode historical-only --dry-run
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_REPO = REPO_ROOT / "data"
STORAGE_ROOT = Path.home() / "ml4t-data"
PIT_FILE = REPO_ROOT / "research" / "outputs" / "sp500_pit" / "sp500_pit_composition.parquet"
CURRENT_SP500 = REPO_ROOT / "data" / "examples" / "symbols" / "sp500_full_2025-11-24.txt"

DEFAULT_START = "2015-01-01"
DEFAULT_END = "2026-06-01"
CHUNK_SIZE = 20


def read_current_sp500() -> set[str]:
    return {
        line.strip()
        for line in CURRENT_SP500.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def has_data(symbol: str, before: str = "2022-01-01") -> bool:
    """Return True if we already have data going back before the given date."""
    all_paths = list(STORAGE_ROOT.glob(f"yahoo_daily_{symbol}/year=*/month=*/data.parquet"))
    all_paths += list(STORAGE_ROOT.glob(f"{symbol}/year=*/month=*/data.parquet"))
    all_paths += list(STORAGE_ROOT.glob(f"equities_daily_{symbol}/year=*/month=*/data.parquet"))
    if not all_paths:
        return False
    # Check if any files are before the cutoff year
    cutoff_year = before[:4]
    return any(f"year={cutoff_year}" in str(p) or any(f"year={y}" in str(p) for y in range(2015, int(cutoff_year))) for p in all_paths)


def build_fetch_launcher(symbols: list[str], start: str, end: str) -> str:
    """Build inline Python that fetches and stores each symbol via DataManager."""
    return "\n".join([
        "import sys",
        "from pathlib import Path",
        "import polars as pl",
        "from ml4t.data import DataManager",
        "from ml4t.data.storage.backend import StorageConfig",
        "from ml4t.data.storage.hive import HiveStorage",
        f"symbols = {symbols!r}",
        f"start = {start!r}",
        f"end = {end!r}",
        f"storage_root = {str(STORAGE_ROOT)!r}",
        "storage = HiveStorage(StorageConfig(base_path=Path(storage_root)))",
        "manager = DataManager(storage=storage)",
        "successes = []",
        "failures = []",
        "no_data = []",
        "print(f'Fetching {len(symbols)} symbols: {start} → {end}', flush=True)",
        "for symbol in symbols:",
        "    try:",
        "        new_df = manager.fetch(symbol, start, end, frequency='daily', provider='yahoo')",
        "        if new_df is None or new_df.is_empty():",
        "            print(f'  {symbol}: NO DATA', flush=True)",
        "            no_data.append(symbol)",
        "            continue",
        "        key = f'equities/daily/{symbol}'",
        "        if storage.exists(key):",
        "            existing = storage.read(key).collect()",
        "            merged = pl.concat([existing, new_df]).unique(subset=['timestamp'], keep='last').sort('timestamp')",
        "        else:",
        "            merged = new_df.sort('timestamp')",
        "        manager.import_data(merged, symbol=symbol, provider='yahoo', frequency='daily', asset_class='equities')",
        "        successes.append(symbol)",
        "        print(f'  {symbol}: {merged.height} rows ({merged[\"timestamp\"].min()} → {merged[\"timestamp\"].max()})', flush=True)",
        "    except Exception as exc:",
        "        exc_str = str(exc).lower()",
        "        if any(kw in exc_str for kw in ('not found or invalid', 'no price data', 'possibly delisted', 'no timezone', 'symbol not found')):",
        "            print(f'  {symbol}: UNAVAILABLE (delisted/no-data)', flush=True)",
        "            no_data.append(symbol)",
        "        else:",
        "            print(f'  {symbol}: FAIL {exc}', flush=True)",
        "            failures.append(symbol)",
        "print(f'Done: {len(successes)} ok, {len(no_data)} no-data, {len(failures)} failed', flush=True)",
        "if failures:",
        "    print('FAILURES: ' + ', '.join(failures), file=sys.stderr)",
        "sys.exit(1 if failures else 0)",
    ])


def chunked(lst: list, size: int) -> list[list]:
    return [lst[i:i + size] for i in range(0, len(lst), size)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("historical-only", "extend-all"),
        default="historical-only",
        help="historical-only: fetch 216 removed tickers only; "
             "extend-all: also extend current SP500 back to START",
    )
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--dry-run", action="store_true", help="Print what would be fetched")
    args = parser.parse_args()

    if not PIT_FILE.exists():
        print(f"ERROR: {PIT_FILE} not found. Run build_sp500_pit_composition.py first.")
        sys.exit(1)

    current = read_current_sp500()
    composition = pl.read_parquet(PIT_FILE)
    all_pit_tickers = set(composition["ticker"].unique().to_list())
    historical_only = sorted(all_pit_tickers - current)

    if args.mode == "historical-only":
        to_fetch = historical_only
        print(f"Mode: historical-only ({len(to_fetch)} tickers removed from S&P500 since 2016)")
    else:
        # Also extend current SP500 back if they don't have pre-2022 data
        needs_extend = [t for t in sorted(current) if not has_data(t, before=args.start)]
        to_fetch = sorted(set(historical_only) | set(needs_extend))
        print(f"Mode: extend-all")
        print(f"  Historical-only tickers: {len(historical_only)}")
        print(f"  Current SP500 needing extension: {len(needs_extend)}")
        print(f"  Total to fetch: {len(to_fetch)}")

    print(f"Date range: {args.start} → {args.end}")
    print(f"Chunk size: {args.chunk_size}")

    if args.dry_run:
        print(f"\n[DRY RUN] Would fetch {len(to_fetch)} tickers:")
        for t in to_fetch[:30]:
            print(f"  {t}")
        if len(to_fetch) > 30:
            print(f"  ... and {len(to_fetch) - 30} more")
        return

    chunks = chunked(to_fetch, args.chunk_size)
    print(f"\nFetching {len(to_fetch)} tickers in {len(chunks)} chunks of {args.chunk_size}...\n")

    total_success = 0
    total_fail = 0
    failed_tickers: list[str] = []

    for i, chunk in enumerate(chunks, 1):
        print(f"=== Chunk {i}/{len(chunks)}: {chunk[0]} … {chunk[-1]} ===")
        launcher = build_fetch_launcher(chunk, args.start, args.end)
        result = subprocess.run(
            ["uv", "run", "python", "-c", launcher],
            cwd=DATA_REPO,
            capture_output=False,
            text=True,
        )
        if result.returncode != 0:
            failed_tickers.extend(chunk)
            total_fail += len(chunk)
        else:
            total_success += len(chunk)
        time.sleep(1)  # polite pause between chunks

    print(f"\n{'='*60}")
    print(f"Fetch complete: {total_success} ok / {total_fail} with errors")
    if failed_tickers:
        print(f"Failed chunks (check individual symbols): {failed_tickers}")

    # Save summary
    summary_path = REPO_ROOT / "research" / "outputs" / "sp500_pit" / "fetch_summary.txt"
    summary_path.write_text(
        f"Fetch mode: {args.mode}\n"
        f"Date range: {args.start} → {args.end}\n"
        f"Tickers attempted: {len(to_fetch)}\n"
        f"Chunks ok: {total_success}\n"
        f"Chunks failed: {total_fail}\n"
    )
    print(f"Summary → {summary_path}")


if __name__ == "__main__":
    main()
