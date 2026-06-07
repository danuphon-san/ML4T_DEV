"""Build a point-in-time S&P500 constituent list from Wikipedia change history.

Wikipedia tracks every addition/removal with dates back to ~2000.  We reconstruct
the exact set of tickers that were IN the index on any given date.

Outputs (in research/outputs/sp500_pit/):
    sp500_changes.parquet    — raw additions/removals table from Wikipedia
    sp500_pit_composition.parquet  — (date, ticker) pairs: all in-index tickers per date
    sp500_pit_summary.json   — coverage stats

Usage:
    uv run python build_sp500_pit_composition.py
    uv run python build_sp500_pit_composition.py --start 2016-01-01 --end 2026-01-01
"""
from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "research" / "outputs" / "sp500_pit"
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# Current SP500 constituents (as of the symbol file timestamp)
SP500_FILE = REPO_ROOT / "data" / "examples" / "symbols" / "sp500_full_2025-11-24.txt"


def fetch_current_sp500(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def fetch_wikipedia_changes() -> pd.DataFrame:
    """Download and parse the 'Changes' table from the Wikipedia S&P500 page."""
    import io
    import requests

    print(f"Fetching {WIKI_URL} ...")
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    resp = requests.get(WIKI_URL, headers=headers, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text), header=0)

    # Table 0 = current constituents, Table 1 = changes history
    changes_raw = tables[1]
    print(f"  Raw changes table shape: {changes_raw.shape}")
    print(f"  Columns: {list(changes_raw.columns)}")

    # Wikipedia columns vary but typically:
    #   Date | Added Ticker | Added Security | Removed Ticker | Removed Security | Reason
    # Flatten multi-level headers if present
    if isinstance(changes_raw.columns, pd.MultiIndex):
        changes_raw.columns = [" ".join(c).strip() for c in changes_raw.columns]

    # Normalize column names to snake_case.
    # Wikipedia format (as of 2025): 'Effective Date', 'Added', 'Added.1', 'Removed', 'Removed.1', 'Reason'
    # where 'Added'='ticker', 'Added.1'='security name', 'Removed'='ticker', 'Removed.1'='security name'
    cols = list(changes_raw.columns)
    col_map: dict[str, str] = {}
    added_seen = removed_seen = 0
    for col in cols:
        lc = col.lower()
        if "date" in lc:
            col_map[col] = "date"
        elif "reason" in lc:
            col_map[col] = "reason"
        elif "added" in lc and "ticker" in lc:
            col_map[col] = "added_ticker"
        elif "added" in lc and "secur" in lc:
            col_map[col] = "added_name"
        elif "remov" in lc and "ticker" in lc:
            col_map[col] = "removed_ticker"
        elif "remov" in lc and "secur" in lc:
            col_map[col] = "removed_name"
        elif col == "Added" or (lc == "added" and added_seen == 0):
            col_map[col] = "added_ticker"
            added_seen += 1
        elif "added" in lc and added_seen == 1:
            col_map[col] = "added_name"
            added_seen += 1
        elif col == "Removed" or (lc == "removed" and removed_seen == 0):
            col_map[col] = "removed_ticker"
            removed_seen += 1
        elif "removed" in lc and removed_seen == 1:
            col_map[col] = "removed_name"
            removed_seen += 1
    changes_raw = changes_raw.rename(columns=col_map)

    required = {"date", "added_ticker", "removed_ticker"}
    missing = required - set(changes_raw.columns)
    if missing:
        raise ValueError(f"Could not parse Wikipedia changes table. Missing columns: {missing}. Got: {list(changes_raw.columns)}")

    changes_raw["date"] = pd.to_datetime(changes_raw["date"], errors="coerce")
    changes = changes_raw.dropna(subset=["date"]).copy()
    changes["added_ticker"] = changes["added_ticker"].str.strip().replace("", pd.NA)
    changes["removed_ticker"] = changes["removed_ticker"].str.strip().replace("", pd.NA)

    print(f"  Parsed {len(changes)} change events")
    print(f"  Date range: {changes['date'].min().date()} → {changes['date'].max().date()}")
    return changes


def reconstruct_composition(
    changes: pd.DataFrame,
    current_tickers: list[str],
    start: date,
    end: date,
) -> pl.DataFrame:
    """Reconstruct point-in-time S&P500 composition for [start, end].

    Algorithm:
      1. Start with the current constituent set (as of symbol file date ~2025-11-24).
      2. Walk changes in reverse chronological order, undoing each event until we
         reach `start`.  After this backward pass, ticker_set = composition at start.
      3. Collect change events in [start, end] for the forward replay.
      4. Walk forward from start to end, emitting (date, ticker) rows and applying
         events as they occur.
    """
    print(f"\nReconstructing composition from {start} to {end} ...")

    # Sort descending (newest first) for the backward unwind pass
    changes_sorted = changes.sort_values("date", ascending=False)

    # Working set = current constituents as of 2025-11-24
    ticker_set: set[str] = set(current_tickers)

    # Collect events in [start, end] for the forward replay (sorted ascending later)
    forward_events: list[tuple[date, str, str]] = []

    for _, row in changes_sorted.iterrows():
        d = row["date"].date()
        added = row.get("added_ticker")
        removed = row.get("removed_ticker")
        added = added if pd.notna(added) and str(added).strip() else None
        removed = removed if pd.notna(removed) and str(removed).strip() else None

        if d >= start:
            # Undo this event to walk backward toward the composition at `start`
            if added:
                ticker_set.discard(added)   # it was added ON or AFTER start → not present before
            if removed:
                ticker_set.add(removed)     # it was removed ON or AFTER start → was present before

            # Also collect for forward replay (events on exactly start are applied on start)
            if d <= end:
                if added:
                    forward_events.append((d, "added", added))
                if removed:
                    forward_events.append((d, "removed", removed))
        else:
            # d < start: no more events relevant to our window
            break

    print(f"  Composition at start ({start}): {len(ticker_set)} tickers")
    print(f"  Change events in [{start}, {end}]: {len(forward_events)}")

    # Build forward event lookup
    events_by_date: dict[date, list[tuple[str, str]]] = {}
    for d, kind, ticker in forward_events:
        events_by_date.setdefault(d, []).append((kind, ticker))

    # Forward replay: emit the composition for each calendar day
    rows_date: list[date] = []
    rows_ticker: list[str] = []
    current_date = start
    one_day = timedelta(days=1)

    while current_date <= end:
        for kind, ticker in events_by_date.get(current_date, []):
            if kind == "added":
                ticker_set.add(ticker)
            else:
                ticker_set.discard(ticker)

        for ticker in ticker_set:
            rows_date.append(current_date)
            rows_ticker.append(ticker)

        current_date += one_day

    return pl.DataFrame({"date": rows_date, "ticker": rows_ticker}).with_columns(
        pl.col("date").cast(pl.Date)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2016-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2026-06-01", help="End date (YYYY-MM-DD)")
    args = parser.parse_args()
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    current_tickers = fetch_current_sp500(SP500_FILE)
    print(f"Current SP500 constituents: {len(current_tickers)}")

    changes = fetch_wikipedia_changes()

    # Save raw changes
    changes_pl = pl.from_pandas(changes.astype(str))
    changes_pl.write_parquet(OUTPUT_DIR / "sp500_changes.parquet")
    print(f"Saved changes → {OUTPUT_DIR}/sp500_changes.parquet")

    composition = reconstruct_composition(changes, current_tickers, start, end)

    composition.write_parquet(OUTPUT_DIR / "sp500_pit_composition.parquet")
    print(f"\nSaved composition → {OUTPUT_DIR}/sp500_pit_composition.parquet")

    # Summary stats
    n_dates = composition["date"].n_unique()
    n_tickers = composition["ticker"].n_unique()
    avg_per_date = len(composition) / n_dates if n_dates > 0 else 0

    # Count tickers that appear historically but are NOT in current SP500
    historical_only = set(composition["ticker"].unique().to_list()) - set(current_tickers)

    summary = {
        "start": str(start),
        "end": str(end),
        "n_calendar_days": n_dates,
        "n_unique_tickers": n_tickers,
        "n_current_constituents": len(current_tickers),
        "n_historical_only_tickers": len(historical_only),
        "avg_constituents_per_day": round(avg_per_date, 1),
        "historical_only_tickers": sorted(historical_only)[:50],  # first 50
    }
    (OUTPUT_DIR / "sp500_pit_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*60}")
    print(f"Point-in-time S&P500 composition ({start} → {end})")
    print(f"  Calendar days:          {n_dates:,}")
    print(f"  Unique tickers total:   {n_tickers}")
    print(f"  Current constituents:   {len(current_tickers)}")
    print(f"  Historical-only:        {len(historical_only)} (were in index but removed)")
    print(f"  Avg constituents/day:   {avg_per_date:.0f}")
    print(f"\nExample historical-only tickers (need to fetch data for these):")
    for t in sorted(historical_only)[:20]:
        print(f"  {t}")


if __name__ == "__main__":
    main()
