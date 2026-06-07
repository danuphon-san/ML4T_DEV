"""Build FF-12 sector mapping for the SP500 PIT universe.

Sources:
  - Wikipedia "List of S&P 500 companies" for current GICS sector by ticker
  - GICS → FF-12 mapping (approximate, sector-level)

For historical-only tickers (delisted / removed from SP500), Wikipedia's current
list does not cover them. These are marked as "Other" and excluded from the
sector-neutral analysis. Document this limitation in the Phase 2 report.

Output: research/outputs/sp500_pit/sp500_ff12_sectors.parquet
  Columns: ticker, gics_sector, ff12_industry

Usage:
    uv run python build_sp500_ff12_sectors.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import io

import pandas as pd
import polars as pl
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
PIT_FILE = REPO_ROOT / "research" / "outputs" / "sp500_pit" / "sp500_pit_composition.parquet"
OUTPUT_FILE = REPO_ROOT / "research" / "outputs" / "sp500_pit" / "sp500_ff12_sectors.parquet"

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# GICS Sector → FF-12 Industry (approximate sector-level mapping)
GICS_TO_FF12 = {
    "Communication Services": "Telcm",
    "Consumer Discretionary": "Shops",
    "Consumer Staples": "NoDur",
    "Energy": "Enrgy",
    "Financials": "Money",
    "Health Care": "Hlth",
    "Industrials": "Manuf",
    "Information Technology": "BusEq",
    "Materials": "Chems",
    "Real Estate": "Money",
    "Utilities": "Utils",
}

FF12_LABELS = {
    "NoDur": "Consumer Non-Durables",
    "Durbl": "Consumer Durables",
    "Manuf": "Manufacturing",
    "Enrgy": "Energy",
    "Chems": "Chemicals",
    "BusEq": "Business Equipment (Technology)",
    "Telcm": "Telecommunications",
    "Utils": "Utilities",
    "Shops": "Wholesale/Retail",
    "Hlth": "Healthcare",
    "Money": "Finance/Real Estate",
    "Other": "Other / Unknown",
}


def fetch_wikipedia_sectors() -> pd.DataFrame:
    """Scrape Wikipedia for current SP500 GICS sector mapping."""
    headers = {"User-Agent": "Mozilla/5.0 (research) ml4t-sectors/1.0"}
    response = requests.get(WIKI_URL, headers=headers, timeout=30)
    response.raise_for_status()
    tables = pd.read_html(io.StringIO(response.text))
    sp500 = tables[0]
    sp500.columns = [c.strip() for c in sp500.columns]
    symbol_col = next(c for c in sp500.columns if c.lower().startswith("symbol"))
    sector_col = next(c for c in sp500.columns if "GICS Sector" in c)
    out = sp500[[symbol_col, sector_col]].rename(
        columns={symbol_col: "ticker", sector_col: "gics_sector"}
    )
    out["ticker"] = out["ticker"].astype(str).str.replace(".", "-", regex=False)
    return out


def main() -> None:
    if not PIT_FILE.exists():
        print(f"ERROR: PIT composition not found at {PIT_FILE}")
        sys.exit(1)

    pit = pl.read_parquet(PIT_FILE)
    pit_tickers = sorted(pit["ticker"].unique().to_list())
    print(f"PIT universe: {len(pit_tickers)} unique tickers")

    print(f"Fetching Wikipedia: {WIKI_URL}")
    wiki = fetch_wikipedia_sectors()
    print(f"Wikipedia returned {len(wiki)} current SP500 entries")

    wiki_pl = pl.from_pandas(wiki).with_columns(
        pl.col("gics_sector").map_elements(
            lambda s: GICS_TO_FF12.get(s, "Other"),
            return_dtype=pl.String,
        ).alias("ff12_industry")
    )

    pit_df = pl.DataFrame({"ticker": pit_tickers})
    merged = pit_df.join(wiki_pl, on="ticker", how="left").with_columns([
        pl.col("gics_sector").fill_null("Unknown"),
        pl.col("ff12_industry").fill_null("Other"),
    ])

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    merged.write_parquet(OUTPUT_FILE)

    n_classified = (merged["ff12_industry"] != "Other").sum()
    print(f"\nClassified: {n_classified}/{len(pit_tickers)} tickers")
    print(f"Unclassified (delisted/historical): {len(pit_tickers) - n_classified}")
    print("\nFF-12 distribution:")
    counts = merged.group_by("ff12_industry").len().sort("len", descending=True)
    for row in counts.iter_rows(named=True):
        label = FF12_LABELS.get(row["ff12_industry"], row["ff12_industry"])
        print(f"  {row['ff12_industry']:6s} ({label:30s}): {row['len']}")

    print(f"\nSaved → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
