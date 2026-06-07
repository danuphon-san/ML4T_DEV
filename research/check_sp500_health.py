from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

from research_universe import STORAGE_ROOT, universe_spec


def read_symbols(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="sp500_full")
    parser.add_argument("--stale-days", type=int, default=3)
    args = parser.parse_args()

    spec = universe_spec(args.prefix)
    symbols = read_symbols(spec.symbol_file)
    metadata_dir = STORAGE_ROOT / ".metadata"
    now = datetime.now()
    stale_cutoff = now - timedelta(days=args.stale_days)

    records: list[dict[str, object]] = []
    missing: list[str] = []
    stale: list[str] = []

    for symbol in symbols:
        path = metadata_dir / f"{symbol}_metadata.json"
        if not path.exists():
            missing.append(symbol)
            continue
        data = json.loads(path.read_text())
        last_update_raw = data.get("last_update")
        last_update = datetime.fromisoformat(last_update_raw) if last_update_raw else None
        is_stale = last_update is None or last_update < stale_cutoff
        if is_stale:
            stale.append(symbol)
        records.append(
            {
                "symbol": symbol,
                "provider": data.get("provider", ""),
                "health_status": data.get("health_status", ""),
                "total_rows": data.get("total_rows", 0),
                "date_range_start": data.get("date_range_start", ""),
                "date_range_end": data.get("date_range_end", ""),
                "last_update": last_update_raw,
                "is_stale": is_stale,
            }
        )

    output_dir = spec.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "prefix": spec.prefix,
        "checked_at": now.isoformat(),
        "symbols_requested": len(symbols),
        "metadata_found": len(records),
        "missing_metadata": missing,
        "stale_symbols": stale,
        "stale_days": args.stale_days,
    }
    (output_dir / f"{spec.prefix}_health_summary.json").write_text(json.dumps(summary, indent=2))
    (output_dir / f"{spec.prefix}_health_records.json").write_text(json.dumps(records, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
