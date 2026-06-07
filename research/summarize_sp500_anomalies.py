from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import polars as pl

from research_universe import universe_spec


def read_symbols(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def main() -> None:
    spec = universe_spec("sp500_full")
    symbols = set(read_symbols(spec.symbol_file))
    reports_dir = Path("/Users/mit/Project/ML4T/data/anomaly_reports")
    rows: list[dict[str, object]] = []
    severity_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()

    for path in sorted(reports_dir.glob("anomaly_report_*.json")):
        report = json.loads(path.read_text())
        symbol = report.get("symbol")
        if symbol not in symbols:
            continue
        anomalies = report.get("anomalies", [])
        for anomaly in anomalies:
            severity = anomaly["severity"]
            anomaly_type = anomaly["type"]
            severity_counts[severity] += 1
            type_counts[anomaly_type] += 1
            rows.append(
                {
                    "symbol": symbol,
                    "timestamp": anomaly["timestamp"],
                    "severity": severity,
                    "type": anomaly_type,
                    "value": anomaly.get("value"),
                    "message": anomaly["message"],
                }
            )

    anomaly_df = pl.DataFrame(rows) if rows else pl.DataFrame(
        schema={
            "symbol": pl.String,
            "timestamp": pl.String,
            "severity": pl.String,
            "type": pl.String,
            "value": pl.Float64,
            "message": pl.String,
        }
    )
    by_symbol = (
        anomaly_df.group_by("symbol")
        .len("anomaly_count")
        .sort("anomaly_count", descending=True)
        if rows
        else pl.DataFrame({"symbol": [], "anomaly_count": []}, schema={"symbol": pl.String, "anomaly_count": pl.UInt32})
    )

    output_dir = spec.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    anomaly_path = output_dir / f"{spec.prefix}_anomaly_summary.parquet"
    by_symbol_path = output_dir / f"{spec.prefix}_anomaly_by_symbol.parquet"
    summary_path = output_dir / f"{spec.prefix}_anomaly_overview.json"
    report_path = output_dir / f"{spec.prefix}_anomaly_report.md"

    anomaly_df.write_parquet(anomaly_path)
    by_symbol.write_parquet(by_symbol_path)

    overview = {
        "prefix": spec.prefix,
        "symbols_requested": len(symbols),
        "symbols_with_reports": anomaly_df.select("symbol").n_unique() if rows else 0,
        "total_anomalies": len(rows),
        "by_severity": dict(severity_counts),
        "by_type": dict(type_counts),
        "top_symbols": by_symbol.head(20).to_dicts(),
    }
    summary_path.write_text(json.dumps(overview, indent=2))

    lines = [
        f"# {spec.prefix} Anomaly Report",
        "",
        f"- Symbols requested: `{overview['symbols_requested']}`",
        f"- Symbols with anomaly reports: `{overview['symbols_with_reports']}`",
        f"- Total anomalies: `{overview['total_anomalies']}`",
        "",
        "## By Severity",
        "",
    ]
    for key, value in sorted(severity_counts.items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## By Type", ""])
    for key, value in sorted(type_counts.items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Top Symbols", ""])
    for row in by_symbol.head(20).to_dicts():
        lines.append(f"- `{row['symbol']}`: `{row['anomaly_count']}`")
    report_path.write_text("\n".join(lines) + "\n")

    print(json.dumps(overview, indent=2))


if __name__ == "__main__":
    main()
