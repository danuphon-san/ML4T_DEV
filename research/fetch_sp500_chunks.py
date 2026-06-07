from __future__ import annotations

import argparse
import json
import subprocess
import sys
from math import ceil
from pathlib import Path

from research_universe import STORAGE_ROOT, universe_spec


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_REPO = REPO_ROOT / "data"


def read_symbols(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def available_symbols(symbols: list[str]) -> list[str]:
    covered: list[str] = []
    for symbol in symbols:
        if (
            any(STORAGE_ROOT.glob(f"yahoo_daily_{symbol}/year=*/month=*/data.parquet"))
            or any(STORAGE_ROOT.glob(f"{symbol}/year=*/month=*/data.parquet"))
            or any(STORAGE_ROOT.glob(f"equities_daily_{symbol}/year=*/month=*/data.parquet"))
        ):
            covered.append(symbol)
    return covered


def missing_symbols(symbols: list[str]) -> list[str]:
    covered = set(available_symbols(symbols))
    return [symbol for symbol in symbols if symbol not in covered]


def chunked(values: list[str], chunk_size: int) -> list[list[str]]:
    return [values[idx : idx + chunk_size] for idx in range(0, len(values), chunk_size)]


def write_chunk_files(
    chunk_dir: Path,
    dataset_name: str,
    symbols: list[str],
    *,
    start: str | None,
    end: str | None,
) -> tuple[Path, Path]:
    symbols_path = chunk_dir / f"{dataset_name}.txt"
    symbols_path.write_text("\n".join(symbols) + "\n")

    dataset_lines = [
        "storage:",
        f"  path: {STORAGE_ROOT}",
        "",
        "validation:",
        "  enabled: true",
        "  strict: false",
        "",
        "datasets:",
        f"  {dataset_name}:",
        "    provider: yahoo",
        "    frequency: daily",
        f"    symbols_file: {symbols_path}",
    ]
    if start:
        dataset_lines.append(f'    start: "{start}"')
    if end:
        dataset_lines.append(f'    end: "{end}"')
    dataset_lines.append("")

    config_path = chunk_dir / f"{dataset_name}.yaml"
    config_path.write_text("\n".join(dataset_lines))
    return symbols_path, config_path


def run_chunk(
    config_path: Path,
    dataset_name: str,
    log_path: Path,
    *,
    update_mode: str,
    start: str | None,
    end: str | None,
    symbols_chunk: list[str],
) -> int:
    if update_mode == "config":
        command = [
            "uv",
            "run",
            "ml4t-data",
            "update-all",
            "-c",
            str(config_path),
            "--dataset",
            dataset_name,
        ]
    elif update_mode == "incremental":
        strategy = update_mode
        launcher = "\n".join(
            [
                "import subprocess",
                "import sys",
                f"symbols = {symbols_chunk!r}",
                f"start = {start!r}",
                f"end = {end!r}",
                f"strategy = {strategy!r}",
                f"storage = {str(STORAGE_ROOT)!r}",
                "failures = []",
                "print(f'updating {len(symbols)} symbols via {strategy}')",
                "for symbol in symbols:",
                "    cmd = ['uv', 'run', 'ml4t-data', 'update', '-s', symbol, '--strategy', strategy, '-p', 'yahoo', '--storage-path', storage]",
                "    if start:",
                "        cmd.extend(['--start', start])",
                "    if end:",
                "        cmd.extend(['--end', end])",
                "    result = subprocess.run(cmd, capture_output=True, text=True)",
                "    sys.stdout.write(result.stdout)",
                "    sys.stderr.write(result.stderr)",
                "    if result.returncode != 0:",
                "        failures.append(symbol)",
                "sys.exit(1 if failures else 0)",
            ]
        )
        command = [
            "uv",
            "run",
            "python",
            "-c",
            launcher,
        ]
    else:
        if not start or not end:
            raise ValueError("start and end are required for backfill and full_refresh modes")
        launcher = "\n".join(
            [
                "import sys",
                "from pathlib import Path",
                "import polars as pl",
                "from ml4t.data import DataManager",
                "from ml4t.data.storage.backend import StorageConfig",
                "from ml4t.data.storage.hive import HiveStorage",
                f"symbols = {symbols_chunk!r}",
                f"start = {start!r}",
                f"end = {end!r}",
                f"storage_root = {str(STORAGE_ROOT)!r}",
                "storage = HiveStorage(StorageConfig(base_path=Path(storage_root)))",
                "manager = DataManager(storage=storage)",
                "failures = []",
                "print(f'loading {len(symbols)} symbols via historical {start} -> {end}')",
                "for symbol in symbols:",
                "    try:",
                "        new_df = manager.fetch(symbol, start, end, frequency='daily', provider='yahoo')",
                "        if new_df is None or new_df.is_empty():",
                "            raise ValueError('no data returned')",
                "        key = f'equities/daily/{symbol}'",
                "        if storage.exists(key):",
                "            existing = storage.read(key).collect()",
                "            merged = pl.concat([existing, new_df]).unique(subset=['timestamp'], keep='last').sort('timestamp')",
                "        else:",
                "            merged = new_df.sort('timestamp')",
                "        manager.import_data(merged, symbol=symbol, provider='yahoo', frequency='daily', asset_class='equities')",
                "        print(f'{symbol}: {merged.height} rows ({merged[\"timestamp\"].min()} -> {merged[\"timestamp\"].max()})')",
                "    except Exception as exc:",
                "        print(f'{symbol}: FAIL {exc}', file=sys.stderr)",
                "        failures.append(symbol)",
                "sys.exit(1 if failures else 0)",
            ]
        )
        command = [
            "uv",
            "run",
            "python",
            "-c",
            launcher,
        ]
    completed = subprocess.run(
        command,
        cwd=DATA_REPO,
        capture_output=True,
        text=True,
    )
    log_path.write_text(completed.stdout + ("\n" + completed.stderr if completed.stderr else ""))
    sys.stdout.write(completed.stdout)
    if completed.stderr:
        sys.stderr.write(completed.stderr)
    return completed.returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="sp500_full")
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--max-chunks", type=int, default=0)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument(
        "--scope",
        choices=("missing", "all"),
        default="missing",
        help="Fetch only uncovered symbols or all symbols in the universe",
    )
    parser.add_argument(
        "--update-mode",
        choices=("config", "incremental", "backfill", "full_refresh"),
        default="config",
        help="Use config-driven update-all or direct per-symbol updates",
    )
    args = parser.parse_args()

    spec = universe_spec(args.prefix)
    all_symbols = read_symbols(spec.symbol_file)
    covered_now = available_symbols(all_symbols)
    pending = missing_symbols(all_symbols) if args.scope == "missing" else list(all_symbols)

    fetch_dir = spec.output_dir / "fetch_chunks"
    fetch_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "prefix": spec.prefix,
        "symbol_file": str(spec.symbol_file),
        "chunk_size": args.chunk_size,
        "start": args.start,
        "end": args.end,
        "scope": args.scope,
        "update_mode": args.update_mode,
        "total_symbols": len(all_symbols),
        "initial_available": len(covered_now),
        "initial_missing": len(all_symbols) - len(covered_now),
        "requested_symbols": len(pending),
        "chunks": [],
    }

    if not pending:
        summary["final_available"] = len(all_symbols)
        summary["final_missing"] = 0
        (spec.output_dir / f"{spec.prefix}_fetch_chunks_summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        return

    chunks = chunked(pending, args.chunk_size)
    if args.max_chunks > 0:
        chunks = chunks[: args.max_chunks]

    total_chunks = len(chunks)
    width = max(3, len(str(max(total_chunks, 1))))

    for idx, symbols_chunk in enumerate(chunks, start=1):
        dataset_name = f"{spec.prefix}_chunk_{idx:0{width}d}"
        _, config_path = write_chunk_files(
            fetch_dir,
            dataset_name,
            symbols_chunk,
            start=args.start,
            end=args.end,
        )
        log_path = fetch_dir / f"{dataset_name}.log"

        print(
            f"[chunk {idx}/{total_chunks}] fetching {len(symbols_chunk)} symbols: "
            f"{symbols_chunk[0]} .. {symbols_chunk[-1]}"
        )
        exit_code = run_chunk(
            config_path,
            dataset_name,
            log_path,
            update_mode=args.update_mode,
            start=args.start,
            end=args.end,
            symbols_chunk=symbols_chunk,
        )
        still_missing = missing_symbols(symbols_chunk)
        completed = [symbol for symbol in symbols_chunk if symbol not in still_missing]
        summary["chunks"].append(
            {
                "dataset": dataset_name,
                "requested": len(symbols_chunk),
                "completed": len(completed),
                "missing_after_chunk": still_missing,
                "log_path": str(log_path),
                "exit_code": exit_code,
            }
        )
        if exit_code != 0:
            break

    final_missing = missing_symbols(all_symbols)
    summary["final_available"] = len(all_symbols) - len(final_missing)
    summary["final_missing"] = len(final_missing)
    summary["remaining_symbols"] = final_missing

    summary_path = spec.output_dir / f"{spec.prefix}_fetch_chunks_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
