from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

from research_universe import universe_spec


REPO_ROOT = Path(__file__).resolve().parents[1]


DERIVED_FETCH_PREFIX = {
    "sp500_10yr": "sp500_full",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="sp500_full")
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--max-chunks", type=int, default=0)
    parser.add_argument(
        "--as-of-date",
        default=(date.today() - timedelta(days=1)).isoformat(),
        help="Safe market-data cutoff date for incremental daily updates",
    )
    parser.add_argument("--rebuild-dataset", action="store_true")
    args = parser.parse_args()

    spec = universe_spec(args.prefix)
    fetch_prefix = DERIVED_FETCH_PREFIX.get(spec.prefix, spec.prefix)
    command = [
        "uv",
        "run",
        "python",
        str(REPO_ROOT / "research" / "fetch_sp500_chunks.py"),
        "--prefix",
        fetch_prefix,
        "--scope",
        "all",
        "--update-mode",
        "incremental",
        "--start",
        args.as_of_date,
        "--end",
        args.as_of_date,
        "--chunk-size",
        str(args.chunk_size),
    ]
    if args.max_chunks > 0:
        command.extend(["--max-chunks", str(args.max_chunks)])

    print(
        f"[{date.today().isoformat()}] incremental SP500 update: "
        f"{spec.prefix} via {fetch_prefix} as-of {args.as_of_date}"
    )
    result = subprocess.run(command, cwd=REPO_ROOT / "data")
    if result.returncode != 0:
        sys.exit(result.returncode)

    if args.rebuild_dataset:
        if spec.prefix == "sp500_10yr":
            rebuild = [
                "uv",
                "run",
                "python",
                "build_sp500_10yr_dataset.py",
                "--model-end",
                args.as_of_date,
            ]
            rebuild_cwd = REPO_ROOT / "research"
        else:
            rebuild = [
                "uv",
                "run",
                "python",
                str(REPO_ROOT / "research" / "build_research_dataset.py"),
                "--prefix",
                spec.prefix,
            ]
            rebuild_cwd = REPO_ROOT / "engineer"
        print(f"[{date.today().isoformat()}] rebuilding research dataset: {spec.prefix}")
        rebuild_result = subprocess.run(rebuild, cwd=rebuild_cwd)
        sys.exit(rebuild_result.returncode)


if __name__ == "__main__":
    main()
