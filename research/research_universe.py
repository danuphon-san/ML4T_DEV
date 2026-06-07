from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
STORAGE_ROOT = Path.home() / "ml4t-data"
FACTOR_FILE = REPO_ROOT / "data" / "data" / "factors" / "fama-french" / "ff5_daily.parquet"
MACRO_FILE = STORAGE_ROOT / "treasury_yields.parquet"
FEATURE_CONFIG = REPO_ROOT / "research" / "configs" / "sp20_core_features.yaml"
OUTPUT_DIR = REPO_ROOT / "research" / "outputs"


@dataclass(frozen=True)
class UniverseSpec:
    prefix: str
    symbol_file: Path | None
    output_dir: Path


def resolve_symbol_file(prefix: str) -> Path | None:
    symbols_dir = REPO_ROOT / "data" / "examples" / "symbols"
    matches = sorted(symbols_dir.glob(f"{prefix}_*.txt"))
    if matches:
        return matches[-1]
    candidate = symbols_dir / f"{prefix}.txt"
    if candidate.exists():
        return candidate
    # Derived prefixes (e.g. sp500_10yr) may not have a symbol file
    return None


def universe_spec(prefix: str) -> UniverseSpec:
    return UniverseSpec(
        prefix=prefix,
        symbol_file=resolve_symbol_file(prefix),
        output_dir=OUTPUT_DIR / prefix,
    )
