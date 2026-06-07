from __future__ import annotations

import json
from pathlib import Path

import yaml

from ml4t.engineer import FeatureCatalog, feature_catalog


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "research" / "outputs" / "engineer"
CONFIG_DIR = REPO_ROOT / "research" / "configs"


def load_feature_names(config_path: Path) -> list[str]:
    config = yaml.safe_load(config_path.read_text())
    return [item["name"] for item in config.get("features", [])]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    catalog = FeatureCatalog()
    all_feature_names = catalog.list()
    all_tags = sorted(
        {
            tag
            for feature_name in all_feature_names
            for tag in catalog.describe(feature_name).get("tags", [])
        }
    )

    config_paths = [
        CONFIG_DIR / "sp20_core_features.yaml",
        CONFIG_DIR / "sp500_engineer_expanded_features.yaml",
    ]
    packs = {
        path.stem: {
            "features": load_feature_names(path),
            "metadata": {name: feature_catalog.describe(name) for name in load_feature_names(path)},
        }
        for path in config_paths
    }

    search_queries = {
        "trend_strength": feature_catalog.search("trend strength", max_results=10),
        "volatility": feature_catalog.search("volatility estimator", max_results=10),
        "spread": feature_catalog.search("spread", max_results=10),
        "microstructure": feature_catalog.search("microstructure", max_results=10),
    }

    snapshot = {
        "categories": feature_catalog.categories(),
        "tags": all_tags,
        "stats": feature_catalog.stats(),
        "normalized_features": feature_catalog.list(normalized=True, limit=50),
        "ta_lib_validated": feature_catalog.list(ta_lib_compatible=True, limit=80),
        "feature_packs": packs,
        "search_queries": search_queries,
    }

    snapshot_path = OUTPUT_DIR / "engineer_discovery_snapshot.json"
    report_path = OUTPUT_DIR / "engineer_discovery_snapshot.md"
    snapshot_path.write_text(json.dumps(snapshot, indent=2))

    lines = [
        "# Engineer Discovery Snapshot",
        "",
        f"- Categories: `{len(snapshot['categories'])}`",
        f"- Tags: `{len(snapshot['tags'])}`",
        f"- Normalized features (sample): `{len(snapshot['normalized_features'])}` listed",
        f"- TA-Lib validated features (sample): `{len(snapshot['ta_lib_validated'])}` listed",
        "",
        "## Feature Packs",
        "",
    ]
    for pack_name, pack in packs.items():
        lines.append(f"- `{pack_name}`: `{len(pack['features'])}` features")
    lines.extend(["", "## Search Highlights", ""])
    for query, results in search_queries.items():
        top = ", ".join(name for name, _score in results[:5])
        lines.append(f"- `{query}`: {top}")

    report_path.write_text("\n".join(lines) + "\n")
    print(json.dumps({"snapshot_path": str(snapshot_path), "report_path": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
