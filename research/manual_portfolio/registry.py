from __future__ import annotations

from pathlib import Path

import yaml

from .models import (
    ArtifactSpec,
    ExecutionPolicy,
    PromotionRegistry,
    RebalanceCadence,
    SizingRule,
    parse_date,
)


def _resolve_path(registry_path: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (registry_path.parent / path).resolve()


def load_promotion_registry(path: Path) -> PromotionRegistry:
    payload = yaml.safe_load(path.read_text())
    cadence = payload["rebalance_cadence"]
    sizing = payload["sizing_rule"]
    execution = payload.get("execution_policy", {})
    artifacts = payload["artifacts"]
    return PromotionRegistry(
        registry_path=path.resolve(),
        active_strategy_id=payload["active_strategy_id"],
        source_prefix=payload["source_prefix"],
        rebalance_cadence=RebalanceCadence(
            kind=cadence["kind"],
            every_n_signals=int(cadence["every_n_signals"]),
            anchor_date=parse_date(cadence.get("anchor_date")),
        ),
        sizing_rule=SizingRule(
            kind=sizing["kind"],
            top_n=int(sizing["top_n"]),
            gross_exposure=float(sizing["gross_exposure"]),
        ),
        execution_policy=ExecutionPolicy(
            quantity_policy=execution.get("quantity_policy", "fractional"),
            min_trade_value=float(execution.get("min_trade_value", 0.0)),
            min_delta_quantity=float(execution.get("min_delta_quantity", 0.0)),
        ),
        signal_artifact=ArtifactSpec(
            path=_resolve_path(path, artifacts["signal"]["path"]),
            date_col=artifacts["signal"].get("date_col", "timestamp"),
            asset_col=artifacts["signal"].get("asset_col", "asset"),
            value_col=artifacts["signal"].get("value_col", "signal"),
        ),
        price_artifact=ArtifactSpec(
            path=_resolve_path(path, artifacts["price"]["path"]),
            date_col=artifacts["price"].get("date_col", "date"),
            asset_col=artifacts["price"].get("asset_col", "asset"),
            value_col=artifacts["price"].get("value_col", "price"),
        ),
        notes=payload.get("notes", ""),
    )
