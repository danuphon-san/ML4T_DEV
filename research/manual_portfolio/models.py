from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat()


def date_to_str(value: date | None) -> str | None:
    return value.isoformat() if value is not None else None


def parse_date(value: str | date | None) -> date | None:
    if value is None or isinstance(value, date):
        return value
    return date.fromisoformat(value)


@dataclass
class HoldingState:
    quantity: float
    avg_cost: float

    def to_dict(self) -> dict[str, float]:
        return {"quantity": self.quantity, "avg_cost": self.avg_cost}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "HoldingState":
        return cls(
            quantity=float(payload["quantity"]), avg_cost=float(payload["avg_cost"])
        )


@dataclass
class PortfolioMetadata:
    portfolio_id: str
    display_name: str
    base_currency: str
    onboarding_mode: str
    created_at: str
    starting_cash: float
    notes: str = ""
    imported_holdings: list[dict[str, Any]] = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PortfolioMetadata":
        return cls(**payload)


@dataclass
class PortfolioState:
    portfolio_id: str
    cash: float
    realized_pnl: float
    holdings: dict[str, HoldingState]
    last_marks: dict[str, dict[str, Any]]
    last_unrealized_pnl: float
    last_processed_date: str | None
    updated_at: str
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "portfolio_id": self.portfolio_id,
            "cash": self.cash,
            "realized_pnl": self.realized_pnl,
            "holdings": {
                symbol: holding.to_dict() for symbol, holding in self.holdings.items()
            },
            "last_marks": self.last_marks,
            "last_unrealized_pnl": self.last_unrealized_pnl,
            "last_processed_date": self.last_processed_date,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PortfolioState":
        holdings = {
            symbol: HoldingState.from_dict(holding)
            for symbol, holding in payload.get("holdings", {}).items()
        }
        return cls(
            portfolio_id=payload["portfolio_id"],
            cash=float(payload["cash"]),
            realized_pnl=float(payload.get("realized_pnl", 0.0)),
            holdings=holdings,
            last_marks=dict(payload.get("last_marks", {})),
            last_unrealized_pnl=float(payload.get("last_unrealized_pnl", 0.0)),
            last_processed_date=payload.get("last_processed_date"),
            updated_at=payload.get("updated_at", utc_now()),
            schema_version=int(payload.get("schema_version", SCHEMA_VERSION)),
        )


@dataclass
class FillRecord:
    fill_id: str
    portfolio_id: str
    trade_date: str
    symbol: str
    side: str
    quantity: float
    fill_price: float
    commission: float
    slippage: float
    notes: str
    recorded_at: str
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RebalanceCadence:
    kind: str
    every_n_signals: int
    anchor_date: date | None = None


@dataclass
class SizingRule:
    kind: str
    top_n: int
    gross_exposure: float


@dataclass
class ExecutionPolicy:
    quantity_policy: str
    min_trade_value: float = 0.0
    min_delta_quantity: float = 0.0


@dataclass
class ArtifactSpec:
    path: Path
    date_col: str
    asset_col: str
    value_col: str


@dataclass
class PromotionRegistry:
    registry_path: Path
    active_strategy_id: str
    source_prefix: str
    rebalance_cadence: RebalanceCadence
    sizing_rule: SizingRule
    execution_policy: ExecutionPolicy
    signal_artifact: ArtifactSpec
    price_artifact: ArtifactSpec
    notes: str = ""
