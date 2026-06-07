"""Manual portfolio workflow service.

This module is the authoritative manual ledger and target-vs-actual service.
It intentionally does not depend on broker runtimes, live reconciliation state,
or live accounting abstractions.
"""

from __future__ import annotations

import uuid
from datetime import date
from math import floor
from pathlib import Path
from typing import Any, Iterable

import polars as pl

from .models import (
    FillRecord,
    HoldingState,
    PortfolioMetadata,
    PortfolioState,
    PromotionRegistry,
    utc_now,
)
from .registry import load_promotion_registry
from .storage import (
    append_jsonl,
    daily_output_dir,
    ensure_dir,
    fills_path,
    list_portfolio_ids,
    load_state,
    read_jsonl,
    save_metadata,
    save_state,
    write_json,
)


def _normalize_symbol(symbol: str) -> str:
    normalized = symbol.strip().upper()
    if not normalized:
        raise ValueError("symbol must not be empty")
    return normalized


def _coerce_holdings(raw_holdings: Iterable[dict[str, Any]]) -> dict[str, HoldingState]:
    holdings: dict[str, HoldingState] = {}
    for index, row in enumerate(raw_holdings):
        if not isinstance(row, dict):
            raise ValueError(f"holding row {index} must be an object")
        missing = {"symbol", "quantity", "avg_cost"} - set(row)
        if missing:
            raise ValueError(
                f"holding row {index} missing required fields: {sorted(missing)}"
            )
        symbol = _normalize_symbol(str(row["symbol"]))
        if symbol in holdings:
            raise ValueError(f"duplicate holding symbol: {symbol}")
        quantity = float(row["quantity"])
        avg_cost = float(row["avg_cost"])
        if quantity <= 0:
            raise ValueError(f"holding quantity must be positive for {symbol}")
        if avg_cost < 0:
            raise ValueError(f"holding avg_cost must be non-negative for {symbol}")
        holdings[symbol] = HoldingState(quantity=quantity, avg_cost=avg_cost)
    return holdings


def onboard_portfolio(
    state_root: Path,
    portfolio_id: str,
    *,
    starting_cash: float,
    display_name: str | None = None,
    imported_holdings: list[dict[str, Any]] | None = None,
    base_currency: str = "USD",
    notes: str = "",
) -> dict[str, Any]:
    if starting_cash < 0:
        raise ValueError("starting_cash must be non-negative")
    onboarding_mode = "import_holdings" if imported_holdings else "cash_only"
    holdings = _coerce_holdings(imported_holdings or [])
    metadata = PortfolioMetadata(
        portfolio_id=portfolio_id,
        display_name=display_name or portfolio_id,
        base_currency=base_currency,
        onboarding_mode=onboarding_mode,
        created_at=utc_now(),
        starting_cash=float(starting_cash),
        notes=notes,
        imported_holdings=imported_holdings or [],
    )
    state = PortfolioState(
        portfolio_id=portfolio_id,
        cash=float(starting_cash),
        realized_pnl=0.0,
        holdings=holdings,
        last_marks={},
        last_unrealized_pnl=0.0,
        last_processed_date=None,
        updated_at=utc_now(),
    )
    save_metadata(state_root, metadata)
    save_state(state_root, state)
    fills_path(state_root, portfolio_id).touch(exist_ok=True)
    return {
        "portfolio_id": portfolio_id,
        "onboarding_mode": onboarding_mode,
        "cash": state.cash,
        "holdings": {symbol: holding.to_dict() for symbol, holding in holdings.items()},
    }


def _apply_fill(state: PortfolioState, fill: FillRecord) -> None:
    """Apply a manual fill to the local ledger.

    This remains the source of truth for manual accounting semantics, including
    cash impact, costs, remaining quantity, and realized P&L.
    """
    symbol = fill.symbol
    quantity = float(fill.quantity)
    commission = float(fill.commission)
    slippage = float(fill.slippage)
    trade_costs = commission + slippage
    if quantity <= 0:
        raise ValueError("quantity must be positive")
    if fill.fill_price <= 0:
        raise ValueError("fill_price must be positive")
    if commission < 0:
        raise ValueError("commission must be non-negative")
    if slippage < 0:
        raise ValueError("slippage must be non-negative")
    if fill.side == "buy":
        total_cost = quantity * fill.fill_price + trade_costs
        state.cash -= total_cost
        current = state.holdings.get(symbol)
        if current is None:
            state.holdings[symbol] = HoldingState(
                quantity=quantity,
                avg_cost=total_cost / quantity,
            )
        else:
            prior_cost = current.quantity * current.avg_cost
            new_quantity = current.quantity + quantity
            current.avg_cost = (prior_cost + total_cost) / new_quantity
            current.quantity = new_quantity
    elif fill.side == "sell":
        current = state.holdings.get(symbol)
        if current is None or current.quantity + 1e-12 < quantity:
            raise ValueError(
                f"insufficient position to sell {quantity} shares of {symbol}"
            )
        proceeds = quantity * fill.fill_price - trade_costs
        realized = (
            (quantity * fill.fill_price) - (quantity * current.avg_cost) - trade_costs
        )
        state.cash += proceeds
        state.realized_pnl += realized
        remaining = current.quantity - quantity
        if remaining <= 1e-12:
            state.holdings.pop(symbol, None)
        else:
            current.quantity = remaining
    else:
        raise ValueError("side must be 'buy' or 'sell'")
    state.updated_at = utc_now()


def record_fill(
    state_root: Path,
    portfolio_id: str,
    *,
    trade_date: date,
    symbol: str,
    side: str,
    quantity: float,
    fill_price: float,
    commission: float = 0.0,
    slippage: float = 0.0,
    notes: str = "",
    fill_id: str | None = None,
) -> dict[str, Any]:
    state = load_state(state_root, portfolio_id)
    symbol = _normalize_symbol(symbol)
    side = side.lower()
    if side not in {"buy", "sell"}:
        raise ValueError("side must be 'buy' or 'sell'")
    if quantity <= 0:
        raise ValueError("quantity must be positive")
    if fill_price <= 0:
        raise ValueError("fill_price must be positive")
    if commission < 0:
        raise ValueError("commission must be non-negative")
    if slippage < 0:
        raise ValueError("slippage must be non-negative")
    fill = FillRecord(
        fill_id=fill_id or str(uuid.uuid4()),
        portfolio_id=portfolio_id,
        trade_date=trade_date.isoformat(),
        symbol=symbol,
        side=side,
        quantity=float(quantity),
        fill_price=float(fill_price),
        commission=float(commission),
        slippage=float(slippage),
        notes=notes,
        recorded_at=utc_now(),
    )
    existing_ids = {
        row["fill_id"] for row in read_jsonl(fills_path(state_root, portfolio_id))
    }
    if fill.fill_id in existing_ids:
        raise ValueError(f"duplicate fill_id: {fill.fill_id}")
    _apply_fill(state, fill)
    append_jsonl(fills_path(state_root, portfolio_id), fill.to_dict())
    save_state(state_root, state)
    return {
        "fill_id": fill.fill_id,
        "portfolio_id": portfolio_id,
        "cash": state.cash,
        "realized_pnl": state.realized_pnl,
        "holdings": {
            asset: holding.to_dict() for asset, holding in state.holdings.items()
        },
    }


def _latest_slice(
    frame: pl.DataFrame, date_col: str, as_of: date
) -> tuple[date, pl.DataFrame]:
    normalized = frame.with_columns(pl.col(date_col).dt.date().alias("__as_of_date"))
    filtered = normalized.filter(pl.col("__as_of_date") <= pl.lit(as_of))
    if filtered.is_empty():
        raise ValueError(f"no rows available on or before {as_of.isoformat()}")
    latest = filtered.select(pl.col("__as_of_date").max()).item()
    latest_frame = filtered.filter(pl.col("__as_of_date") == latest).drop(
        "__as_of_date"
    )
    return latest, latest_frame


def _load_registry(path: Path | PromotionRegistry) -> PromotionRegistry:
    if isinstance(path, PromotionRegistry):
        return path
    return load_promotion_registry(path)


def _validate_registry(registry: PromotionRegistry) -> None:
    if registry.rebalance_cadence.kind != "every_n_signal_dates":
        raise ValueError(
            f"unsupported rebalance cadence: {registry.rebalance_cadence.kind}"
        )
    if registry.rebalance_cadence.every_n_signals <= 0:
        raise ValueError("rebalance_cadence.every_n_signals must be positive")
    if registry.sizing_rule.kind != "top_n_equal_weight":
        raise ValueError(f"unsupported sizing rule: {registry.sizing_rule.kind}")
    if registry.sizing_rule.top_n <= 0:
        raise ValueError("sizing_rule.top_n must be positive")
    if not 0 <= registry.sizing_rule.gross_exposure <= 1:
        raise ValueError("sizing_rule.gross_exposure must be between 0 and 1")
    if registry.execution_policy.quantity_policy not in {"fractional", "whole_share"}:
        raise ValueError(
            "execution_policy.quantity_policy must be 'fractional' or 'whole_share'"
        )
    if registry.execution_policy.min_trade_value < 0:
        raise ValueError("execution_policy.min_trade_value must be non-negative")
    if registry.execution_policy.min_delta_quantity < 0:
        raise ValueError("execution_policy.min_delta_quantity must be non-negative")
    for label, artifact in (
        ("signal", registry.signal_artifact),
        ("price", registry.price_artifact),
    ):
        if not artifact.path.exists():
            raise ValueError(f"{label} artifact does not exist: {artifact.path}")
        columns = set(pl.read_parquet(artifact.path, n_rows=0).columns)
        required = {artifact.date_col, artifact.asset_col, artifact.value_col}
        missing = sorted(required - columns)
        if missing:
            raise ValueError(f"{label} artifact missing required columns: {missing}")


def _read_artifact(artifact_label: str, registry: PromotionRegistry) -> pl.DataFrame:
    artifact = (
        registry.signal_artifact
        if artifact_label == "signal"
        else registry.price_artifact
    )
    return pl.read_parquet(artifact.path).select(
        pl.col(artifact.date_col).alias(f"{artifact_label}_date"),
        pl.col(artifact.asset_col).cast(pl.Utf8).alias("asset"),
        pl.col(artifact.value_col)
        .cast(pl.Float64)
        .alias("signal" if artifact_label == "signal" else "price"),
    )


def _load_target_inputs(
    registry: PromotionRegistry, as_of: date
) -> tuple[date, pl.DataFrame]:
    _validate_registry(registry)
    signals = _read_artifact("signal", registry)
    prices = _read_artifact("price", registry)
    signal_date, latest_signals = _latest_slice(signals, "signal_date", as_of)
    price_date, latest_prices = _latest_slice(prices, "price_date", as_of)
    merged = latest_signals.join(latest_prices, on="asset", how="inner")
    if merged.is_empty():
        raise ValueError(
            "no overlapping signal and price rows for target generation "
            f"on signal date {signal_date.isoformat()} and price date {price_date.isoformat()}"
        )
    return signal_date, merged


def _artifact_warnings(as_of: date, joined_frame: pl.DataFrame) -> list[str]:
    warnings: list[str] = []
    signal_date = joined_frame.select(pl.col("signal_date").dt.date().max()).item()
    price_date = joined_frame.select(pl.col("price_date").dt.date().max()).item()
    if signal_date < as_of:
        warnings.append(
            f"latest signal date {signal_date.isoformat()} is before as_of {as_of.isoformat()}"
        )
    if price_date < as_of:
        warnings.append(
            f"latest price date {price_date.isoformat()} is before as_of {as_of.isoformat()}"
        )
    return warnings


def _is_rebalance_day(registry: PromotionRegistry, signal_date: date) -> bool:
    signals = pl.read_parquet(registry.signal_artifact.path).select(
        pl.col(registry.signal_artifact.date_col).dt.date().alias("signal_date")
    )
    unique_dates = sorted(signals.get_column("signal_date").unique().to_list())
    if signal_date not in unique_dates:
        return False
    anchor = registry.rebalance_cadence.anchor_date or unique_dates[0]
    if signal_date < anchor:
        return False
    try:
        anchor_index = unique_dates.index(anchor)
    except ValueError:
        anchor_index = 0
    current_index = unique_dates.index(signal_date)
    offset = current_index - anchor_index
    return offset >= 0 and offset % registry.rebalance_cadence.every_n_signals == 0


def _build_actual_snapshot(
    state: PortfolioState, latest_prices: dict[str, float], as_of: date
) -> dict[str, Any]:
    holdings_rows: list[dict[str, Any]] = []
    total_market_value = 0.0
    total_unrealized = 0.0
    marks: dict[str, dict[str, Any]] = {}
    for asset, holding in sorted(state.holdings.items()):
        mark = float(latest_prices.get(asset, 0.0))
        market_value = holding.quantity * mark
        cost_basis = holding.quantity * holding.avg_cost
        unrealized = market_value - cost_basis
        total_market_value += market_value
        total_unrealized += unrealized
        marks[asset] = {"price": mark, "as_of": as_of.isoformat()}
        holdings_rows.append(
            {
                "asset": asset,
                "quantity": holding.quantity,
                "avg_cost": holding.avg_cost,
                "mark_price": mark,
                "market_value": market_value,
                "cost_basis": cost_basis,
                "unrealized_pnl": unrealized,
            }
        )
    equity = state.cash + total_market_value
    return {
        "as_of": as_of.isoformat(),
        "cash": state.cash,
        "equity": equity,
        "realized_pnl": state.realized_pnl,
        "unrealized_pnl": total_unrealized,
        "holdings": holdings_rows,
        "last_marks": marks,
    }


def _build_target_snapshot(
    registry: PromotionRegistry,
    joined_frame: pl.DataFrame,
    signal_date: date,
    equity: float,
) -> dict[str, Any]:
    top_n = registry.sizing_rule.top_n
    gross_exposure = registry.sizing_rule.gross_exposure
    ranked = joined_frame.sort(["signal", "asset"], descending=[True, False]).head(
        top_n
    )
    selected_count = ranked.height
    if selected_count == 0:
        raise ValueError("target snapshot cannot be empty")
    weight = gross_exposure / selected_count
    rows: list[dict[str, Any]] = []
    total_target_exposure = 0.0
    for row in ranked.iter_rows(named=True):
        price = float(row["price"])
        raw_target_value = equity * weight
        raw_target_quantity = raw_target_value / price if price else 0.0
        if registry.execution_policy.quantity_policy == "whole_share":
            target_quantity = float(floor(raw_target_quantity))
            target_value = target_quantity * price
        else:
            target_quantity = raw_target_quantity
            target_value = raw_target_value
        total_target_exposure += target_value
        rows.append(
            {
                "asset": row["asset"],
                "signal": float(row["signal"]),
                "mark_price": price,
                "target_weight": weight,
                "target_value": target_value,
                "target_quantity": target_quantity,
                "raw_target_quantity": raw_target_quantity,
            }
        )
    residual_cash = equity - total_target_exposure
    return {
        "strategy_id": registry.active_strategy_id,
        "signal_date": signal_date.isoformat(),
        "gross_exposure": gross_exposure,
        "quantity_policy": registry.execution_policy.quantity_policy,
        "cash_target": equity * (1.0 - gross_exposure),
        "total_target_exposure": total_target_exposure,
        "residual_cash": residual_cash,
        "positions": rows,
    }


def _build_rebalance_plan(
    actual_snapshot: dict[str, Any],
    target_snapshot: dict[str, Any],
    registry: PromotionRegistry,
) -> dict[str, Any]:
    actual_holdings = {row["asset"]: row for row in actual_snapshot["holdings"]}
    target_holdings = {row["asset"]: row for row in target_snapshot["positions"]}
    assets = sorted(set(actual_holdings) | set(target_holdings))
    rows: list[dict[str, Any]] = []
    instructions: list[dict[str, Any]] = []
    current_exposure = 0.0
    for asset in assets:
        actual = actual_holdings.get(asset, {})
        target = target_holdings.get(asset, {})
        actual_qty = float(actual.get("quantity", 0.0))
        target_qty = float(target.get("target_quantity", 0.0))
        delta_qty = target_qty - actual_qty
        mark_price = float(target.get("mark_price", actual.get("mark_price", 0.0)))
        actual_value = float(actual.get("market_value", 0.0))
        target_value = float(target.get("target_value", 0.0))
        estimated_trade_value = abs(delta_qty) * mark_price
        current_exposure += actual_value
        suppressed_reason = ""
        if abs(delta_qty) <= 1e-9:
            action = "no-op"
        elif abs(delta_qty) < registry.execution_policy.min_delta_quantity:
            action = "no-op"
            suppressed_reason = "below_min_delta_quantity"
        elif estimated_trade_value < registry.execution_policy.min_trade_value:
            action = "no-op"
            suppressed_reason = "below_min_trade_value"
        elif target_qty <= 1e-9 and actual_qty > 1e-9:
            action = "close"
        elif delta_qty > 0:
            action = "buy"
        else:
            action = "sell"
        row = {
            "asset": asset,
            "symbol": asset,
            "action": action,
            "actual_quantity": actual_qty,
            "current_quantity": actual_qty,
            "target_quantity": target_qty,
            "delta_quantity": delta_qty,
            "mark_price": mark_price,
            "estimated_trade_value": estimated_trade_value,
            "actual_value": actual_value,
            "current_value": actual_value,
            "target_value": target_value,
            "suppressed_reason": suppressed_reason,
        }
        rows.append(row)
        if action != "no-op":
            instructions.append(row)
    return {
        "rows": rows,
        "instructions": instructions,
        "summary": {
            "total_target_exposure": target_snapshot["total_target_exposure"],
            "current_exposure": current_exposure,
            "residual_cash": target_snapshot["residual_cash"],
            "actionable_trade_count": len(instructions),
        },
    }


def portfolio_status(
    state_root: Path,
    promotion_registry_path: Path,
    portfolio_id: str,
    *,
    as_of: date | None = None,
) -> dict[str, Any]:
    """Compare promoted target holdings against manual actual holdings."""
    registry = _load_registry(promotion_registry_path)
    state = load_state(state_root, portfolio_id)
    as_of = as_of or date.today()
    signal_date, joined_frame = _load_target_inputs(registry, as_of)
    warnings = _artifact_warnings(as_of, joined_frame)
    latest_prices = {
        row["asset"]: float(row["price"])
        for row in joined_frame.select("asset", "price").unique().iter_rows(named=True)
    }
    actual_snapshot = _build_actual_snapshot(state, latest_prices, as_of)
    target_snapshot = _build_target_snapshot(
        registry, joined_frame, signal_date, actual_snapshot["equity"]
    )
    rebalance_plan = _build_rebalance_plan(actual_snapshot, target_snapshot, registry)
    return {
        "portfolio_id": portfolio_id,
        "strategy_id": registry.active_strategy_id,
        "as_of": as_of.isoformat(),
        "rebalance_date": signal_date.isoformat(),
        "run_status": "warning" if warnings else "ok",
        "warnings": warnings,
        "blocking_reasons": [],
        "actual_snapshot": actual_snapshot,
        "target_snapshot": target_snapshot,
        "rebalance_plan": rebalance_plan,
        "is_rebalance_day": _is_rebalance_day(registry, signal_date),
    }


def _write_daily_markdown(output_dir: Path, status: dict[str, Any]) -> None:
    actual = status.get("actual_snapshot") or {}
    lines = [
        f"# Daily Summary: {status['portfolio_id']}",
        "",
        f"- As of: `{status['as_of']}`",
        f"- Strategy: `{status['strategy_id']}`",
        f"- Run status: `{status['run_status']}`",
    ]
    if status["run_status"] == "blocked":
        lines.extend(["", "## Blocking Reasons", ""])
        lines.extend(f"- {reason}" for reason in status["blocking_reasons"])
        (output_dir / "daily_summary.md").write_text("\n".join(lines) + "\n")
        return
    lines.extend(
        [
            f"- Rebalance date: `{status['rebalance_date']}`",
            f"- Rebalance day: `{status['is_rebalance_day']}`",
            f"- Equity: `{actual['equity']:.2f}`",
            f"- Cash: `{actual['cash']:.2f}`",
            f"- Realized P&L: `{actual['realized_pnl']:.2f}`",
            f"- Unrealized P&L: `{actual['unrealized_pnl']:.2f}`",
        ]
    )
    if status["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in status["warnings"])
    lines.extend(
        [
            "",
            "## Rebalance Instructions",
            "",
        ]
    )
    instructions = status["rebalance_plan"]["instructions"]
    if status["is_rebalance_day"] and instructions:
        for row in instructions:
            lines.append(
                f"- {row['action']} `{row['symbol']}`: delta_qty `{row['delta_quantity']:.6f}`, "
                f"current `{row['current_quantity']:.6f}` -> target `{row['target_quantity']:.6f}`, "
                f"mark `{row['mark_price']:.2f}`, est_value `{row['estimated_trade_value']:.2f}`"
            )
    elif status["is_rebalance_day"]:
        lines.append("- No action required.")
    else:
        lines.append(
            f"- Suppressed because today is not a rebalance date for `{status['rebalance_date']}`."
        )
    (output_dir / "daily_summary.md").write_text("\n".join(lines) + "\n")


def _blocked_daily_payload(
    *,
    portfolio_id: str,
    strategy_id: str,
    as_of: date,
    reason: str,
) -> dict[str, Any]:
    return {
        "portfolio_id": portfolio_id,
        "strategy_id": strategy_id,
        "as_of": as_of.isoformat(),
        "rebalance_date": None,
        "is_rebalance_day": False,
        "run_status": "blocked",
        "warnings": [],
        "blocking_reasons": [reason],
        "actual_snapshot": None,
        "target_snapshot": None,
        "rebalance_plan": {
            "rows": [],
            "instructions": [],
            "suppressed": True,
            "suppression_reason": "blocked",
            "summary": {
                "total_target_exposure": 0.0,
                "current_exposure": 0.0,
                "residual_cash": 0.0,
                "actionable_trade_count": 0,
            },
        },
    }


def daily_run(
    state_root: Path,
    promotion_registry_path: Path,
    *,
    as_of: date | None = None,
    portfolio_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Produce daily target, actual, and rebalance outputs per portfolio."""
    registry = _load_registry(promotion_registry_path)
    as_of = as_of or date.today()
    portfolio_ids = portfolio_ids or list_portfolio_ids(state_root)
    results: dict[str, Any] = {}
    for portfolio_id in portfolio_ids:
        output_dir = ensure_dir(
            daily_output_dir(state_root, portfolio_id, as_of.isoformat())
        )
        try:
            status = portfolio_status(
                state_root, registry.registry_path, portfolio_id, as_of=as_of
            )
            state = load_state(state_root, portfolio_id)
            state.last_marks = status["actual_snapshot"]["last_marks"]
            state.last_unrealized_pnl = status["actual_snapshot"]["unrealized_pnl"]
            state.last_processed_date = as_of.isoformat()
            state.updated_at = utc_now()
            save_state(state_root, state)

            plan = status["rebalance_plan"]
            instructions = plan["instructions"] if status["is_rebalance_day"] else []
            output_payload = {
                "portfolio_id": portfolio_id,
                "strategy_id": registry.active_strategy_id,
                "as_of": as_of.isoformat(),
                "rebalance_date": status["rebalance_date"],
                "is_rebalance_day": status["is_rebalance_day"],
                "run_status": status["run_status"],
                "warnings": status["warnings"],
                "blocking_reasons": [],
                "actual_snapshot": status["actual_snapshot"],
                "target_snapshot": status["target_snapshot"],
                "rebalance_plan": {
                    "rows": plan["rows"],
                    "instructions": instructions,
                    "suppressed": not status["is_rebalance_day"],
                    "suppression_reason": ""
                    if status["is_rebalance_day"]
                    else "not_rebalance_day",
                    "summary": {
                        **plan["summary"],
                        "actionable_trade_count": len(instructions),
                    },
                },
            }
            write_json(output_dir / "target_snapshot.json", status["target_snapshot"])
            write_json(output_dir / "actual_snapshot.json", status["actual_snapshot"])
        except Exception as exc:
            output_payload = _blocked_daily_payload(
                portfolio_id=portfolio_id,
                strategy_id=registry.active_strategy_id,
                as_of=as_of,
                reason=str(exc),
            )
        write_json(output_dir / "rebalance_plan.json", output_payload["rebalance_plan"])
        write_json(output_dir / "daily_run.json", output_payload)
        _write_daily_markdown(output_dir, output_payload)
        results[portfolio_id] = output_payload
    return {
        "as_of": as_of.isoformat(),
        "strategy_id": registry.active_strategy_id,
        "run_status": (
            "blocked"
            if results
            and all(row["run_status"] == "blocked" for row in results.values())
            else "warning"
            if any(row["run_status"] != "ok" for row in results.values())
            else "ok"
        ),
        "portfolios": results,
    }
