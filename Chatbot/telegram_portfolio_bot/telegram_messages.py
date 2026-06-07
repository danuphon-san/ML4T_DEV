from __future__ import annotations

from datetime import date
from typing import Any


MAX_MESSAGE_LENGTH = 4000


def _round_money(value: float) -> str:
    return f"{value:,.2f}"


def _round_qty(value: float) -> str:
    return f"{value:,.4f}"


def split_message(text: str, *, limit: int = MAX_MESSAGE_LENGTH) -> list[str]:
    if len(text) <= limit:
        return [text]
    parts: list[str] = []
    current = ""
    for line in text.splitlines():
        candidate = line if not current else f"{current}\n{line}"
        if len(candidate) <= limit:
            current = candidate
            continue
        if current:
            parts.append(current)
        current = line
    if current:
        parts.append(current)
    return parts


def format_daily_summary(portfolio: dict[str, Any]) -> str:
    actual = portfolio["actual_snapshot"]
    return "\n".join(
        [
            f"Daily summary {portfolio['as_of']}",
            f"Portfolio: {portfolio['portfolio_id']}",
            f"Strategy: {portfolio['strategy_id']}",
            f"Equity: {_round_money(float(actual['equity']))}",
            f"Cash: {_round_money(float(actual['cash']))}",
            f"Realized P&L: {_round_money(float(actual['realized_pnl']))}",
            f"Unrealized P&L: {_round_money(float(actual['unrealized_pnl']))}",
        ]
    )


def format_drift_summary(portfolio: dict[str, Any], *, max_rows: int = 5) -> str | None:
    rows = portfolio["rebalance_plan"]["rows"]
    ranked = sorted(
        rows,
        key=lambda row: abs(float(row["target_value"]) - float(row["actual_value"])),
        reverse=True,
    )
    top_rows = [
        row
        for row in ranked
        if abs(float(row["target_value"]) - float(row["actual_value"])) > 1e-9
    ][:max_rows]
    if not top_rows:
        return None
    lines = [
        f"Drift summary {portfolio['as_of']}",
        f"Portfolio: {portfolio['portfolio_id']}",
    ]
    for row in top_rows:
        target_value = float(row["target_value"])
        actual_value = float(row["actual_value"])
        delta_value = target_value - actual_value
        lines.append(
            f"{row['asset']}: {row['action']} "
            f"delta_qty {_round_qty(float(row['delta_quantity']))}, "
            f"value gap {_round_money(delta_value)}"
        )
    return "\n".join(lines)


def format_rebalance_instructions(portfolio: dict[str, Any]) -> str:
    instructions = portfolio["rebalance_plan"]["instructions"]
    if not instructions:
        return (
            f"Rebalance instructions {portfolio['as_of']}\n"
            f"Portfolio: {portfolio['portfolio_id']}\n"
            "No rebalance actions."
        )
    lines = [
        f"Rebalance instructions {portfolio['as_of']}",
        f"Portfolio: {portfolio['portfolio_id']}",
    ]
    for row in instructions:
        lines.append(
            f"{row['action']} {row['asset']} "
            f"{_round_qty(float(row['delta_quantity']))} "
            f"(actual {_round_qty(float(row['actual_quantity']))} -> "
            f"target {_round_qty(float(row['target_quantity']))})"
        )
    return "\n".join(lines)


def format_fill_confirmation(
    *,
    portfolio_id: str,
    trade_date: date,
    side: str,
    symbol: str,
    quantity: float,
    fill_price: float,
    commission: float,
    slippage: float,
    fill_result: dict[str, Any],
) -> str:
    return "\n".join(
        [
            f"Fill confirmation {trade_date.isoformat()}",
            f"Portfolio: {portfolio_id}",
            f"Trade: {side.upper()} {symbol} {_round_qty(quantity)} @ {_round_money(fill_price)}",
            f"Commission: {_round_money(commission)}",
            f"Slippage: {_round_money(slippage)}",
            f"Cash: {_round_money(float(fill_result['cash']))}",
            f"Realized P&L: {_round_money(float(fill_result['realized_pnl']))}",
        ]
    )


def format_status_reply(status: dict[str, Any]) -> str:
    actual = status["actual_snapshot"]
    rows = sorted(
        status["rebalance_plan"]["rows"],
        key=lambda row: abs(float(row["delta_quantity"])),
        reverse=True,
    )
    headline = "No drift"
    if rows and abs(float(rows[0]["delta_quantity"])) > 1e-9:
        top = rows[0]
        headline = (
            f"Top drift: {top['asset']} {top['action']} "
            f"{_round_qty(float(top['delta_quantity']))}"
        )
    return "\n".join(
        [
            f"Status {status['as_of']}",
            f"Portfolio: {status['portfolio_id']}",
            f"Strategy: {status['strategy_id']}",
            f"Equity: {_round_money(float(actual['equity']))}",
            f"Cash: {_round_money(float(actual['cash']))}",
            f"Realized P&L: {_round_money(float(actual['realized_pnl']))}",
            f"Unrealized P&L: {_round_money(float(actual['unrealized_pnl']))}",
            headline,
        ]
    )


def format_start_reply(portfolio_ids: list[str], *, mini_app_enabled: bool) -> str:
    if not portfolio_ids:
        lines = ["Bot is online.", "Authorized portfolios: none."]
    else:
        lines = ["Bot is online.", "Authorized portfolios: " + ", ".join(portfolio_ids)]
    if mini_app_enabled:
        lines.append("Use the Open Portfolio App button for the Mini App dashboard.")
    return "\n".join(lines)


def format_portfolios_reply(portfolio_ids: list[str]) -> str:
    if not portfolio_ids:
        return "Authorized portfolios: none."
    return "Authorized portfolios: " + ", ".join(portfolio_ids)


def fill_usage() -> str:
    return (
        "Usage: /fill <portfolio_id> <buy|sell> <symbol> <qty> <price> "
        "[commission] [slippage] [notes...]"
    )


def help_text() -> str:
    return "\n".join(
        [
            "/start",
            "/app",
            "/portfolios",
            "/status <portfolio_id>",
            "/whoami",
            "/adduser <password> <portfolio_id>",
            fill_usage(),
            "/help",
        ]
    )


def format_whoami_reply(*, chat_id: int | None, user_id: int | None) -> str:
    chat_value = "unknown" if chat_id is None else str(chat_id)
    user_value = "unknown" if user_id is None else str(user_id)
    return "\n".join(
        [
            "Telegram identity",
            f"chat_id: {chat_value}",
            f"user_id: {user_value}",
        ]
    )


def add_user_usage() -> str:
    return "Usage: /adduser <password> <portfolio_id>"


def format_add_user_prompt(*, portfolio_id: str) -> str:
    return "\n".join(
        [
            f"Creating portfolio {portfolio_id}",
            "Reply with initial cash amount.",
        ]
    )


def format_add_user_success(*, portfolio_id: str, portfolio_ids: list[str]) -> str:
    return "\n".join(
        [
            f"Access granted for {portfolio_id}",
            "Authorized portfolios: " + ", ".join(portfolio_ids),
        ]
    )
