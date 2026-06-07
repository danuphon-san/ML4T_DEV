from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl

from aiohttp import web

from manual_portfolio.service import portfolio_status, record_fill
from manual_portfolio.storage import load_metadata, portfolio_dir, read_json, tail_jsonl


ASSETS_DIR = Path(__file__).resolve().parent / "webapp"
MINI_APP_CONFIG_KEY = web.AppKey("mini_app_config", Any)
MINI_APP_CONTROLLER_KEY = web.AppKey("mini_app_controller", Any)


class MiniAppAuthError(ValueError):
    """Raised when Telegram Mini App auth cannot be verified."""


@dataclass(frozen=True)
class MiniAppIdentity:
    user_id: int
    chat_id: int | None
    username: str | None
    first_name: str | None
    last_name: str | None
    auth_date: int | None


def _secret_key(bot_token: str) -> bytes:
    return hmac.new(b"WebAppData", bot_token.encode("utf-8"), hashlib.sha256).digest()


def parse_and_verify_init_data(init_data: str, *, bot_token: str) -> MiniAppIdentity:
    if not init_data:
        raise MiniAppAuthError("Missing Telegram Mini App auth.")
    try:
        pairs = parse_qsl(init_data, keep_blank_values=True, strict_parsing=True)
    except ValueError as exc:
        raise MiniAppAuthError("Malformed Telegram init data.") from exc
    payload = dict(pairs)
    provided_hash = payload.pop("hash", "")
    if not provided_hash:
        raise MiniAppAuthError("Missing Telegram auth hash.")

    data_check_string = "\n".join(f"{key}={value}" for key, value in sorted(payload.items()))
    expected_hash = hmac.new(
        _secret_key(bot_token),
        data_check_string.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(expected_hash, provided_hash):
        raise MiniAppAuthError("Invalid Telegram auth hash.")

    raw_user = payload.get("user")
    if not raw_user:
        raise MiniAppAuthError("Missing Telegram user.")
    try:
        user = json.loads(raw_user)
    except json.JSONDecodeError as exc:
        raise MiniAppAuthError("Invalid Telegram user payload.") from exc
    if not isinstance(user, dict) or "id" not in user:
        raise MiniAppAuthError("Invalid Telegram user payload.")

    chat_id: int | None = None
    raw_receiver = payload.get("receiver")
    if raw_receiver:
        try:
            receiver = json.loads(raw_receiver)
        except json.JSONDecodeError as exc:
            raise MiniAppAuthError("Invalid Telegram receiver payload.") from exc
        if isinstance(receiver, dict) and receiver.get("id") is not None:
            chat_id = int(receiver["id"])

    auth_date_raw = payload.get("auth_date")
    auth_date = int(auth_date_raw) if auth_date_raw else None
    return MiniAppIdentity(
        user_id=int(user["id"]),
        chat_id=chat_id,
        username=user.get("username"),
        first_name=user.get("first_name"),
        last_name=user.get("last_name"),
        auth_date=auth_date,
    )


def _init_data_from_request(request: web.Request) -> str:
    header_value = request.headers.get("X-Telegram-Init-Data", "").strip()
    if header_value:
        return header_value
    query_value = request.query.get("tgWebAppData", "").strip()
    if query_value:
        return query_value
    raise MiniAppAuthError("Missing Telegram Mini App auth.")


def require_mini_app_identity(request: web.Request, *, bot_token: str) -> MiniAppIdentity:
    return parse_and_verify_init_data(_init_data_from_request(request), bot_token=bot_token)


def _top_drift_row(status: dict[str, Any]) -> dict[str, Any] | None:
    rows = sorted(
        status["rebalance_plan"]["rows"],
        key=lambda row: abs(float(row["delta_quantity"])),
        reverse=True,
    )
    if not rows:
        return None
    top = rows[0]
    if abs(float(top["delta_quantity"])) <= 1e-9:
        return None
    return {
        "asset": top["asset"],
        "action": top["action"],
        "delta_quantity": float(top["delta_quantity"]),
        "target_quantity": float(top["target_quantity"]),
        "actual_quantity": float(top["actual_quantity"]),
    }


def build_overview_payload(
    *,
    status: dict[str, Any],
    display_name: str,
    base_currency: str,
) -> dict[str, Any]:
    actual = status["actual_snapshot"]
    holdings_market_value = sum(float(row["market_value"]) for row in actual["holdings"])
    instructions = [
        row for row in status["rebalance_plan"]["instructions"] if row["action"] != "no-op"
    ]
    return {
        "portfolio_id": status["portfolio_id"],
        "display_name": display_name,
        "strategy_id": status["strategy_id"],
        "base_currency": base_currency,
        "as_of": status["as_of"],
        "rebalance_date": status["rebalance_date"],
        "run_status": status["run_status"],
        "warnings": status["warnings"],
        "is_rebalance_day": bool(status["is_rebalance_day"]),
        "equity": float(actual["equity"]),
        "cash": float(actual["cash"]),
        "holdings_market_value": holdings_market_value,
        "realized_pnl": float(actual["realized_pnl"]),
        "unrealized_pnl": float(actual["unrealized_pnl"]),
        "position_count": len(actual["holdings"]),
        "rebalance_action_count": len(instructions),
        "top_drift": _top_drift_row(status),
    }


def build_holdings_payload(*, status: dict[str, Any], display_name: str, base_currency: str) -> dict[str, Any]:
    actual = status["actual_snapshot"]
    holdings = sorted(
        [
            {
                "asset": row["asset"],
                "quantity": float(row["quantity"]),
                "avg_cost": float(row["avg_cost"]),
                "mark_price": float(row["mark_price"]),
                "market_value": float(row["market_value"]),
                "cost_basis": float(row["cost_basis"]),
                "unrealized_pnl": float(row["unrealized_pnl"]),
            }
            for row in actual["holdings"]
        ],
        key=lambda row: row["market_value"],
        reverse=True,
    )
    return {
        "portfolio_id": status["portfolio_id"],
        "display_name": display_name,
        "base_currency": base_currency,
        "as_of": status["as_of"],
        "cash": float(actual["cash"]),
        "equity": float(actual["equity"]),
        "holdings_market_value": sum(item["market_value"] for item in holdings),
        "holdings": holdings,
    }


def build_rebalance_payload(
    *,
    status: dict[str, Any],
    display_name: str,
    base_currency: str,
) -> dict[str, Any]:
    rows = sorted(
        [
            {
                "asset": row["asset"],
                "action": row["action"],
                "actual_quantity": float(row["actual_quantity"]),
                "target_quantity": float(row["target_quantity"]),
                "delta_quantity": float(row["delta_quantity"]),
                "actual_value": float(row["actual_value"]),
                "target_value": float(row["target_value"]),
            }
            for row in status["rebalance_plan"]["rows"]
        ],
        key=lambda row: abs(row["delta_quantity"]),
        reverse=True,
    )
    return {
        "portfolio_id": status["portfolio_id"],
        "display_name": display_name,
        "base_currency": base_currency,
        "as_of": status["as_of"],
        "rebalance_date": status["rebalance_date"],
        "is_rebalance_day": bool(status["is_rebalance_day"]),
        "suppressed": not bool(status["is_rebalance_day"]),
        "rows": rows,
        "instructions": [row for row in rows if row["action"] != "no-op"],
    }


def _load_latest_daily_run(state_root: Path, portfolio_id: str) -> dict[str, Any] | None:
    daily_root = portfolio_dir(state_root, portfolio_id) / "daily"
    if not daily_root.exists():
        return None
    dated_dirs = sorted(path for path in daily_root.iterdir() if path.is_dir())
    if not dated_dirs:
        return None
    latest_path = dated_dirs[-1] / "daily_run.json"
    if not latest_path.exists():
        return None
    return read_json(latest_path)


def build_activity_payload(
    *,
    state_root: Path,
    portfolio_id: str,
    display_name: str,
) -> dict[str, Any]:
    recent_fills = []
    for row in reversed(tail_jsonl(portfolio_dir(state_root, portfolio_id) / "fills.jsonl", 10)):
        recent_fills.append(
            {
                "fill_id": row["fill_id"],
                "trade_date": row["trade_date"],
                "symbol": row["symbol"],
                "side": row["side"],
                "quantity": float(row["quantity"]),
                "fill_price": float(row["fill_price"]),
                "commission": float(row["commission"]),
                "slippage": float(row["slippage"]),
                "notes": row["notes"],
                "recorded_at": row["recorded_at"],
            }
        )
    latest_daily_run = _load_latest_daily_run(state_root, portfolio_id)
    latest_daily_summary: dict[str, Any] | None = None
    if latest_daily_run is not None:
        actual = latest_daily_run["actual_snapshot"]
        latest_daily_summary = {
            "as_of": latest_daily_run["as_of"],
            "strategy_id": latest_daily_run["strategy_id"],
            "is_rebalance_day": bool(latest_daily_run["is_rebalance_day"]),
            "equity": float(actual["equity"]),
            "cash": float(actual["cash"]),
            "realized_pnl": float(actual["realized_pnl"]),
            "unrealized_pnl": float(actual["unrealized_pnl"]),
            "rebalance_instruction_count": len(latest_daily_run["rebalance_plan"]["instructions"]),
        }
    return {
        "portfolio_id": portfolio_id,
        "display_name": display_name,
        "recent_fills": recent_fills,
        "latest_daily_run": latest_daily_summary,
    }


def _fill_id_from_client_request(
    *,
    portfolio_id: str,
    user_id: int,
    client_request_id: str | None,
) -> str | None:
    if client_request_id is None:
        return None
    normalized = client_request_id.strip()
    if not normalized:
        raise ValueError("client_request_id must not be empty")
    return f"webapp-{portfolio_id}-{user_id}-{normalized}"


def _asset_version() -> str:
    digest = hashlib.sha256()
    for filename in ("app.css", "app.js"):
        digest.update((ASSETS_DIR / filename).read_bytes())
    return digest.hexdigest()[:12]


def render_index_html(*, title: str, mini_app_path: str, api_base_path: str) -> str:
    template = (ASSETS_DIR / "index.html").read_text(encoding="utf-8")
    config_json = json.dumps({"title": title, "apiBasePath": api_base_path})
    return (
        template.replace("__MINI_APP_BOOTSTRAP__", config_json)
        .replace("__MINI_APP_PATH__", mini_app_path)
        .replace("__ASSET_VERSION__", _asset_version())
    )


async def index_handler(request: web.Request) -> web.Response:
    config = request.app[MINI_APP_CONFIG_KEY]
    return web.Response(
        text=render_index_html(
            title=config.mini_app_title,
            mini_app_path=config.mini_app_path,
            api_base_path=f"{config.mini_app_path}/api",
        ),
        content_type="text/html",
    )


def register_static_assets(app: web.Application, *, mini_app_path: str) -> None:
    app.router.add_static(f"{mini_app_path}/assets", ASSETS_DIR)


def _portfolio_context(
    request: web.Request,
    *,
    portfolio_id: str,
) -> tuple[MiniAppIdentity, dict[str, Any], dict[str, Any]]:
    config = request.app[MINI_APP_CONFIG_KEY]
    controller = request.app[MINI_APP_CONTROLLER_KEY]
    identity = require_mini_app_identity(request, bot_token=config.bot_token)
    if not controller.ensure_portfolio_access(
        portfolio_id,
        chat_id=identity.chat_id,
        user_id=identity.user_id,
    ):
        raise web.HTTPForbidden(text="Access denied.")
    metadata = load_metadata(config.state_root, portfolio_id).to_dict()
    status = portfolio_status(config.state_root, config.promotion_registry_path, portfolio_id)
    return identity, metadata, status


async def me_handler(request: web.Request) -> web.Response:
    config = request.app[MINI_APP_CONFIG_KEY]
    controller = request.app[MINI_APP_CONTROLLER_KEY]
    identity = require_mini_app_identity(request, bot_token=config.bot_token)
    portfolio_ids = controller.authorized_portfolios(chat_id=identity.chat_id, user_id=identity.user_id)
    portfolios = []
    for portfolio_id in portfolio_ids:
        try:
            metadata = load_metadata(config.state_root, portfolio_id)
        except FileNotFoundError:
            continue
        portfolios.append(
            {
                "portfolio_id": portfolio_id,
                "display_name": metadata.display_name,
                "base_currency": metadata.base_currency,
            }
        )
    return web.json_response(
        {
            "user": {
                "user_id": identity.user_id,
                "chat_id": identity.chat_id,
                "username": identity.username,
                "first_name": identity.first_name,
                "last_name": identity.last_name,
            },
            "mini_app_url": config.public_mini_app_url,
            "portfolios": portfolios,
        }
    )


async def overview_handler(request: web.Request) -> web.Response:
    _, metadata, status = _portfolio_context(request, portfolio_id=request.match_info["portfolio_id"])
    return web.json_response(
        build_overview_payload(
            status=status,
            display_name=str(metadata["display_name"]),
            base_currency=str(metadata["base_currency"]),
        )
    )


async def holdings_handler(request: web.Request) -> web.Response:
    _, metadata, status = _portfolio_context(request, portfolio_id=request.match_info["portfolio_id"])
    return web.json_response(
        build_holdings_payload(
            status=status,
            display_name=str(metadata["display_name"]),
            base_currency=str(metadata["base_currency"]),
        )
    )


async def rebalance_handler(request: web.Request) -> web.Response:
    _, metadata, status = _portfolio_context(request, portfolio_id=request.match_info["portfolio_id"])
    return web.json_response(
        build_rebalance_payload(
            status=status,
            display_name=str(metadata["display_name"]),
            base_currency=str(metadata["base_currency"]),
        )
    )


async def activity_handler(request: web.Request) -> web.Response:
    config = request.app[MINI_APP_CONFIG_KEY]
    _identity, metadata, _status = _portfolio_context(request, portfolio_id=request.match_info["portfolio_id"])
    return web.json_response(
        build_activity_payload(
            state_root=config.state_root,
            portfolio_id=request.match_info["portfolio_id"],
            display_name=str(metadata["display_name"]),
        )
    )


async def fill_submit_handler(request: web.Request) -> web.Response:
    portfolio_id = request.match_info["portfolio_id"]
    identity, metadata, _status = _portfolio_context(request, portfolio_id=portfolio_id)
    payload = await request.json()
    try:
        result = record_fill(
            request.app[MINI_APP_CONFIG_KEY].state_root,
            portfolio_id,
            trade_date=date.fromisoformat(str(payload["trade_date"])),
            symbol=str(payload["symbol"]),
            side=str(payload["side"]),
            quantity=float(payload["quantity"]),
            fill_price=float(payload["fill_price"]),
            commission=float(payload.get("commission", 0.0)),
            slippage=float(payload.get("slippage", 0.0)),
            notes=str(payload.get("notes", "")),
            fill_id=_fill_id_from_client_request(
                portfolio_id=portfolio_id,
                user_id=identity.user_id,
                client_request_id=payload.get("client_request_id"),
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise web.HTTPBadRequest(text=str(exc) or "Invalid fill payload.") from exc

    updated_status = portfolio_status(
        request.app[MINI_APP_CONFIG_KEY].state_root,
        request.app[MINI_APP_CONFIG_KEY].promotion_registry_path,
        portfolio_id,
    )
    return web.json_response(
        {
            "fill": {
                "fill_id": result["fill_id"],
                "portfolio_id": result["portfolio_id"],
                "cash": float(result["cash"]),
                "realized_pnl": float(result["realized_pnl"]),
            },
            "overview": build_overview_payload(
                status=updated_status,
                display_name=str(metadata["display_name"]),
                base_currency=str(metadata["base_currency"]),
            ),
        }
    )


def register_mini_app_routes(app: web.Application, *, mini_app_path: str) -> None:
    api_base = f"{mini_app_path}/api"
    app.router.add_get(mini_app_path, index_handler)
    app.router.add_get(f"{mini_app_path}/", index_handler)
    register_static_assets(app, mini_app_path=mini_app_path)
    app.router.add_get(f"{api_base}/me", me_handler)
    app.router.add_get(f"{api_base}/portfolios/{{portfolio_id}}/overview", overview_handler)
    app.router.add_get(f"{api_base}/portfolios/{{portfolio_id}}/holdings", holdings_handler)
    app.router.add_get(f"{api_base}/portfolios/{{portfolio_id}}/rebalance", rebalance_handler)
    app.router.add_get(f"{api_base}/portfolios/{{portfolio_id}}/activity", activity_handler)
    app.router.add_post(f"{api_base}/portfolios/{{portfolio_id}}/fills", fill_submit_handler)
