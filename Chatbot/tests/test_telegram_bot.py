from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import subprocess
import sys
from datetime import date, datetime, time as clock_time, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

import polars as pl
import pytest
import yaml
from aiohttp.test_utils import TestClient, TestServer

from manual_portfolio.cli import (
    _default_signal_refresh_command,
    _post_close_run_context,
    daily_workflow_main,
    daily_run_main,
    record_fill_main,
    scheduled_daily_run_main,
)
from manual_portfolio.service import daily_run, onboard_portfolio, record_fill
from manual_portfolio.storage import fills_path, load_state, read_jsonl
from telegram_portfolio_bot.telegram_access import load_telegram_access_map
from telegram_portfolio_bot.telegram_bot import (
    TelegramBotController,
    build_telegram_fill_id,
    create_bot_app,
)
from telegram_portfolio_bot.telegram_config import load_telegram_bot_config
from telegram_portfolio_bot.telegram_notifications import (
    NotificationResult,
    TelegramNotifier,
)


def test_default_signal_refresh_command_uses_validated_sp500_10yr_path() -> None:
    assert _default_signal_refresh_command(source_prefix="sp500_10yr") == [
        "uv",
        "run",
        "python",
        "backtest_walkforward_10yr.py",
        "--prefix",
        "sp500_10yr",
        "--min-train-years",
        "2",
    ]


def test_default_signal_refresh_command_keeps_legacy_long_only_path() -> None:
    assert _default_signal_refresh_command(source_prefix="sp100_seed") == [
        "uv",
        "run",
        "python",
        "backtest_long_only.py",
        "--prefix",
        "sp100_seed",
    ]


def _write_registry(
    path: Path,
    *,
    active_strategy_id: str,
    signal_path: Path,
    price_path: Path,
    every_n_signals: int = 1,
    anchor_date: str = "2026-01-01",
    top_n: int = 2,
    gross_exposure: float = 1.0,
) -> Path:
    payload = {
        "schema_version": 1,
        "active_strategy_id": active_strategy_id,
        "source_prefix": "test",
        "rebalance_cadence": {
            "kind": "every_n_signal_dates",
            "every_n_signals": every_n_signals,
            "anchor_date": anchor_date,
        },
        "sizing_rule": {
            "kind": "top_n_equal_weight",
            "top_n": top_n,
            "gross_exposure": gross_exposure,
        },
        "artifacts": {
            "signal": {
                "path": signal_path.name,
                "date_col": "timestamp",
                "asset_col": "asset",
                "value_col": "signal",
            },
            "price": {
                "path": price_path.name,
                "date_col": "date",
                "asset_col": "asset",
                "value_col": "price",
            },
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path


def _write_market_data(tmp_path: Path) -> tuple[Path, Path]:
    signal_path = tmp_path / "signals.parquet"
    price_path = tmp_path / "prices.parquet"
    pl.DataFrame(
        {
            "timestamp": [
                date(2026, 1, 1),
                date(2026, 1, 1),
                date(2026, 1, 1),
                date(2026, 1, 2),
                date(2026, 1, 2),
                date(2026, 1, 2),
                date(2026, 1, 3),
                date(2026, 1, 3),
                date(2026, 1, 3),
            ],
            "asset": ["AAPL", "MSFT", "NVDA"] * 3,
            "signal": [10.0, 9.0, 1.0, 8.0, 7.0, 2.0, 1.0, 11.0, 10.0],
        }
    ).with_columns(pl.col("timestamp").cast(pl.Datetime)).write_parquet(signal_path)
    pl.DataFrame(
        {
            "date": [
                date(2026, 1, 1),
                date(2026, 1, 1),
                date(2026, 1, 1),
                date(2026, 1, 2),
                date(2026, 1, 2),
                date(2026, 1, 2),
                date(2026, 1, 3),
                date(2026, 1, 3),
                date(2026, 1, 3),
            ],
            "asset": ["AAPL", "MSFT", "NVDA"] * 3,
            "price": [100.0, 200.0, 50.0] * 3,
        }
    ).with_columns(pl.col("date").cast(pl.Datetime)).write_parquet(price_path)
    return signal_path, price_path


def _write_access_map(path: Path, *, portfolios: dict[str, dict[str, Any]]) -> Path:
    path.write_text(yaml.safe_dump({"portfolios": portfolios}, sort_keys=False))
    return path


def _write_bot_config(
    path: Path,
    *,
    state_root: Path,
    registry_path: Path,
    access_map_path: Path,
    strict_portfolio_validation: bool = False,
    register_webhook_on_startup: bool = False,
    bot_token: str = "123456:test-token",
    public_webhook_url: str = "https://example.com/telegram/webhook",
    add_user_password: str | None = None,
) -> Path:
    payload = {
        "bot_token": bot_token,
        "public_webhook_url": public_webhook_url,
        "webhook_secret_token": "secret-token",
        "add_user_password": add_user_password,
        "webhook_path": "/telegram/webhook",
        "bind_host": "127.0.0.1",
        "bind_port": 8081,
        "state_root": str(state_root),
        "promotion_registry_path": str(registry_path),
        "access_map_path": str(access_map_path),
        "register_webhook_on_startup": register_webhook_on_startup,
        "strict_portfolio_validation": strict_portfolio_validation,
        "dedupe_journal_path": str(
            state_root / ".telegram" / "processed_updates.jsonl"
        ),
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path


def _telegram_update(
    *,
    update_id: int,
    text: str,
    chat_id: int = 1001,
    user_id: int = 2001,
    message_id: int = 11,
    timestamp: int = 1767225600,
) -> dict[str, Any]:
    return {
        "update_id": update_id,
        "message": {
            "message_id": message_id,
            "date": timestamp,
            "text": text,
            "chat": {"id": chat_id, "type": "private"},
            "from": {"id": user_id, "is_bot": False, "first_name": "Ops"},
        },
    }


async def _post_webhook(
    config_path: Path, payload: dict[str, Any], *, secret: str | None = "secret-token"
) -> tuple[int, str]:
    app = create_bot_app(config_path=config_path)
    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()
    try:
        headers = {}
        if secret is not None:
            headers["X-Telegram-Bot-Api-Secret-Token"] = secret
        response = await client.post("/telegram/webhook", json=payload, headers=headers)
        return response.status, await response.text()
    finally:
        await client.close()


def _mini_app_init_data(
    *,
    bot_token: str,
    user_id: int = 2001,
    username: str = "ops",
    first_name: str = "Ops",
    chat_id: int | None = None,
    auth_date: int = 1767225600,
) -> str:
    payload: dict[str, str] = {
        "auth_date": str(auth_date),
        "query_id": "AAHdF6IQAAAAAN0XohDhrOrc",
        "user": json.dumps(
            {
                "id": user_id,
                "first_name": first_name,
                "username": username,
                "language_code": "en",
            },
            separators=(",", ":"),
        ),
    }
    if chat_id is not None:
        payload["receiver"] = json.dumps(
            {
                "id": chat_id,
                "type": "private",
                "first_name": first_name,
                "username": username,
            },
            separators=(",", ":"),
        )
    data_check_string = "\n".join(
        f"{key}={value}" for key, value in sorted(payload.items())
    )
    secret = hmac.new(b"WebAppData", bot_token.encode("utf-8"), hashlib.sha256).digest()
    payload["hash"] = hmac.new(
        secret, data_check_string.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    return urlencode(payload)


class FakeSender:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.sent: list[tuple[int, str]] = []

    async def send(self, *, chat_id: int, text: str) -> None:
        if self.fail:
            raise RuntimeError("telegram send failed")
        self.sent.append((chat_id, text))


def _setup_portfolio_env(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    state_root = tmp_path / "state"
    signal_path, price_path = _write_market_data(tmp_path)
    registry_path = _write_registry(
        tmp_path / "registry.yaml",
        active_strategy_id="alpha",
        signal_path=signal_path,
        price_path=price_path,
        every_n_signals=2,
    )
    onboard_portfolio(state_root, "p1", starting_cash=1000.0)
    access_map_path = _write_access_map(
        tmp_path / "telegram_access_map.yaml",
        portfolios={
            "p1": {
                "chats": [1001],
                "users": [2001],
                "delivery_chat_id": 1001,
            }
        },
    )
    config_path = _write_bot_config(
        tmp_path / "telegram_bot.yaml",
        state_root=state_root,
        registry_path=registry_path,
        access_map_path=access_map_path,
    )
    return state_root, registry_path, access_map_path, config_path


def test_access_map_enforces_joint_chat_and_user_authorization(tmp_path: Path) -> None:
    _, _, access_map_path, _ = _setup_portfolio_env(tmp_path)
    access_map = load_telegram_access_map(access_map_path)

    assert access_map.authorized_portfolios(chat_id=1001, user_id=2001) == ["p1"]
    assert access_map.authorized_portfolios(chat_id=1001, user_id=9999) == []
    assert access_map.authorized_portfolios(chat_id=9999, user_id=2001) == []


def test_unauthorized_status_and_fill_are_denied(tmp_path: Path) -> None:
    state_root, registry_path, access_map_path, config_path = _setup_portfolio_env(
        tmp_path
    )
    controller = TelegramBotController(
        config=load_telegram_bot_config(config_path),
        access_map=load_telegram_access_map(access_map_path),
    )

    assert controller.status_text("p1", chat_id=9999, user_id=2001) == "Access denied."
    text = controller.record_fill_from_command(
        "p1",
        side="buy",
        symbol="AAPL",
        quantity=1.0,
        fill_price=100.0,
        commission=0.0,
        slippage=0.0,
        notes="",
        context=type(
            "Ctx",
            (),
            {
                "chat_id": 9999,
                "user_id": 2001,
                "message_id": 1,
                "message_date": date(2026, 1, 2),
            },
        )(),
    )
    assert text == "Access denied."
    assert read_jsonl(fills_path(state_root, "p1")) == []


def test_fill_via_controller_writes_same_journal_and_deterministic_fill_id(
    tmp_path: Path,
) -> None:
    state_root, _, access_map_path, config_path = _setup_portfolio_env(tmp_path)
    controller = TelegramBotController(
        config=load_telegram_bot_config(config_path),
        access_map=load_telegram_access_map(access_map_path),
    )
    context = type(
        "Ctx",
        (),
        {
            "chat_id": 1001,
            "user_id": 2001,
            "message_id": 17,
            "message_date": date(2026, 1, 2),
        },
    )()

    text = controller.record_fill_from_command(
        "p1",
        side="buy",
        symbol="aapl",
        quantity=2.0,
        fill_price=100.0,
        commission=1.0,
        slippage=0.5,
        notes="telegram test",
        context=context,
    )

    rows = read_jsonl(fills_path(state_root, "p1"))
    assert "Fill recorded 2026-01-02" in text
    assert rows[0]["fill_id"] == build_telegram_fill_id(
        portfolio_id="p1",
        chat_id=1001,
        message_id=17,
        message_date=date(2026, 1, 2),
    )
    assert rows[0]["notes"] == "telegram test"


def test_daily_run_notifications_and_fill_confirmation(tmp_path: Path) -> None:
    state_root, registry_path, access_map_path, config_path = _setup_portfolio_env(
        tmp_path
    )
    record_fill(
        state_root,
        "p1",
        trade_date=date(2026, 1, 1),
        symbol="AAPL",
        side="buy",
        quantity=2.0,
        fill_price=100.0,
    )
    result = daily_run(state_root, registry_path, as_of=date(2026, 1, 3))

    sender = FakeSender()
    notifier = TelegramNotifier(
        config=load_telegram_bot_config(config_path),
        access_map=load_telegram_access_map(access_map_path),
        sender=sender,
    )
    notifications = asyncio.run(notifier.send_daily_run_notifications(result))
    fill_result = record_fill(
        state_root,
        "p1",
        trade_date=date(2026, 1, 3),
        symbol="AAPL",
        side="sell",
        quantity=1.0,
        fill_price=105.0,
        commission=1.0,
        slippage=0.5,
    )
    fill_notification = asyncio.run(
        notifier.send_fill_confirmation(
            portfolio_id="p1",
            trade_date=date(2026, 1, 3),
            side="sell",
            symbol="AAPL",
            quantity=1.0,
            fill_price=105.0,
            commission=1.0,
            slippage=0.5,
            fill_result=fill_result,
        )
    )

    assert [item.message_type for item in notifications] == [
        "daily_summary",
        "drift_summary",
        "rebalance_instructions",
    ]
    assert fill_notification is not None
    assert fill_notification.message_type == "fill_confirmation"
    assert len(sender.sent) == 4
    assert "Daily summary 2026-01-03" in sender.sent[0][1]
    assert "Fill confirmation 2026-01-03" in sender.sent[-1][1]


def test_daily_run_notifications_skip_rebalance_messages_when_not_rebalance_day(
    tmp_path: Path,
) -> None:
    state_root, registry_path, access_map_path, config_path = _setup_portfolio_env(
        tmp_path
    )
    result = daily_run(state_root, registry_path, as_of=date(2026, 1, 2))

    sender = FakeSender()
    notifier = TelegramNotifier(
        config=load_telegram_bot_config(config_path),
        access_map=load_telegram_access_map(access_map_path),
        sender=sender,
    )
    notifications = asyncio.run(notifier.send_daily_run_notifications(result))

    assert result["portfolios"]["p1"]["is_rebalance_day"] is False
    assert [item.message_type for item in notifications] == ["daily_summary"]
    assert len(sender.sent) == 1


def test_notification_failure_does_not_prevent_daily_outputs(tmp_path: Path) -> None:
    state_root, registry_path, access_map_path, config_path = _setup_portfolio_env(
        tmp_path
    )
    result = daily_run(state_root, registry_path, as_of=date(2026, 1, 3))
    sender = FakeSender(fail=True)
    notifier = TelegramNotifier(
        config=load_telegram_bot_config(config_path),
        access_map=load_telegram_access_map(access_map_path),
        sender=sender,
    )

    notifications = asyncio.run(notifier.send_daily_run_notifications(result))

    assert any(item.ok is False for item in notifications)
    assert (state_root / "p1" / "daily" / "2026-01-03" / "daily_run.json").exists()


def test_webhook_rejects_missing_or_invalid_secret(tmp_path: Path) -> None:
    _, _, _, config_path = _setup_portfolio_env(tmp_path)

    missing_status, _ = asyncio.run(
        _post_webhook(
            config_path, _telegram_update(update_id=1, text="/start"), secret=None
        )
    )
    invalid_status, _ = asyncio.run(
        _post_webhook(
            config_path, _telegram_update(update_id=2, text="/start"), secret="wrong"
        )
    )

    assert missing_status == 401
    assert invalid_status == 401


def test_webhook_start_portfolios_and_status_reflect_access(tmp_path: Path) -> None:
    _, _, _, config_path = _setup_portfolio_env(tmp_path)

    _, start_body = asyncio.run(
        _post_webhook(config_path, _telegram_update(update_id=1, text="/start"))
    )
    _, portfolio_body = asyncio.run(
        _post_webhook(config_path, _telegram_update(update_id=2, text="/portfolios"))
    )
    _, status_body = asyncio.run(
        _post_webhook(config_path, _telegram_update(update_id=3, text="/status p1"))
    )

    assert "Authorized portfolios: p1" in start_body
    assert "Authorized portfolios: p1" in portfolio_body
    assert "Status" in status_body
    assert "Portfolio: p1" in status_body
    assert "Open Portfolio App" in start_body


def test_webhook_whoami_returns_chat_and_user_ids(tmp_path: Path) -> None:
    _, _, _, config_path = _setup_portfolio_env(tmp_path)

    _, body = asyncio.run(
        _post_webhook(
            config_path,
            _telegram_update(
                update_id=4,
                text="/whoami",
                chat_id=1001,
                user_id=2001,
            ),
        )
    )

    assert "Telegram identity" in body
    assert "chat_id" in body
    assert "1001" in body
    assert "user_id" in body
    assert "2001" in body


def test_adduser_prompts_for_initial_cash_and_onboards_portfolio(
    tmp_path: Path,
) -> None:
    state_root = tmp_path / "state"
    signal_path, price_path = _write_market_data(tmp_path)
    registry_path = _write_registry(
        tmp_path / "registry.yaml",
        active_strategy_id="alpha",
        signal_path=signal_path,
        price_path=price_path,
    )
    access_map_path = _write_access_map(
        tmp_path / "telegram_access_map.yaml",
        portfolios={
            "other": {
                "chats": [9],
                "users": [9],
                "delivery_chat_id": 9,
            }
        },
    )
    config_path = _write_bot_config(
        tmp_path / "telegram_bot.yaml",
        state_root=state_root,
        registry_path=registry_path,
        access_map_path=access_map_path,
        add_user_password="let-me-in",
    )

    async def run_flow() -> tuple[str, str]:
        app = create_bot_app(config_path=config_path)
        server = TestServer(app)
        client = TestClient(server)
        await client.start_server()
        try:
            headers = {"X-Telegram-Bot-Api-Secret-Token": "secret-token"}
            prompt_response = await client.post(
                "/telegram/webhook",
                json=_telegram_update(
                    update_id=20,
                    text="/adduser let-me-in new-p1",
                    chat_id=1001,
                    user_id=2001,
                ),
                headers=headers,
            )
            body_response = await client.post(
                "/telegram/webhook",
                json=_telegram_update(
                    update_id=21,
                    text="25000",
                    chat_id=1001,
                    user_id=2001,
                    message_id=12,
                ),
                headers=headers,
            )
            return await prompt_response.text(), await body_response.text()
        finally:
            await client.close()

    prompt_body, body = asyncio.run(run_flow())

    access_map = load_telegram_access_map(access_map_path)
    state = load_state(state_root, "new-p1")
    assert "Creating portfolio new-p1" in prompt_body
    assert "Reply with initial cash amount." in prompt_body
    assert "Access granted for new-p1" in body
    assert access_map.authorized_portfolios(chat_id=1001, user_id=2001) == ["new-p1"]
    assert access_map.delivery_chat_id_for("new-p1") == 1001
    assert state.cash == 25000.0


def test_adduser_rejects_wrong_password(tmp_path: Path) -> None:
    state_root, registry_path, access_map_path, _ = _setup_portfolio_env(tmp_path)
    config_path = _write_bot_config(
        tmp_path / "telegram_bot.yaml",
        state_root=state_root,
        registry_path=registry_path,
        access_map_path=access_map_path,
        add_user_password="let-me-in",
    )

    _, body = asyncio.run(
        _post_webhook(
            config_path,
            _telegram_update(
                update_id=21,
                text="/adduser nope p1",
                chat_id=3001,
                user_id=4001,
            ),
        )
    )

    access_map = load_telegram_access_map(access_map_path)
    assert "Access denied." in body
    assert access_map.authorized_portfolios(chat_id=3001, user_id=4001) == []


def test_adduser_existing_portfolio_grants_access_immediately(tmp_path: Path) -> None:
    state_root, registry_path, access_map_path, _ = _setup_portfolio_env(tmp_path)
    config_path = _write_bot_config(
        tmp_path / "telegram_bot.yaml",
        state_root=state_root,
        registry_path=registry_path,
        access_map_path=access_map_path,
        add_user_password="let-me-in",
    )

    _, body = asyncio.run(
        _post_webhook(
            config_path,
            _telegram_update(
                update_id=22,
                text="/adduser let-me-in p1",
                chat_id=3001,
                user_id=4001,
            ),
        )
    )

    access_map = load_telegram_access_map(access_map_path)
    assert "Access granted for p1" in body
    assert access_map.authorized_portfolios(chat_id=3001, user_id=4001) == ["p1"]


def test_webhook_fill_buy_sell_and_malformed_usage(tmp_path: Path) -> None:
    state_root, _, _, config_path = _setup_portfolio_env(tmp_path)

    _, buy_body = asyncio.run(
        _post_webhook(
            config_path,
            _telegram_update(
                update_id=10,
                text="/fill p1 buy AAPL 2 100 1.25 0.5 opening lot",
                message_id=501,
            ),
        )
    )
    _, sell_body = asyncio.run(
        _post_webhook(
            config_path,
            _telegram_update(
                update_id=11,
                text="/fill p1 sell AAPL 1 110 1 0.25 trim",
                message_id=502,
            ),
        )
    )
    _, bad_body = asyncio.run(
        _post_webhook(
            config_path,
            _telegram_update(update_id=12, text="/fill p1 buy", message_id=503),
        )
    )

    rows = read_jsonl(fills_path(state_root, "p1"))
    state = load_state(state_root, "p1")
    assert "Fill recorded" in buy_body
    assert "Fill recorded" in sell_body
    assert "Usage: /fill" in bad_body
    assert len(rows) == 2
    assert rows[0]["notes"] == "opening lot"
    assert rows[1]["notes"] == "trim"
    assert state.holdings["AAPL"].quantity == 1.0


def test_duplicate_webhook_update_does_not_duplicate_fill(tmp_path: Path) -> None:
    state_root, _, _, config_path = _setup_portfolio_env(tmp_path)
    payload = _telegram_update(
        update_id=77,
        text="/fill p1 buy AAPL 2 100",
        message_id=700,
    )

    _, first_body = asyncio.run(_post_webhook(config_path, payload))
    _, second_body = asyncio.run(_post_webhook(config_path, payload))

    assert "Fill recorded" in first_body
    assert second_body.strip() == "{}"
    assert len(read_jsonl(fills_path(state_root, "p1"))) == 1


def test_mini_app_requires_valid_auth(tmp_path: Path) -> None:
    _, _, _, config_path = _setup_portfolio_env(tmp_path)

    async def run_case() -> tuple[int, str]:
        app = create_bot_app(config_path=config_path)
        server = TestServer(app)
        client = TestClient(server)
        await client.start_server()
        try:
            response = await client.get("/mini-app/api/me")
            return response.status, await response.text()
        finally:
            await client.close()

    status, body = asyncio.run(run_case())
    assert status == 401
    assert "Missing Telegram Mini App auth." in body


def test_mini_app_endpoints_return_portfolio_data(tmp_path: Path) -> None:
    _, _, _, config_path = _setup_portfolio_env(tmp_path)
    init_data = _mini_app_init_data(
        bot_token="123456:test-token", user_id=2001, chat_id=1001
    )

    async def run_case() -> dict[str, Any]:
        app = create_bot_app(config_path=config_path)
        server = TestServer(app)
        client = TestClient(server)
        await client.start_server()
        try:
            headers = {"X-Telegram-Init-Data": init_data}
            me = await (await client.get("/mini-app/api/me", headers=headers)).json()
            overview = await (
                await client.get(
                    "/mini-app/api/portfolios/p1/overview", headers=headers
                )
            ).json()
            holdings = await (
                await client.get(
                    "/mini-app/api/portfolios/p1/holdings", headers=headers
                )
            ).json()
            rebalance = await (
                await client.get(
                    "/mini-app/api/portfolios/p1/rebalance", headers=headers
                )
            ).json()
            activity = await (
                await client.get(
                    "/mini-app/api/portfolios/p1/activity", headers=headers
                )
            ).json()
            html = await (await client.get("/mini-app")).text()
            return {
                "me": me,
                "overview": overview,
                "holdings": holdings,
                "rebalance": rebalance,
                "activity": activity,
                "html": html,
            }
        finally:
            await client.close()

    payload = asyncio.run(run_case())
    assert payload["me"]["portfolios"][0]["portfolio_id"] == "p1"
    assert payload["overview"]["portfolio_id"] == "p1"
    assert payload["overview"]["cash"] == 1000.0
    assert payload["overview"]["run_status"] == "warning"
    assert any("latest signal date" in item for item in payload["overview"]["warnings"])
    assert any("latest price date" in item for item in payload["overview"]["warnings"])
    assert payload["holdings"]["holdings"] == []
    assert payload["rebalance"]["rows"][0]["asset"]
    assert payload["activity"]["recent_fills"] == []
    assert "window.__PORTFOLIO_APP_CONFIG__" in payload["html"]
    assert payload["html"].count('id="fill-panel"') == 1
    assert payload["html"].count("Record Fill") == 2
    assert "/assets/app.css?v=" in payload["html"]
    assert "/assets/app.js?v=" in payload["html"]
    assert "__ASSET_VERSION__" not in payload["html"]


def test_mini_app_status_button_uses_existing_rebalance_view() -> None:
    webapp_root = (
        Path(__file__).resolve().parents[1] / "telegram_portfolio_bot" / "webapp"
    )
    html = (webapp_root / "index.html").read_text()
    js = (webapp_root / "app.js").read_text()

    assert 'id="jump-status"' in html
    assert "Portfolio Status" in html
    assert 'id="jump-rebalance"' in html
    assert 'id="jump-fill"' in html
    assert 'elements.jumpStatus.addEventListener("click", async () => {' in js
    assert "await loadPortfolioData();" in js
    assert 'activateTab("rebalance");' in js


def test_mini_app_fill_submission_uses_existing_ledger_path(tmp_path: Path) -> None:
    state_root, _, _, config_path = _setup_portfolio_env(tmp_path)
    init_data = _mini_app_init_data(
        bot_token="123456:test-token", user_id=2001, chat_id=1001
    )

    async def run_case() -> tuple[dict[str, Any], int, str]:
        app = create_bot_app(config_path=config_path)
        server = TestServer(app)
        client = TestClient(server)
        await client.start_server()
        try:
            headers = {"X-Telegram-Init-Data": init_data}
            payload = {
                "trade_date": "2026-01-02",
                "symbol": "AAPL",
                "side": "buy",
                "quantity": 2,
                "fill_price": 100,
                "commission": 1,
                "slippage": 0.5,
                "notes": "mini app fill",
                "client_request_id": "abc-123",
            }
            response = await client.post(
                "/mini-app/api/portfolios/p1/fills",
                headers=headers,
                json=payload,
            )
            duplicate = await client.post(
                "/mini-app/api/portfolios/p1/fills",
                headers=headers,
                json=payload,
            )
            return await response.json(), duplicate.status, await duplicate.text()
        finally:
            await client.close()

    response, duplicate_status, duplicate_body = asyncio.run(run_case())
    fills = read_jsonl(fills_path(state_root, "p1"))
    assert response["fill"]["fill_id"] == "webapp-p1-2001-abc-123"
    assert fills[0]["fill_id"] == "webapp-p1-2001-abc-123"
    assert fills[0]["notes"] == "mini app fill"
    assert response["overview"]["cash"] == 798.5
    assert duplicate_status == 400
    assert "duplicate fill_id" in duplicate_body


def test_daily_run_cli_stays_computation_only_when_notify_flag_is_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    state_root, registry_path, _, _ = _setup_portfolio_env(tmp_path)

    monkeypatch.setattr(
        "sys.argv",
        [
            "daily-run",
            "--state-root",
            str(state_root),
            "--promotion-registry",
            str(registry_path),
            "--as-of",
            "2026-01-03",
            "--notify-telegram",
        ],
    )

    daily_run_main()

    payload = json.loads(capsys.readouterr().out)
    assert "telegram_notifications" not in payload
    assert (state_root / "p1" / "daily" / "2026-01-03" / "daily_run.json").exists()


def test_daily_workflow_runs_update_then_signal_before_summary_and_then_notifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    state_root, registry_path, access_map_path, config_path = _setup_portfolio_env(
        tmp_path
    )
    events: list[str] = []

    def fake_update_run(
        command: list[str], **_: Any
    ) -> subprocess.CompletedProcess[str]:
        events.append(command[0])
        return subprocess.CompletedProcess(command, 0)

    def fake_daily_run(command: list[str], **_: Any) -> tuple[int, dict[str, Any]]:
        events.append(f"daily:{command[0]}")
        return 0, daily_run(state_root, registry_path, as_of=date(2026, 1, 3))

    async def fake_notify(**kwargs: Any) -> list[NotificationResult]:
        events.append("notify")
        assert kwargs["result"]["artifacts"]["p1"]["daily_run"].endswith(
            "daily_run.json"
        )
        return [
            NotificationResult(
                portfolio_id="p1",
                message_type="daily_summary",
                ok=True,
                detail="sent",
            )
        ]

    monkeypatch.setattr("manual_portfolio.cli.subprocess.run", fake_update_run)
    monkeypatch.setattr("manual_portfolio.cli._run_capture_json", fake_daily_run)
    monkeypatch.setattr(
        "manual_portfolio.cli._send_daily_run_notifications", fake_notify
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "daily-workflow",
            "--state-root",
            str(state_root),
            "--promotion-registry",
            str(registry_path),
            "--as-of",
            "2026-01-03",
            "--notify-telegram",
            "--telegram-config",
            str(config_path),
            "--telegram-access-map",
            str(access_map_path),
            "--update-command",
            "mock-update",
            "--signal-command",
            "mock-signal",
        ],
    )

    daily_workflow_main()

    payload = json.loads(capsys.readouterr().out)
    assert events == ["mock-update", "mock-signal", "daily:daily-run", "notify"]
    assert payload["promoted_artifacts"]["price"].endswith("prices.parquet")
    assert payload["promoted_artifacts"]["signal"].endswith("signals.parquet")
    assert payload["telegram_notifications"][0]["ok"] is True


def test_daily_workflow_stops_on_update_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_root, registry_path, _, _ = _setup_portfolio_env(tmp_path)
    events: list[str] = []

    def fake_update_run(
        command: list[str], **_: Any
    ) -> subprocess.CompletedProcess[str]:
        events.append(command[0])
        return subprocess.CompletedProcess(command, 9)

    def fake_daily_run(command: list[str], **_: Any) -> tuple[int, dict[str, Any]]:
        events.append("daily")
        return 0, {}

    monkeypatch.setattr("manual_portfolio.cli.subprocess.run", fake_update_run)
    monkeypatch.setattr("manual_portfolio.cli._run_capture_json", fake_daily_run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "daily-workflow",
            "--state-root",
            str(state_root),
            "--promotion-registry",
            str(registry_path),
            "--as-of",
            "2026-01-03",
            "--update-command",
            "mock-update",
            "--signal-command",
            "mock-signal",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        daily_workflow_main()

    assert exc_info.value.code == 9
    assert events == ["mock-update"]


def test_daily_workflow_stops_on_signal_refresh_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_root, registry_path, _, _ = _setup_portfolio_env(tmp_path)
    events: list[str] = []

    def fake_run(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        events.append(command[0])
        return subprocess.CompletedProcess(
            command,
            7 if command[0] == "mock-signal" else 0,
        )

    def fake_daily_run(command: list[str], **_: Any) -> tuple[int, dict[str, Any]]:
        events.append("daily")
        return 0, {}

    monkeypatch.setattr("manual_portfolio.cli.subprocess.run", fake_run)
    monkeypatch.setattr("manual_portfolio.cli._run_capture_json", fake_daily_run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "daily-workflow",
            "--state-root",
            str(state_root),
            "--promotion-registry",
            str(registry_path),
            "--as-of",
            "2026-01-03",
            "--update-command",
            "mock-update",
            "--signal-command",
            "mock-signal",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        daily_workflow_main()

    assert exc_info.value.code == 7
    assert events == ["mock-update", "mock-signal"]


def test_daily_workflow_stops_when_promoted_price_artifact_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_root, registry_path, _, _ = _setup_portfolio_env(tmp_path)
    payload = yaml.safe_load(registry_path.read_text())
    (registry_path.parent / payload["artifacts"]["price"]["path"]).unlink()
    registry_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    events: list[str] = []

    def fake_run(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        events.append(command[0])
        return subprocess.CompletedProcess(command, 0)

    def fake_daily_run(command: list[str], **_: Any) -> tuple[int, dict[str, Any]]:
        events.append("daily")
        return 0, {}

    monkeypatch.setattr("manual_portfolio.cli.subprocess.run", fake_run)
    monkeypatch.setattr("manual_portfolio.cli._run_capture_json", fake_daily_run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "daily-workflow",
            "--state-root",
            str(state_root),
            "--promotion-registry",
            str(registry_path),
            "--as-of",
            "2026-01-03",
            "--update-command",
            "mock-update",
            "--signal-command",
            "mock-signal",
        ],
    )

    with pytest.raises(FileNotFoundError, match="missing promoted price artifact"):
        daily_workflow_main()

    assert events == ["mock-update"]


def test_daily_workflow_stops_when_promoted_price_artifact_is_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_root, registry_path, _, _ = _setup_portfolio_env(tmp_path)
    events: list[str] = []

    def fake_run(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        events.append(command[0])
        return subprocess.CompletedProcess(command, 0)

    def fake_daily_run(command: list[str], **_: Any) -> tuple[int, dict[str, Any]]:
        events.append("daily")
        return 0, {}

    monkeypatch.setattr("manual_portfolio.cli.subprocess.run", fake_run)
    monkeypatch.setattr("manual_portfolio.cli._run_capture_json", fake_daily_run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "daily-workflow",
            "--state-root",
            str(state_root),
            "--promotion-registry",
            str(registry_path),
            "--as-of",
            "2026-01-04",
            "--update-command",
            "mock-update",
            "--signal-command",
            "mock-signal",
        ],
    )

    with pytest.raises(ValueError, match="promoted price artifact is stale"):
        daily_workflow_main()

    assert events == ["mock-update"]


def test_daily_workflow_stops_when_promoted_signal_artifact_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_root, registry_path, _, _ = _setup_portfolio_env(tmp_path)
    payload = yaml.safe_load(registry_path.read_text())
    (registry_path.parent / payload["artifacts"]["signal"]["path"]).unlink()
    registry_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    events: list[str] = []

    def fake_run(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        events.append(command[0])
        return subprocess.CompletedProcess(command, 0)

    def fake_daily_run(command: list[str], **_: Any) -> tuple[int, dict[str, Any]]:
        events.append("daily")
        return 0, {}

    monkeypatch.setattr("manual_portfolio.cli.subprocess.run", fake_run)
    monkeypatch.setattr("manual_portfolio.cli._run_capture_json", fake_daily_run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "daily-workflow",
            "--state-root",
            str(state_root),
            "--promotion-registry",
            str(registry_path),
            "--as-of",
            "2026-01-03",
            "--update-command",
            "mock-update",
            "--signal-command",
            "mock-signal",
        ],
    )

    with pytest.raises(FileNotFoundError, match="missing promoted signal artifact"):
        daily_workflow_main()

    assert events == ["mock-update", "mock-signal"]


def test_daily_workflow_stops_when_promoted_signal_artifact_is_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_root, registry_path, _, _ = _setup_portfolio_env(tmp_path)
    payload = yaml.safe_load(registry_path.read_text())
    price_path = registry_path.parent / payload["artifacts"]["price"]["path"]
    pl.DataFrame(
        {
            "date": [date(2026, 1, 4)],
            "asset": ["AAPL"],
            "price": [100.0],
        }
    ).with_columns(pl.col("date").cast(pl.Datetime)).write_parquet(price_path)
    events: list[str] = []

    def fake_run(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        events.append(command[0])
        return subprocess.CompletedProcess(command, 0)

    def fake_daily_run(command: list[str], **_: Any) -> tuple[int, dict[str, Any]]:
        events.append("daily")
        return 0, {}

    monkeypatch.setattr("manual_portfolio.cli.subprocess.run", fake_run)
    monkeypatch.setattr("manual_portfolio.cli._run_capture_json", fake_daily_run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "daily-workflow",
            "--state-root",
            str(state_root),
            "--promotion-registry",
            str(registry_path),
            "--as-of",
            "2026-01-04",
            "--update-command",
            "mock-update",
            "--signal-command",
            "mock-signal",
        ],
    )

    with pytest.raises(ValueError, match="promoted signal artifact is stale"):
        daily_workflow_main()

    assert events == ["mock-update", "mock-signal"]


def test_daily_workflow_stops_on_daily_run_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_root, registry_path, _, _ = _setup_portfolio_env(tmp_path)
    events: list[str] = []

    def fake_update_run(
        command: list[str], **_: Any
    ) -> subprocess.CompletedProcess[str]:
        events.append(command[0])
        return subprocess.CompletedProcess(command, 0)

    def fake_daily_run(command: list[str], **_: Any) -> tuple[int, None]:
        events.append("daily")
        return 8, None

    monkeypatch.setattr("manual_portfolio.cli.subprocess.run", fake_update_run)
    monkeypatch.setattr("manual_portfolio.cli._run_capture_json", fake_daily_run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "daily-workflow",
            "--state-root",
            str(state_root),
            "--promotion-registry",
            str(registry_path),
            "--as-of",
            "2026-01-03",
            "--update-command",
            "mock-update",
            "--signal-command",
            "mock-signal",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        daily_workflow_main()

    assert exc_info.value.code == 8
    assert events == ["mock-update", "mock-signal", "daily"]


def test_daily_workflow_smoke_with_mocked_update_and_telegram(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    state_root, registry_path, access_map_path, config_path = _setup_portfolio_env(
        tmp_path
    )

    async def fake_notify(**_: Any) -> list[NotificationResult]:
        return [
            NotificationResult(
                portfolio_id="p1",
                message_type="daily_summary",
                ok=True,
                detail="sent",
            )
        ]

    monkeypatch.setattr(
        "manual_portfolio.cli._send_daily_run_notifications", fake_notify
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "daily-workflow",
            "--state-root",
            str(state_root),
            "--promotion-registry",
            str(registry_path),
            "--as-of",
            "2026-01-03",
            "--notify-telegram",
            "--telegram-config",
            str(config_path),
            "--telegram-access-map",
            str(access_map_path),
            "--update-command",
            sys.executable,
            "-c",
            "pass",
            "--signal-command",
            sys.executable,
            "-c",
            "pass",
        ],
    )

    daily_workflow_main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["promoted_artifacts"]["price"].endswith("prices.parquet")
    assert payload["promoted_artifacts"]["signal"].endswith("signals.parquet")
    assert payload["artifacts"]["p1"]["daily_run"].endswith("daily_run.json")
    assert payload["artifacts"]["p1"]["rebalance_plan"].endswith("rebalance_plan.json")
    assert payload["telegram_notifications"][0]["message_type"] == "daily_summary"


def test_scheduled_daily_run_computes_new_york_as_of_and_wait() -> None:
    as_of, wait_seconds = _post_close_run_context(
        now=datetime(2026, 1, 3, 15, 55, tzinfo=ZoneInfo("America/New_York")),
        market_close=clock_time(hour=16),
        post_close_buffer=timedelta(minutes=15),
    )

    assert as_of == date(2026, 1, 3)
    assert wait_seconds == 20 * 60


def test_scheduled_daily_run_invokes_daily_workflow_with_telegram_notify(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_root, registry_path, access_map_path, config_path = _setup_portfolio_env(
        tmp_path
    )
    commands: list[list[str]] = []

    def fake_run(
        command: list[str], *, check: bool
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        assert check is False
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("manual_portfolio.cli.subprocess.run", fake_run)
    monkeypatch.setattr(
        "manual_portfolio.cli._post_close_run_context",
        lambda **_: (date(2026, 1, 3), 0.0),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "scheduled-daily-run",
            "--state-root",
            str(state_root),
            "--promotion-registry",
            str(registry_path),
            "--telegram-config",
            str(config_path),
            "--telegram-access-map",
            str(access_map_path),
            "--portfolio-id",
            "p1",
        ],
    )

    scheduled_daily_run_main()

    assert commands == [
        [
            "daily-workflow",
            "--state-root",
            str(state_root),
            "--promotion-registry",
            str(registry_path),
            "--as-of",
            "2026-01-03",
            "--notify-telegram",
            "--telegram-config",
            str(config_path),
            "--telegram-access-map",
            str(access_map_path),
            "--portfolio-id",
            "p1",
        ]
    ]


def test_scheduled_daily_run_exits_nonzero_when_daily_run_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_root, registry_path, access_map_path, config_path = _setup_portfolio_env(
        tmp_path
    )

    def fake_run(
        command: list[str], *, check: bool
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 7)

    monkeypatch.setattr("manual_portfolio.cli.subprocess.run", fake_run)
    monkeypatch.setattr(
        "manual_portfolio.cli._post_close_run_context",
        lambda **_: (date(2026, 1, 3), 0.0),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "scheduled-daily-run",
            "--state-root",
            str(state_root),
            "--promotion-registry",
            str(registry_path),
            "--telegram-config",
            str(config_path),
            "--telegram-access-map",
            str(access_map_path),
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        scheduled_daily_run_main()

    assert exc_info.value.code == 7


def test_record_fill_cli_notifies_after_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    state_root, _, _, _ = _setup_portfolio_env(tmp_path)

    async def fake_notify(**kwargs: Any) -> NotificationResult:
        assert read_jsonl(fills_path(state_root, "p1"))
        return NotificationResult(
            portfolio_id=kwargs["portfolio_id"],
            message_type="fill_confirmation",
            ok=True,
            detail="sent",
        )

    monkeypatch.setattr(
        "manual_portfolio.cli._send_fill_confirmation_notification", fake_notify
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "record-fill",
            "--state-root",
            str(state_root),
            "--portfolio-id",
            "p1",
            "--trade-date",
            "2026-01-02",
            "--symbol",
            "AAPL",
            "--side",
            "buy",
            "--quantity",
            "1",
            "--fill-price",
            "100",
            "--notify-telegram",
        ],
    )

    record_fill_main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["telegram_notifications"][0]["ok"] is True
    assert len(read_jsonl(fills_path(state_root, "p1"))) == 1


def test_startup_validation_fails_fast_for_bad_config_and_access_map(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_root = tmp_path / "state"
    signal_path, price_path = _write_market_data(tmp_path)
    registry_path = _write_registry(
        tmp_path / "registry.yaml",
        active_strategy_id="alpha",
        signal_path=signal_path,
        price_path=price_path,
    )
    access_map_path = _write_access_map(
        tmp_path / "telegram_access_map.yaml",
        portfolios={"missing": {"chats": [1], "users": [2], "delivery_chat_id": 1}},
    )
    config_path = _write_bot_config(
        tmp_path / "telegram_bot.yaml",
        state_root=state_root,
        registry_path=registry_path,
        access_map_path=access_map_path,
        strict_portfolio_validation=True,
    )

    with pytest.raises(ValueError, match="unknown portfolio_id"):
        create_bot_app(config_path=config_path)

    env_config_path = tmp_path / "telegram_env.yaml"
    env_config_path.write_text(
        yaml.safe_dump(
            {
                "bot_token": "env:TELEGRAM_BOT_TOKEN",
                "public_webhook_url": "https://example.com/telegram/webhook",
                "webhook_secret_token": "env:TELEGRAM_WEBHOOK_SECRET",
                "access_map_path": str(access_map_path),
            },
            sort_keys=False,
        )
    )
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_WEBHOOK_SECRET", raising=False)
    with pytest.raises(ValueError, match="missing required environment variable"):
        load_telegram_bot_config(env_config_path)

    missing_url_path = tmp_path / "telegram_missing_url.yaml"
    missing_url_path.write_text(
        yaml.safe_dump(
            {
                "bot_token": "token",
                "public_webhook_url": "",
                "webhook_secret_token": "secret",
                "access_map_path": str(access_map_path),
            },
            sort_keys=False,
        )
    )
    with pytest.raises(ValueError, match="public_webhook_url is required"):
        load_telegram_bot_config(missing_url_path)

    malformed_access_map_path = tmp_path / "telegram_bad_access.yaml"
    malformed_access_map_path.write_text(yaml.safe_dump({"portfolios": []}))
    config_path = _write_bot_config(
        tmp_path / "telegram_malformed.yaml",
        state_root=state_root,
        registry_path=registry_path,
        access_map_path=malformed_access_map_path,
    )
    with pytest.raises(ValueError, match="non-empty 'portfolios' mapping"):
        create_bot_app(config_path=config_path)
