from __future__ import annotations

import asyncio
import logging
import secrets
import shlex
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from aiohttp import web
from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.methods import SendMessage
from aiogram.types import KeyboardButton, MenuButtonWebApp, Message, ReplyKeyboardMarkup, WebAppInfo
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application

from manual_portfolio.service import onboard_portfolio, portfolio_status, record_fill
from manual_portfolio.storage import append_jsonl, ensure_dir, read_jsonl
from .telegram_access import TelegramAccessMap, load_telegram_access_map, upsert_access_entry
from .telegram_config import TelegramBotConfig, load_telegram_bot_config
from .telegram_mini_app import (
    MINI_APP_CONFIG_KEY,
    MINI_APP_CONTROLLER_KEY,
    MiniAppAuthError,
    register_mini_app_routes,
)
from .telegram_messages import (
    add_user_usage,
    fill_usage,
    format_add_user_prompt,
    format_add_user_success,
    format_portfolios_reply,
    format_start_reply,
    format_status_reply,
    format_whoami_reply,
    help_text,
)


LOGGER = logging.getLogger(__name__)
CONFIG_KEY = web.AppKey("telegram_config", TelegramBotConfig)
ACCESS_MAP_KEY = web.AppKey("telegram_access_map", TelegramAccessMap)
DISPATCHER_KEY = web.AppKey("telegram_dispatcher", Dispatcher)
BOT_KEY = web.AppKey("telegram_bot", Bot)
CONTROLLER_KEY = web.AppKey("telegram_controller", Any)


@dataclass(frozen=True)
class CommandContext:
    chat_id: int | None
    user_id: int | None
    message_id: int | None
    message_date: date


@dataclass(frozen=True)
class PendingOnboarding:
    portfolio_id: str
    chat_id: int
    user_id: int


@dataclass
class ProcessedUpdateJournal:
    path: Path
    max_entries: int = 2048

    def __post_init__(self) -> None:
        ensure_dir(self.path.parent)
        self.path.touch(exist_ok=True)
        rows = read_jsonl(self.path)
        recent_ids = [int(row["update_id"]) for row in rows[-self.max_entries :]]
        self._seen = set(recent_ids)
        self._queue = deque(recent_ids, maxlen=self.max_entries)

    def mark_seen(self, update_id: int) -> bool:
        if update_id in self._seen:
            return False
        if len(self._queue) == self._queue.maxlen:
            evicted = self._queue.popleft()
            self._seen.discard(evicted)
        self._queue.append(update_id)
        self._seen.add(update_id)
        append_jsonl(
            self.path,
            {
                "update_id": update_id,
                "recorded_at": datetime.now(tz=UTC).replace(microsecond=0).isoformat(),
            },
        )
        return True


class DeduplicatingRequestHandler(SimpleRequestHandler):
    def __init__(self, *args: Any, update_journal: ProcessedUpdateJournal, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.update_journal = update_journal

    async def _load_update(self, request: web.Request) -> dict[str, Any]:
        return await request.json(loads=self.bot.session.json_loads)

    async def _background_feed_update(self, bot: Bot, update: dict[str, Any]) -> None:
        result = await self.dispatcher.feed_raw_update(bot=bot, update=update, **self.data)
        if result is not None:
            await self.dispatcher.silent_call_request(bot=bot, result=result)

    async def handle(self, request: web.Request) -> web.Response:
        bot = await self.resolve_bot(request)
        if not self.verify_secret(request.headers.get("X-Telegram-Bot-Api-Secret-Token", ""), bot):
            return web.Response(body="Unauthorized", status=401)
        update = await self._load_update(request)
        update_id = update.get("update_id")
        if not isinstance(update_id, int):
            return web.Response(body="Bad Request", status=400)
        if not self.update_journal.mark_seen(update_id):
            return web.json_response({})
        if self.handle_in_background:
            task = asyncio.create_task(self._background_feed_update(bot=bot, update=update))
            self._background_feed_update_tasks.add(task)
            task.add_done_callback(self._background_feed_update_tasks.discard)
            return web.json_response({})
        result = await self.dispatcher.feed_webhook_update(bot, update, **self.data)
        return web.Response(body=self._build_response_writer(bot=bot, result=result))


@dataclass
class TelegramBotController:
    config: TelegramBotConfig
    access_map: TelegramAccessMap
    pending_onboarding: dict[tuple[int, int], PendingOnboarding] = field(default_factory=dict)

    def authorized_portfolios(self, *, chat_id: int | None, user_id: int | None) -> list[str]:
        return self.access_map.authorized_portfolios(chat_id=chat_id, user_id=user_id)

    def add_user_access(
        self,
        portfolio_id: str,
        *,
        password: str,
        context: CommandContext,
    ) -> str:
        if self.config.add_user_password is None:
            return "Add user is disabled."
        if not secrets.compare_digest(password, self.config.add_user_password):
            return "Access denied."
        if context.chat_id is None or context.user_id is None:
            return "Unable to identify chat or user."
        if not (self.config.state_root / portfolio_id).exists():
            self.pending_onboarding[(context.chat_id, context.user_id)] = PendingOnboarding(
                portfolio_id=portfolio_id,
                chat_id=context.chat_id,
                user_id=context.user_id,
            )
            return format_add_user_prompt(portfolio_id=portfolio_id)
        self.access_map = upsert_access_entry(
            self.config.access_map_path,
            portfolio_id=portfolio_id,
            chat_id=context.chat_id,
            user_id=context.user_id,
            delivery_chat_id=context.chat_id,
        )
        portfolio_ids = self.authorized_portfolios(chat_id=context.chat_id, user_id=context.user_id)
        return format_add_user_success(portfolio_id=portfolio_id, portfolio_ids=portfolio_ids)

    def handle_pending_onboarding(self, *, text: str, context: CommandContext) -> str | None:
        if context.chat_id is None or context.user_id is None:
            return None
        key = (context.chat_id, context.user_id)
        pending = self.pending_onboarding.get(key)
        if pending is None:
            return None
        try:
            starting_cash = float(text.strip())
        except ValueError:
            return "Initial cash must be a number."
        if starting_cash < 0:
            return "Initial cash must be non-negative."
        onboard_portfolio(
            self.config.state_root,
            pending.portfolio_id,
            starting_cash=starting_cash,
        )
        self.access_map = upsert_access_entry(
            self.config.access_map_path,
            portfolio_id=pending.portfolio_id,
            chat_id=context.chat_id,
            user_id=context.user_id,
            delivery_chat_id=context.chat_id,
        )
        self.pending_onboarding.pop(key, None)
        portfolio_ids = self.authorized_portfolios(chat_id=context.chat_id, user_id=context.user_id)
        return format_add_user_success(
            portfolio_id=pending.portfolio_id,
            portfolio_ids=portfolio_ids,
        )

    def ensure_portfolio_access(
        self,
        portfolio_id: str,
        *,
        chat_id: int | None,
        user_id: int | None,
    ) -> bool:
        return self.access_map.is_authorized(
            portfolio_id,
            chat_id=chat_id,
            user_id=user_id,
        )

    def status_text(self, portfolio_id: str, *, chat_id: int | None, user_id: int | None) -> str:
        if not self.ensure_portfolio_access(portfolio_id, chat_id=chat_id, user_id=user_id):
            return "Access denied."
        try:
            status = portfolio_status(
                self.config.state_root,
                self.config.promotion_registry_path,
                portfolio_id,
            )
        except Exception:
            LOGGER.exception("telegram status command failed: portfolio=%s", portfolio_id)
            return "Unable to load portfolio status."
        return format_status_reply(status)

    def record_fill_from_command(
        self,
        portfolio_id: str,
        *,
        side: str,
        symbol: str,
        quantity: float,
        fill_price: float,
        commission: float,
        slippage: float,
        notes: str,
        context: CommandContext,
    ) -> str:
        if not self.ensure_portfolio_access(portfolio_id, chat_id=context.chat_id, user_id=context.user_id):
            return "Access denied."
        fill_id = build_telegram_fill_id(
            portfolio_id=portfolio_id,
            chat_id=context.chat_id,
            message_id=context.message_id,
            message_date=context.message_date,
        )
        result = record_fill(
            self.config.state_root,
            portfolio_id,
            trade_date=context.message_date,
            symbol=symbol,
            side=side,
            quantity=quantity,
            fill_price=fill_price,
            commission=commission,
            slippage=slippage,
            notes=notes,
            fill_id=fill_id,
        )
        return "\n".join(
            [
                f"Fill recorded {context.message_date.isoformat()}",
                f"Portfolio: {portfolio_id}",
                f"Fill ID: {result['fill_id']}",
                f"Cash: {float(result['cash']):,.2f}",
                f"Realized P&L: {float(result['realized_pnl']):,.2f}",
            ]
        )


def build_telegram_fill_id(
    *,
    portfolio_id: str,
    chat_id: int | None,
    message_id: int | None,
    message_date: date,
) -> str:
    chat_part = "na" if chat_id is None else str(chat_id)
    message_part = "na" if message_id is None else str(message_id)
    return f"tg-{portfolio_id}-{message_date.isoformat()}-{chat_part}-{message_part}"


def _message_context(message: Message) -> CommandContext:
    user_id = message.from_user.id if message.from_user is not None else None
    message_date = message.date.astimezone(UTC).date()
    return CommandContext(
        chat_id=message.chat.id if message.chat is not None else None,
        user_id=user_id,
        message_id=message.message_id,
        message_date=message_date,
    )


def _mini_app_keyboard(config: TelegramBotConfig) -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [
                KeyboardButton(
                    text="Open Portfolio App",
                    web_app=WebAppInfo(url=config.public_mini_app_url),
                )
            ]
        ],
        resize_keyboard=True,
    )


def _parse_fill_args(raw_text: str) -> tuple[str, str, str, float, float, float, float, str]:
    try:
        parts = shlex.split(raw_text)
    except ValueError as exc:
        raise ValueError(f"invalid command syntax: {exc}") from exc
    if len(parts) < 6:
        raise ValueError(fill_usage())
    _, portfolio_id, side, symbol, qty_raw, price_raw, *tail = parts
    commission = 0.0
    slippage = 0.0
    notes_start = 0
    try:
        quantity = float(qty_raw)
        fill_price = float(price_raw)
    except ValueError as exc:
        raise ValueError(fill_usage()) from exc
    if tail:
        try:
            commission = float(tail[0])
            notes_start = 1
        except ValueError:
            return portfolio_id, side.lower(), symbol, quantity, fill_price, 0.0, 0.0, " ".join(tail)
    if len(tail) > 1:
        try:
            slippage = float(tail[1])
            notes_start = 2
        except ValueError:
            return portfolio_id, side.lower(), symbol, quantity, fill_price, commission, 0.0, " ".join(tail[1:])
    notes = " ".join(tail[notes_start:])
    return portfolio_id, side.lower(), symbol, quantity, fill_price, commission, slippage, notes


def create_router(controller: TelegramBotController) -> Router:
    router = Router()

    @router.message(CommandStart())
    async def start_handler(message: Message) -> SendMessage:
        context = _message_context(message)
        portfolio_ids = controller.authorized_portfolios(chat_id=context.chat_id, user_id=context.user_id)
        return SendMessage(
            chat_id=message.chat.id,
            text=format_start_reply(portfolio_ids, mini_app_enabled=True),
            reply_markup=_mini_app_keyboard(controller.config),
        )

    @router.message(Command("app"))
    async def app_handler(message: Message) -> SendMessage:
        return SendMessage(
            chat_id=message.chat.id,
            text="Open the Portfolio Mini App.",
            reply_markup=_mini_app_keyboard(controller.config),
        )

    @router.message(Command("portfolios"))
    async def portfolios_handler(message: Message) -> SendMessage:
        context = _message_context(message)
        portfolio_ids = controller.authorized_portfolios(chat_id=context.chat_id, user_id=context.user_id)
        return SendMessage(chat_id=message.chat.id, text=format_portfolios_reply(portfolio_ids))

    @router.message(Command("status"))
    async def status_handler(message: Message) -> SendMessage:
        context = _message_context(message)
        parts = (message.text or "").split(maxsplit=1)
        if len(parts) != 2:
            text = "Usage: /status <portfolio_id>"
        else:
            text = controller.status_text(parts[1].strip(), chat_id=context.chat_id, user_id=context.user_id)
        return SendMessage(chat_id=message.chat.id, text=text)

    @router.message(Command("whoami"))
    async def whoami_handler(message: Message) -> SendMessage:
        context = _message_context(message)
        return SendMessage(
            chat_id=message.chat.id,
            text=format_whoami_reply(chat_id=context.chat_id, user_id=context.user_id),
        )

    @router.message(Command("adduser"))
    async def adduser_handler(message: Message) -> SendMessage:
        context = _message_context(message)
        parts = (message.text or "").split(maxsplit=2)
        if len(parts) != 3:
            text = add_user_usage()
        else:
            _, password, portfolio_id = parts
            text = controller.add_user_access(
                portfolio_id.strip(),
                password=password,
                context=context,
            )
        return SendMessage(chat_id=message.chat.id, text=text)

    @router.message(Command("fill"))
    async def fill_handler(message: Message) -> SendMessage:
        context = _message_context(message)
        try:
            (
                portfolio_id,
                side,
                symbol,
                quantity,
                fill_price,
                commission,
                slippage,
                notes,
            ) = _parse_fill_args(message.text or "")
            text = controller.record_fill_from_command(
                portfolio_id,
                side=side,
                symbol=symbol,
                quantity=quantity,
                fill_price=fill_price,
                commission=commission,
                slippage=slippage,
                notes=notes,
                context=context,
            )
        except ValueError as exc:
            text = str(exc) if str(exc) else fill_usage()
        except Exception:
            LOGGER.exception("telegram fill command crashed")
            text = "Unable to record fill."
        return SendMessage(chat_id=message.chat.id, text=text)

    @router.message(Command("help"))
    async def help_handler(message: Message) -> SendMessage:
        return SendMessage(chat_id=message.chat.id, text=help_text())

    @router.message(F.text)
    async def fallback_handler(message: Message) -> SendMessage:
        context = _message_context(message)
        pending_text = controller.handle_pending_onboarding(
            text=message.text or "",
            context=context,
        )
        if pending_text is not None:
            return SendMessage(chat_id=message.chat.id, text=pending_text)
        return SendMessage(chat_id=message.chat.id, text=help_text())

    return router


async def _register_telegram_surfaces(bot: Bot, config: TelegramBotConfig) -> None:
    if not config.register_webhook_on_startup:
        return
    await bot.set_webhook(
        url=config.public_webhook_url,
        allowed_updates=["message"],
        secret_token=config.webhook_secret_token,
        drop_pending_updates=False,
    )
    await bot.set_chat_menu_button(
        menu_button=MenuButtonWebApp(
            text=config.mini_app_title,
            web_app=WebAppInfo(url=config.public_mini_app_url),
        )
    )


async def _healthcheck(_: web.Request) -> web.Response:
    return web.json_response({"ok": True})


def validate_bot_startup(config: TelegramBotConfig, access_map: TelegramAccessMap) -> None:
    access_map.validate(
        state_root=config.state_root,
        strict_portfolio_validation=config.strict_portfolio_validation,
    )
    if not config.public_webhook_url.startswith("https://"):
        raise ValueError("public_webhook_url must use https")
    if not config.public_mini_app_url.startswith("https://"):
        raise ValueError("public_mini_app_url must use https")


def create_bot_app(
    *,
    config_path: Path,
    access_map_path: Path | None = None,
    bot: Bot | None = None,
) -> web.Application:
    config = load_telegram_bot_config(config_path)
    access_map = load_telegram_access_map(access_map_path or config.access_map_path)
    validate_bot_startup(config, access_map)
    controller = TelegramBotController(config=config, access_map=access_map)
    dp = Dispatcher()
    dp.include_router(create_router(controller))
    bot_instance = bot or Bot(token=config.bot_token)

    async def on_startup(*_: Any) -> None:
        await _register_telegram_surfaces(bot_instance, config)

    dp.startup.register(on_startup)

    @web.middleware
    async def mini_app_error_middleware(request: web.Request, handler: Any) -> web.StreamResponse:
        try:
            return await handler(request)
        except MiniAppAuthError as exc:
            raise web.HTTPUnauthorized(text=str(exc)) from exc

    app = web.Application()
    app.router.add_get("/health", _healthcheck)
    app.middlewares.append(mini_app_error_middleware)
    handler = DeduplicatingRequestHandler(
        dispatcher=dp,
        bot=bot_instance,
        handle_in_background=False,
        secret_token=config.webhook_secret_token,
        update_journal=ProcessedUpdateJournal(config.dedupe_journal_path),
    )
    handler.register(app, config.webhook_path)
    setup_application(app, dp, bot=bot_instance)
    app[CONFIG_KEY] = config
    app[ACCESS_MAP_KEY] = access_map
    app[DISPATCHER_KEY] = dp
    app[BOT_KEY] = bot_instance
    app[CONTROLLER_KEY] = controller
    app[MINI_APP_CONFIG_KEY] = config
    app[MINI_APP_CONTROLLER_KEY] = controller
    register_mini_app_routes(app, mini_app_path=config.mini_app_path)
    return app


def run_bot(*, config_path: Path, access_map_path: Path | None = None) -> None:
    app = create_bot_app(config_path=config_path, access_map_path=access_map_path)
    config = app[CONFIG_KEY]
    web.run_app(app, host=config.bind_host, port=config.bind_port)
