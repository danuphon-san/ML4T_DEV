from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Protocol

from aiogram import Bot

from .telegram_access import TelegramAccessMap, load_telegram_access_map
from .telegram_config import TelegramBotConfig, load_telegram_bot_config
from .telegram_messages import (
    format_daily_summary,
    format_drift_summary,
    format_fill_confirmation,
    format_rebalance_instructions,
    split_message,
)


LOGGER = logging.getLogger(__name__)


class MessageSender(Protocol):
    async def send(self, *, chat_id: int, text: str) -> None: ...


@dataclass
class BotMessageSender:
    bot: Bot

    async def send(self, *, chat_id: int, text: str) -> None:
        for chunk in split_message(text):
            await self.bot.send_message(chat_id=chat_id, text=chunk)


@dataclass
class NotificationResult:
    portfolio_id: str
    message_type: str
    ok: bool
    detail: str


@dataclass
class TelegramNotifier:
    config: TelegramBotConfig
    access_map: TelegramAccessMap
    sender: MessageSender

    async def send_portfolio_message(
        self,
        *,
        portfolio_id: str,
        message_type: str,
        text: str,
    ) -> NotificationResult:
        chat_id = self.access_map.delivery_chat_id_for(portfolio_id)
        if chat_id is None:
            detail = "missing delivery chat"
            LOGGER.error("telegram notification skipped: portfolio=%s type=%s detail=%s", portfolio_id, message_type, detail)
            return NotificationResult(
                portfolio_id=portfolio_id,
                message_type=message_type,
                ok=False,
                detail=detail,
            )
        try:
            await self.sender.send(chat_id=chat_id, text=text)
        except Exception as exc:
            detail = str(exc)
            LOGGER.exception(
                "telegram notification failed: portfolio=%s type=%s",
                portfolio_id,
                message_type,
            )
            return NotificationResult(
                portfolio_id=portfolio_id,
                message_type=message_type,
                ok=False,
                detail=detail,
            )
        return NotificationResult(
            portfolio_id=portfolio_id,
            message_type=message_type,
            ok=True,
            detail="sent",
        )

    async def send_daily_run_notifications(self, result: dict) -> list[NotificationResult]:
        notifications: list[NotificationResult] = []
        toggles = self.config.notification_toggles
        for portfolio_id, portfolio in result["portfolios"].items():
            if toggles.daily_summary:
                notifications.append(
                    await self.send_portfolio_message(
                        portfolio_id=portfolio_id,
                        message_type="daily_summary",
                        text=format_daily_summary(portfolio),
                    )
                )
            if toggles.drift_summary and portfolio["is_rebalance_day"]:
                drift_text = format_drift_summary(portfolio)
                if drift_text is not None:
                    notifications.append(
                        await self.send_portfolio_message(
                            portfolio_id=portfolio_id,
                            message_type="drift_summary",
                            text=drift_text,
                        )
                    )
            if toggles.rebalance_instructions and portfolio["is_rebalance_day"]:
                notifications.append(
                    await self.send_portfolio_message(
                        portfolio_id=portfolio_id,
                        message_type="rebalance_instructions",
                        text=format_rebalance_instructions(portfolio),
                    )
                )
        return notifications

    async def send_fill_confirmation(
        self,
        *,
        portfolio_id: str,
        trade_date: date,
        side: str,
        symbol: str,
        quantity: float,
        fill_price: float,
        commission: float,
        slippage: float,
        fill_result: dict,
    ) -> NotificationResult | None:
        if not self.config.notification_toggles.fill_confirmations:
            return None
        return await self.send_portfolio_message(
            portfolio_id=portfolio_id,
            message_type="fill_confirmation",
            text=format_fill_confirmation(
                portfolio_id=portfolio_id,
                trade_date=trade_date,
                side=side,
                symbol=symbol,
                quantity=quantity,
                fill_price=fill_price,
                commission=commission,
                slippage=slippage,
                fill_result=fill_result,
            ),
        )


async def send_text_notification(
    *,
    config_path: Path,
    access_map_path: Path | None,
    portfolio_id: str,
    text: str,
    message_type: str,
) -> NotificationResult:
    config = load_telegram_bot_config(config_path)
    access_map = load_telegram_access_map(access_map_path or config.access_map_path)
    access_map.validate(
        state_root=config.state_root,
        strict_portfolio_validation=config.strict_portfolio_validation,
    )
    async with Bot(token=config.bot_token) as bot:
        notifier = TelegramNotifier(
            config=config,
            access_map=access_map,
            sender=BotMessageSender(bot),
        )
        return await notifier.send_portfolio_message(
            portfolio_id=portfolio_id,
            message_type=message_type,
            text=text,
        )


def send_text_notification_sync(
    *,
    config_path: Path,
    access_map_path: Path | None,
    portfolio_id: str,
    text: str,
    message_type: str,
) -> NotificationResult:
    return asyncio.run(
        send_text_notification(
            config_path=config_path,
            access_map_path=access_map_path,
            portfolio_id=portfolio_id,
            text=text,
            message_type=message_type,
        )
    )
