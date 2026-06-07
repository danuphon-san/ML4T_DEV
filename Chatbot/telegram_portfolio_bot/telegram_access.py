from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from manual_portfolio.storage import list_portfolio_ids


@dataclass(frozen=True)
class PortfolioAccessEntry:
    portfolio_id: str
    chat_ids: frozenset[int]
    user_ids: frozenset[int]
    delivery_chat_id: int | None = None


@dataclass(frozen=True)
class TelegramAccessMap:
    portfolios: dict[str, PortfolioAccessEntry]

    def authorized_portfolios(self, *, chat_id: int | None, user_id: int | None) -> list[str]:
        authorized: list[str] = []
        for portfolio_id, entry in self.portfolios.items():
            chat_ok = chat_id is not None and chat_id in entry.chat_ids
            user_ok = user_id is not None and user_id in entry.user_ids
            if chat_id is not None and user_id is not None:
                if chat_ok and user_ok:
                    authorized.append(portfolio_id)
                continue
            if chat_id is not None and chat_ok:
                authorized.append(portfolio_id)
                continue
            if user_id is not None and user_ok:
                authorized.append(portfolio_id)
        return sorted(authorized)

    def is_authorized(self, portfolio_id: str, *, chat_id: int | None, user_id: int | None) -> bool:
        return portfolio_id in self.authorized_portfolios(chat_id=chat_id, user_id=user_id)

    def delivery_chat_id_for(self, portfolio_id: str) -> int | None:
        entry = self.portfolios.get(portfolio_id)
        if entry is None:
            return None
        return entry.delivery_chat_id

    def validate(self, *, state_root: Path, strict_portfolio_validation: bool) -> None:
        known_portfolios = set(list_portfolio_ids(state_root))
        for portfolio_id, entry in self.portfolios.items():
            if not entry.chat_ids and not entry.user_ids:
                raise ValueError(f"access map entry must include chats or users: {portfolio_id}")
            if entry.delivery_chat_id is not None and entry.delivery_chat_id not in entry.chat_ids:
                raise ValueError(
                    f"delivery_chat_id must also appear in chats for portfolio: {portfolio_id}"
                )
            if strict_portfolio_validation and portfolio_id not in known_portfolios:
                raise ValueError(f"unknown portfolio_id in access map: {portfolio_id}")


def _read_structured_file(path: Path) -> dict[str, Any]:
    raw = path.read_text()
    if path.suffix.lower() == ".json":
        payload = json.loads(raw)
    else:
        payload = yaml.safe_load(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"access map must be a mapping: {path}")
    return payload


def _coerce_ids(raw: Any, *, field_name: str) -> frozenset[int]:
    if raw is None:
        return frozenset()
    if not isinstance(raw, list):
        raise ValueError(f"{field_name} must be a list")
    ids: set[int] = set()
    for value in raw:
        ids.add(int(value))
    return frozenset(ids)


def load_telegram_access_map(path: Path) -> TelegramAccessMap:
    payload = _read_structured_file(path)
    portfolios_payload = payload.get("portfolios")
    if not isinstance(portfolios_payload, dict) or not portfolios_payload:
        raise ValueError("access map must include a non-empty 'portfolios' mapping")
    portfolios: dict[str, PortfolioAccessEntry] = {}
    for portfolio_id, raw_entry in portfolios_payload.items():
        if not isinstance(raw_entry, dict):
            raise ValueError(f"portfolio entry must be a mapping: {portfolio_id}")
        chat_ids = _coerce_ids(raw_entry.get("chats"), field_name=f"{portfolio_id}.chats")
        user_ids = _coerce_ids(raw_entry.get("users"), field_name=f"{portfolio_id}.users")
        delivery_chat_raw = raw_entry.get("delivery_chat_id")
        delivery_chat_id = int(delivery_chat_raw) if delivery_chat_raw is not None else None
        portfolios[portfolio_id] = PortfolioAccessEntry(
            portfolio_id=portfolio_id,
            chat_ids=chat_ids,
            user_ids=user_ids,
            delivery_chat_id=delivery_chat_id,
        )
    return TelegramAccessMap(portfolios=portfolios)


def upsert_access_entry(
    path: Path,
    *,
    portfolio_id: str,
    chat_id: int | None,
    user_id: int | None,
    delivery_chat_id: int | None,
) -> TelegramAccessMap:
    payload = {"portfolios": {}}
    if path.exists():
        payload = _read_structured_file(path)
    portfolios_payload = payload.setdefault("portfolios", {})
    if not isinstance(portfolios_payload, dict):
        raise ValueError("access map must include a 'portfolios' mapping")
    raw_entry = portfolios_payload.get(portfolio_id, {})
    if not isinstance(raw_entry, dict):
        raise ValueError(f"portfolio entry must be a mapping: {portfolio_id}")

    chats = {int(value) for value in raw_entry.get("chats", [])}
    users = {int(value) for value in raw_entry.get("users", [])}
    if chat_id is not None:
        chats.add(int(chat_id))
    if user_id is not None:
        users.add(int(user_id))

    updated_entry: dict[str, Any] = {
        "chats": sorted(chats),
        "users": sorted(users),
    }
    existing_delivery = raw_entry.get("delivery_chat_id")
    if delivery_chat_id is not None:
        updated_entry["delivery_chat_id"] = int(delivery_chat_id)
    elif existing_delivery is not None:
        updated_entry["delivery_chat_id"] = int(existing_delivery)

    portfolios_payload[portfolio_id] = updated_entry
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return load_telegram_access_map(path)
