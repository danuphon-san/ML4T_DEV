from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import yaml


def _default_state_root() -> Path:
    return Path(__file__).resolve().parents[2] / "research" / "state" / "manual_portfolios"


def _default_registry_path() -> Path:
    return Path(__file__).resolve().parents[2] / "research" / "configs" / "manual_active_strategy.yaml"


def default_bot_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "telegram_bot.yaml"


def default_access_map_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "telegram_access_map.yaml"


@dataclass(frozen=True)
class TelegramNotificationToggles:
    daily_summary: bool = True
    drift_summary: bool = True
    rebalance_instructions: bool = True
    fill_confirmations: bool = True


@dataclass(frozen=True)
class TelegramBotConfig:
    bot_token: str
    public_webhook_url: str
    webhook_secret_token: str
    add_user_password: str | None
    webhook_path: str
    mini_app_path: str
    mini_app_title: str
    bind_host: str
    bind_port: int
    state_root: Path
    promotion_registry_path: Path
    access_map_path: Path
    register_webhook_on_startup: bool
    strict_portfolio_validation: bool
    dedupe_journal_path: Path
    notification_toggles: TelegramNotificationToggles

    @property
    def public_mini_app_url(self) -> str:
        parts = urlsplit(self.public_webhook_url)
        return urlunsplit((parts.scheme, parts.netloc, self.mini_app_path, "", ""))


def _load_structured_file(path: Path) -> dict[str, Any]:
    raw = path.read_text()
    if path.suffix.lower() == ".json":
        payload = json.loads(raw)
    else:
        payload = yaml.safe_load(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"config file must contain a mapping: {path}")
    return payload


def _resolve_env_ref(raw: Any, *, field_name: str) -> str:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{field_name} must be a non-empty string")
    env_name: str | None = None
    if raw.startswith("env:"):
        env_name = raw.removeprefix("env:").strip()
    elif raw.startswith("${") and raw.endswith("}"):
        env_name = raw[2:-1].strip()
    if env_name is None:
        return raw
    if not env_name:
        raise ValueError(f"{field_name} env reference is empty")
    value = os.getenv(env_name)
    if not value:
        raise ValueError(f"missing required environment variable for {field_name}: {env_name}")
    return value


def _resolve_optional_env_ref(raw: Any, *, field_name: str) -> str | None:
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{field_name} must be a non-empty string")
    env_name: str | None = None
    if raw.startswith("env:"):
        env_name = raw.removeprefix("env:").strip()
    elif raw.startswith("${") and raw.endswith("}"):
        env_name = raw[2:-1].strip()
    if env_name is None:
        return raw
    if not env_name:
        raise ValueError(f"{field_name} env reference is empty")
    value = os.getenv(env_name)
    if not value:
        return None
    return value


def _coerce_bool(payload: dict[str, Any], key: str, *, default: bool) -> bool:
    raw = payload.get(key, default)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        normalized = raw.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{key} must be a boolean")


def _coerce_path(base_dir: Path, raw: Any, *, default: Path | None = None) -> Path:
    if raw is None:
        if default is None:
            raise ValueError("path value is required")
        return default
    path = Path(str(raw))
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def load_telegram_bot_config(path: Path) -> TelegramBotConfig:
    payload = _load_structured_file(path)
    base_dir = path.parent
    toggles_payload = payload.get("notification_toggles", {})
    if toggles_payload is None:
        toggles_payload = {}
    if not isinstance(toggles_payload, dict):
        raise ValueError("notification_toggles must be a mapping")
    webhook_path = str(payload.get("webhook_path", "/telegram/webhook")).strip()
    if not webhook_path.startswith("/"):
        raise ValueError("webhook_path must start with '/'")
    mini_app_path = str(payload.get("mini_app_path", "/mini-app")).strip()
    if not mini_app_path.startswith("/"):
        raise ValueError("mini_app_path must start with '/'")
    public_webhook_url = str(payload.get("public_webhook_url", "")).strip()
    if not public_webhook_url:
        raise ValueError("public_webhook_url is required")
    state_root = _coerce_path(base_dir, payload.get("state_root"), default=_default_state_root())
    return TelegramBotConfig(
        bot_token=_resolve_env_ref(payload.get("bot_token"), field_name="bot_token"),
        public_webhook_url=public_webhook_url,
        webhook_secret_token=_resolve_env_ref(
            payload.get("webhook_secret_token"),
            field_name="webhook_secret_token",
        ),
        add_user_password=_resolve_optional_env_ref(
            payload.get("add_user_password"),
            field_name="add_user_password",
        ),
        webhook_path=webhook_path,
        mini_app_path=mini_app_path.rstrip("/") or "/mini-app",
        mini_app_title=str(payload.get("mini_app_title", "Portfolio App")).strip() or "Portfolio App",
        bind_host=str(payload.get("bind_host", "127.0.0.1")),
        bind_port=int(payload.get("bind_port", 8080)),
        state_root=state_root,
        promotion_registry_path=_coerce_path(
            base_dir,
            payload.get("promotion_registry_path"),
            default=_default_registry_path(),
        ),
        access_map_path=_coerce_path(
            base_dir,
            payload.get("access_map_path"),
            default=default_access_map_path(),
        ),
        register_webhook_on_startup=_coerce_bool(
            payload,
            "register_webhook_on_startup",
            default=True,
        ),
        strict_portfolio_validation=_coerce_bool(
            payload,
            "strict_portfolio_validation",
            default=False,
        ),
        dedupe_journal_path=_coerce_path(
            base_dir,
            payload.get("dedupe_journal_path"),
            default=state_root / ".telegram" / "processed_updates.jsonl",
        ),
        notification_toggles=TelegramNotificationToggles(
            daily_summary=_coerce_bool(toggles_payload, "daily_summary", default=True),
            drift_summary=_coerce_bool(toggles_payload, "drift_summary", default=True),
            rebalance_instructions=_coerce_bool(
                toggles_payload,
                "rebalance_instructions",
                default=True,
            ),
            fill_confirmations=_coerce_bool(
                toggles_payload,
                "fill_confirmations",
                default=True,
            ),
        ),
    )
