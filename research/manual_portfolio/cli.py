from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import subprocess
import sys
import time
from datetime import date, datetime, time as clock_time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

from .models import ArtifactSpec
from .registry import load_promotion_registry
from .service import daily_run, onboard_portfolio, portfolio_status, record_fill

ROOT = Path(__file__).resolve().parents[2]
CHATBOT_ROOT = ROOT / "Chatbot"


DEFAULT_STATE_ROOT = Path(__file__).resolve().parents[1] / "state" / "manual_portfolios"
DEFAULT_REGISTRY_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "manual_active_strategy.yaml"
)


def _load_telegram_symbol(module_name: str, symbol_name: str):
    if str(CHATBOT_ROOT) not in sys.path:
        sys.path.insert(0, str(CHATBOT_ROOT))
    module = importlib.import_module(module_name)
    return getattr(module, symbol_name)


def _default_bot_config_path() -> Path:
    return _load_telegram_symbol(
        "telegram_portfolio_bot.telegram_config",
        "default_bot_config_path",
    )()


def _default_access_map_path() -> Path:
    return _load_telegram_symbol(
        "telegram_portfolio_bot.telegram_config",
        "default_access_map_path",
    )()


DEFAULT_BOT_CONFIG_PATH = _default_bot_config_path()
DEFAULT_ACCESS_MAP_PATH = _default_access_map_path()
NEW_YORK_TZ = ZoneInfo("America/New_York")
RESEARCH_ROOT = Path(__file__).resolve().parents[1]


def _date_arg(raw: str) -> date:
    return date.fromisoformat(raw)


def _load_holdings(path: Path | None) -> list[dict]:
    if path is None:
        return []
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError("holdings file must be a JSON array")
    return payload


def onboard_portfolio_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
    parser.add_argument("--portfolio-id", required=True)
    parser.add_argument("--display-name")
    parser.add_argument("--starting-cash", required=True, type=float)
    parser.add_argument("--holdings-file", type=Path)
    parser.add_argument("--notes", default="")
    args = parser.parse_args()
    result = onboard_portfolio(
        args.state_root,
        args.portfolio_id,
        starting_cash=args.starting_cash,
        display_name=args.display_name,
        imported_holdings=_load_holdings(args.holdings_file),
        notes=args.notes,
    )
    print(json.dumps(result, indent=2))


def record_fill_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
    parser.add_argument("--portfolio-id", required=True)
    parser.add_argument("--trade-date", required=True, type=_date_arg)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--side", required=True, choices=["buy", "sell"])
    parser.add_argument("--quantity", required=True, type=float)
    parser.add_argument("--fill-price", required=True, type=float)
    parser.add_argument("--commission", type=float, default=0.0)
    parser.add_argument("--slippage", type=float, default=0.0)
    parser.add_argument("--notes", default="")
    parser.add_argument("--fill-id")
    parser.add_argument("--notify-telegram", action="store_true")
    parser.add_argument("--telegram-config", type=Path, default=DEFAULT_BOT_CONFIG_PATH)
    parser.add_argument(
        "--telegram-access-map", type=Path, default=DEFAULT_ACCESS_MAP_PATH
    )
    args = parser.parse_args()
    result = record_fill(
        args.state_root,
        args.portfolio_id,
        trade_date=args.trade_date,
        symbol=args.symbol,
        side=args.side,
        quantity=args.quantity,
        fill_price=args.fill_price,
        commission=args.commission,
        slippage=args.slippage,
        notes=args.notes,
        fill_id=args.fill_id,
    )
    if args.notify_telegram:
        try:
            notification = asyncio.run(
                _send_fill_confirmation_notification(
                    config_path=args.telegram_config,
                    access_map_path=args.telegram_access_map,
                    portfolio_id=args.portfolio_id,
                    trade_date=args.trade_date,
                    side=args.side,
                    symbol=args.symbol,
                    quantity=args.quantity,
                    fill_price=args.fill_price,
                    commission=args.commission,
                    slippage=args.slippage,
                    fill_result=result,
                )
            )
            result["telegram_notifications"] = (
                [] if notification is None else [notification.__dict__]
            )
        except Exception as exc:
            result["telegram_notifications"] = [
                {
                    "portfolio_id": args.portfolio_id,
                    "message_type": "fill_confirmation",
                    "ok": False,
                    "detail": str(exc),
                }
            ]
    print(json.dumps(result, indent=2))


def portfolio_status_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
    parser.add_argument(
        "--promotion-registry", type=Path, default=DEFAULT_REGISTRY_PATH
    )
    parser.add_argument("--portfolio-id", required=True)
    parser.add_argument("--as-of", type=_date_arg)
    args = parser.parse_args()
    result = portfolio_status(
        args.state_root,
        args.promotion_registry,
        args.portfolio_id,
        as_of=args.as_of,
    )
    print(json.dumps(result, indent=2))


def daily_run_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
    parser.add_argument(
        "--promotion-registry", type=Path, default=DEFAULT_REGISTRY_PATH
    )
    parser.add_argument("--as-of", type=_date_arg)
    parser.add_argument("--portfolio-id", action="append", dest="portfolio_ids")
    parser.add_argument(
        "--notify-telegram", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument("--telegram-config", type=Path, default=DEFAULT_BOT_CONFIG_PATH)
    parser.add_argument(
        "--telegram-access-map", type=Path, default=DEFAULT_ACCESS_MAP_PATH
    )
    args = parser.parse_args()
    result = daily_run(
        args.state_root,
        args.promotion_registry,
        as_of=args.as_of,
        portfolio_ids=args.portfolio_ids,
    )
    print(json.dumps(result, indent=2))


def _post_close_run_context(
    *,
    now: datetime | None = None,
    market_close: clock_time = clock_time(hour=16, minute=0),
    post_close_buffer: timedelta = timedelta(minutes=15),
) -> tuple[date, float]:
    current = now.astimezone(NEW_YORK_TZ) if now else datetime.now(NEW_YORK_TZ)
    ready_at = (
        datetime.combine(current.date(), market_close, tzinfo=NEW_YORK_TZ)
        + post_close_buffer
    )
    wait_seconds = max(0.0, (ready_at - current).total_seconds())
    return current.date(), wait_seconds


def _scheduled_daily_run_command(args: argparse.Namespace, as_of: date) -> list[str]:
    command = [
        "daily-workflow",
        "--state-root",
        str(args.state_root),
        "--promotion-registry",
        str(args.promotion_registry),
        "--as-of",
        as_of.isoformat(),
        "--notify-telegram",
        "--telegram-config",
        str(args.telegram_config),
        "--telegram-access-map",
        str(args.telegram_access_map),
    ]
    for portfolio_id in args.portfolio_ids or []:
        command.extend(["--portfolio-id", portfolio_id])
    return command


def scheduled_daily_run_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
    parser.add_argument(
        "--promotion-registry", type=Path, default=DEFAULT_REGISTRY_PATH
    )
    parser.add_argument("--portfolio-id", action="append", dest="portfolio_ids")
    parser.add_argument("--telegram-config", type=Path, default=DEFAULT_BOT_CONFIG_PATH)
    parser.add_argument(
        "--telegram-access-map", type=Path, default=DEFAULT_ACCESS_MAP_PATH
    )
    parser.add_argument("--market-close", default="16:00")
    parser.add_argument("--post-close-buffer-minutes", type=float, default=15.0)
    parser.add_argument("--no-wait", action="store_true")
    args = parser.parse_args()

    hour, minute = (int(part) for part in args.market_close.split(":", maxsplit=1))
    as_of, wait_seconds = _post_close_run_context(
        market_close=clock_time(hour=hour, minute=minute),
        post_close_buffer=timedelta(minutes=args.post_close_buffer_minutes),
    )
    if wait_seconds > 0 and not args.no_wait:
        time.sleep(wait_seconds)

    completed = subprocess.run(_scheduled_daily_run_command(args, as_of), check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def _daily_run_command(args: argparse.Namespace, as_of: date) -> list[str]:
    command = [
        "daily-run",
        "--state-root",
        str(args.state_root),
        "--promotion-registry",
        str(args.promotion_registry),
        "--as-of",
        as_of.isoformat(),
    ]
    for portfolio_id in args.portfolio_ids or []:
        command.extend(["--portfolio-id", portfolio_id])
    return command


def _default_data_update_command(*, source_prefix: str, as_of: date) -> list[str]:
    return [
        "uv",
        "run",
        "python",
        "update_sp500_daily.py",
        "--prefix",
        source_prefix,
        "--rebuild-dataset",
        "--as-of-date",
        as_of.isoformat(),
    ]


def _default_signal_refresh_command(*, source_prefix: str) -> list[str]:
    if source_prefix == "sp500_10yr":
        return [
            "uv",
            "run",
            "python",
            "backtest_walkforward_10yr.py",
            "--prefix",
            source_prefix,
            "--min-train-years",
            "2",
        ]
    return [
        "uv",
        "run",
        "python",
        "backtest_long_only.py",
        "--prefix",
        source_prefix,
    ]


def _split_workflow_command_overrides(
    argv: list[str],
) -> tuple[list[str], list[str] | None, list[str] | None]:
    cleaned: list[str] = []
    update_command: list[str] | None = None
    signal_command: list[str] | None = None
    index = 0
    while index < len(argv):
        token = argv[index]
        if token not in {"--update-command", "--signal-command"}:
            cleaned.append(token)
            index += 1
            continue
        target: list[str] = []
        index += 1
        while index < len(argv) and argv[index] not in {
            "--update-command",
            "--signal-command",
        }:
            target.append(argv[index])
            index += 1
        if token == "--update-command":
            update_command = target
        else:
            signal_command = target
    return cleaned, update_command, signal_command


def _promoted_artifact_latest_date(label: str, artifact: ArtifactSpec) -> date:
    latest = (
        pl.scan_parquet(artifact.path)
        .select(pl.col(artifact.date_col).dt.date().max().alias("latest_date"))
        .collect()
        .item()
    )
    if latest is None:
        raise ValueError(f"promoted {label} artifact has no dated rows: {artifact.path}")
    return latest


def _validated_artifact_path(label: str, artifact: ArtifactSpec, *, as_of: date) -> str:
    if not artifact.path.exists():
        raise FileNotFoundError(f"missing promoted {label} artifact: {artifact.path}")
    latest = _promoted_artifact_latest_date(label, artifact)
    if latest < as_of:
        raise ValueError(
            f"promoted {label} artifact is stale: latest {label} date "
            f"{latest.isoformat()} is before as_of {as_of.isoformat()}: {artifact.path}"
        )
    return str(artifact.path)


def _collect_daily_artifacts(
    *,
    state_root: Path,
    as_of: date,
    result: dict,
) -> dict[str, dict[str, str]]:
    artifacts: dict[str, dict[str, str]] = {}
    for portfolio_id in result["portfolios"]:
        output_dir = state_root / portfolio_id / "daily" / as_of.isoformat()
        daily_run_path = output_dir / "daily_run.json"
        rebalance_plan_path = output_dir / "rebalance_plan.json"
        if not daily_run_path.exists():
            raise FileNotFoundError(f"missing daily artifact: {daily_run_path}")
        if not rebalance_plan_path.exists():
            raise FileNotFoundError(
                f"missing rebalance artifact: {rebalance_plan_path}"
            )
        artifacts[portfolio_id] = {
            "daily_run": str(daily_run_path),
            "rebalance_plan": str(rebalance_plan_path),
        }
    return artifacts


def _run_capture_json(command: list[str], *, cwd: Path) -> tuple[int, dict | None]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    if completed.returncode != 0:
        return completed.returncode, None
    return completed.returncode, json.loads(completed.stdout)


def daily_workflow_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
    parser.add_argument(
        "--promotion-registry", type=Path, default=DEFAULT_REGISTRY_PATH
    )
    parser.add_argument("--as-of", type=_date_arg)
    parser.add_argument("--portfolio-id", action="append", dest="portfolio_ids")
    parser.add_argument("--notify-telegram", action="store_true")
    parser.add_argument("--telegram-config", type=Path, default=DEFAULT_BOT_CONFIG_PATH)
    parser.add_argument(
        "--telegram-access-map", type=Path, default=DEFAULT_ACCESS_MAP_PATH
    )
    parser.add_argument(
        "--update-command",
        action="store_true",
        help="Command run before signal refresh; pass command values after this flag.",
    )
    parser.add_argument(
        "--signal-command",
        action="store_true",
        help="Command run before daily-run; pass this flag last.",
    )
    cleaned_argv, update_command, signal_command = _split_workflow_command_overrides(
        sys.argv[1:]
    )
    args = parser.parse_args(cleaned_argv)

    as_of = args.as_of or datetime.now(NEW_YORK_TZ).date()
    registry = load_promotion_registry(args.promotion_registry)
    update_completed = subprocess.run(
        update_command
        or _default_data_update_command(
            source_prefix=registry.source_prefix,
            as_of=as_of,
        ),
        cwd=RESEARCH_ROOT,
        check=False,
    )
    if update_completed.returncode != 0:
        raise SystemExit(update_completed.returncode)

    promoted_artifacts = {
        "price": _validated_artifact_path(
            "price",
            registry.price_artifact,
            as_of=as_of,
        ),
    }

    signal_completed = subprocess.run(
        signal_command
        or _default_signal_refresh_command(source_prefix=registry.source_prefix),
        cwd=RESEARCH_ROOT,
        check=False,
    )
    if signal_completed.returncode != 0:
        raise SystemExit(signal_completed.returncode)

    promoted_artifacts["signal"] = _validated_artifact_path(
        "signal",
        registry.signal_artifact,
        as_of=as_of,
    )

    daily_returncode, result = _run_capture_json(
        _daily_run_command(args, as_of),
        cwd=RESEARCH_ROOT,
    )
    if daily_returncode != 0:
        raise SystemExit(daily_returncode)
    if result is None:
        raise SystemExit("daily-run completed without a result payload")

    result["artifacts"] = _collect_daily_artifacts(
        state_root=args.state_root,
        as_of=as_of,
        result=result,
    )
    result["promoted_artifacts"] = promoted_artifacts

    if args.notify_telegram:
        try:
            notifications = asyncio.run(
                _send_daily_run_notifications(
                    config_path=args.telegram_config,
                    access_map_path=args.telegram_access_map,
                    result=result,
                )
            )
            result["telegram_notifications"] = [item.__dict__ for item in notifications]
        except Exception as exc:
            result["telegram_notifications"] = [
                {
                    "portfolio_id": portfolio_id,
                    "message_type": "daily_run",
                    "ok": False,
                    "detail": str(exc),
                }
                for portfolio_id in result["portfolios"]
            ]
    print(json.dumps(result, indent=2))


async def _build_notifier(config_path: Path, access_map_path: Path | None):
    Bot = _load_telegram_symbol("aiogram", "Bot")
    load_telegram_bot_config = _load_telegram_symbol(
        "telegram_portfolio_bot.telegram_config",
        "load_telegram_bot_config",
    )
    load_telegram_access_map = _load_telegram_symbol(
        "telegram_portfolio_bot.telegram_access",
        "load_telegram_access_map",
    )
    TelegramNotifier = _load_telegram_symbol(
        "telegram_portfolio_bot.telegram_notifications",
        "TelegramNotifier",
    )
    BotMessageSender = _load_telegram_symbol(
        "telegram_portfolio_bot.telegram_notifications",
        "BotMessageSender",
    )
    config = load_telegram_bot_config(config_path)
    access_map = load_telegram_access_map(access_map_path or config.access_map_path)
    access_map.validate(
        state_root=config.state_root,
        strict_portfolio_validation=config.strict_portfolio_validation,
    )
    bot = Bot(token=config.bot_token)
    return TelegramNotifier(
        config=config,
        access_map=access_map,
        sender=BotMessageSender(bot),
    )


async def _send_daily_run_notifications(
    *,
    config_path: Path,
    access_map_path: Path | None,
    result: dict,
):
    notifier = await _build_notifier(config_path, access_map_path)
    bot = notifier.sender.bot
    try:
        return await notifier.send_daily_run_notifications(result)
    finally:
        await bot.session.close()


async def _send_fill_confirmation_notification(
    *,
    config_path: Path,
    access_map_path: Path | None,
    portfolio_id: str,
    trade_date: date,
    side: str,
    symbol: str,
    quantity: float,
    fill_price: float,
    commission: float,
    slippage: float,
    fill_result: dict,
):
    notifier = await _build_notifier(config_path, access_map_path)
    bot = notifier.sender.bot
    try:
        return await notifier.send_fill_confirmation(
            portfolio_id=portfolio_id,
            trade_date=trade_date,
            side=side,
            symbol=symbol,
            quantity=quantity,
            fill_price=fill_price,
            commission=commission,
            slippage=slippage,
            fill_result=fill_result,
        )
    finally:
        await bot.session.close()


def run_telegram_bot_main() -> None:
    run_bot = _load_telegram_symbol("telegram_portfolio_bot.telegram_bot", "run_bot")
    parser = argparse.ArgumentParser()
    parser.add_argument("--telegram-config", type=Path, default=DEFAULT_BOT_CONFIG_PATH)
    parser.add_argument("--telegram-access-map", type=Path)
    args = parser.parse_args()
    run_bot(
        config_path=args.telegram_config,
        access_map_path=args.telegram_access_map,
    )


def send_telegram_notification_main() -> None:
    send_text_notification_sync = _load_telegram_symbol(
        "telegram_portfolio_bot.telegram_notifications",
        "send_text_notification_sync",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--telegram-config", type=Path, default=DEFAULT_BOT_CONFIG_PATH)
    parser.add_argument("--telegram-access-map", type=Path)
    parser.add_argument("--portfolio-id", required=True)
    parser.add_argument("--message-type", default="operator")
    parser.add_argument("--text", required=True)
    args = parser.parse_args()
    result = send_text_notification_sync(
        config_path=args.telegram_config,
        access_map_path=args.telegram_access_map,
        portfolio_id=args.portfolio_id,
        text=args.text,
        message_type=args.message_type,
    )
    print(json.dumps(result.__dict__, indent=2))
