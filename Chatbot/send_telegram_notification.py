from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = ROOT / "research"


def _bootstrap_research_imports() -> None:
    if str(RESEARCH_ROOT) not in sys.path:
        sys.path.insert(0, str(RESEARCH_ROOT))


def main() -> None:
    _bootstrap_research_imports()
    from telegram_portfolio_bot.telegram_config import default_bot_config_path
    from telegram_portfolio_bot.telegram_notifications import send_text_notification_sync

    parser = argparse.ArgumentParser()
    parser.add_argument("--telegram-config", type=Path, default=default_bot_config_path())
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


if __name__ == "__main__":
    main()
