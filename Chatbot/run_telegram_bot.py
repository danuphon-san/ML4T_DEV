from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = ROOT / "research"


def _bootstrap_research_imports() -> None:
    if str(RESEARCH_ROOT) not in sys.path:
        sys.path.insert(0, str(RESEARCH_ROOT))


def main() -> None:
    _bootstrap_research_imports()
    from telegram_portfolio_bot.telegram_bot import run_bot
    from telegram_portfolio_bot.telegram_config import default_bot_config_path

    parser = argparse.ArgumentParser()
    parser.add_argument("--telegram-config", type=Path, default=default_bot_config_path())
    parser.add_argument("--telegram-access-map", type=Path)
    args = parser.parse_args()
    run_bot(
        config_path=args.telegram_config,
        access_map_path=args.telegram_access_map,
    )


if __name__ == "__main__":
    main()
