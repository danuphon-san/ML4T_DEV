from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CHATBOT_ROOT = ROOT / "Chatbot"
RESEARCH_ROOT = ROOT / "research"

for path in (CHATBOT_ROOT, RESEARCH_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
