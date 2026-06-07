from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from .models import PortfolioMetadata, PortfolioState


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _json_text(
    payload: dict[str, Any],
    *,
    indent: int | None,
) -> str:
    separators = (",", ": ") if indent is not None else (",", ":")
    text = json.dumps(payload, indent=indent, sort_keys=True, separators=separators)
    return f"{text}\n"


def _atomic_write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    temp_path: str | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
            temp_path = handle.name
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and os.path.exists(temp_path):
            os.unlink(temp_path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_text(path, _json_text(payload, indent=2))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(_json_text(payload, indent=None))
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def tail_jsonl(path: Path, limit: int) -> list[dict[str, Any]]:
    if limit < 0:
        raise ValueError("limit must be non-negative")
    if limit == 0:
        return []
    rows = read_jsonl(path)
    return rows[-limit:]


def portfolio_dir(state_root: Path, portfolio_id: str) -> Path:
    return state_root / portfolio_id


def metadata_path(state_root: Path, portfolio_id: str) -> Path:
    return portfolio_dir(state_root, portfolio_id) / "metadata.json"


def state_path(state_root: Path, portfolio_id: str) -> Path:
    return portfolio_dir(state_root, portfolio_id) / "state.json"


def fills_path(state_root: Path, portfolio_id: str) -> Path:
    return portfolio_dir(state_root, portfolio_id) / "fills.jsonl"


def daily_output_dir(state_root: Path, portfolio_id: str, as_of: str) -> Path:
    return portfolio_dir(state_root, portfolio_id) / "daily" / as_of


def load_metadata(state_root: Path, portfolio_id: str) -> PortfolioMetadata:
    return PortfolioMetadata.from_dict(
        read_json(metadata_path(state_root, portfolio_id))
    )


def load_state(state_root: Path, portfolio_id: str) -> PortfolioState:
    return PortfolioState.from_dict(read_json(state_path(state_root, portfolio_id)))


def save_metadata(state_root: Path, metadata: PortfolioMetadata) -> None:
    ensure_dir(portfolio_dir(state_root, metadata.portfolio_id))
    write_json(metadata_path(state_root, metadata.portfolio_id), metadata.to_dict())


def save_state(state_root: Path, state: PortfolioState) -> None:
    ensure_dir(portfolio_dir(state_root, state.portfolio_id))
    write_json(state_path(state_root, state.portfolio_id), state.to_dict())


def list_portfolio_ids(state_root: Path) -> list[str]:
    if not state_root.exists():
        return []
    return sorted(path.name for path in state_root.iterdir() if path.is_dir())
