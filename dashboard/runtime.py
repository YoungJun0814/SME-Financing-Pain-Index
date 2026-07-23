from __future__ import annotations

from dataclasses import dataclass
from os import environ
from typing import Mapping


@dataclass(frozen=True)
class RuntimeSettings:
    host: str
    port: int
    debug: bool


def _parse_port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as exc:
        raise ValueError("PORT must be an integer between 1 and 65535.") from exc
    if not 1 <= port <= 65535:
        raise ValueError("PORT must be an integer between 1 and 65535.")
    return port


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError("DASH_DEBUG must be one of true/false, yes/no, on/off, or 1/0.")


def load_runtime_settings(values: Mapping[str, str] | None = None) -> RuntimeSettings:
    source = environ if values is None else values
    host = source.get("DASH_HOST", "127.0.0.1").strip()
    if not host:
        raise ValueError("DASH_HOST cannot be empty.")
    return RuntimeSettings(
        host=host,
        port=_parse_port(source.get("PORT", "8050")),
        debug=_parse_bool(source.get("DASH_DEBUG", "false")),
    )
