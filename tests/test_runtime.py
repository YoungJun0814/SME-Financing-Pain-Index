from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PATH = PROJECT_ROOT / "dashboard" / "runtime.py"


def load_runtime_module():
    spec = importlib.util.spec_from_file_location("dashboard_runtime", RUNTIME_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_runtime_settings_have_safe_local_defaults() -> None:
    runtime = load_runtime_module()

    settings = runtime.load_runtime_settings({})

    assert settings.host == "127.0.0.1"
    assert settings.port == 8050
    assert settings.debug is False


def test_runtime_settings_accept_container_environment() -> None:
    runtime = load_runtime_module()

    settings = runtime.load_runtime_settings(
        {"DASH_HOST": "0.0.0.0", "PORT": "9000", "DASH_DEBUG": "yes"}
    )

    assert settings.host == "0.0.0.0"
    assert settings.port == 9000
    assert settings.debug is True


@pytest.mark.parametrize("port", ["0", "65536", "not-a-number"])
def test_runtime_settings_reject_invalid_ports(port: str) -> None:
    runtime = load_runtime_module()

    with pytest.raises(ValueError, match="PORT"):
        runtime.load_runtime_settings({"PORT": port})


def test_container_assets_use_production_safeguards() -> None:
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
    dockerignore = (PROJECT_ROOT / ".dockerignore").read_text(encoding="utf-8")

    assert "gunicorn" in dockerfile
    assert "USER appuser" in dockerfile
    assert "HEALTHCHECK" in dockerfile
    assert "dashboard.wsgi:application" in dockerfile
    assert "data/processed" in dockerfile
    assert "data/raw" in dockerignore
    assert ".git" in dockerignore
