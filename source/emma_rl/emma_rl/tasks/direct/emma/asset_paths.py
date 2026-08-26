"""Resolve curated runtime assets from the repository asset directory."""

from __future__ import annotations

import os
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[6]
ASSET_ROOT = REPOSITORY_ROOT / "assets"


def resolve_asset(filename: str, override_env: str | None = None) -> str:
    """Return an absolute asset path and fail early when it is unavailable."""

    override = os.environ.get(override_env) if override_env else None
    path = Path(override).expanduser() if override else ASSET_ROOT / filename
    path = path.resolve()
    if not path.is_file():
        hint = f" Set {override_env} to override the path." if override_env else ""
        raise FileNotFoundError(f"Required asset not found: {path}.{hint}")
    return str(path)
