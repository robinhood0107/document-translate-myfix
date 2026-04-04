from __future__ import annotations

import os
from datetime import datetime


def build_export_timestamp(now: datetime | None = None) -> str:
    current = now or datetime.now()
    return current.strftime("%b-%d-%Y_%I-%M-%S%p")


def reserve_export_run_token(
    base_dir: str,
    base_timestamp: str,
    cache: dict[str, str] | None = None,
) -> str:
    abs_dir = os.path.abspath(base_dir)
    if cache is not None:
        cached = cache.get(abs_dir)
        if cached:
            return cached

    suffix = 0
    while True:
        token = base_timestamp if suffix == 0 else f"{base_timestamp}_{suffix:03d}"
        run_root = os.path.join(base_dir, f"comic_translate_{token}")
        try:
            os.makedirs(run_root, exist_ok=False)
            if cache is not None:
                cache[abs_dir] = token
            return token
        except FileExistsError:
            suffix += 1


def export_run_root(base_dir: str, token: str) -> str:
    return os.path.join(base_dir, f"comic_translate_{token}")
