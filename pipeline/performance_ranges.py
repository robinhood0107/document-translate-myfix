from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator


def _nvtx_enabled() -> bool:
    value = str(os.environ.get("CT_PERFORMANCE_NVTX", "") or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


@contextmanager
def performance_range(name: str) -> Iterator[None]:
    """Emit an optional NVTX range without adding a product dependency.

    Nsight capture is a lab-only workflow. Product runs pay only the environment
    check unless ``CT_PERFORMANCE_NVTX`` is explicitly enabled. Import or CUDA
    failures are deliberately fail-open because telemetry must never stop a
    translation run.
    """

    if not _nvtx_enabled():
        yield
        return

    range_pop = None
    try:
        import torch

        nvtx = getattr(getattr(torch, "cuda", None), "nvtx", None)
        range_push = getattr(nvtx, "range_push", None)
        range_pop = getattr(nvtx, "range_pop", None)
        if callable(range_push) and callable(range_pop):
            range_push(str(name or "comic-translate"))
    except Exception:
        range_pop = None

    try:
        yield
    finally:
        if callable(range_pop):
            try:
                range_pop()
            except Exception:
                pass
