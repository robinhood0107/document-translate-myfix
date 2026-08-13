#!/usr/bin/env python3
"""Deliberately unavailable A5 entry point.

A future implementation must bind exact synthetic/E1/visual/ONNX/product-stack
evidence and invoke the promoted PR3-PR6 product runner.  The prior lab-only
factorized implementation was intentionally removed so deleting one guard can
never make it consume the holdout.
"""

from __future__ import annotations


A5_UNAVAILABLE_MESSAGE = (
    "A5 unavailable until verified product-stack evidence sealer/runner is implemented"
)


def main(argv: list[str] | None = None) -> int:
    del argv
    raise RuntimeError(A5_UNAVAILABLE_MESSAGE)


if __name__ == "__main__":
    raise SystemExit(main())
