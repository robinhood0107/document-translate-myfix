#!/usr/bin/env python3
"""Finalize optional inpaint evaluation pages from source-only decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.inpaint_eval_contract import (  # noqa: E402
    InpaintEvalManifestError,
    finalize_optional_eval_manifest,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Seal source-only required/none decisions into a derived private "
            "inpaint evaluation manifest."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        manifest = finalize_optional_eval_manifest(
            args.manifest,
            args.decisions,
            args.output,
        )
    except InpaintEvalManifestError as exc:
        print(
            json.dumps(exc.as_record(), ensure_ascii=False, sort_keys=True),
            file=sys.stderr,
        )
        return 1
    print(
        json.dumps(
            {
                "corpus_id": manifest.corpus_id,
                "expected_count": manifest.expected_count,
                "manifest_sha256": manifest.manifest_sha256,
                "parent_manifest_sha256": manifest.parent_manifest_sha256,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
