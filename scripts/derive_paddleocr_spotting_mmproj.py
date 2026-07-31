from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.ocr.paddleocr_vl_spotting.gguf_metadata import (
    GGUFMetadataError,
    find_metadata_entry,
    patch_uint32_metadata,
)


METADATA_KEY = "clip.vision.image_max_pixels"
CROP_IMAGE_MAX_PIXELS = 1_003_520
SPOTTING_IMAGE_MAX_PIXELS = 1_605_632
CROP_MMPROJ_SHA256 = (
    "204d757d7610d9b3faab10d506d69e5b244e32bf765e2bab2d0167e65e0a058a"
)
SPOTTING_MMPROJ_SHA256 = (
    "8e011479092c5e82c8c1c2d85d52b9ac48df12183c5c7bc3190190732259db09"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def derive_spotting_projector(source: Path, output: Path) -> None:
    source = source.resolve()
    output = output.resolve()
    if source == output:
        raise ValueError(
            "The crop OCR projector must never be modified in place."
        )
    if not source.is_file():
        raise FileNotFoundError(source)
    source_sha = _sha256(source)
    if source_sha != CROP_MMPROJ_SHA256:
        raise ValueError(
            "Source projector SHA-256 does not match the official crop OCR "
            f"projector: {source_sha}"
        )
    source_entry = find_metadata_entry(source, METADATA_KEY)
    if int(source_entry.value) != CROP_IMAGE_MAX_PIXELS:
        raise ValueError(
            "Source projector does not contain the official crop OCR pixel "
            f"budget: {source_entry.value}"
        )
    if output.exists():
        if (
            _sha256(output) == SPOTTING_MMPROJ_SHA256
            and int(find_metadata_entry(output, METADATA_KEY).value)
            == SPOTTING_IMAGE_MAX_PIXELS
        ):
            return
        raise FileExistsError(
            f"Refusing to overwrite an unverified projector: {output}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f"{output.name}.partial")
    if partial.exists():
        partial.unlink()
    try:
        with source.open("rb") as source_stream, partial.open(
            "xb"
        ) as output_stream:
            shutil.copyfileobj(
                source_stream,
                output_stream,
                length=8 * 1024 * 1024,
            )
            output_stream.flush()
            os.fsync(output_stream.fileno())
        patch_uint32_metadata(
            partial,
            key=METADATA_KEY,
            expected_value=CROP_IMAGE_MAX_PIXELS,
            replacement_value=SPOTTING_IMAGE_MAX_PIXELS,
        )
        output_sha = _sha256(partial)
        if output_sha != SPOTTING_MMPROJ_SHA256:
            raise ValueError(
                "Derived Spotting projector SHA-256 does not match the "
                f"official product contract: {output_sha}"
            )
        os.replace(partial, output)
    finally:
        if partial.exists():
            partial.unlink()


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Derive a dedicated PaddleOCR-VL Spotting projector without "
            "modifying the official crop OCR projector."
        )
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv if argv is not None else sys.argv[1:]))
    try:
        derive_spotting_projector(args.source, args.output)
    except (GGUFMetadataError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        "Prepared official PaddleOCR-VL Spotting projector: "
        f"{args.output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
