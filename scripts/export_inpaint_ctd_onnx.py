#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import onnx
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.masking.ctd_vendor.ctd import TextDetBase  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export CTD at one fixed ONNX size.")
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-size", type=int, default=1280)
    parser.add_argument("--opset", type=int, default=12)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = args.source_model.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.partial")
    model = TextDetBase(
        str(source),
        device="cpu",
        half=False,
        act="leaky",
    ).eval()
    example = torch.zeros(
        (1, 3, int(args.input_size), int(args.input_size)),
        dtype=torch.float32,
    )
    with torch.inference_mode():
        torch.onnx.export(
            model,
            example,
            str(temporary),
            export_params=True,
            opset_version=int(args.opset),
            do_constant_folding=True,
            input_names=["images"],
            output_names=["blk", "seg", "det"],
            dynamo=False,
        )
    exported = onnx.load(str(temporary))
    onnx.checker.check_model(exported)
    temporary.replace(output)
    metadata = {
        "schema_version": "inpaint-ctd-fixed-onnx-export-v1",
        "source_model": source.name,
        "source_sha256": _sha256(source),
        "output_model": output.name,
        "output_sha256": _sha256(output),
        "input_size": int(args.input_size),
        "opset": int(args.opset),
        "torch_version": torch.__version__,
        "onnx_version": onnx.__version__,
    }
    sidecar = output.with_suffix(output.suffix + ".json")
    sidecar_temporary = sidecar.with_name(f".{sidecar.name}.partial")
    sidecar_temporary.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    sidecar_temporary.replace(sidecar)
    print(json.dumps(metadata, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
