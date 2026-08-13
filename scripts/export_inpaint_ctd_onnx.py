#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.masking.ctd_vendor.ctd import TextDetBase  # noqa: E402
from benchmarking.inpaint_detector_bakeoff.synthetic_detector import (  # noqa: E402
    source_dependency_provenance,
    validate_checkpoint_provenance,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _current_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def load_validated_synthetic_checkpoint(
    *,
    checkpoint_path: Path,
    base_model_path: Path,
    font_paths: tuple[Path, ...],
    expected_code_commit: str,
) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("synthetic checkpoint root must be an object")
    validate_checkpoint_provenance(
        checkpoint,
        checkpoint_path=checkpoint_path,
        base_model_path=base_model_path,
        font_paths=font_paths,
        expected_code_commit=expected_code_commit,
    )
    return checkpoint


def apply_synthetic_checkpoint(
    model: TextDetBase,
    checkpoint: dict[str, object],
) -> None:
    state_dict = checkpoint.get("text_seg_state_dict")
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("synthetic checkpoint lacks text_seg_state_dict")
    model.text_seg.load_state_dict(state_dict, strict=True)
    model.eval()


def validate_export_parity(
    model: TextDetBase,
    onnx_path: Path,
    example: torch.Tensor,
) -> dict[str, object]:
    import onnxruntime as ort

    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    height, width = int(example.shape[-2]), int(example.shape[-1])
    x_axis = torch.linspace(0.0, 1.0, width, dtype=example.dtype).view(1, 1, 1, width)
    y_axis = torch.linspace(0.0, 1.0, height, dtype=example.dtype).view(1, 1, height, 1)
    gradient = ((x_axis + y_axis) * 0.5).expand_as(example).contiguous()
    checker = torch.empty_like(example)
    checker[..., 0::2, 0::2] = 1.0 / 255.0
    checker[..., 1::2, 1::2] = 0.5
    checker[..., 0::2, 1::2] = 1.0
    checker[..., 1::2, 0::2] = 2.0 / 255.0
    parity_inputs = (
        ("low_signal", torch.full_like(example, 1.0 / 255.0)),
        ("gradient", gradient),
        ("boundary_checker", checker),
    )
    records: list[dict[str, object]] = []
    for input_id, value in parity_inputs:
        input_array = np.ascontiguousarray(value.cpu().numpy())
        with torch.inference_mode():
            torch_outputs = tuple(
                np.ascontiguousarray(output.detach().cpu().numpy())
                for output in model(value)
            )
        onnx_outputs = tuple(
            np.ascontiguousarray(output)
            for output in session.run(None, {"images": input_array})
        )
        if len(torch_outputs) != 3 or len(onnx_outputs) != 3:
            raise RuntimeError("CTD ONNX parity requires exactly three outputs")
        for name, left, right in zip(
            ("blk", "seg", "det"), torch_outputs, onnx_outputs, strict=True
        ):
            if left.shape != right.shape:
                raise RuntimeError(f"CTD ONNX {name} output shape mismatch")
            if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
                raise RuntimeError(f"CTD ONNX {name} output contains NaN or Inf")
        torch_binary = (
            np.clip(torch_outputs[1] * 255.0, 0, 255).astype(np.uint8) > 0
        )
        onnx_binary = (
            np.clip(onnx_outputs[1] * 255.0, 0, 255).astype(np.uint8) > 0
        )
        xor_count = int(np.count_nonzero(torch_binary ^ onnx_binary))
        if xor_count:
            raise RuntimeError(
                f"CTD ONNX final binary segmentation parity failed: XOR={xor_count}"
            )
        records.append(
            {
                "input_id": input_id,
                "input_sha256": hashlib.sha256(input_array.tobytes()).hexdigest(),
                "segmentation_binary_xor_pixel_count": xor_count,
                "output_shapes": {
                    name: list(output.shape)
                    for name, output in zip(
                        ("blk", "seg", "det"), onnx_outputs, strict=True
                    )
                },
                "maximum_absolute_error": {
                    name: float(
                        np.max(
                            np.abs(
                                left.astype(np.float32) - right.astype(np.float32)
                            )
                        )
                    )
                    for name, left, right in zip(
                        ("blk", "seg", "det"),
                        torch_outputs,
                        onnx_outputs,
                        strict=True,
                    )
                },
            }
        )
    return {
        "provider": "CPUExecutionProvider",
        "onnxruntime_version": str(ort.__version__),
        "input_count": len(records),
        "input_contract": "deterministic_nonzero_graph_parity_v2",
        "segmentation_binary_xor_pixel_count": sum(
            int(record["segmentation_binary_xor_pixel_count"])
            for record in records
        ),
        "inputs": records,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export CTD at one fixed ONNX size.")
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument(
        "--synthetic-checkpoint",
        type=Path,
        help="Validated synthetic-only text-seg checkpoint to bake into ONNX.",
    )
    parser.add_argument(
        "--font",
        action="append",
        default=[],
        help="Exact repeatable font inputs from synthetic fine-tuning.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-size", type=int, default=1280)
    parser.add_argument("--opset", type=int, default=12)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = args.source_model.resolve()
    output = args.output.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    sidecar = output.with_suffix(output.suffix + ".json")
    temporary = output.with_name(f".{output.name}.partial")
    sidecar_temporary = sidecar.with_name(f".{sidecar.name}.partial")
    if any(
        path.exists()
        for path in (output, sidecar, temporary, sidecar_temporary)
    ):
        raise FileExistsError("ONNX export output and sidecar must be fresh")
    if int(args.input_size) < 32 or int(args.input_size) % 32:
        raise ValueError("ONNX input size must be at least 32 and divisible by 32")
    if int(args.opset) < 12:
        raise ValueError("ONNX opset must be at least 12")
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = (
        args.synthetic_checkpoint.resolve() if args.synthetic_checkpoint else None
    )
    checkpoint: dict[str, object] | None = None
    checkpoint_sha256 = ""
    checkpoint_code_commit = ""
    checkpoint_training_contract = ""
    if checkpoint_path is not None:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        checkpoint = load_validated_synthetic_checkpoint(
            checkpoint_path=checkpoint_path,
            base_model_path=source,
            font_paths=tuple(Path(value).resolve() for value in args.font),
            expected_code_commit=_current_commit(),
        )
        checkpoint_sha256 = _sha256(checkpoint_path)
        checkpoint_code_commit = str(checkpoint["code_commit"])
        checkpoint_training_contract = str(checkpoint["training_contract"])
    model = TextDetBase(
        str(source),
        device="cpu",
        half=False,
        act="leaky",
    ).eval()
    if checkpoint is not None:
        apply_synthetic_checkpoint(model, checkpoint)
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
    import onnx

    exported = onnx.load(str(temporary))
    onnx.checker.check_model(exported)
    parity = validate_export_parity(model, temporary, example)
    temporary.replace(output)
    metadata = {
        "schema_version": "inpaint-ctd-fixed-onnx-export-v2",
        "source_model": source.name,
        "source_sha256": _sha256(source),
        "output_model": output.name,
        "output_sha256": _sha256(output),
        "input_size": int(args.input_size),
        "opset": int(args.opset),
        "synthetic_checkpoint": checkpoint_path.name if checkpoint_path else None,
        "synthetic_checkpoint_sha256": checkpoint_sha256,
        "checkpoint_code_commit": checkpoint_code_commit,
        "checkpoint_training_contract": checkpoint_training_contract,
        "export_code_commit": _current_commit(),
        "exporter_sha256": _sha256(Path(__file__).resolve()),
        "source_dependency_sha256": source_dependency_provenance(),
        "preprocessing_contract": {
            "input_layout": "NCHW",
            "input_dtype": "float32",
            "input_size": int(args.input_size),
            "input_scale": 1.0 / 255.0,
            "activation": "leaky",
            "output_names": ["blk", "seg", "det"],
        },
        "python_to_onnx_parity": parity,
        "torch_version": torch.__version__,
        "onnx_version": onnx.__version__,
    }
    metadata["preprocessing_contract_sha256"] = _canonical_sha256(
        metadata["preprocessing_contract"]
    )
    sidecar_temporary.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    sidecar_temporary.replace(sidecar)
    print(json.dumps(metadata, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
