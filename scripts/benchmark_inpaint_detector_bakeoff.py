#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarking.inpaint_detector_bakeoff.ballons_ctbd import (  # noqa: E402
    BallonsCTBDReference,
    CTBDSettings,
)
from benchmarking.inpaint_detector_bakeoff.ballons_ctd import (  # noqa: E402
    BallonsCTDFullPageReference,
    BallonsCTDOriginalReference,
)
from benchmarking.inpaint_detector_bakeoff.contracts import (  # noqa: E402
    RoleCandidateSpec,
    binary_mask,
    mask_sha256,
)
from benchmarking.inpaint_detector_bakeoff.stage1 import (  # noqa: E402
    load_stage1_manifest,
    run_stage1,
    validate_source_only_manifest_v4,
)
from benchmarking.inpaint_detector_bakeoff.synthetic_detector import (  # noqa: E402
    CANDIDATE_ID as SYNTHETIC_FINETUNE_CANDIDATE_ID,
    CTDSyntheticFineTuneReference,
    cuda_peak_memory_provenance,
    evaluation_runtime_provenance,
)
from benchmarking.inpaint_detector_bakeoff.tiled_detector import (  # noqa: E402
    TiledCandidateReference,
    TiledInferenceSettings,
)
from modules.masking.ctd_refiner import CTDRefinerSettings  # noqa: E402
from modules.utils.download import ModelDownloader, ModelID  # noqa: E402
from scripts.validation_artifact_harness import (  # noqa: E402
    select_managed_output_directory,
)


FAMILY = "inpaint-detector-bakeoff-v2"
CATEGORY = "40-inpaint-mask-render"
EVALUATOR_PATH = Path(__file__).resolve()
STAGE1_PATH = ROOT / "benchmarking" / "inpaint_detector_bakeoff" / "stage1.py"
SYNTHETIC_DETECTOR_PATH = (
    ROOT / "benchmarking" / "inpaint_detector_bakeoff" / "synthetic_detector.py"
)
TILED_DETECTOR_PATH = (
    ROOT / "benchmarking" / "inpaint_detector_bakeoff" / "tiled_detector.py"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _text_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _current_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _existing_edit_paths(manifest_path: Path) -> dict[str, str]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths: dict[str, str] = {}
    for page in payload.get("pages", []):
        if not isinstance(page, dict):
            continue
        value = page.get("existing_source_edit_mask", page.get("baseline_mask"))
        if isinstance(value, dict):
            value = value.get("path")
        if isinstance(value, str) and value.strip():
            artifact = Path(value.strip())
            if not artifact.is_absolute():
                artifact = manifest_path.resolve().parent / artifact
            paths[str(page.get("page_id") or "")] = str(artifact.resolve())
    return paths


def _ownership_path(args: argparse.Namespace, page) -> Path:
    if args.ownership_root is not None:
        return args.ownership_root.resolve() / f"{page.page_id}_ownership.png"
    value = str(page.ownership_mask or "").strip()
    if not value:
        raise ValueError("ownership-ROI CTD requires an ownership mask")
    return Path(value)


def _candidate(args: argparse.Namespace):
    if args.candidate in {
        "ctd-synthetic-finetune",
        "ctd-synthetic-finetune-tiled",
    }:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for CTD synthetic fine-tune")
        model_path = Path(
            args.model
            or ModelDownloader.get_file_path(
                ModelID.CTD_TORCH,
                "comictextdetector.pt",
            )
        )
        adapter = CTDSyntheticFineTuneReference(
            model_path,
            Path(args.checkpoint),
            device=args.device,
            detect_size=int(args.detect_size),
            dilate_size=3,
            max_batches=int(args.max_batches),
            font_paths=tuple(Path(value) for value in args.font),
            expected_code_commit=_current_commit(),
        )
        infer = adapter.infer
        if args.candidate.endswith("-tiled"):
            infer = TiledCandidateReference(
                infer,
                TiledInferenceSettings(
                    tile_sizes=tuple(args.tile_size),
                    overlap=float(args.tile_overlap),
                ),
            ).infer
        return infer, model_path, None

    if args.candidate == "ballons-ctbd":
        model_path = Path(args.model or ModelDownloader.get_file_path(
            ModelID.RTDETR_V2_ONNX,
            "detector.onnx",
        ))
        providers = [args.provider]
        adapter = BallonsCTBDReference(
            str(model_path),
            providers,
            CTBDSettings(
                confidence_threshold=float(args.confidence),
                inpaint_mask_dilate=int(args.ctbd_dilate),
            ),
            disable_cpu_fallback=args.provider != "CPUExecutionProvider",
        )
        if not adapter.providers or adapter.providers[0] != args.provider:
            raise RuntimeError(
                "CTBD effective provider differs from the requested provider"
            )
        return adapter.infer, model_path, None

    model_path = Path(args.model or ModelDownloader.get_file_path(
        ModelID.CTD_TORCH if args.device != "cpu" else ModelID.CTD_ONNX,
        "comictextdetector.pt" if args.device != "cpu" else "comictextdetector.pt.onnx",
    ))
    if args.candidate == "ballons-ctd-original":
        if not args.ballons_root:
            raise ValueError("--ballons-root is required for the original CTD reference")
        adapter = BallonsCTDOriginalReference(
            ballons_root=args.ballons_root,
            model_path=str(model_path),
            device=args.device,
            detect_size=int(args.detect_size),
            dilate_size=3,
        )

        def original_infer(image_bgr):
            return adapter.infer(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

        return original_infer, model_path, None

    adapter = BallonsCTDFullPageReference(
        CTDRefinerSettings(
            detect_size=int(args.detect_size),
            det_rearrange_max_batches=int(args.max_batches),
            device=args.device,
            mask_dilate_size=0,
        ),
        dilate_size=3,
        model_path=model_path,
    )

    def infer(image_bgr):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return adapter.infer(image_rgb)

    if args.candidate == "ballons-ctd-tiled":
        infer = TiledCandidateReference(
            infer,
            TiledInferenceSettings(
                tile_sizes=tuple(args.tile_size),
                overlap=float(args.tile_overlap),
            ),
        ).infer

    def infer_ownership_roi(page, image_bgr):
        ownership_path = _ownership_path(args, page)
        ownership = cv2.imdecode(
            np.fromfile(ownership_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE
        )
        if ownership is None or ownership.size == 0:
            raise FileNotFoundError(ownership_path)
        return adapter.infer_with_ownership_rois(
            cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
            ownership,
        )

    return (
        infer,
        model_path,
        infer_ownership_roi if args.candidate == "ballons-ctd-text-roi" else None,
    )


def _source_and_ownership_sha256(args: argparse.Namespace, page) -> str:
    ownership_path = _ownership_path(args, page)
    digest = hashlib.sha256()
    for path in (Path(page.source_image), ownership_path):
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run source-only detector mask bake-off without creating candidate images.",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        choices=(
            "ballons-ctd",
            "ballons-ctd-text-roi",
            "ballons-ctd-original",
            "ballons-ctbd",
            "ctd-synthetic-finetune",
            "ctd-synthetic-finetune-tiled",
            "ballons-ctd-tiled",
        ),
        required=True,
    )
    parser.add_argument("--variant", choices=("raw", "refined", "dilated"), default="raw")
    parser.add_argument("--model", default="")
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Synthetic-only CTD text-seg fine-tune checkpoint.",
    )
    parser.add_argument(
        "--font",
        action="append",
        default=[],
        help=(
            "Exact local font inputs recorded by the synthetic fine-tune; "
            "repeat in the original training order."
        ),
    )
    parser.add_argument("--ballons-root", default="")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Private detector-output cache. The cache is provenance-keyed and fail-closed.",
    )
    parser.add_argument("--code-commit", default="")
    parser.add_argument("--provider", default="CUDAExecutionProvider")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--confidence", type=float, default=0.3)
    parser.add_argument("--ctbd-dilate", type=int, default=4)
    parser.add_argument("--detect-size", type=int, default=1280)
    parser.add_argument("--max-batches", type=int, default=4)
    parser.add_argument(
        "--tile-size",
        action="append",
        type=int,
        default=None,
        help="Source-space square tile size; repeat for multiscale tiling.",
    )
    parser.add_argument("--tile-overlap", type=float, default=0.2)
    parser.add_argument(
        "--ownership-root",
        type=Path,
        help=(
            "Optional sparse authoritative ownership masks named "
            "<page_id>_ownership.png for the ownership-ROI CTD variant."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.tile_size = list(args.tile_size or [768])
    manifest_path = args.manifest.resolve()
    manifest_binding = validate_source_only_manifest_v4(manifest_path)
    current_commit = _current_commit()
    if args.code_commit and args.code_commit != current_commit:
        raise ValueError("--code-commit must equal the current Git HEAD")
    if args.output_dir is not None and args.output_dir.resolve().exists():
        raise FileExistsError(
            "detector evaluation output directory must be fresh and absent: "
            f"{args.output_dir.resolve()}"
        )
    output_root, managed = select_managed_output_directory(
        family=FAMILY,
        category=CATEGORY,
        explicit_output_directory=args.output_dir,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        pages = load_stage1_manifest(manifest_path)
        if sorted(page.page_id for page in pages) != manifest_binding["page_ids"]:
            raise ValueError("loaded page inventory differs from the sealed manifest")
        evaluation_runtime: dict[str, object]
        if args.candidate == "ballons-ctbd":
            import onnxruntime as ort

            available = list(ort.get_available_providers())
            if args.provider not in available:
                raise RuntimeError(
                    f"requested ONNX provider is unavailable: {args.provider}"
                )
            evaluation_runtime = {
                "requested_provider": args.provider,
                "onnxruntime": str(ort.__version__),
                "available_providers": available,
            }
        else:
            evaluation_runtime = evaluation_runtime_provenance(args.device)
            if args.device.startswith("cuda"):
                import torch

                torch.cuda.reset_peak_memory_stats(args.device)
        infer, model_path, page_infer = _candidate(args)
        if args.candidate == "ballons-ctbd":
            adapter = getattr(infer, "__self__", None)
            configured = list(getattr(adapter, "providers", ()))
            if not configured or configured[0] != args.provider:
                raise RuntimeError(
                    "CTBD effective provider differs from the requested provider"
                )
            evaluation_runtime.update(
                {
                    "runtime_provider": configured[0],
                    "configured_providers": configured,
                    "cpu_ep_fallback_disabled": bool(
                        getattr(adapter, "disable_cpu_fallback", False)
                    ),
                }
            )
        preprocessing_contract = {
            "candidate": args.candidate,
            "detect_size": int(args.detect_size),
            "max_batches": int(args.max_batches),
            "confidence": float(args.confidence),
            "ctbd_dilate": int(args.ctbd_dilate),
            "device": args.device,
            "ownership_roi": args.candidate == "ballons-ctd-text-roi",
            "tile_sizes": (
                list(args.tile_size) if args.candidate.endswith("-tiled") else []
            ),
            "tile_overlap": (
                float(args.tile_overlap) if args.candidate.endswith("-tiled") else 0.0
            ),
            "ownership_contract": (
                "external-page-id-mask"
                if args.ownership_root
                else "manifest-sparse-mask"
            ),
            "checkpoint_sha256": (
                _sha256(Path(args.checkpoint).resolve())
                if args.checkpoint
                else ""
            ),
            "font_assets": [
                {
                    "name": Path(value).resolve().name,
                    "sha256": _sha256(Path(value).resolve()),
                    "size_bytes": Path(value).resolve().stat().st_size,
                }
                for value in args.font
            ],
        }
        candidate_id = (
            SYNTHETIC_FINETUNE_CANDIDATE_ID
            if args.candidate == "ctd-synthetic-finetune"
            else args.candidate
        )
        model_identity_sha256 = _text_sha256(
            {
                "model_sha256": _sha256(model_path),
                "checkpoint_sha256": preprocessing_contract["checkpoint_sha256"],
            }
        )
        candidate_spec = RoleCandidateSpec(
            candidate_id=candidate_id,
            provider=args.candidate,
            role="seed",
            variant="native-bundle-v2",
            code_commit=current_commit,
            model_sha256=model_identity_sha256,
            runtime_provider=(
                args.provider if args.candidate == "ballons-ctbd" else args.device
            ),
            preprocessing_contract_sha256=_text_sha256(preprocessing_contract),
        )
        cache_root = args.cache_dir.resolve() if args.cache_dir else output_root / "detector_cache"
        native_root = output_root / "native_masks"
        variant_outputs: dict[str, list[dict[str, object]]] = {
            "raw": [],
            "refined": [],
            "dilated": [],
        }
        output_artifacts: list[dict[str, object]] = []

        def record_mask_artifact(
            path: Path,
            mask: np.ndarray,
            *,
            page_id: str,
            role: str,
            variant_name: str,
        ) -> dict[str, object]:
            decoded = cv2.imdecode(
                np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE
            )
            if decoded is None or decoded.size == 0:
                raise OSError(f"failed to decode written mask artifact: {path}")
            normalized = binary_mask(decoded, mask.shape)
            if not np.array_equal(normalized, binary_mask(mask)):
                raise RuntimeError(f"written mask artifact differs from memory: {path}")
            record = {
                "page_id": page_id,
                "role": role,
                "variant": variant_name,
                "relative_path": path.relative_to(output_root).as_posix(),
                "artifact_sha256": _sha256(path),
                "binary_mask_sha256": mask_sha256(normalized),
                "pixel_count": int(cv2.countNonZero(normalized)),
            }
            output_artifacts.append(record)
            return record

        def write_native_masks(page, result) -> None:
            for variant_name, mask in (
                ("raw", result.raw_mask),
                ("refined", result.refined_mask),
                ("dilated", result.dilated_mask),
            ):
                path = native_root / variant_name / f"{page.page_id}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(path), mask):
                    raise OSError(f"failed to write native detector mask: {path}")
                variant_outputs[variant_name].append(
                    record_mask_artifact(
                        path,
                        mask,
                        page_id=page.page_id,
                        role="native_detector_mask",
                        variant_name=variant_name,
                    )
                )
            _write_json(
                native_root / "metadata" / f"{page.page_id}.json",
                {
                    **result.parity_record(),
                    "boxes": [
                        {
                            "xyxy": list(box.xyxy),
                            "label": box.label,
                            "score": float(box.score),
                            "provider": box.provider,
                        }
                        for box in result.boxes
                    ],
                },
            )

        rows, summary, edits = run_stage1(
            pages,
            infer,
            variant=args.variant,
            existing_edit_paths=_existing_edit_paths(manifest_path),
            candidate_spec=candidate_spec,
            cache_root=cache_root,
            result_sink=write_native_masks,
            page_infer=page_infer,
            cache_input_sha256=(
                (lambda page: _source_and_ownership_sha256(args, page))
                if args.candidate == "ballons-ctd-text-roi"
                else None
            ),
        )
        edit_root = output_root / "positive_edit_masks"
        edit_root.mkdir(parents=True, exist_ok=True)
        edit_outputs: list[dict[str, object]] = []
        for page_id, mask in edits.items():
            path = edit_root / f"{page_id}_positive_edit.png"
            if not cv2.imwrite(str(path), mask):
                raise OSError(f"failed to write edit mask for {page_id}")
            edit_outputs.append(
                record_mask_artifact(
                    path,
                    mask,
                    page_id=page_id,
                    role="positive_edit_mask",
                    variant_name=args.variant,
                )
            )
        variant_output_identity = {
            name: {
                "output_mask_set_sha256": _text_sha256(
                    [
                        {
                            "page_id": record["page_id"],
                            "binary_mask_sha256": record["binary_mask_sha256"],
                            "pixel_count": record["pixel_count"],
                        }
                        for record in sorted(
                            records, key=lambda row: str(row["page_id"])
                        )
                    ]
                ),
                "page_count": len(records),
                "provenance": "native_detector_output",
                "independent_output": True,
            }
            for name, records in variant_outputs.items()
        }
        if args.candidate == "ctd-synthetic-finetune":
            raw_identity = variant_output_identity["raw"]["output_mask_set_sha256"]
            refined_identity = variant_output_identity["refined"][
                "output_mask_set_sha256"
            ]
            if raw_identity != refined_identity:
                raise RuntimeError(
                    "synthetic fine-tune refined output must exactly reuse raw"
                )
            variant_output_identity["raw"].update(
                {"provenance": "native_finetuned_ctd_text_seg"}
            )
            variant_output_identity["refined"].update(
                {
                    "provenance": "exact_identity_reuse",
                    "independent_output": False,
                    "source_variant": "raw",
                    "source_output_mask_set_sha256": raw_identity,
                }
            )
            variant_output_identity["dilated"].update(
                {
                    "provenance": "elliptical_native3_from_raw",
                    "source_variant": "raw",
                }
            )
        artifact_inventory = {
            "schema_version": "inpaint-detector-output-artifact-inventory-v1",
            "records": sorted(
                output_artifacts,
                key=lambda value: (
                    str(value["role"]),
                    str(value["variant"]),
                    str(value["page_id"]),
                ),
            ),
        }
        artifact_inventory["inventory_sha256"] = _text_sha256(
            artifact_inventory["records"]
        )
        artifact_inventory_path = output_root / "output-artifact-inventory.json"
        _write_json(artifact_inventory_path, artifact_inventory)
        if args.candidate != "ballons-ctbd":
            import torch

            if args.device.startswith("cuda"):
                torch.cuda.synchronize(args.device)
            evaluation_runtime.update(cuda_peak_memory_provenance(args.device))
        inference_count = sum(
            1
            for row in rows
            if not bool(dict(row.get("runtime") or {}).get("cache_hit", False))
        )
        cache_hit_count = len(rows) - inference_count
        evaluation_runtime["inference_count"] = inference_count
        evaluation_runtime["cache_hit_count"] = cache_hit_count
        if args.device.startswith("cuda") and args.candidate != "ballons-ctbd":
            if inference_count < 1:
                raise RuntimeError(
                    "CUDA evaluation requires at least one current-process inference"
                )
            invalid_devices = sorted(
                {
                    str(dict(row.get("runtime") or {}).get("device") or "")
                    for row in rows
                    if not str(
                        dict(row.get("runtime") or {}).get("device") or ""
                    ).startswith("cuda")
                }
            )
            if invalid_devices:
                raise RuntimeError(
                    "CUDA evaluation produced a non-CUDA detector runtime: "
                    + ", ".join(invalid_devices)
                )
        result = {
            "schema_version": "inpaint-detector-bakeoff-stage1-v1",
            "candidate": candidate_id,
            "variant": args.variant,
            "manifest_sha256": _sha256(manifest_path),
            "manifest_binding": manifest_binding,
            "page_ids": list(manifest_binding["page_ids"]),
            "model": {
                "name": model_path.name,
                "size_bytes": model_path.stat().st_size,
                "sha256": _sha256(model_path),
                "provider": args.provider if args.candidate == "ballons-ctbd" else args.device,
                "checkpoint_sha256": (
                    _sha256(Path(args.checkpoint).resolve())
                    if args.checkpoint
                    else ""
                ),
            },
            "role_candidate": {
                "candidate_id": candidate_spec.candidate_id,
                "provider": candidate_spec.provider,
                "role": candidate_spec.role,
                "variant": candidate_spec.variant,
                "code_commit": candidate_spec.code_commit,
                "model_sha256": candidate_spec.model_sha256,
                "runtime_provider": candidate_spec.runtime_provider,
                "preprocessing_contract_sha256": candidate_spec.preprocessing_contract_sha256,
                "status": candidate_spec.status,
            },
            "detector_cache_root": str(cache_root),
            "variant_output_identity": variant_output_identity,
            "positive_edit_output_identity": {
                "output_mask_set_sha256": _text_sha256(
                    [
                        {
                            "page_id": record["page_id"],
                            "binary_mask_sha256": record["binary_mask_sha256"],
                            "pixel_count": record["pixel_count"],
                        }
                        for record in sorted(
                            edit_outputs, key=lambda row: str(row["page_id"])
                        )
                    ]
                ),
                "page_count": len(edit_outputs),
            },
            "output_artifact_inventory": {
                "relative_path": artifact_inventory_path.relative_to(
                    output_root
                ).as_posix(),
                "artifact_sha256": _sha256(artifact_inventory_path),
                "inventory_sha256": artifact_inventory["inventory_sha256"],
                "artifact_count": len(output_artifacts),
            },
            "evaluation_provenance": {
                "code_commit": current_commit,
                "evaluator_sha256": _sha256(EVALUATOR_PATH),
                "stage1_sha256": _sha256(STAGE1_PATH),
            "synthetic_detector_sha256": _sha256(SYNTHETIC_DETECTOR_PATH),
                "tiled_detector_sha256": _sha256(TILED_DETECTOR_PATH),
                "runtime": evaluation_runtime,
            },
            "summary": summary,
            "pages": rows,
        }
        _write_json(output_root / "stage1-results.json", result)
        if managed is not None:
            managed.complete(
                metadata={
                    "candidate": args.candidate,
                    "variant": args.variant,
                    "manifest_sha256": result["manifest_sha256"],
                    "summary": summary,
                }
            )
            mismatches = managed.verify()
            if mismatches:
                raise RuntimeError("managed artifact verification failed: " + "; ".join(mismatches))
            print(str(managed.run_root))
        else:
            print(str(output_root))
        return 0
    except BaseException as error:
        if managed is not None:
            managed.fail(
                error,
                metadata={
                    "candidate": args.candidate,
                    "variant": args.variant,
                    "manifest": manifest_path.name,
                },
            )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
