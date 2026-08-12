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
from benchmarking.inpaint_detector_bakeoff.contracts import RoleCandidateSpec  # noqa: E402
from benchmarking.inpaint_detector_bakeoff.stage1 import (  # noqa: E402
    load_stage1_manifest,
    run_stage1,
)
from modules.masking.ctd_refiner import CTDRefinerSettings  # noqa: E402
from modules.utils.download import ModelDownloader, ModelID  # noqa: E402
from scripts.validation_artifact_harness import (  # noqa: E402
    select_managed_output_directory,
)


FAMILY = "inpaint-detector-bakeoff-v2"
CATEGORY = "40-inpaint-mask-render"


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
            paths[str(page.get("page_id") or "")] = value.strip()
    return paths


def _ownership_path(args: argparse.Namespace, page) -> Path:
    if args.ownership_root is not None:
        return args.ownership_root.resolve() / f"{page.page_id}_ownership.png"
    value = str(page.ownership_mask or "").strip()
    if not value:
        raise ValueError("ownership-ROI CTD requires an ownership mask")
    return Path(value)


def _candidate(args: argparse.Namespace):
    if args.candidate == "ballons-ctbd":
        model_path = Path(args.model or ModelDownloader.get_file_path(
            ModelID.RTDETR_V2_ONNX,
            "detector.onnx",
        ))
        providers = [args.provider]
        if args.provider != "CPUExecutionProvider":
            providers.append("CPUExecutionProvider")
        adapter = BallonsCTBDReference(
            str(model_path),
            providers,
            CTBDSettings(
                confidence_threshold=float(args.confidence),
                inpaint_mask_dilate=int(args.ctbd_dilate),
            ),
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

    def infer_ownership_roi(page, image_bgr):
        ownership_path = _ownership_path(args, page)
        ownership = cv2.imread(str(ownership_path), cv2.IMREAD_GRAYSCALE)
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
        ),
        required=True,
    )
    parser.add_argument("--variant", choices=("raw", "refined", "dilated"), default="raw")
    parser.add_argument("--model", default="")
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
    manifest_path = args.manifest.resolve()
    output_root, managed = select_managed_output_directory(
        family=FAMILY,
        category=CATEGORY,
        explicit_output_directory=args.output_dir,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        pages = load_stage1_manifest(manifest_path)
        infer, model_path, page_infer = _candidate(args)
        preprocessing_contract = {
            "candidate": args.candidate,
            "detect_size": int(args.detect_size),
            "max_batches": int(args.max_batches),
            "confidence": float(args.confidence),
            "ctbd_dilate": int(args.ctbd_dilate),
            "device": args.device,
            "ownership_roi": args.candidate == "ballons-ctd-text-roi",
            "ownership_contract": (
                "external-page-id-mask"
                if args.ownership_root
                else "manifest-sparse-mask"
            ),
        }
        candidate_spec = RoleCandidateSpec(
            candidate_id=args.candidate,
            provider=args.candidate,
            role="seed",
            variant="native-bundle-v2",
            code_commit=args.code_commit or _current_commit(),
            model_sha256=_sha256(model_path),
            runtime_provider=(
                args.provider if args.candidate == "ballons-ctbd" else args.device
            ),
            preprocessing_contract_sha256=_text_sha256(preprocessing_contract),
        )
        cache_root = args.cache_dir.resolve() if args.cache_dir else output_root / "detector_cache"
        native_root = output_root / "native_masks"

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
        for page_id, mask in edits.items():
            if not cv2.imwrite(str(edit_root / f"{page_id}_positive_edit.png"), mask):
                raise OSError(f"failed to write edit mask for {page_id}")
        result = {
            "schema_version": "inpaint-detector-bakeoff-stage1-v1",
            "candidate": args.candidate,
            "variant": args.variant,
            "manifest_sha256": _sha256(manifest_path),
            "model": {
                "name": model_path.name,
                "size_bytes": model_path.stat().st_size,
                "sha256": _sha256(model_path),
                "provider": args.provider if args.candidate == "ballons-ctbd" else args.device,
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
