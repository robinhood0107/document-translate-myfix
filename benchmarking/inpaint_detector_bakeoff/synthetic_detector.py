from __future__ import annotations

import hashlib
import math
from pathlib import Path
import platform
import subprocess
import time
from typing import Mapping

import cv2
import numpy as np

from modules.masking.ctd_refiner import CTDRefiner, CTDRefinerSettings

from .contracts import CandidateMaskResult, binary_mask
from .synthetic_training import (
    supported_font_phrase_pairs,
    synthetic_training_digest,
)


CHECKPOINT_SCHEMA = "inpaint-ctd-synthetic-finetune-checkpoint-v4"
CANDIDATE_ID = "ctd-synthetic-low-contrast-finetune-v4"
TRAINING_CONTRACT = "deterministic_synthetic_only_no_holdout_training"
CHECKPOINT_SELECTION_CONTRACT = (
    "pareto_instance_seed_recall_all_page_false_claim_"
    "text_page_false_claim_no_text_false_claim_pixel_quality_v3"
)
SOURCE_DEPENDENCY_ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _is_code_commit(value: str) -> bool:
    return len(value) in {40, 64} and all(
        character in "0123456789abcdef" for character in value
    )


def source_dependency_provenance() -> dict[str, str]:
    """Hash the CTD implementation closure that affects train/eval/export."""

    paths = sorted(
        (SOURCE_DEPENDENCY_ROOT / "modules" / "masking" / "ctd_vendor").rglob(
            "*.py"
        )
    )
    paths.extend(
        [
            SOURCE_DEPENDENCY_ROOT / "modules" / "masking" / "ctd_refiner.py",
            SOURCE_DEPENDENCY_ROOT / "modules" / "utils" / "mask_roi.py",
            SOURCE_DEPENDENCY_ROOT / "modules" / "utils" / "textblock.py",
        ]
    )
    return {
        path.relative_to(SOURCE_DEPENDENCY_ROOT).as_posix(): _sha256(path)
        for path in sorted(set(paths))
    }


def current_code_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip().lower()
    if not _is_code_commit(commit):
        raise RuntimeError("unable to resolve the current Git commit")
    return commit


def font_asset_provenance(font_paths: tuple[Path, ...]) -> list[dict[str, object]]:
    normalized = tuple(path.resolve() for path in font_paths)
    supported = supported_font_phrase_pairs(normalized)
    phrases_by_path: dict[Path, list[str]] = {path: [] for path in normalized}
    for path, phrase in supported:
        phrases_by_path[path].append(phrase)
    return [
        {
            "name": path.name,
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
            "supported_phrases": phrases_by_path[path],
        }
        for path in normalized
    ]


def evaluation_runtime_provenance(device: str) -> dict[str, object]:
    import torch

    normalized = str(device).strip()
    if not normalized:
        raise ValueError("evaluation device must not be empty")
    cuda_requested = normalized.startswith("cuda")
    if cuda_requested and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but CUDA is unavailable")
    runtime: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
        "opencv": str(cv2.__version__),
        "requested_device": normalized,
        "runtime_provider": "torch_cuda" if cuda_requested else "torch_cpu",
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda": str(torch.version.cuda or ""),
        "cudnn": str(torch.backends.cudnn.version() or ""),
    }
    if cuda_requested:
        index = torch.cuda.current_device() if normalized == "cuda" else torch.device(normalized).index
        if index is None:
            index = torch.cuda.current_device()
        runtime.update(
            {
                "cuda_device_index": int(index),
                "cuda_device_name": str(torch.cuda.get_device_name(index)),
                "cuda_device_capability": list(torch.cuda.get_device_capability(index)),
            }
        )
    return runtime


def cuda_peak_memory_provenance(device: str) -> dict[str, int]:
    import torch

    if not str(device).startswith("cuda"):
        return {
            "peak_memory_allocated_bytes": 0,
            "peak_memory_reserved_bytes": 0,
        }
    return {
        "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def _validated_int(
    parameters: Mapping[str, object],
    field: str,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    value = parameters.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"CTD synthetic fine-tune {field} must be an integer")
    if value < minimum:
        raise ValueError(
            f"CTD synthetic fine-tune {field} must be at least {minimum}"
        )
    if maximum is not None and value > maximum:
        raise ValueError(
            f"CTD synthetic fine-tune {field} must be at most {maximum}"
        )
    return int(value)


def _validated_float(
    parameters: Mapping[str, object],
    field: str,
    *,
    positive: bool,
) -> float:
    value = parameters.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"CTD synthetic fine-tune {field} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"CTD synthetic fine-tune {field} must be finite")
    if positive and normalized <= 0.0:
        raise ValueError(f"CTD synthetic fine-tune {field} must be positive")
    if not positive and normalized < 0.0:
        raise ValueError(f"CTD synthetic fine-tune {field} must be non-negative")
    return normalized


def validate_training_hyperparameters(
    parameters: Mapping[str, object],
) -> dict[str, int | float | str | bool]:
    normalized: dict[str, int | float | str | bool] = {
        "seed": _validated_int(
            parameters,
            "seed",
            minimum=0,
            maximum=2**32 - 1,
        ),
        "train_samples": _validated_int(parameters, "train_samples", minimum=1),
        "dev_samples": _validated_int(parameters, "dev_samples", minimum=1),
        "image_size": _validated_int(parameters, "image_size", minimum=32),
        "epochs": _validated_int(parameters, "epochs", minimum=1),
        "batch_size": _validated_int(parameters, "batch_size", minimum=1),
        "learning_rate": _validated_float(
            parameters, "learning_rate", positive=True
        ),
        "anchor_weight": _validated_float(
            parameters, "anchor_weight", positive=False
        ),
        "distillation_weight": _validated_float(
            parameters, "distillation_weight", positive=True
        ),
        "adam_beta1": _validated_float(
            parameters, "adam_beta1", positive=False
        ),
        "adam_beta2": _validated_float(
            parameters, "adam_beta2", positive=False
        ),
        "adam_epsilon": _validated_float(
            parameters, "adam_epsilon", positive=True
        ),
        "weight_decay": _validated_float(
            parameters, "weight_decay", positive=False
        ),
        "raw_probability_threshold": _validated_float(
            parameters, "raw_probability_threshold", positive=True
        ),
        "evaluation_batch_size": _validated_int(
            parameters, "evaluation_batch_size", minimum=1
        ),
    }
    if int(normalized["image_size"]) % 32:
        raise ValueError("CTD synthetic fine-tune image_size must be divisible by 32")
    for field in ("adam_beta1", "adam_beta2"):
        if float(normalized[field]) >= 1.0:
            raise ValueError(f"CTD synthetic fine-tune {field} must be below 1")
    if float(normalized["raw_probability_threshold"]) > 1.0:
        raise ValueError(
            "CTD synthetic fine-tune raw_probability_threshold must not exceed 1"
        )
    if parameters.get("optimizer") != "AdamW":
        raise ValueError("CTD synthetic fine-tune optimizer must be AdamW")
    if parameters.get("adam_amsgrad") is not False:
        raise ValueError("CTD synthetic fine-tune adam_amsgrad must be false")
    normalized["optimizer"] = "AdamW"
    normalized["adam_amsgrad"] = False
    return normalized


def _validate_runtime_versions(value: object) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("CTD synthetic fine-tune lacks runtime_versions")
    required = (
        "python",
        "python_implementation",
        "platform",
        "pillow",
        "freetype",
        "torch",
        "numpy",
        "opencv",
        "device",
    )
    for field in required:
        if not str(value.get(field) or ""):
            raise ValueError(
                f"CTD synthetic fine-tune runtime_versions lacks {field}"
            )
    for field in ("cuda", "cudnn"):
        if field not in value or not isinstance(value[field], str):
            raise ValueError(
                f"CTD synthetic fine-tune runtime_versions lacks {field}"
            )
    for field in (
        "peak_memory_allocated_bytes",
        "peak_memory_reserved_bytes",
    ):
        amount = value.get(field)
        if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
            raise ValueError(
                f"CTD synthetic fine-tune runtime_versions lacks valid {field}"
            )
    allocated = int(value["peak_memory_allocated_bytes"])
    reserved = int(value["peak_memory_reserved_bytes"])
    if reserved < allocated:
        raise ValueError("CTD synthetic fine-tune peak reserved VRAM is below allocated")
    if str(value["device"]).startswith("cuda"):
        if allocated <= 0 or reserved <= 0:
            raise ValueError("CUDA training runtime lacks measured peak VRAM")
        if not str(value["cuda"]):
            raise ValueError("CUDA training runtime lacks a CUDA version")
        if not str(value.get("cuda_device_name") or ""):
            raise ValueError("CUDA training runtime lacks the device name")
        capability = value.get("cuda_device_capability")
        if (
            not isinstance(capability, list)
            or len(capability) != 2
            or not all(isinstance(component, int) for component in capability)
        ):
            raise ValueError("CUDA training runtime lacks device capability")
    elif allocated != 0 or reserved != 0:
        raise ValueError("CPU training runtime must record zero CUDA peak VRAM")


def _validate_font_assets(value: object) -> None:
    if not isinstance(value, list):
        raise ValueError("CTD synthetic fine-tune lacks font_assets")
    for asset in value:
        if not isinstance(asset, Mapping):
            raise ValueError("CTD synthetic fine-tune font asset must be an object")
        if not str(asset.get("name") or ""):
            raise ValueError("CTD synthetic fine-tune font asset lacks name")
        if not _is_sha256(str(asset.get("sha256") or "")):
            raise ValueError("CTD synthetic fine-tune font asset lacks SHA-256")
        size_bytes = asset.get("size_bytes")
        if (
            isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes < 1
        ):
            raise ValueError("CTD synthetic fine-tune font asset has invalid size")
        supported_phrases = asset.get("supported_phrases")
        if not isinstance(supported_phrases, list) or not all(
            isinstance(phrase, str) and phrase
            for phrase in supported_phrases
        ):
            raise ValueError(
                "CTD synthetic fine-tune font asset lacks supported phrases"
            )


def validate_checkpoint_provenance(
    checkpoint: Mapping[str, object],
    *,
    checkpoint_path: Path,
    base_model_path: Path,
    font_paths: tuple[Path, ...] = (),
    expected_code_commit: str | None = None,
) -> None:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError("unsupported CTD synthetic fine-tune checkpoint")
    state_dict = checkpoint.get("text_seg_state_dict")
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("CTD synthetic fine-tune checkpoint lacks text_seg_state_dict")
    expected = {
        "base_model_sha256": _sha256(base_model_path),
        "generator_sha256": _sha256(
            Path(__file__).with_name("synthetic_training.py")
        ),
        "detector_sha256": _sha256(Path(__file__)),
        "trainer_sha256": _sha256(
            Path(__file__).resolve().parents[2]
            / "scripts"
            / "train_inpaint_synthetic_detector_v4.py"
        ),
    }
    for field, current in expected.items():
        declared = str(checkpoint.get(field) or "")
        if not _is_sha256(declared) or declared != current:
            raise ValueError(
                f"CTD synthetic fine-tune {field} differs from runtime input"
            )
    if checkpoint.get("source_dependency_sha256") != source_dependency_provenance():
        raise ValueError(
            "CTD synthetic fine-tune source dependency closure differs from runtime"
        )
    if str(checkpoint.get("training_contract") or "") != TRAINING_CONTRACT:
        raise ValueError("CTD synthetic fine-tune lacks the synthetic-only contract")
    if (
        str(checkpoint.get("checkpoint_selection_contract") or "")
        != CHECKPOINT_SELECTION_CONTRACT
    ):
        raise ValueError("CTD synthetic fine-tune lacks the safety-first selection contract")
    hyperparameters = checkpoint.get("training_hyperparameters")
    if not isinstance(hyperparameters, Mapping):
        raise ValueError("CTD synthetic fine-tune lacks training_hyperparameters")
    normalized_hyperparameters = validate_training_hyperparameters(hyperparameters)
    seed = int(normalized_hyperparameters["seed"])
    train_samples = int(normalized_hyperparameters["train_samples"])
    dev_samples = int(normalized_hyperparameters["dev_samples"])
    expected_checkpoint_fields = {
        "seed": seed,
        "image_size": int(normalized_hyperparameters["image_size"]),
        "train_seed_first": seed,
        "train_seed_last": seed + train_samples - 1,
        "dev_seed_first": seed + train_samples,
        "dev_seed_last": seed + train_samples + dev_samples - 1,
    }
    for field, expected_value in expected_checkpoint_fields.items():
        value = checkpoint.get(field)
        if isinstance(value, bool) or value != expected_value:
            raise ValueError(
                f"CTD synthetic fine-tune {field} differs from hyperparameters"
            )
    epoch = checkpoint.get("epoch")
    if (
        isinstance(epoch, bool)
        or not isinstance(epoch, int)
        or epoch < 1
        or epoch > int(normalized_hyperparameters["epochs"])
    ):
        raise ValueError("CTD synthetic fine-tune checkpoint epoch is invalid")
    normalized_fonts = tuple(path.resolve() for path in font_paths)
    declared_fonts = checkpoint.get("font_assets")
    actual_fonts = font_asset_provenance(normalized_fonts)
    if declared_fonts != actual_fonts:
        raise ValueError("CTD synthetic fine-tune font assets differ from runtime inputs")
    shape = (
        int(normalized_hyperparameters["image_size"]),
        int(normalized_hyperparameters["image_size"]),
    )
    train_seeds = tuple(range(seed, seed + train_samples))
    dev_seeds = tuple(
        range(seed + train_samples, seed + train_samples + dev_samples)
    )
    expected_datasets = {
        "train_dataset_sha256": synthetic_training_digest(
            train_seeds,
            shape=shape,
            font_paths=normalized_fonts,
        ),
        "dev_dataset_sha256": synthetic_training_digest(
            dev_seeds,
            shape=shape,
            font_paths=normalized_fonts,
        ),
    }
    for field, expected_digest in expected_datasets.items():
        declared_digest = str(checkpoint.get(field) or "")
        if not _is_sha256(declared_digest) or declared_digest != expected_digest:
            raise ValueError(
                f"CTD synthetic fine-tune {field} differs from regenerated data"
            )
    code_commit = str(checkpoint.get("code_commit") or "")
    if not _is_code_commit(code_commit):
        raise ValueError("CTD synthetic fine-tune lacks a valid code_commit")
    runtime_commit = (expected_code_commit or current_code_commit()).lower()
    if code_commit != runtime_commit:
        raise ValueError(
            "CTD synthetic fine-tune code_commit differs from current Git HEAD"
        )
    _validate_runtime_versions(checkpoint.get("runtime_versions"))
    _validate_font_assets(declared_fonts)


class _FineTunedCTDRefiner(CTDRefiner):
    def __init__(
        self,
        settings: CTDRefinerSettings,
        *,
        base_model_path: Path,
        checkpoint: Mapping[str, object],
    ) -> None:
        super().__init__(settings)
        self._base_model_path = str(base_model_path.resolve())
        self._checkpoint = checkpoint
        self._fine_tune_loaded = False

    def _choose_model_path(self) -> str:
        return self._base_model_path

    def _ensure_model(self) -> None:
        super()._ensure_model()
        if self._fine_tune_loaded:
            return
        if self.backend != "torch" or not hasattr(self.net, "text_seg"):
            raise RuntimeError("CTD synthetic fine-tune requires the PyTorch backend")
        state_dict = self._checkpoint.get("text_seg_state_dict")
        if not isinstance(state_dict, Mapping):
            raise ValueError("CTD synthetic fine-tune checkpoint lacks text_seg_state_dict")
        self.net.text_seg.load_state_dict(state_dict, strict=True)
        self.net.eval()
        self._fine_tune_loaded = True


class CTDSyntheticFineTuneReference:
    """Lab-only CTD reference with a synthetic-fine-tuned segmentation head."""

    def __init__(
        self,
        base_model_path: str | Path,
        checkpoint_path: str | Path,
        *,
        device: str = "cpu",
        detect_size: int = 1280,
        dilate_size: int = 3,
        max_batches: int = 4,
        font_paths: tuple[str | Path, ...] = (),
        expected_code_commit: str | None = None,
    ) -> None:
        import torch

        self.device = str(device).strip()
        if not self.device:
            raise ValueError("CTD synthetic fine-tune device must not be empty")
        self.detect_size = int(detect_size)
        self.dilate_size = int(dilate_size)
        self.max_batches = int(max_batches)
        if self.detect_size < 1:
            raise ValueError("CTD synthetic fine-tune detect_size must be positive")
        if self.dilate_size < 0:
            raise ValueError("CTD synthetic fine-tune dilate_size must be non-negative")
        if self.max_batches < 1:
            raise ValueError("CTD synthetic fine-tune max_batches must be positive")
        self.base_model_path = Path(base_model_path).resolve()
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.font_paths = tuple(Path(path).resolve() for path in font_paths)
        self.evaluation_runtime = evaluation_runtime_provenance(self.device)
        if self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(self.device)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(self.checkpoint_path)
        self.checkpoint_sha256 = _sha256(self.checkpoint_path)
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(checkpoint, Mapping):
            raise ValueError("CTD synthetic fine-tune checkpoint root must be an object")
        validate_checkpoint_provenance(
            checkpoint,
            checkpoint_path=self.checkpoint_path,
            base_model_path=self.base_model_path,
            font_paths=self.font_paths,
            expected_code_commit=expected_code_commit,
        )
        self.checkpoint = checkpoint
        settings = CTDRefinerSettings(
            device=self.device,
            detect_size=self.detect_size,
            mask_dilate_size=0,
            det_rearrange_max_batches=self.max_batches,
        )
        self.refiner = _FineTunedCTDRefiner(
            settings,
            base_model_path=self.base_model_path,
            checkpoint=checkpoint,
        )

    def infer(self, image_bgr: np.ndarray) -> CandidateMaskResult:
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("CTD synthetic fine-tune expects a BGR image")
        started = time.perf_counter()
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        raw = binary_mask(self.refiner._infer_raw_mask(image_rgb))
        if self.dilate_size:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.dilate_size * 2 + 1, self.dilate_size * 2 + 1),
                (self.dilate_size, self.dilate_size),
            )
            dilated = binary_mask(cv2.dilate(raw, kernel), raw.shape)
        else:
            dilated = raw.copy()
        return CandidateMaskResult(
            candidate_id=CANDIDATE_ID,
            raw_mask=raw,
            refined_mask=raw.copy(),
            dilated_mask=dilated,
            runtime={
                "seconds": time.perf_counter() - started,
                "device": self.device,
                "detect_size": self.detect_size,
                "max_batches": self.max_batches,
                "base_model_sha256": str(self.checkpoint["base_model_sha256"]),
                "checkpoint_sha256": self.checkpoint_sha256,
                "generator_sha256": str(self.checkpoint["generator_sha256"]),
                "detector_sha256": str(self.checkpoint["detector_sha256"]),
                "trainer_sha256": str(self.checkpoint["trainer_sha256"]),
                "source_dependency_sha256": dict(
                    self.checkpoint["source_dependency_sha256"]
                ),
                "training_contract": str(self.checkpoint["training_contract"]),
                "checkpoint_selection_contract": str(
                    self.checkpoint["checkpoint_selection_contract"]
                ),
                "training_hyperparameters": dict(
                    self.checkpoint["training_hyperparameters"]
                ),
                "train_dataset_sha256": str(
                    self.checkpoint["train_dataset_sha256"]
                ),
                "dev_dataset_sha256": str(
                    self.checkpoint["dev_dataset_sha256"]
                ),
                "training_runtime_versions": dict(
                    self.checkpoint["runtime_versions"]
                ),
                "code_commit": str(self.checkpoint["code_commit"]),
                "raw_output_contract": "native_finetuned_ctd_text_seg",
                "refined_output_contract": "exact_identity_reuse_of_raw",
                "dilated_output_contract": "elliptical_native3_from_raw",
                "raw_output_sha256": hashlib.sha256(
                    raw.tobytes(order="C")
                ).hexdigest(),
                "refined_output_sha256": hashlib.sha256(
                    raw.tobytes(order="C")
                ).hexdigest(),
                "evaluation_runtime": dict(self.evaluation_runtime),
                **cuda_peak_memory_provenance(self.device),
            },
        )
