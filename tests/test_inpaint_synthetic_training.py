from __future__ import annotations

import json
from pathlib import Path
import subprocess
import struct
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from benchmarking.inpaint_detector_bakeoff.synthetic_training import (
    BACKGROUND_KINDS,
    TEXT_STYLES,
    _shift_mask,
    supported_font_phrase_pairs,
    synthetic_training_digest,
    synthetic_training_sample,
)
from benchmarking.inpaint_detector_bakeoff.stage1 import (
    load_stage1_manifest,
    manifest_page_artifact_sha256,
    source_manifest_page_inventory_sha256,
    validate_source_only_manifest_v4,
)
from benchmarking.inpaint_detector_bakeoff.synthetic_detector import (
    CHECKPOINT_SELECTION_CONTRACT,
    CHECKPOINT_SCHEMA,
    TRAINING_CONTRACT,
    _sha256,
    cuda_peak_memory_provenance,
    evaluation_runtime_provenance,
    source_dependency_provenance,
    validate_checkpoint_provenance,
    validate_training_hyperparameters,
)
from benchmarking.inpaint_detector_bakeoff import synthetic_detector
from scripts import benchmark_inpaint_detector_bakeoff
from scripts import export_inpaint_ctd_onnx
from scripts.build_inpaint_generalization_synthetic_v4 import (
    main as build_generalization_synthetic_main,
)
from scripts.reseal_inpaint_source_manifest_v4 import reseal_manifest
from scripts.train_inpaint_synthetic_detector_v4 import (
    CTD_RAW_PROBABILITY_THRESHOLD,
    _pareto_epochs,
    _runtime_versions,
    _selection_key,
    _selection_metric_summary,
    main as train_synthetic_detector_main,
)


def test_synthetic_training_sample_is_deterministic_and_nonempty() -> None:
    first = synthetic_training_sample(1234, shape=(96, 104))
    second = synthetic_training_sample(1234, shape=(96, 104))

    assert first.sample_id == second.sample_id
    assert np.array_equal(first.image, second.image)
    assert np.array_equal(first.target, second.target)
    assert first.image.shape == (96, 104, 3)
    assert first.target.shape == (96, 104)
    assert np.any(first.target)
    assert first.has_text is True


def test_synthetic_training_covers_all_background_and_text_families() -> None:
    observed_backgrounds = set()
    observed_styles = set()
    for seed in range(256):
        sample = synthetic_training_sample(seed, shape=(96, 96))
        observed_backgrounds.add(sample.background_kind)
        observed_styles.add(sample.text_style)
        assert sample.has_text == (seed % 5 != 0)
        assert np.any(sample.target) == sample.has_text
        assert np.any(sample.image != sample.background) == sample.has_text

    assert observed_backgrounds == set(BACKGROUND_KINDS)
    assert observed_styles == set(TEXT_STYLES)


def test_shift_mask_clips_at_crop_edge_without_wrapping() -> None:
    mask = np.zeros((12, 12), np.uint8)
    mask[0:3, 0:3] = 255

    shifted = _shift_mask(mask, -1, -1)

    assert np.count_nonzero(shifted) == 4
    assert np.count_nonzero(shifted[-3:, :]) == 0
    assert np.count_nonzero(shifted[:, -3:]) == 0


def test_synthetic_training_rejects_missing_font_asset(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        synthetic_training_sample(
            1234,
            shape=(96, 96),
            font_paths=(tmp_path / "missing.ttf",),
        )


def _write_cmap12_font(path: Path, codepoints: set[int]) -> None:
    groups = b"".join(
        struct.pack(">III", codepoint, codepoint, index + 1)
        for index, codepoint in enumerate(sorted(codepoints))
    )
    subtable = struct.pack(
        ">HHIII",
        12,
        0,
        16 + len(groups),
        0,
        len(codepoints),
    ) + groups
    cmap = struct.pack(">HHHHI", 0, 1, 3, 10, 12) + subtable
    sfnt = (
        struct.pack(">IHHHH", 0x00010000, 1, 0, 0, 0)
        + b"cmap"
        + struct.pack(">III", 0, 28, len(cmap))
        + cmap
    )
    path.write_bytes(sfnt)


def test_font_phrase_selection_uses_cmap_supported_pairs(tmp_path: Path) -> None:
    japanese = tmp_path / "japanese.ttf"
    korean = tmp_path / "korean.ttf"
    unsupported = tmp_path / "latin-only.ttf"
    _write_cmap12_font(japanese, {ord(character) for character in "文字"})
    _write_cmap12_font(korean, {ord(character) for character in "한글"})
    _write_cmap12_font(unsupported, {ord("A")})

    pairs = supported_font_phrase_pairs((japanese, korean, unsupported))

    assert pairs == (
        (japanese.resolve(), "文字"),
        (korean.resolve(), "한글"),
    )


def test_font_phrase_selection_rejects_all_tofu_assets(tmp_path: Path) -> None:
    unsupported = tmp_path / "latin-only.ttf"
    _write_cmap12_font(unsupported, {ord("A")})

    with pytest.raises(ValueError, match="support none"):
        supported_font_phrase_pairs((unsupported,))


def test_synthetic_training_dataset_digest_is_ordered_and_repeatable() -> None:
    first = synthetic_training_digest((101, 102, 103), shape=(64, 64))
    repeated = synthetic_training_digest((101, 102, 103), shape=(64, 64))
    reordered = synthetic_training_digest((103, 102, 101), shape=(64, 64))

    assert first == repeated
    assert first != reordered


def _checkpoint_payload(base_model, checkpoint):
    checkpoint.write_bytes(b"checkpoint")
    generator = Path(synthetic_detector.__file__).with_name("synthetic_training.py")
    trainer = (
        Path(synthetic_detector.__file__).resolve().parents[2]
        / "scripts"
        / "train_inpaint_synthetic_detector_v4.py"
    )
    hyperparameters = _valid_training_hyperparameters()
    seed = int(hyperparameters["seed"])
    train_count = int(hyperparameters["train_samples"])
    dev_count = int(hyperparameters["dev_samples"])
    shape = (int(hyperparameters["image_size"]),) * 2
    code_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(synthetic_detector.__file__).resolve().parents[2],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "schema_version": CHECKPOINT_SCHEMA,
        "text_seg_state_dict": {"placeholder": b"test"},
        "base_model_sha256": _sha256(base_model),
        "generator_sha256": _sha256(generator),
        "detector_sha256": _sha256(Path(synthetic_detector.__file__)),
        "trainer_sha256": _sha256(trainer),
        "source_dependency_sha256": source_dependency_provenance(),
        "training_contract": TRAINING_CONTRACT,
        "checkpoint_selection_contract": CHECKPOINT_SELECTION_CONTRACT,
        "training_hyperparameters": hyperparameters,
        "seed": 41371,
        "image_size": 32,
        "epoch": 1,
        "train_seed_first": 41371,
        "train_seed_last": seed + train_count - 1,
        "dev_seed_first": seed + train_count,
        "dev_seed_last": seed + train_count + dev_count - 1,
        "train_dataset_sha256": synthetic_training_digest(
            tuple(range(seed, seed + train_count)), shape=shape
        ),
        "dev_dataset_sha256": synthetic_training_digest(
            tuple(range(seed + train_count, seed + train_count + dev_count)),
            shape=shape,
        ),
        "runtime_versions": {
            "python": "3.12.0",
            "python_implementation": "CPython",
            "platform": "Windows-11",
            "pillow": "11.0.0",
            "freetype": "2.13.3",
            "torch": "2.7.0",
            "numpy": "2.2.0",
            "opencv": "4.11.0",
            "device": "cpu",
            "cuda": "",
            "cudnn": "",
            "peak_memory_allocated_bytes": 0,
            "peak_memory_reserved_bytes": 0,
        },
        "code_commit": code_commit,
        "font_assets": [],
    }


def test_synthetic_detector_checkpoint_provenance_matches_runtime(tmp_path) -> None:
    base_model = tmp_path / "base.pt"
    base_model.write_bytes(b"base-model")
    checkpoint = tmp_path / "checkpoint.pt"
    payload = _checkpoint_payload(base_model, checkpoint)

    validate_checkpoint_provenance(
        payload,
        checkpoint_path=checkpoint,
        base_model_path=base_model,
    )


def test_synthetic_detector_checkpoint_rejects_stale_runtime_sha(tmp_path) -> None:
    base_model = tmp_path / "base.pt"
    base_model.write_bytes(b"base-model")
    checkpoint = tmp_path / "checkpoint.pt"
    payload = _checkpoint_payload(base_model, checkpoint)
    payload["detector_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="detector_sha256 differs"):
        validate_checkpoint_provenance(
            payload,
            checkpoint_path=checkpoint,
            base_model_path=base_model,
        )


def _valid_training_hyperparameters() -> dict[str, object]:
    return {
        "seed": 41371,
        "train_samples": 1,
        "dev_samples": 1,
        "image_size": 32,
        "epochs": 2,
        "batch_size": 4,
        "learning_rate": 1e-4,
        "anchor_weight": 1e-3,
        "distillation_weight": 1.0,
        "optimizer": "AdamW",
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1e-8,
        "weight_decay": 0.01,
        "adam_amsgrad": False,
        "raw_probability_threshold": 1.0 / 255.0,
        "evaluation_batch_size": 8,
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("seed", -1, "at least"),
        ("seed", 2**32, "at most"),
        ("seed", True, "integer"),
        ("train_samples", 0, "at least"),
        ("dev_samples", 0, "at least"),
        ("image_size", 319, "divisible"),
        ("epochs", 0, "at least"),
        ("batch_size", 0, "at least"),
        ("learning_rate", 0.0, "positive"),
        ("learning_rate", float("nan"), "finite"),
        ("anchor_weight", -1.0, "non-negative"),
        ("distillation_weight", 0.0, "positive"),
        ("distillation_weight", float("inf"), "finite"),
        ("adam_beta1", 1.0, "below 1"),
        ("adam_epsilon", 0.0, "positive"),
        ("raw_probability_threshold", 1.1, "must not exceed"),
        ("evaluation_batch_size", 0, "at least"),
    ),
)
def test_training_hyperparameters_reject_unsafe_numeric_values(
    field: str,
    value: object,
    message: str,
) -> None:
    parameters = _valid_training_hyperparameters()
    parameters[field] = value  # type: ignore[assignment]

    with pytest.raises(ValueError, match=message):
        validate_training_hyperparameters(parameters)


def test_checkpoint_requires_dataset_and_runtime_provenance(tmp_path: Path) -> None:
    base_model = tmp_path / "base.pt"
    base_model.write_bytes(b"base-model")
    checkpoint = tmp_path / "checkpoint.pt"
    payload = _checkpoint_payload(base_model, checkpoint)
    del payload["dev_dataset_sha256"]

    with pytest.raises(ValueError, match="dev_dataset_sha256"):
        validate_checkpoint_provenance(
            payload,
            checkpoint_path=checkpoint,
            base_model_path=base_model,
        )


def test_checkpoint_rejects_source_dependency_closure_drift(tmp_path: Path) -> None:
    base_model = tmp_path / "base.pt"
    base_model.write_bytes(b"base-model")
    checkpoint = tmp_path / "checkpoint.pt"
    payload = _checkpoint_payload(base_model, checkpoint)
    payload["source_dependency_sha256"] = {"changed.py": "0" * 64}

    with pytest.raises(ValueError, match="source dependency closure"):
        validate_checkpoint_provenance(
            payload,
            checkpoint_path=checkpoint,
            base_model_path=base_model,
        )


def test_checkpoint_rejects_regenerated_dataset_mismatch(tmp_path: Path) -> None:
    base_model = tmp_path / "base.pt"
    base_model.write_bytes(b"base-model")
    checkpoint = tmp_path / "checkpoint.pt"
    payload = _checkpoint_payload(base_model, checkpoint)
    payload["train_dataset_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="differs from regenerated data"):
        validate_checkpoint_provenance(
            payload,
            checkpoint_path=checkpoint,
            base_model_path=base_model,
        )


def test_checkpoint_rejects_code_commit_other_than_current_head(
    tmp_path: Path,
) -> None:
    base_model = tmp_path / "base.pt"
    base_model.write_bytes(b"base-model")
    checkpoint = tmp_path / "checkpoint.pt"
    payload = _checkpoint_payload(base_model, checkpoint)
    payload["code_commit"] = "0" * 40

    with pytest.raises(ValueError, match="current Git HEAD"):
        validate_checkpoint_provenance(
            payload,
            checkpoint_path=checkpoint,
            base_model_path=base_model,
        )


def test_checkpoint_rejects_font_inputs_other_than_training_assets(
    tmp_path: Path,
) -> None:
    base_model = tmp_path / "base.pt"
    base_model.write_bytes(b"base-model")
    checkpoint = tmp_path / "checkpoint.pt"
    payload = _checkpoint_payload(base_model, checkpoint)
    font = tmp_path / "font.ttf"
    _write_cmap12_font(font, {ord(character) for character in "文字"})

    with pytest.raises(ValueError, match="font assets differ"):
        validate_checkpoint_provenance(
            payload,
            checkpoint_path=checkpoint,
            base_model_path=base_model,
            font_paths=(font,),
        )


def test_checkpoint_rejects_unreproducible_hyperparameters(tmp_path: Path) -> None:
    base_model = tmp_path / "base.pt"
    base_model.write_bytes(b"base-model")
    checkpoint = tmp_path / "checkpoint.pt"
    payload = _checkpoint_payload(base_model, checkpoint)
    payload["training_hyperparameters"] = {
        **payload["training_hyperparameters"],
        "learning_rate": float("nan"),
    }

    with pytest.raises(ValueError, match="learning_rate must be finite"):
        validate_checkpoint_provenance(
            payload,
            checkpoint_path=checkpoint,
            base_model_path=base_model,
        )


def test_checkpoint_requires_pillow_and_freetype_runtime_versions(
    tmp_path: Path,
) -> None:
    base_model = tmp_path / "base.pt"
    base_model.write_bytes(b"base-model")
    checkpoint = tmp_path / "checkpoint.pt"
    payload = _checkpoint_payload(base_model, checkpoint)
    payload["runtime_versions"] = dict(payload["runtime_versions"])
    del payload["runtime_versions"]["freetype"]

    with pytest.raises(ValueError, match="runtime_versions lacks freetype"):
        validate_checkpoint_provenance(
            payload,
            checkpoint_path=checkpoint,
            base_model_path=base_model,
        )


@pytest.mark.parametrize(
    ("device", "allocated", "reserved", "message"),
    (
        ("cpu", 1, 1, "CPU training runtime"),
        ("cuda", 0, 0, "measured peak VRAM"),
        ("cuda", 20, 10, "reserved VRAM is below allocated"),
    ),
)
def test_checkpoint_rejects_inconsistent_peak_vram_provenance(
    tmp_path: Path,
    device: str,
    allocated: int,
    reserved: int,
    message: str,
) -> None:
    base_model = tmp_path / "base.pt"
    base_model.write_bytes(b"base-model")
    checkpoint = tmp_path / "checkpoint.pt"
    payload = _checkpoint_payload(base_model, checkpoint)
    runtime = dict(payload["runtime_versions"])
    runtime.update(
        {
            "device": device,
            "cuda": "13.0" if device == "cuda" else "",
            "cudnn": "9100" if device == "cuda" else "",
            "peak_memory_allocated_bytes": allocated,
            "peak_memory_reserved_bytes": reserved,
        }
    )
    if device == "cuda":
        runtime["cuda_device_name"] = "fixture-gpu"
        runtime["cuda_device_capability"] = [8, 9]
    payload["runtime_versions"] = runtime

    with pytest.raises(ValueError, match=message):
        validate_checkpoint_provenance(
            payload,
            checkpoint_path=checkpoint,
            base_model_path=base_model,
        )


def test_training_runtime_records_python_pillow_and_freetype() -> None:
    import torch

    runtime = _runtime_versions(torch, "cpu")

    assert runtime["python"]
    assert runtime["python_implementation"]
    assert runtime["pillow"]
    assert runtime["freetype"]
    assert runtime["device"] == "cpu"


def test_evaluation_runtime_records_cuda_provider_gpu_and_peak() -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA runtime is unavailable")

    torch.cuda.reset_peak_memory_stats("cuda")
    runtime = evaluation_runtime_provenance("cuda")
    peak = cuda_peak_memory_provenance("cuda")

    assert runtime["runtime_provider"] == "torch_cuda"
    assert runtime["cuda"]
    assert runtime["cudnn"]
    assert runtime["cuda_device_name"]
    assert runtime["cuda_device_capability"]
    assert peak["peak_memory_allocated_bytes"] >= 0
    assert peak["peak_memory_reserved_bytes"] >= 0


def test_checkpoint_selection_includes_text_and_no_text_false_claim_axes() -> None:
    safe = {
        "instance_seed_recall": 1.0,
        "all_page_false_pixel_count": 5,
        "text_page_false_pixel_count": 2,
        "no_text_false_pixel_count": 3,
        "precision": 0.89,
        "recall": 0.91,
        "f1": 0.90,
    }
    more_false = {
        **safe,
        "all_page_false_pixel_count": 6,
        "no_text_false_pixel_count": 4,
        "f1": 0.99,
    }
    missed_instance = {
        **safe,
        "instance_seed_recall": 0.99,
        "all_page_false_pixel_count": 0,
        "text_page_false_pixel_count": 0,
        "no_text_false_pixel_count": 0,
        "f1": 1.0,
    }
    text_page_unsafe = {
        **safe,
        "text_page_false_pixel_count": 3,
        "no_text_false_pixel_count": 2,
        "f1": 0.99,
    }

    assert _selection_key(safe) > _selection_key(more_false)
    assert _selection_key(safe) > _selection_key(missed_instance)
    assert _selection_key(safe) > _selection_key(text_page_unsafe)


def test_training_retains_nondominated_safety_recall_epochs() -> None:
    high_recall = {
        "epoch": 1,
        "instance_seed_recall": 1.0,
        "all_page_false_pixel_count": 20,
        "text_page_false_pixel_count": 16,
        "no_text_false_pixel_count": 4,
        "recall": 0.95,
        "precision": 0.80,
    }
    safer = {
        "epoch": 2,
        "instance_seed_recall": 0.95,
        "all_page_false_pixel_count": 2,
        "text_page_false_pixel_count": 2,
        "no_text_false_pixel_count": 0,
        "recall": 0.90,
        "precision": 0.98,
    }
    dominated = {
        "epoch": 3,
        "instance_seed_recall": 0.90,
        "all_page_false_pixel_count": 5,
        "text_page_false_pixel_count": 4,
        "no_text_false_pixel_count": 1,
        "recall": 0.80,
        "precision": 0.70,
    }

    assert _pareto_epochs([high_recall, safer, dominated]) == (1, 2)


def test_training_pareto_retains_text_vs_no_text_safety_tradeoff() -> None:
    text_safe = {
        "epoch": 1,
        "instance_seed_recall": 1.0,
        "all_page_false_pixel_count": 10,
        "text_page_false_pixel_count": 1,
        "no_text_false_pixel_count": 9,
        "recall": 0.95,
        "precision": 0.90,
    }
    no_text_safe = {
        **text_safe,
        "epoch": 2,
        "text_page_false_pixel_count": 9,
        "no_text_false_pixel_count": 1,
    }
    dominated = {
        **text_safe,
        "epoch": 3,
        "all_page_false_pixel_count": 12,
        "text_page_false_pixel_count": 10,
        "no_text_false_pixel_count": 2,
    }

    assert _pareto_epochs([text_safe, no_text_safe, dominated]) == (1, 2)


def test_selection_summary_distinguishes_selected_from_max_f1() -> None:
    selected = {"f1": 0.80, "instance_seed_recall": 1.0}
    history = [
        {"epoch": 1, "loss": 0.3, **selected},
        {"epoch": 2, "loss": 0.2, "f1": 0.95, "instance_seed_recall": 0.9},
    ]

    summary = _selection_metric_summary(selected, history)

    assert summary["selected_dev_f1"] == pytest.approx(0.80)
    assert summary["max_dev_f1"] == pytest.approx(0.95)
    assert summary["max_dev_f1_epoch"] == 2
    assert "best_dev_f1" not in summary


def test_benchmark_passes_max_batches_to_synthetic_adapter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    class FakeReference:
        def __init__(self, model_path, checkpoint_path, **kwargs) -> None:
            captured["model_path"] = model_path
            captured["checkpoint_path"] = checkpoint_path
            captured.update(kwargs)

        def infer(self, image):
            return image

    monkeypatch.setattr(
        benchmark_inpaint_detector_bakeoff,
        "CTDSyntheticFineTuneReference",
        FakeReference,
    )
    args = SimpleNamespace(
        candidate="ctd-synthetic-finetune",
        checkpoint=str(tmp_path / "checkpoint.pt"),
        model=str(tmp_path / "base.pt"),
        device="cuda",
        detect_size=1536,
        max_batches=7,
        font=[],
    )

    benchmark_inpaint_detector_bakeoff._candidate(args)

    assert captured["detect_size"] == 1536
    assert captured["max_batches"] == 7


@pytest.mark.parametrize(
    ("provider", "fallback_disabled"),
    (
        ("CUDAExecutionProvider", True),
        ("CPUExecutionProvider", False),
    ),
)
def test_ctbd_candidate_uses_only_the_claimed_provider_and_disables_cuda_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    provider: str,
    fallback_disabled: bool,
) -> None:
    captured: dict[str, object] = {}

    class FakeReference:
        def __init__(self, model_path, providers, settings, **kwargs) -> None:
            captured.update(
                {
                    "model_path": model_path,
                    "providers": providers,
                    "settings": settings,
                    **kwargs,
                }
            )
            self.providers = list(providers)

        def infer(self, image):
            return image

    monkeypatch.setattr(
        benchmark_inpaint_detector_bakeoff,
        "BallonsCTBDReference",
        FakeReference,
    )
    args = SimpleNamespace(
        candidate="ballons-ctbd",
        model=str(tmp_path / "detector.onnx"),
        provider=provider,
        confidence=0.3,
        ctbd_dilate=4,
    )

    benchmark_inpaint_detector_bakeoff._candidate(args)

    assert captured["providers"] == [provider]
    assert captured["disable_cpu_fallback"] is fallback_disabled


def test_synthetic_detector_hashes_checkpoint_once_at_initialization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import torch

    base_model = tmp_path / "base.pt"
    base_model.write_bytes(b"base-model")
    checkpoint = tmp_path / "checkpoint.pt"
    payload = _checkpoint_payload(base_model, checkpoint)
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: payload)
    real_sha256 = synthetic_detector._sha256
    checkpoint_hash_calls = 0

    def counting_sha256(path: Path) -> str:
        nonlocal checkpoint_hash_calls
        if Path(path).resolve() == checkpoint.resolve():
            checkpoint_hash_calls += 1
        return real_sha256(path)

    monkeypatch.setattr(synthetic_detector, "_sha256", counting_sha256)
    reference = synthetic_detector.CTDSyntheticFineTuneReference(
        base_model,
        checkpoint,
        device="cpu",
        detect_size=128,
        max_batches=6,
    )
    reference.refiner._infer_raw_mask = lambda image: np.zeros(
        image.shape[:2], dtype=np.uint8
    )

    first = reference.infer(np.zeros((16, 16, 3), dtype=np.uint8))
    second = reference.infer(np.zeros((16, 16, 3), dtype=np.uint8))

    assert checkpoint_hash_calls == 1
    assert reference.refiner.settings.det_rearrange_max_batches == 6
    assert first.runtime["checkpoint_sha256"] == second.runtime["checkpoint_sha256"]
    assert first.runtime["training_hyperparameters"] == payload[
        "training_hyperparameters"
    ]


def test_synthetic_checkpoint_payload_round_trips_with_weights_only(
    tmp_path: Path,
) -> None:
    import torch

    base_model = tmp_path / "base.pt"
    base_model.write_bytes(b"base-model")
    checkpoint = tmp_path / "checkpoint.pt"
    payload = _checkpoint_payload(base_model, checkpoint)
    payload["text_seg_state_dict"] = {"weight": torch.zeros(1)}
    torch.save(payload, checkpoint)

    reference = synthetic_detector.CTDSyntheticFineTuneReference(
        base_model,
        checkpoint,
        device="cpu",
        detect_size=128,
    )

    assert reference.checkpoint["train_dataset_sha256"] == payload[
        "train_dataset_sha256"
    ]
    assert reference.checkpoint["runtime_versions"]["freetype"] == "2.13.3"


def test_synthetic_detector_validates_max_batches_before_loading_assets(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="max_batches must be positive"):
        synthetic_detector.CTDSyntheticFineTuneReference(
            tmp_path / "missing-base.pt",
            tmp_path / "missing-checkpoint.pt",
            max_batches=0,
        )


def test_training_evaluation_uses_native_ctd_raw_mask_threshold() -> None:
    assert CTD_RAW_PROBABILITY_THRESHOLD == pytest.approx(1.0 / 255.0)


def test_synthetic_v4_manifest_seals_every_artifact_and_inventory(
    tmp_path: Path,
) -> None:
    output = tmp_path / "synthetic-v4"

    assert build_generalization_synthetic_main(
        ["--output-dir", str(output)]
    ) == 0
    manifest = output / "synthetic-inpaint-generalization-v4.json"
    binding = validate_source_only_manifest_v4(manifest)

    assert binding["page_count"] == 18
    assert len(binding["page_ids"]) == 18
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["page_count"] == 18
    assert payload["page_inventory_sha256"] == binding["page_inventory_sha256"]
    assert all(page["source_sha256"] for page in payload["pages"])
    assert all(page["artifact_sha256"] for page in payload["pages"])


def test_synthetic_v4_manifest_rejects_source_byte_drift(tmp_path: Path) -> None:
    output = tmp_path / "synthetic-v4"
    build_generalization_synthetic_main(["--output-dir", str(output)])
    manifest = output / "synthetic-inpaint-generalization-v4.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    source = Path(payload["pages"][0]["path"])
    source.write_bytes(source.read_bytes() + b"drift")

    with pytest.raises(ValueError, match="artifact SHA inventory differs"):
        validate_source_only_manifest_v4(manifest)


def test_synthetic_v4_builder_requires_fresh_output(tmp_path: Path) -> None:
    output = tmp_path / "already-exists"
    output.mkdir()

    with pytest.raises(FileExistsError, match="fresh and absent"):
        build_generalization_synthetic_main(["--output-dir", str(output)])


def _rewrite_synthetic_manifest_with_relative_paths(manifest: Path) -> None:
    payload = json.loads(manifest.read_text(encoding="utf-8"))

    def relative(value: object) -> object:
        if not isinstance(value, str) or not value:
            return value
        artifact = Path(value)
        return artifact.relative_to(manifest.parent).as_posix()

    for page in payload["pages"]:
        for field in (
            "path",
            "target_text_mask",
            "preserve_mask",
            "protected_structure_mask",
            "ambiguous_structure_mask",
            "ownership_mask",
            "bubble_interior_mask",
            "corner_protect_mask",
            "claim_seed_mask",
            "existing_source_edit_mask",
            "baseline",
            "baseline_mask",
            "known_background",
        ):
            if page.get(field):
                page[field] = relative(page[field])
        for instance in page["target_instances"]:
            instance["mask_path"] = relative(instance["mask_path"])
        for region in page["regions"]:
            for field in (
                "bubble_interior_mask",
                "ownership_mask",
                "protected_structure_mask",
                "ambiguous_structure_mask",
                "corner_protect_mask",
            ):
                region[field] = relative(region[field])
        page["artifact_sha256"] = manifest_page_artifact_sha256(manifest, page)
        page["source_sha256"] = page["artifact_sha256"]["path"]
    manifest.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    seal_path = manifest.with_suffix(manifest.suffix + ".seal.json")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    seal["manifest_sha256"] = _sha256(manifest)
    seal_path.write_text(
        json.dumps(seal, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _refresh_source_manifest_seal(manifest: Path, payload: dict[str, object]) -> None:
    pages = payload["pages"]
    assert isinstance(pages, list)
    for page in pages:
        assert isinstance(page, dict)
        page["artifact_sha256"] = manifest_page_artifact_sha256(manifest, page)
        page["source_sha256"] = page["artifact_sha256"]["path"]
    payload["page_inventory_sha256"] = source_manifest_page_inventory_sha256(
        pages
    )
    manifest.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    seal_path = manifest.with_suffix(manifest.suffix + ".seal.json")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    seal["manifest_sha256"] = _sha256(manifest)
    seal_path.write_text(
        json.dumps(seal, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def test_source_only_loader_consumes_manifest_relative_hashed_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "synthetic-v4"
    build_generalization_synthetic_main(["--output-dir", str(output)])
    manifest = output / "synthetic-inpaint-generalization-v4.json"
    _rewrite_synthetic_manifest_with_relative_paths(manifest)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    binding = validate_source_only_manifest_v4(manifest)
    pages = load_stage1_manifest(manifest)

    assert binding["page_count"] == 18
    assert all(Path(page.source_image).is_absolute() for page in pages)
    assert all(Path(page.source_image).is_file() for page in pages)
    assert all(
        Path(instance.mask_path).is_absolute()
        for page in pages
        for instance in page.target_instances
    )


@pytest.mark.parametrize("level", ("root", "page"))
def test_source_only_manifest_requires_candidate_seen_false_exactly(
    tmp_path: Path,
    level: str,
) -> None:
    output = tmp_path / "synthetic-v4"
    build_generalization_synthetic_main(["--output-dir", str(output)])
    manifest = output / "synthetic-inpaint-generalization-v4.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if level == "root":
        payload.pop("candidate_seen")
    else:
        payload["pages"][0].pop("candidate_seen")
    _refresh_source_manifest_seal(manifest, payload)

    with pytest.raises(ValueError, match="candidate"):
        validate_source_only_manifest_v4(manifest)


def test_source_only_manifest_requires_page_frozen_flag_exactly(
    tmp_path: Path,
) -> None:
    output = tmp_path / "synthetic-v4"
    build_generalization_synthetic_main(["--output-dir", str(output)])
    manifest = output / "synthetic-inpaint-generalization-v4.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["pages"][0].pop("annotation_frozen_before_candidate")
    _refresh_source_manifest_seal(manifest, payload)

    with pytest.raises(ValueError, match="was not frozen"):
        validate_source_only_manifest_v4(manifest)


def test_source_only_manifest_authenticates_seal_before_artifact_reads(
    tmp_path: Path,
) -> None:
    output = tmp_path / "synthetic-v4"
    build_generalization_synthetic_main(["--output-dir", str(output)])
    manifest = output / "synthetic-inpaint-generalization-v4.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["pages"][0]["path"] = str(tmp_path / "must-not-be-opened.png")
    manifest.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="seal SHA differs"):
        validate_source_only_manifest_v4(manifest)


def test_source_only_manifest_rejects_non_binary_annotation_mask(
    tmp_path: Path,
) -> None:
    output = tmp_path / "synthetic-v4"
    build_generalization_synthetic_main(["--output-dir", str(output)])
    manifest = output / "synthetic-inpaint-generalization-v4.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    page = payload["pages"][0]
    mask = np.zeros((int(page["height"]), int(page["width"])), np.uint8)
    mask[0, 0] = 127
    corrupt = output / "non-binary-claim-seed.png"
    assert cv2.imwrite(str(corrupt), mask)
    page["claim_seed_mask"] = str(corrupt)
    _refresh_source_manifest_seal(manifest, payload)

    with pytest.raises(ValueError, match="not binary"):
        validate_source_only_manifest_v4(manifest)


def test_source_only_manifest_requires_page_protection_even_with_region_mask(
    tmp_path: Path,
) -> None:
    output = tmp_path / "synthetic-v4"
    build_generalization_synthetic_main(["--output-dir", str(output)])
    manifest = output / "synthetic-inpaint-generalization-v4.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    page = payload["pages"][0]
    region_protect = np.zeros(
        (int(page["height"]), int(page["width"])), np.uint8
    )
    region_protect[-1, -1] = 255
    region_protect_path = output / "region-only-protect.png"
    assert cv2.imwrite(str(region_protect_path), region_protect)
    page["regions"][0]["protected_structure_mask"] = str(region_protect_path)
    page.pop("protected_structure_mask")
    _refresh_source_manifest_seal(manifest, payload)

    with pytest.raises(ValueError, match="missing fields: protected_structure_mask"):
        validate_source_only_manifest_v4(manifest)


def test_source_only_manifest_rejects_invalid_region_route_class(
    tmp_path: Path,
) -> None:
    output = tmp_path / "synthetic-v4"
    build_generalization_synthetic_main(["--output-dir", str(output)])
    manifest = output / "synthetic-inpaint-generalization-v4.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["pages"][0]["regions"][0]["bubble_route_class"] = "candidate_clean"
    _refresh_source_manifest_seal(manifest, payload)

    with pytest.raises(ValueError, match="invalid region bubble_route_class"):
        validate_source_only_manifest_v4(manifest)


def test_manifest_artifact_inventory_rejects_paired_source_sha_mismatch(
    tmp_path: Path,
) -> None:
    output = tmp_path / "synthetic-v4"
    build_generalization_synthetic_main(["--output-dir", str(output)])
    manifest = output / "synthetic-inpaint-generalization-v4.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    page = payload["pages"][0]
    reference = Path(page["path"])
    page["paired_reference"] = {
        "path": str(reference),
        "source_sha256": "0" * 64,
        "reference_sha256": _sha256(reference),
        "proposal_only": True,
    }

    with pytest.raises(ValueError, match="paired reference source SHA differs"):
        manifest_page_artifact_sha256(manifest, page)


def test_source_manifest_inventory_binds_source_and_artifact_identities() -> None:
    first = {
        "page_id": "same-id",
        "source_sha256": "1" * 64,
        "artifact_sha256": {"path": "1" * 64},
    }
    source_changed = {
        **first,
        "source_sha256": "2" * 64,
        "artifact_sha256": {"path": "2" * 64},
    }
    artifact_changed = {
        **first,
        "artifact_sha256": {"path": "1" * 64, "target": "3" * 64},
    }

    identity = source_manifest_page_inventory_sha256([first])
    assert identity != source_manifest_page_inventory_sha256([source_changed])
    assert identity != source_manifest_page_inventory_sha256([artifact_changed])


def test_source_only_manifest_rejects_mismatched_mask_shape(tmp_path: Path) -> None:
    output = tmp_path / "synthetic-v4"
    build_generalization_synthetic_main(["--output-dir", str(output)])
    manifest = output / "synthetic-inpaint-generalization-v4.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    target = output / "wrong-shape-target.png"
    assert cv2.imwrite(str(target), np.zeros((8, 8), np.uint8))
    payload["pages"][0]["target_text_mask"] = str(target)
    _refresh_source_manifest_seal(manifest, payload)

    with pytest.raises(ValueError, match="shape mismatch"):
        validate_source_only_manifest_v4(manifest)


def test_source_only_manifest_rejects_target_structure_overlap(
    tmp_path: Path,
) -> None:
    output = tmp_path / "synthetic-v4"
    build_generalization_synthetic_main(["--output-dir", str(output)])
    manifest = output / "synthetic-inpaint-generalization-v4.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    page = payload["pages"][0]
    page["protected_structure_mask"] = page["target_text_mask"]
    _refresh_source_manifest_seal(manifest, payload)

    with pytest.raises(ValueError, match="evaluation masks overlap"):
        validate_source_only_manifest_v4(manifest)


def test_source_only_manifest_rejects_target_outside_page_ownership(
    tmp_path: Path,
) -> None:
    output = tmp_path / "synthetic-v4"
    build_generalization_synthetic_main(["--output-dir", str(output)])
    manifest = output / "synthetic-inpaint-generalization-v4.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    page = payload["pages"][0]
    empty_ownership = output / "page-only-empty-ownership.png"
    assert cv2.imwrite(
        str(empty_ownership),
        np.zeros((int(page["height"]), int(page["width"])), np.uint8),
    )
    page["ownership_mask"] = str(empty_ownership)
    _refresh_source_manifest_seal(manifest, payload)

    with pytest.raises(ValueError, match="outside page ownership"):
        validate_source_only_manifest_v4(manifest)


def test_synthetic_v4_manifest_binds_page_bubble_interior_bytes(
    tmp_path: Path,
) -> None:
    output = tmp_path / "synthetic-v4"
    build_generalization_synthetic_main(["--output-dir", str(output)])
    manifest = output / "synthetic-inpaint-generalization-v4.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    interior = Path(payload["pages"][0]["bubble_interior_mask"])
    interior.write_bytes(interior.read_bytes() + b"drift")

    with pytest.raises(ValueError, match="artifact SHA inventory differs"):
        validate_source_only_manifest_v4(manifest)


def test_metadata_only_reseal_preserves_frozen_annotation_fields(
    tmp_path: Path,
) -> None:
    original = tmp_path / "synthetic-v4"
    build_generalization_synthetic_main(["--output-dir", str(original)])
    source = original / "synthetic-inpaint-generalization-v4.json"
    before = json.loads(source.read_text(encoding="utf-8"))
    before.pop("page_count")
    before.pop("page_ids")
    before.pop("page_inventory_sha256")
    source.write_text(
        json.dumps(before, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    source_seal_path = source.with_suffix(source.suffix + ".seal.json")
    source_seal = json.loads(source_seal_path.read_text(encoding="utf-8"))
    source_seal["manifest_sha256"] = _sha256(source)
    source_seal_path.write_text(
        json.dumps(source_seal, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    output = reseal_manifest(source, tmp_path / "resealed")
    after = json.loads(output.read_text(encoding="utf-8"))

    assert validate_source_only_manifest_v4(output)["page_count"] == 18
    for old, new in zip(before["pages"], after["pages"], strict=True):
        assert {
            key: value
            for key, value in old.items()
            if key not in {"artifact_sha256", "source_sha256"}
        } == {
            key: value
            for key, value in new.items()
            if key not in {"artifact_sha256", "source_sha256"}
        }


def test_metadata_only_reseal_rejects_artifact_byte_drift(
    tmp_path: Path,
) -> None:
    original = tmp_path / "synthetic-v4"
    build_generalization_synthetic_main(["--output-dir", str(original)])
    source = original / "synthetic-inpaint-generalization-v4.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    artifact = Path(payload["pages"][0]["target_text_mask"])
    artifact.write_bytes(artifact.read_bytes() + b"drift")

    with pytest.raises(ValueError, match="artifact bytes differ from the original"):
        reseal_manifest(source, tmp_path / "must-not-reseal-drift")


def test_metadata_only_reseal_relocates_relative_artifact_paths(
    tmp_path: Path,
) -> None:
    original = tmp_path / "synthetic-relative"
    build_generalization_synthetic_main(["--output-dir", str(original)])
    source = original / "synthetic-inpaint-generalization-v4.json"
    _rewrite_synthetic_manifest_with_relative_paths(source)

    output = reseal_manifest(source, tmp_path / "resealed-relative")
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert validate_source_only_manifest_v4(output)["page_count"] == 18
    assert all(Path(page["path"]).is_absolute() for page in payload["pages"])
    assert all(Path(page["path"]).is_file() for page in payload["pages"])


def test_metadata_only_reseal_rejects_post_candidate_source_seal(
    tmp_path: Path,
) -> None:
    original = tmp_path / "synthetic-v4"
    build_generalization_synthetic_main(["--output-dir", str(original)])
    source = original / "synthetic-inpaint-generalization-v4.json"
    seal_path = source.with_suffix(source.suffix + ".seal.json")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    seal["candidate_generated"] = True
    seal_path.write_text(json.dumps(seal, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="predate candidate generation"):
        reseal_manifest(source, tmp_path / "must-not-reseal")


def test_detector_evaluator_requires_fresh_explicit_output(tmp_path: Path) -> None:
    corpus = tmp_path / "synthetic-v4"
    build_generalization_synthetic_main(["--output-dir", str(corpus)])
    output = tmp_path / "existing-evaluation"
    output.mkdir()

    with pytest.raises(FileExistsError, match="fresh and absent"):
        benchmark_inpaint_detector_bakeoff.main(
            [
                "--manifest",
                str(corpus / "synthetic-inpaint-generalization-v4.json"),
                "--candidate",
                "ctd-synthetic-finetune",
                "--checkpoint",
                str(tmp_path / "missing.pt"),
                "--output-dir",
                str(output),
            ]
        )


def test_synthetic_training_requires_fresh_explicit_output(tmp_path: Path) -> None:
    output = tmp_path / "existing-training"
    output.mkdir()

    with pytest.raises(FileExistsError, match="fresh and absent"):
        train_synthetic_detector_main(
            [
                "--base-model",
                str(tmp_path / "missing.pt"),
                "--output-dir",
                str(output),
                "--train-samples",
                "1",
                "--dev-samples",
                "1",
                "--image-size",
                "32",
                "--epochs",
                "1",
            ]
        )


class _ParityModel:
    def __init__(self, torch) -> None:
        self.torch = torch

    def __call__(self, example):
        return (
            self.torch.zeros((1, 1, 1), dtype=self.torch.float32),
            self.torch.tensor([[[[0.0, 0.5]]]], dtype=self.torch.float32),
            self.torch.zeros((1, 2, 1, 2), dtype=self.torch.float32),
        )


def test_onnx_export_parity_requires_final_binary_xor_zero(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import onnxruntime as ort
    import torch

    class Session:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def run(self, _outputs, _inputs):
            return [
                np.zeros((1, 1, 1), np.float32),
                np.array([[[[0.0, 0.5]]]], np.float32),
                np.zeros((1, 2, 1, 2), np.float32),
            ]

    monkeypatch.setattr(ort, "InferenceSession", Session)

    result = export_inpaint_ctd_onnx.validate_export_parity(
        _ParityModel(torch),
        tmp_path / "model.onnx",
        torch.zeros((1, 3, 32, 32)),
    )

    assert result["segmentation_binary_xor_pixel_count"] == 0
    assert result["input_count"] == 3
    assert {row["input_id"] for row in result["inputs"]} == {
        "low_signal",
        "gradient",
        "boundary_checker",
    }
    assert all(len(row["input_sha256"]) == 64 for row in result["inputs"])


def test_onnx_export_parity_rejects_binary_mask_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import onnxruntime as ort
    import torch

    class Session:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def run(self, _outputs, _inputs):
            return [
                np.zeros((1, 1, 1), np.float32),
                np.zeros((1, 1, 1, 2), np.float32),
                np.zeros((1, 2, 1, 2), np.float32),
            ]

    monkeypatch.setattr(ort, "InferenceSession", Session)

    with pytest.raises(RuntimeError, match="XOR=1"):
        export_inpaint_ctd_onnx.validate_export_parity(
            _ParityModel(torch),
            tmp_path / "model.onnx",
            torch.zeros((1, 3, 32, 32)),
        )


def test_onnx_export_parity_rejects_nonfinite_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import onnxruntime as ort
    import torch

    class Session:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def run(self, _outputs, _inputs):
            return [
                np.zeros((1, 1, 1), np.float32),
                np.array([[[[np.nan, 0.5]]]], np.float32),
                np.zeros((1, 2, 1, 2), np.float32),
            ]

    monkeypatch.setattr(ort, "InferenceSession", Session)

    with pytest.raises(RuntimeError, match="NaN or Inf"):
        export_inpaint_ctd_onnx.validate_export_parity(
            _ParityModel(torch),
            tmp_path / "model.onnx",
            torch.zeros((1, 3, 32, 32)),
        )


def test_onnx_export_applies_finetuned_text_seg_state_strictly() -> None:
    observed: dict[str, object] = {}

    class TextSeg:
        def load_state_dict(self, state, *, strict):
            observed["state"] = state
            observed["strict"] = strict

    class Model:
        text_seg = TextSeg()

        def eval(self):
            observed["eval"] = True
            return self

    state = {"weight": object()}
    export_inpaint_ctd_onnx.apply_synthetic_checkpoint(
        Model(),
        {"text_seg_state_dict": state},
    )

    assert observed == {"state": state, "strict": True, "eval": True}


@pytest.mark.parametrize(
    "existing_name",
    (
        "model.onnx",
        "model.onnx.json",
        ".model.onnx.partial",
        ".model.onnx.json.partial",
    ),
)
def test_onnx_export_requires_fresh_output(
    tmp_path: Path,
    existing_name: str,
) -> None:
    source = tmp_path / "source.pt"
    source.write_bytes(b"source")
    output = tmp_path / "model.onnx"
    (tmp_path / existing_name).write_bytes(b"existing")

    with pytest.raises(FileExistsError, match="must be fresh"):
        export_inpaint_ctd_onnx.main(
            ["--source-model", str(source), "--output", str(output)]
        )
