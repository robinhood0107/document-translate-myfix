from __future__ import annotations

from pathlib import Path
import struct
from types import SimpleNamespace

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
from benchmarking.inpaint_detector_bakeoff.synthetic_detector import (
    CHECKPOINT_SELECTION_CONTRACT,
    CHECKPOINT_SCHEMA,
    TRAINING_CONTRACT,
    _sha256,
    validate_checkpoint_provenance,
    validate_training_hyperparameters,
)
from benchmarking.inpaint_detector_bakeoff import synthetic_detector
from scripts import benchmark_inpaint_detector_bakeoff
from scripts.train_inpaint_synthetic_detector_v4 import (
    CTD_RAW_PROBABILITY_THRESHOLD,
    _pareto_epochs,
    _runtime_versions,
    _selection_key,
    _selection_metric_summary,
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
    return {
        "schema_version": CHECKPOINT_SCHEMA,
        "text_seg_state_dict": {"placeholder": b"test"},
        "base_model_sha256": _sha256(base_model),
        "generator_sha256": _sha256(generator),
        "detector_sha256": _sha256(Path(synthetic_detector.__file__)),
        "trainer_sha256": _sha256(trainer),
        "training_contract": TRAINING_CONTRACT,
        "checkpoint_selection_contract": CHECKPOINT_SELECTION_CONTRACT,
        "training_hyperparameters": _valid_training_hyperparameters(),
        "seed": 41371,
        "image_size": 320,
        "epoch": 1,
        "train_seed_first": 41371,
        "train_seed_last": 41498,
        "dev_seed_first": 41499,
        "dev_seed_last": 41530,
        "train_dataset_sha256": "1" * 64,
        "dev_dataset_sha256": "2" * 64,
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
        },
        "code_commit": "a" * 40,
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
        "train_samples": 128,
        "dev_samples": 32,
        "image_size": 320,
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


def test_training_runtime_records_python_pillow_and_freetype() -> None:
    import torch

    runtime = _runtime_versions(torch, "cpu")

    assert runtime["python"]
    assert runtime["python_implementation"]
    assert runtime["pillow"]
    assert runtime["freetype"]
    assert runtime["device"] == "cpu"


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
    )

    benchmark_inpaint_detector_bakeoff._candidate(args)

    assert captured["detect_size"] == 1536
    assert captured["max_batches"] == 7


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

    assert reference.checkpoint["train_dataset_sha256"] == "1" * 64
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
