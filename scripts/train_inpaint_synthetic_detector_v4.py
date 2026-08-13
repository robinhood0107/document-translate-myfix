#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

import cv2
import numpy as np


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarking.inpaint_detector_bakeoff.synthetic_detector import (  # noqa: E402
    ARCHITECTURE_HEAD_ONLY,
    ARCHITECTURE_LAST_BACKBONE_STAGE,
    CHECKPOINT_SELECTION_CONTRACT,
    CHECKPOINT_SCHEMA,
    LAST_BACKBONE_STAGE_INDICES,
    TRAINING_CONTRACT,
    apply_synthetic_checkpoint_weights,
    font_asset_provenance,
    source_dependency_provenance,
    validate_training_hyperparameters,
)
from benchmarking.inpaint_detector_bakeoff.synthetic_training import (  # noqa: E402
    synthetic_training_digest,
    synthetic_training_sample,
)
from modules.masking.ctd_vendor.ctd import TextDetBase  # noqa: E402
from modules.masking.ctd_vendor.ctd.basemodel import (  # noqa: E402
    TEXTDET_MASK,
)
from scripts.validation_artifact_harness import (  # noqa: E402
    select_managed_output_directory,
)


FAMILY = "inpaint-ctd-synthetic-finetune-training-v4"
CATEGORY = "40-inpaint-mask-render"
CTD_RAW_PROBABILITY_THRESHOLD = 1.0 / 255.0
EVALUATION_BATCH_SIZE = 8
ADAM_BETAS = (0.9, 0.999)
ADAM_EPSILON = 1e-8
ADAM_WEIGHT_DECAY = 0.01
GENERATOR_PATH = ROOT / "benchmarking" / "inpaint_detector_bakeoff" / "synthetic_training.py"
DETECTOR_PATH = ROOT / "benchmarking" / "inpaint_detector_bakeoff" / "synthetic_detector.py"
TRAINER_PATH = Path(__file__).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _training_hyperparameters(
    args: argparse.Namespace,
) -> dict[str, int | float | str | bool]:
    return validate_training_hyperparameters(
        {
            "seed": args.seed,
            "train_samples": args.train_samples,
            "dev_samples": args.dev_samples,
            "image_size": args.image_size,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "anchor_weight": args.anchor_weight,
            "distillation_weight": args.distillation_weight,
            "optimizer": "AdamW",
            "adam_beta1": ADAM_BETAS[0],
            "adam_beta2": ADAM_BETAS[1],
            "adam_epsilon": ADAM_EPSILON,
            "weight_decay": ADAM_WEIGHT_DECAY,
            "adam_amsgrad": False,
            "raw_probability_threshold": CTD_RAW_PROBABILITY_THRESHOLD,
            "evaluation_batch_size": args.evaluation_batch_size,
            "architecture_mode": args.architecture_mode,
            "backbone_learning_rate_scale": args.backbone_learning_rate_scale,
        }
    )


def _runtime_versions(torch, device: str) -> dict[str, object]:
    import PIL
    from PIL import features

    runtime: dict[str, object] = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "pillow": str(PIL.__version__),
        "freetype": str(features.version_module("freetype2") or "unavailable"),
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
        "opencv": str(cv2.__version__),
        "device": device,
        "cuda": str(torch.version.cuda or ""),
        "cudnn": str(torch.backends.cudnn.version() or ""),
        "peak_memory_allocated_bytes": 0,
        "peak_memory_reserved_bytes": 0,
    }
    if device.startswith("cuda"):
        runtime["cuda_device_name"] = str(torch.cuda.get_device_name(device))
        runtime["cuda_device_capability"] = list(
            torch.cuda.get_device_capability(device)
        )
    return runtime


def _runtime_with_peak_memory(
    torch,
    runtime: dict[str, object],
    device: str,
) -> dict[str, object]:
    current = dict(runtime)
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
        current["peak_memory_allocated_bytes"] = int(
            torch.cuda.max_memory_allocated(device)
        )
        current["peak_memory_reserved_bytes"] = int(
            torch.cuda.max_memory_reserved(device)
        )
    return current


def _tensor_batch(
    torch,
    seeds: list[int],
    shape: tuple[int, int],
    device: str,
    font_paths: tuple[Path, ...],
):
    images = []
    targets = []
    for seed in seeds:
        sample = synthetic_training_sample(seed, shape=shape, font_paths=font_paths)
        images.append(
            torch.from_numpy(sample.image.transpose(2, 0, 1).copy()).float() / 255.0
        )
        targets.append(
            torch.from_numpy((sample.target > 0).astype(np.float32)[None, :, :])
        )
    return torch.stack(images).to(device), torch.stack(targets).to(device)


def _segmentation_probability(torch, model: TextDetBase, image):
    with torch.no_grad():
        _blocks, features = model.blk_det(image, detect=True)
    return model.text_seg(*features, forward_mode=TEXTDET_MASK)


def _dice_loss(torch, probability, target):
    numerator = 2.0 * torch.sum(probability * target, dim=(1, 2, 3)) + 1.0
    denominator = (
        torch.sum(probability, dim=(1, 2, 3))
        + torch.sum(target, dim=(1, 2, 3))
        + 1.0
    )
    return 1.0 - torch.mean(numerator / denominator)


def _evaluate(
    torch,
    model,
    seeds: list[int],
    shape: tuple[int, int],
    device: str,
    font_paths: tuple[Path, ...],
    evaluation_batch_size: int = EVALUATION_BATCH_SIZE,
):
    intersections = target_pixels = predicted_pixels = 0
    seeded_instances = required_instances = 0
    all_page_false_pixels = 0
    text_page_false_pixels = 0
    no_text_false_pixels = 0
    by_background: dict[str, list[int]] = {}
    by_style: dict[str, list[int]] = {}
    _set_evaluation_state(model)
    with torch.no_grad():
        for offset in range(0, len(seeds), evaluation_batch_size):
            batch_seeds = seeds[offset : offset + evaluation_batch_size]
            samples = [
                synthetic_training_sample(seed, shape=shape, font_paths=font_paths)
                for seed in batch_seeds
            ]
            image = torch.stack(
                [
                    torch.from_numpy(sample.image.transpose(2, 0, 1).copy())
                    .float()
                    .div(255.0)
                    for sample in samples
                ]
            ).to(device)
            target = torch.stack(
                [
                    torch.from_numpy(
                        (sample.target > 0).astype(np.float32)[None, :, :]
                    )
                    for sample in samples
                ]
            ).to(device)
            predicted = (
                _segmentation_probability(torch, model, image)
                >= CTD_RAW_PROBABILITY_THRESHOLD
            )
            truth = target > 0.5
            intersections += int(torch.count_nonzero(predicted & truth).item())
            target_pixels += int(torch.count_nonzero(truth).item())
            predicted_pixels += int(torch.count_nonzero(predicted).item())
            for index, sample in enumerate(samples):
                predicted_local = predicted[index, 0]
                truth_local = truth[index, 0]
                overlap = int(torch.count_nonzero(predicted_local & truth_local).item())
                truth_count = int(torch.count_nonzero(truth_local).item())
                false_count = int(
                    torch.count_nonzero(predicted_local & ~truth_local).item()
                )
                all_page_false_pixels += false_count
                if sample.has_text:
                    required_instances += 1
                    seeded_instances += int(overlap > 0)
                    text_page_false_pixels += false_count
                    by_background.setdefault(sample.background_kind, [0, 0])
                    by_style.setdefault(sample.text_style, [0, 0])
                    by_background[sample.background_kind][0] += overlap
                    by_background[sample.background_kind][1] += truth_count
                    by_style[sample.text_style][0] += overlap
                    by_style[sample.text_style][1] += truth_count
                else:
                    no_text_false_pixels += false_count
    precision = intersections / predicted_pixels if predicted_pixels else 0.0
    recall = intersections / target_pixels if target_pixels else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "instance_seed_recall": (
            seeded_instances / required_instances if required_instances else 0.0
        ),
        "required_instance_count": required_instances,
        "missed_instance_count": required_instances - seeded_instances,
        "all_page_false_pixel_count": all_page_false_pixels,
        "text_page_false_pixel_count": text_page_false_pixels,
        "no_text_false_pixel_count": no_text_false_pixels,
        "background_recall": {
            key: overlap / total if total else 0.0
            for key, (overlap, total) in sorted(by_background.items())
        },
        "style_recall": {
            key: overlap / total if total else 0.0
            for key, (overlap, total) in sorted(by_style.items())
        },
    }


def _selection_key(
    metrics: dict[str, object],
) -> tuple[float, int, int, int, float, float, float]:
    """Recommend one checkpoint without discarding the Pareto epoch set.

    The detector track exists to recover completely missed instances. A higher
    pixel F1 must not win by sacrificing instance recall. Equal-recall epochs
    first minimize false claims over *all* pages, then independently compare
    false claims on text-bearing and no-text pages. Every nondominated epoch is
    retained for synthetic/E1 evaluation; this key only chooses the convenient
    default.
    """

    return (
        float(metrics["instance_seed_recall"]),
        -int(metrics["all_page_false_pixel_count"]),
        -int(metrics["text_page_false_pixel_count"]),
        -int(metrics["no_text_false_pixel_count"]),
        float(metrics["recall"]),
        float(metrics["precision"]),
        float(metrics["f1"]),
    )


def _metrics_dominate(
    left: dict[str, object],
    right: dict[str, object],
) -> bool:
    comparisons = (
        float(left["instance_seed_recall"])
        >= float(right["instance_seed_recall"]),
        int(left["all_page_false_pixel_count"])
        <= int(right["all_page_false_pixel_count"]),
        int(left["text_page_false_pixel_count"])
        <= int(right["text_page_false_pixel_count"]),
        int(left["no_text_false_pixel_count"])
        <= int(right["no_text_false_pixel_count"]),
        float(left["recall"]) >= float(right["recall"]),
        float(left["precision"]) >= float(right["precision"]),
    )
    strict = (
        float(left["instance_seed_recall"])
        > float(right["instance_seed_recall"])
        or int(left["all_page_false_pixel_count"])
        < int(right["all_page_false_pixel_count"])
        or int(left["text_page_false_pixel_count"])
        < int(right["text_page_false_pixel_count"])
        or int(left["no_text_false_pixel_count"])
        < int(right["no_text_false_pixel_count"])
        or float(left["recall"]) > float(right["recall"])
        or float(left["precision"]) > float(right["precision"])
    )
    return all(comparisons) and strict


def _pareto_epochs(history: list[dict[str, object]]) -> tuple[int, ...]:
    return tuple(
        int(row["epoch"])
        for row in history
        if not any(
            other is not row and _metrics_dominate(other, row)
            for other in history
        )
    )


def _selection_metric_summary(
    selected_dev_metrics: dict[str, object],
    history: list[dict[str, object]],
) -> dict[str, object]:
    max_f1_row = max(history, key=lambda row: float(row["f1"]))
    return {
        "selected_dev_f1": float(selected_dev_metrics["f1"]),
        "selected_dev_metrics": dict(selected_dev_metrics),
        "max_dev_f1": float(max_f1_row["f1"]),
        "max_dev_f1_epoch": int(max_f1_row["epoch"]),
        "max_dev_f1_metrics": {
            key: value
            for key, value in max_f1_row.items()
            if key not in {"epoch", "loss"}
        },
    }


def _backbone_stage_state_dict(model) -> dict[str, object]:
    return {
        str(index): model.blk_det.model[index].state_dict()
        for index in LAST_BACKBONE_STAGE_INDICES
    }


def _configure_trainable_parameters(model, architecture_mode: str):
    for parameter in model.blk_det.parameters():
        parameter.requires_grad_(False)
    for parameter in model.text_det.parameters():
        parameter.requires_grad_(False)
    for parameter in model.text_seg.parameters():
        parameter.requires_grad_(True)
    backbone_parameters = []
    if architecture_mode == ARCHITECTURE_LAST_BACKBONE_STAGE:
        for index in LAST_BACKBONE_STAGE_INDICES:
            module = model.blk_det.model[index]
            for parameter in module.parameters():
                parameter.requires_grad_(True)
                backbone_parameters.append(parameter)
    elif architecture_mode != ARCHITECTURE_HEAD_ONLY:
        raise ValueError(f"unsupported architecture mode: {architecture_mode}")
    return list(model.text_seg.parameters()), backbone_parameters


def _set_training_state(model, architecture_mode: str) -> None:
    model.blk_det.eval()
    model.text_det.eval()
    model.text_seg.train()
    if architecture_mode == ARCHITECTURE_LAST_BACKBONE_STAGE:
        for index in LAST_BACKBONE_STAGE_INDICES:
            model.blk_det.model[index].train()


def _set_evaluation_state(model) -> None:
    model.blk_det.eval()
    model.text_det.eval()
    model.text_seg.eval()


def train(args: argparse.Namespace, output: Path) -> dict[str, object]:
    import torch

    hyperparameters = _training_hyperparameters(args)
    seed = int(hyperparameters["seed"])
    device = str(args.device).strip()
    if not device:
        raise ValueError("device must not be empty")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA training requested but CUDA is unavailable")
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    base_model_path = args.base_model.resolve()
    if not base_model_path.is_file():
        raise FileNotFoundError(base_model_path)
    shape = (
        int(hyperparameters["image_size"]),
        int(hyperparameters["image_size"]),
    )
    font_paths = tuple(Path(value).resolve() for value in args.font)
    font_assets = font_asset_provenance(font_paths)
    train_seeds = list(
        range(seed, seed + int(hyperparameters["train_samples"]))
    )
    dev_seeds = list(
        range(
            seed + int(hyperparameters["train_samples"]),
            seed
            + int(hyperparameters["train_samples"])
            + int(hyperparameters["dev_samples"]),
        )
    )
    train_dataset_sha256 = synthetic_training_digest(
        train_seeds,
        shape=shape,
        font_paths=font_paths,
    )
    dev_dataset_sha256 = synthetic_training_digest(
        dev_seeds,
        shape=shape,
        font_paths=font_paths,
    )
    runtime_versions = _runtime_versions(torch, device)
    code_commit = _code_commit()
    base_model_sha256 = _sha256(base_model_path)
    generator_sha256 = _sha256(GENERATOR_PATH)
    detector_sha256 = _sha256(DETECTOR_PATH)
    trainer_sha256 = _sha256(TRAINER_PATH)
    source_dependency_sha256 = source_dependency_provenance()

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    model = TextDetBase(
        str(base_model_path),
        device=device,
        half=False,
        act="leaky",
    )
    architecture_mode = str(hyperparameters["architecture_mode"])
    head_parameters, backbone_parameters = _configure_trainable_parameters(
        model, architecture_mode
    )
    _set_training_state(model, architecture_mode)
    teacher_seg = copy.deepcopy(model.text_seg).eval()
    for parameter in teacher_seg.parameters():
        parameter.requires_grad_(False)
    teacher_backbone = None
    if architecture_mode == ARCHITECTURE_LAST_BACKBONE_STAGE:
        teacher_backbone = copy.deepcopy(model.blk_det).eval()
        for parameter in teacher_backbone.parameters():
            parameter.requires_grad_(False)
    anchor = {
        f"text_seg.{name}": value.detach().clone()
        for name, value in model.text_seg.named_parameters()
    }
    if architecture_mode == ARCHITECTURE_LAST_BACKBONE_STAGE:
        anchor.update(
            {
                f"blk_det.{index}.{name}": value.detach().clone()
                for index in LAST_BACKBONE_STAGE_INDICES
                for name, value in model.blk_det.model[index].named_parameters()
            }
        )
    parameter_groups: list[dict[str, object]] = [{"params": head_parameters}]
    if backbone_parameters:
        parameter_groups.append(
            {
                "params": backbone_parameters,
                "lr": float(hyperparameters["learning_rate"])
                * float(hyperparameters["backbone_learning_rate_scale"]),
            }
        )
    optimizer = torch.optim.AdamW(
        parameter_groups,
        lr=float(hyperparameters["learning_rate"]),
        betas=(
            float(hyperparameters["adam_beta1"]),
            float(hyperparameters["adam_beta2"]),
        ),
        eps=float(hyperparameters["adam_epsilon"]),
        weight_decay=float(hyperparameters["weight_decay"]),
        amsgrad=bool(hyperparameters["adam_amsgrad"]),
    )
    bce = torch.nn.BCELoss()
    started = time.perf_counter()
    history = []
    best_selection_key: (
        tuple[float, int, int, int, float, float, float] | None
    ) = None
    selected_checkpoint: Path | None = None
    checkpoints: list[dict[str, object]] = []
    baseline_dev_metrics = _evaluate(
        torch,
        model,
        dev_seeds,
        shape,
        device,
        font_paths,
        int(hyperparameters["evaluation_batch_size"]),
    )
    for epoch in range(1, int(hyperparameters["epochs"]) + 1):
        _set_training_state(model, architecture_mode)
        random.Random(seed + epoch).shuffle(train_seeds)
        losses = []
        batch_size = int(hyperparameters["batch_size"])
        for offset in range(0, len(train_seeds), batch_size):
            image, target = _tensor_batch(
                torch,
                train_seeds[offset : offset + batch_size],
                shape,
                device,
                font_paths,
            )
            optimizer.zero_grad(set_to_none=True)
            if teacher_backbone is None:
                with torch.no_grad():
                    _blocks, features = model.blk_det(image, detect=True)
                    teacher_features = features
            else:
                _blocks, features = model.blk_det(image, detect=True)
                with torch.no_grad():
                    _teacher_blocks, teacher_features = teacher_backbone(
                        image, detect=True
                    )
            with torch.no_grad():
                teacher_probability = teacher_seg(
                    *teacher_features, forward_mode=TEXTDET_MASK
                )
            probability = model.text_seg(*features, forward_mode=TEXTDET_MASK)
            data_loss = bce(probability.clamp(1e-5, 1.0 - 1e-5), target)
            data_loss = data_loss + _dice_loss(torch, probability, target)
            outside = (target <= 0.5).float()
            distillation_loss = torch.sum(
                ((probability - teacher_probability) ** 2) * outside
            ) / torch.clamp(torch.sum(outside), min=1.0)
            anchor_loss = sum(
                torch.mean((value - anchor[f"text_seg.{name}"]) ** 2)
                for name, value in model.text_seg.named_parameters()
            )
            if architecture_mode == ARCHITECTURE_LAST_BACKBONE_STAGE:
                anchor_loss = anchor_loss + sum(
                    torch.mean(
                        (value - anchor[f"blk_det.{index}.{name}"]) ** 2
                    )
                    for index in LAST_BACKBONE_STAGE_INDICES
                    for name, value in model.blk_det.model[
                        index
                    ].named_parameters()
                )
            loss = (
                data_loss
                + float(hyperparameters["distillation_weight"])
                * distillation_loss
                + float(hyperparameters["anchor_weight"]) * anchor_loss
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        metrics = _evaluate(
            torch,
            model,
            dev_seeds,
            shape,
            device,
            font_paths,
            int(hyperparameters["evaluation_batch_size"]),
        )
        row = {"epoch": epoch, "loss": float(np.mean(losses)), **metrics}
        history.append(row)
        selection_key = _selection_key(metrics)
        checkpoint = output / f"ctd-synthetic-finetune-epoch-{epoch:03d}.pt"
        checkpoint_payload = {
            "schema_version": CHECKPOINT_SCHEMA,
            "text_seg_state_dict": model.text_seg.state_dict(),
            "architecture_mode": architecture_mode,
            "backbone_stage_indices": (
                list(LAST_BACKBONE_STAGE_INDICES)
                if architecture_mode == ARCHITECTURE_LAST_BACKBONE_STAGE
                else []
            ),
            "backbone_stage_state_dict": (
                _backbone_stage_state_dict(model)
                if architecture_mode == ARCHITECTURE_LAST_BACKBONE_STAGE
                else None
            ),
            "base_model_sha256": base_model_sha256,
            "image_size": int(hyperparameters["image_size"]),
            "epoch": epoch,
            "seed": seed,
            "train_seed_first": min(train_seeds),
            "train_seed_last": max(train_seeds),
            "dev_seed_first": dev_seeds[0],
            "dev_seed_last": dev_seeds[-1],
            "generator_sha256": generator_sha256,
            "detector_sha256": detector_sha256,
            "trainer_sha256": trainer_sha256,
            "source_dependency_sha256": source_dependency_sha256,
            "training_contract": TRAINING_CONTRACT,
            "checkpoint_selection_contract": CHECKPOINT_SELECTION_CONTRACT,
            "training_hyperparameters": hyperparameters,
            "train_dataset_sha256": train_dataset_sha256,
            "dev_dataset_sha256": dev_dataset_sha256,
            "runtime_versions": _runtime_with_peak_memory(
                torch, runtime_versions, device
            ),
            "code_commit": code_commit,
            "determinism_contract": {
                "torch_deterministic_algorithms": True,
                "cudnn_benchmark": False,
                "cudnn_deterministic": True,
                "cublas_workspace_config": os.environ.get(
                    "CUBLAS_WORKSPACE_CONFIG", ""
                ),
            },
            "font_assets": font_assets,
            "dev_metrics": metrics,
            "baseline_dev_metrics": baseline_dev_metrics,
        }
        torch.save(checkpoint_payload, checkpoint)
        checkpoints.append(
            {
                "epoch": epoch,
                "checkpoint": checkpoint.name,
                "checkpoint_sha256": _sha256(checkpoint),
                "dev_metrics": metrics,
            }
        )
        if best_selection_key is None or selection_key > best_selection_key:
            best_selection_key = selection_key
            selected_checkpoint = checkpoint
    if selected_checkpoint is None:
        raise RuntimeError("training produced no checkpoint")
    pareto_epochs = frozenset(_pareto_epochs(history))
    for row in checkpoints:
        row["pareto"] = int(row["epoch"]) in pareto_epochs
        row["recommended"] = Path(str(row["checkpoint"])) == Path(
            selected_checkpoint.name
        )
    saved = torch.load(selected_checkpoint, map_location=device, weights_only=True)
    apply_synthetic_checkpoint_weights(model, saved)
    model.text_seg.eval()
    selected_dev_metrics = dict(saved["dev_metrics"])
    selection_metric_summary = _selection_metric_summary(
        selected_dev_metrics,
        history,
    )
    previews = []
    with torch.no_grad():
        for preview_seed in dev_seeds[:8]:
            sample = synthetic_training_sample(
                preview_seed, shape=shape, font_paths=font_paths
            )
            tensor = (
                torch.from_numpy(sample.image.transpose(2, 0, 1).copy())
                .float()
                .div(255.0)
                .unsqueeze(0)
                .to(device)
            )
            predicted = (
                _segmentation_probability(torch, model, tensor)[0, 0]
                >= CTD_RAW_PROBABILITY_THRESHOLD
            ).cpu().numpy()
            overlay = sample.image.copy()
            overlay[predicted] = (0, 0, 255)
            previews.append(np.concatenate((sample.image, overlay), axis=1))
    preview_path = output / "synthetic-dev-preview.png"
    if previews and not cv2.imwrite(
        str(preview_path), np.concatenate(previews, axis=0)
    ):
        raise OSError(preview_path)
    runtime_versions = _runtime_with_peak_memory(torch, runtime_versions, device)
    return {
        "schema_version": "inpaint-ctd-synthetic-finetune-training-results-v4",
        "device": device,
        "seed": seed,
        "train_seed_first": min(train_seeds),
        "train_seed_last": max(train_seeds),
        "dev_seed_first": min(dev_seeds),
        "dev_seed_last": max(dev_seeds),
        "train_sample_count": len(train_seeds),
        "dev_sample_count": len(dev_seeds),
        "image_size": int(hyperparameters["image_size"]),
        "architecture_mode": architecture_mode,
        "epochs": int(hyperparameters["epochs"]),
        "batch_size": int(hyperparameters["batch_size"]),
        "learning_rate": float(hyperparameters["learning_rate"]),
        "anchor_weight": float(hyperparameters["anchor_weight"]),
        "distillation_weight": float(hyperparameters["distillation_weight"]),
        "training_hyperparameters": hyperparameters,
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint": selected_checkpoint.name,
        "selected_checkpoint": selected_checkpoint.name,
        "selected_checkpoint_epoch": int(saved["epoch"]),
        "checkpoint_sha256": _sha256(selected_checkpoint),
        "checkpoints": checkpoints,
        "pareto_epochs": sorted(pareto_epochs),
        "base_model_sha256": base_model_sha256,
        **selection_metric_summary,
        "checkpoint_selection_contract": CHECKPOINT_SELECTION_CONTRACT,
        "baseline_dev_metrics": baseline_dev_metrics,
        "history": history,
        "code_commit": code_commit,
        "generator_sha256": generator_sha256,
        "detector_sha256": detector_sha256,
        "trainer_sha256": trainer_sha256,
        "source_dependency_sha256": source_dependency_sha256,
        "train_dataset_sha256": train_dataset_sha256,
        "dev_dataset_sha256": dev_dataset_sha256,
        "runtime_versions": runtime_versions,
        "data_contract": TRAINING_CONTRACT,
        "determinism_contract": {
            "torch_deterministic_algorithms": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
        },
        "architecture_contract": (
            "existing_ctd_last_backbone_stage_and_text_seg_finetune"
            if architecture_mode == ARCHITECTURE_LAST_BACKBONE_STAGE
            else "existing_ctd_frozen_backbone_finetuned_text_seg"
        ),
        "evaluation_output_contract": {
            "raw": "native_finetuned_ctd_text_seg",
            "refined": "exact_identity_reuse_of_raw",
            "dilated": "elliptical_native3_from_raw",
        },
        "font_assets": font_assets,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune the existing CTD segmentation head on synthetic-only data."
    )
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=41371)
    parser.add_argument("--train-samples", type=int, default=2048)
    parser.add_argument("--dev-samples", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--evaluation-batch-size", type=int, default=EVALUATION_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--architecture-mode",
        choices=(ARCHITECTURE_HEAD_ONLY, ARCHITECTURE_LAST_BACKBONE_STAGE),
        default=ARCHITECTURE_HEAD_ONLY,
    )
    parser.add_argument("--backbone-learning-rate-scale", type=float, default=0.1)
    parser.add_argument("--anchor-weight", type=float, default=1e-3)
    parser.add_argument("--distillation-weight", type=float, default=1.0)
    parser.add_argument(
        "--font",
        action="append",
        default=[],
        help=(
            "Optional local CJK font asset for deterministic synthetic glyphs; "
            "repeatable and recorded by SHA, never copied into the repository."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _training_hyperparameters(args)
    if args.output_dir is not None and args.output_dir.resolve().exists():
        raise FileExistsError(
            "training output directory must be fresh and absent: "
            f"{args.output_dir.resolve()}"
        )
    output, managed = select_managed_output_directory(
        family=FAMILY,
        category=CATEGORY,
        explicit_output_directory=args.output_dir,
    )
    output.mkdir(parents=True, exist_ok=True)
    try:
        payload = train(args, output)
        result_path = output / "training-results.json"
        result_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if managed is not None:
            managed.complete(
                metadata={
                    "checkpoint_sha256": payload["checkpoint_sha256"],
                    "base_model_sha256": payload["base_model_sha256"],
                    "selected_dev_f1": payload["selected_dev_f1"],
                    "max_dev_f1": payload["max_dev_f1"],
                    "data_contract": payload["data_contract"],
                    "architecture_contract": payload["architecture_contract"],
                }
            )
            mismatches = managed.verify()
            if mismatches:
                raise RuntimeError(
                    "managed artifact verification failed: " + "; ".join(mismatches)
                )
            print(managed.run_root)
        else:
            print(output)
        return 0
    except BaseException as error:
        if managed is not None:
            managed.fail(error)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
