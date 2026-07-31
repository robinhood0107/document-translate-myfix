#!/usr/bin/env python3
"""Isolated CUDA FP32 inference adapter for the official ZITS++ model.

This module is executed inside a disposable CUDA container by the benchmark
client.  It deliberately avoids the historical training dependencies
(``pytorch_lightning``, compiled NMS, and metric packages) and loads only the
official inference modules and checkpoint prefixes needed by model_512.

The process uses a JSON-lines protocol so the model is loaded exactly once for
all frozen benchmark cases.  All paths in requests are relative to the mounted
exchange directory.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
import types
from pathlib import Path
from typing import Any, Mapping, Sequence


SOURCE_COMMIT = "de8dd48b17aedd15824842adb7bcca7535daba84"
MODEL_SHA256 = "e30d2073ba63af42836ac611214ed984db7ec739a1eef019451df6a34f566f57"
LSM_SHA256 = "6f72a60ec895f11830763069a40cb548dfb0ba77aca5282ffeea8afc72dc1723"
CONFIG_SHA256 = "e46dd48b4715f0044debfa0faca1c5af0c149b1cb1dffa29afb042687f13e4f2"
LICENSE_SHA256 = "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"


def _json_line(payload: Mapping[str, Any]) -> None:
    sys.stdout.write(
        json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
    )
    sys.stdout.flush()


def _safe_exchange_path(root: Path, value: str, *, must_exist: bool) -> Path:
    relative = str(value).strip()
    if not relative:
        raise ValueError("exchange path is empty")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("exchange path escapes the mounted root") from exc
    if candidate == root:
        raise ValueError("exchange path resolves to the mounted root")
    if must_exist and not candidate.is_file():
        raise FileNotFoundError(f"exchange input is missing: {value}")
    return candidate


def _install_compatibility_shims() -> None:
    """Provide the two tiny APIs imported by the official model source."""

    import torch
    from torch import nn

    class CfgNode(dict):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs.pop("new_allowed", None)
            super().__init__(*args, **kwargs)

        def __getattr__(self, name: str) -> Any:
            try:
                return self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

        def __setattr__(self, name: str, value: Any) -> None:
            self[name] = value

    class DropPath(nn.Module):
        def __init__(self, drop_prob: float = 0.0) -> None:
            super().__init__()
            self.drop_prob = float(drop_prob)

        def forward(self, value):
            if self.drop_prob <= 0.0 or not self.training:
                return value
            keep = 1.0 - self.drop_prob
            shape = (value.shape[0],) + (1,) * (value.ndim - 1)
            random_tensor = keep + torch.rand(
                shape,
                dtype=value.dtype,
                device=value.device,
            )
            return value.div(keep) * random_tensor.floor()

    yacs = types.ModuleType("yacs")
    yacs_config = types.ModuleType("yacs.config")
    yacs_config.CfgNode = CfgNode
    yacs.config = yacs_config
    timm = types.ModuleType("timm")
    timm_models = types.ModuleType("timm.models")
    timm_layers = types.ModuleType("timm.models.layers")
    timm_layers.DropPath = DropPath
    timm_layers.trunc_normal_ = nn.init.trunc_normal_
    timm_models.layers = timm_layers
    timm.models = timm_models
    sys.modules.update(
        {
            "yacs": yacs,
            "yacs.config": yacs_config,
            "timm": timm,
            "timm.models": timm_models,
            "timm.models.layers": timm_layers,
        }
    )


def _install_checkpoint_unpickle_shim() -> None:
    """Allow the trusted official Lightning checkpoint metadata to unpickle."""

    lightning = types.ModuleType("pytorch_lightning")
    callbacks = types.ModuleType("pytorch_lightning.callbacks")
    model_checkpoint = types.ModuleType(
        "pytorch_lightning.callbacks.model_checkpoint"
    )

    class ModelCheckpoint:
        pass

    ModelCheckpoint.__module__ = (
        "pytorch_lightning.callbacks.model_checkpoint"
    )
    model_checkpoint.ModelCheckpoint = ModelCheckpoint
    callbacks.model_checkpoint = model_checkpoint
    lightning.callbacks = callbacks
    sys.modules.update(
        {
            "pytorch_lightning": lightning,
            "pytorch_lightning.callbacks": callbacks,
            "pytorch_lightning.callbacks.model_checkpoint": model_checkpoint,
        }
    )


def _resize(image, width: int, height: int, *, nearest: bool = False):
    import cv2

    interpolation = cv2.INTER_NEAREST if nearest else (
        cv2.INTER_AREA
        if image.shape[0] > height and image.shape[1] > width
        else cv2.INTER_LINEAR
    )
    return cv2.resize(image, (width, height), interpolation=interpolation)


def _to_tensor(image, *, normalize: bool = False):
    import numpy as np
    import torch

    array = np.asarray(image)
    if array.ndim == 2:
        array = array[:, :, None]
    tensor = torch.from_numpy(
        np.ascontiguousarray(array.transpose(2, 0, 1))
    ).float() / 255.0
    if normalize:
        tensor = tensor * 2.0 - 1.0
    return tensor


def _masked_position_encoding(mask):
    import cv2
    import numpy as np

    original = np.asarray(mask, dtype=np.uint8).copy()
    original_height, original_width = original.shape[:2]
    working = _resize(original, 256, 256)
    working[working > 0] = 255
    known = 1.0 - (working.astype(np.float32) / 255.0)
    positions = np.zeros((256, 256), dtype=np.int32)
    directions = np.zeros((256, 256, 4), dtype=np.int32)
    ones = np.ones((3, 3), dtype=np.float32)
    directional_filters = (
        np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=np.float32),
        np.array([[0, 0, 0], [1, 1, 0], [1, 1, 0]], dtype=np.float32),
        np.array([[0, 1, 1], [0, 1, 1], [0, 0, 0]], dtype=np.float32),
        np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]], dtype=np.float32),
    )
    distance = 0
    while bool(np.any(known < 1.0)):
        distance += 1
        expanded = cv2.filter2D(known, -1, ones)
        expanded[expanded > 0] = 1
        newly_known = expanded - known
        positions[newly_known == 1] = distance
        for index, kernel in enumerate(directional_filters):
            direction = cv2.filter2D(known, -1, kernel)
            direction[direction > 0] = 1
            direction -= known
            directions[direction == 1, index] = 1
        known = expanded
        if distance > 256:
            raise RuntimeError("masked position encoding did not converge")

    relative = positions / 128.0
    relative = np.clip((relative * 128).astype(np.int32), 0, 127)
    if (original_height, original_width) != (256, 256):
        relative = _resize(relative, original_width, original_height, nearest=True)
        relative[original == 0] = 0
        directions = _resize(
            directions,
            original_width,
            original_height,
            nearest=True,
        )
        directions[original == 0, :] = 0
    return relative, positions, directions


def _load_prefix(module, state: Mapping[str, Any], prefix: str) -> None:
    marker = prefix + "."
    selected = {
        key[len(marker) :]: value
        for key, value in state.items()
        if key.startswith(marker)
    }
    if not selected:
        raise RuntimeError(f"checkpoint prefix is missing: {prefix}")
    missing, unexpected = module.load_state_dict(selected, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint prefix {prefix} differs: "
            f"missing={list(missing)}, unexpected={list(unexpected)}"
        )


class ZITSPlusPlusModel:
    def __init__(
        self,
        *,
        source_root: Path,
        model_checkpoint: Path,
        lsm_checkpoint: Path,
        test_size: int,
    ) -> None:
        import torch
        import yaml

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; CPU fallback is forbidden")
        self.torch = torch
        self.test_size = int(test_size)
        self.device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_grad_enabled(False)

        _install_compatibility_shims()
        _install_checkpoint_unpickle_shim()
        sys.path.insert(0, str(source_root))
        from networks.generators import FTRModel
        from networks.tsr import EdgeLineGPT256RelBCE_edge_pred_infer
        from networks.upsample import StructureUpsampling4
        from trainers.lsm_hawp.detector import WireframeDetector

        config_path = model_checkpoint.parent.parent / "config.yml"
        with config_path.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        generator_args = dict(config["g_args"])
        if bool(generator_args.get("use_gradient")):
            raise RuntimeError("model_512 unexpectedly requires gradient prior")

        with contextlib.redirect_stdout(sys.stderr):
            structure = StructureUpsampling4()
            edge_line = EdgeLineGPT256RelBCE_edge_pred_infer()
            generator = FTRModel(config=generator_args)
            wireframe = WireframeDetector(is_cuda=True)

        checkpoint = torch.load(
            str(model_checkpoint),
            map_location="cpu",
            weights_only=False,
        )
        state = checkpoint["state_dict"]
        _load_prefix(structure, state, "structure_upsample")
        _load_prefix(edge_line, state, "edgeline_tsr")
        _load_prefix(generator, state, "ftr_ema")
        del state
        del checkpoint
        lsm = torch.load(
            str(lsm_checkpoint),
            map_location="cpu",
            weights_only=False,
        )
        wireframe.load_state_dict(lsm["model"], strict=True)
        del lsm

        self.structure = structure.to(self.device, dtype=torch.float32).eval()
        self.edge_line = edge_line.to(self.device, dtype=torch.float32).eval()
        self.generator = generator.to(self.device, dtype=torch.float32).eval()
        self.wireframe = wireframe.to(
            self.device,
            dtype=torch.float32,
        ).eval()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        non_fp32 = [
            name
            for name, parameter in self.generator.named_parameters()
            if parameter.dtype != torch.float32
        ]
        if non_fp32:
            raise RuntimeError(
                "ZITS++ generator has non-FP32 parameters: "
                + ", ".join(non_fp32[:5])
            )

    def _wireframe_prior(self, image_512, mask_512):
        import cv2
        import numpy as np
        import torch
        import torch.nn.functional as functional

        mean = torch.tensor(
            [109.730, 103.832, 98.681],
            device=self.device,
        ).reshape(1, 3, 1, 1)
        std = torch.tensor(
            [22.275, 22.124, 23.229],
            device=self.device,
        ).reshape(1, 3, 1, 1)
        image_255 = image_512 * 255.0
        resized_mask = functional.interpolate(
            mask_512,
            size=image_255.shape[2:],
            mode="nearest",
        )
        masked = image_255 * (1 - resized_mask) + resized_mask * 127.5
        normalized = (masked - mean) / std
        line_maps = []
        line_count = 0
        for batch_index in range(normalized.shape[0]):
            output = self.wireframe(normalized[batch_index : batch_index + 1])
            canvas = np.zeros((256, 256), dtype=np.uint8)
            if int(output.get("num_proposals", 0) or 0) > 0:
                lines = output["lines_pred"].detach().cpu().numpy()
                scores = output["lines_score"].detach().cpu().numpy()
                for line, score in zip(lines, scores):
                    if float(score) <= 0.85:
                        continue
                    x1, y1, x2, y2 = (
                        int(np.clip(round(float(line[0]) * 256), 0, 255)),
                        int(np.clip(round(float(line[1]) * 256), 0, 255)),
                        int(np.clip(round(float(line[2]) * 256), 0, 255)),
                        int(np.clip(round(float(line[3]) * 256), 0, 255)),
                    )
                    cv2.line(
                        canvas,
                        (x1, y1),
                        (x2, y2),
                        255,
                        1,
                        cv2.LINE_AA,
                    )
                    line_count += 1
            line_maps.append(_to_tensor(canvas).unsqueeze(0))
        return torch.cat(line_maps, dim=0).to(self.device), line_count

    def inpaint(self, image, mask):
        import cv2
        import numpy as np
        import torch
        import torch.nn.functional as functional
        from trainers.nms_torch import get_nms

        source = np.asarray(image, dtype=np.uint8)
        source_mask = (np.asarray(mask) > 0).astype(np.uint8) * 255
        height, width = source.shape[:2]
        resized_image = _resize(
            source,
            self.test_size,
            self.test_size,
        )
        resized_mask = _resize(
            source_mask,
            self.test_size,
            self.test_size,
            nearest=True,
        )
        resized_mask = (resized_mask > 0).astype(np.uint8) * 255
        image_256 = _resize(resized_image, 256, 256)
        mask_256 = _resize(resized_mask, 256, 256)
        mask_256[mask_256 > 0] = 255
        image_512 = _resize(resized_image, 512, 512)
        mask_512 = _resize(resized_mask, 512, 512, nearest=True)
        mask_512[mask_512 > 0] = 255

        batch = {
            "image": _to_tensor(resized_image, normalize=True)
            .unsqueeze(0)
            .to(self.device),
            "img_256": _to_tensor(image_256, normalize=True)
            .unsqueeze(0)
            .to(self.device),
            "mask": _to_tensor(resized_mask)
            .unsqueeze(0)
            .to(self.device),
            "mask_256": _to_tensor(mask_256)
            .unsqueeze(0)
            .to(self.device),
            "mask_512": _to_tensor(mask_512)
            .unsqueeze(0)
            .to(self.device),
            "img_512": _to_tensor(image_512)
            .unsqueeze(0)
            .to(self.device),
        }
        relative, absolute, direction = _masked_position_encoding(
            resized_mask
        )
        batch["rel_pos"] = torch.from_numpy(relative).long().unsqueeze(0).to(
            self.device
        )
        batch["abs_pos"] = torch.from_numpy(absolute).long().unsqueeze(0).to(
            self.device
        )
        batch["direct"] = torch.from_numpy(direction).long().unsqueeze(0).to(
            self.device
        )
        batch["line_256"], line_count = self._wireframe_prior(
            batch["img_512"],
            batch["mask_512"],
        )

        edge, line = self.edge_line(
            batch["img_256"],
            batch["line_256"],
            masks=batch["mask_256"],
        )
        line = batch["line_256"] * (1 - batch["mask_256"]) + (
            line * batch["mask_256"]
        )
        current_size = 256
        edge_nms = edge
        while current_size * 2 <= self.test_size:
            line = self.structure(line)[0]
            edge_nms = get_nms(edge, binary_threshold=50)
            edge_nms = self.structure(edge_nms)[0]
            edge_nms = torch.sigmoid((edge_nms + 2) * 2)
            line = torch.sigmoid((line + 2) * 2)
            current_size *= 2
        edge_nms = functional.interpolate(
            edge_nms,
            size=(self.test_size, self.test_size),
            mode="bilinear",
            align_corners=False,
        )
        edge = functional.interpolate(
            edge,
            size=(self.test_size, self.test_size),
            mode="bilinear",
            align_corners=False,
        )
        edge = torch.where(edge >= 0.25, edge_nms, edge)
        line = functional.interpolate(
            line,
            size=(self.test_size, self.test_size),
            mode="bilinear",
            align_corners=False,
        )
        batch["edge"] = edge.detach()
        batch["line"] = line.detach()

        generated = self.generator(batch)
        combined = batch["image"] * (1 - batch["mask"]) + (
            generated * batch["mask"]
        )
        combined = (
            torch.clamp((combined + 1.0) * 127.5, 0, 255)
            .permute(0, 2, 3, 1)[0]
            .round()
            .byte()
            .cpu()
            .numpy()
        )
        restored = cv2.resize(
            combined,
            (width, height),
            interpolation=cv2.INTER_CUBIC,
        )
        final = source.copy()
        active = source_mask > 0
        final[active] = restored[active]
        return final, {
            "wireframe_line_count": int(line_count),
            "test_size": self.test_size,
            "max_memory_allocated_bytes": int(
                torch.cuda.max_memory_allocated(self.device)
            ),
            "max_memory_reserved_bytes": int(
                torch.cuda.max_memory_reserved(self.device)
            ),
        }


def serve(args: argparse.Namespace) -> int:
    import cv2
    import numpy as np
    import torch

    source_root = Path(args.source_root).resolve()
    model_checkpoint = Path(args.model_checkpoint).resolve()
    lsm_checkpoint = Path(args.lsm_checkpoint).resolve()
    exchange_root = Path(args.exchange_root).resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"official source root is missing: {source_root}")
    exchange_root.mkdir(parents=True, exist_ok=True)

    load_started = time.perf_counter()
    model = ZITSPlusPlusModel(
        source_root=source_root,
        model_checkpoint=model_checkpoint,
        lsm_checkpoint=lsm_checkpoint,
        test_size=args.test_size,
    )
    load_seconds = time.perf_counter() - load_started
    properties = torch.cuda.get_device_properties(0)
    _json_line(
        {
            "status": "ready",
            "actual_device": "cuda:0",
            "actual_precision": "fp32",
            "cpu_fallback_used": False,
            "fp32_promotion_eligible": True,
            "gpu_name": properties.name,
            "gpu_total_memory_bytes": int(properties.total_memory),
            "model_load_seconds": round(load_seconds, 6),
            "source_commit": SOURCE_COMMIT,
            "model_sha256": MODEL_SHA256,
            "lsm_sha256": LSM_SHA256,
            "config_sha256": CONFIG_SHA256,
            "license_sha256": LICENSE_SHA256,
            "test_size": int(args.test_size),
        }
    )
    for raw_line in sys.stdin:
        try:
            request = json.loads(raw_line)
            if request.get("command") == "shutdown":
                _json_line({"status": "shutdown"})
                return 0
            request_id = str(request.get("request_id") or "")
            image_path = _safe_exchange_path(
                exchange_root,
                str(request.get("image") or ""),
                must_exist=True,
            )
            mask_path = _safe_exchange_path(
                exchange_root,
                str(request.get("mask") or ""),
                must_exist=True,
            )
            output_path = _safe_exchange_path(
                exchange_root,
                str(request.get("output") or ""),
                must_exist=False,
            )
            image_bgr = cv2.imdecode(
                np.fromfile(str(image_path), dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            mask = cv2.imdecode(
                np.fromfile(str(mask_path), dtype=np.uint8),
                cv2.IMREAD_GRAYSCALE,
            )
            if image_bgr is None or mask is None:
                raise ValueError("unable to decode exchange image or mask")
            image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            torch.cuda.reset_peak_memory_stats(0)
            started = time.perf_counter()
            with torch.inference_mode():
                cleaned, diagnostics = model.inpaint(image, mask)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            output_path.parent.mkdir(parents=True, exist_ok=True)
            encoded_source = cv2.cvtColor(cleaned, cv2.COLOR_RGB2BGR)
            ok, encoded = cv2.imencode(".png", encoded_source)
            if not ok:
                raise RuntimeError("unable to encode ZITS++ output")
            encoded.tofile(str(output_path))
            _json_line(
                {
                    "status": "completed",
                    "request_id": request_id,
                    "inference_seconds": round(elapsed, 6),
                    "diagnostics": diagnostics,
                }
            )
        except Exception as exc:
            _json_line(
                {
                    "status": "failed",
                    "request_id": str(
                        locals().get("request", {}).get("request_id") or ""
                    ),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ZITS++ isolated CUDA FP32 JSON-lines adapter."
    )
    parser.add_argument("command", choices=("serve",))
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--model-checkpoint", required=True)
    parser.add_argument("--lsm-checkpoint", required=True)
    parser.add_argument("--exchange-root", required=True)
    parser.add_argument("--test-size", type=int, default=512)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "serve":
        return serve(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
