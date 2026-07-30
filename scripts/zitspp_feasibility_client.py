#!/usr/bin/env python3
"""Host-side client for the isolated ZITS++ benchmark adapter."""

from __future__ import annotations

import hashlib
import json
import os
import queue
import secrets
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = ROOT / "scripts" / "zitspp_feasibility_adapter.py"
SOURCE_COMMIT = "de8dd48b17aedd15824842adb7bcca7535daba84"
MODEL_SHA256 = "e30d2073ba63af42836ac611214ed984db7ec739a1eef019451df6a34f566f57"
LSM_SHA256 = "6f72a60ec895f11830763069a40cb548dfb0ba77aca5282ffeea8afc72dc1723"
CONFIG_SHA256 = "e46dd48b4715f0044debfa0faca1c5af0c149b1cb1dffa29afb042687f13e4f2"
LICENSE_SHA256 = "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
DEFAULT_DOCKER_IMAGE = (
    "ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/"
    "paddleocr-genai-vllm-server@"
    "sha256:d0d32c04a2119613d25a0a4c292e165ccc107954b74580613cf59e378037f8f5"
)


class ZITSConfigurationError(RuntimeError):
    """Raised when the external official model contract is incomplete."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_text(command: list[str], *, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.stdout.strip()


def validate_external_contract(
    *,
    source_root: Path,
    model_checkpoint: Path,
    lsm_checkpoint: Path,
) -> dict[str, Any]:
    source_root = source_root.expanduser().resolve()
    model_checkpoint = model_checkpoint.expanduser().resolve()
    lsm_checkpoint = lsm_checkpoint.expanduser().resolve()
    config_path = model_checkpoint.parent.parent / "config.yml"
    license_path = source_root / "LICENSE"
    if not (source_root / ".git").is_dir():
        raise ZITSConfigurationError(
            f"official ZITS++ Git checkout is missing: {source_root}"
        )
    head = _run_text(["git", "rev-parse", "HEAD"], cwd=source_root)
    if head != SOURCE_COMMIT:
        raise ZITSConfigurationError(
            f"official ZITS++ commit differs: {head}"
        )
    if _run_text(["git", "status", "--porcelain"], cwd=source_root):
        raise ZITSConfigurationError("official ZITS++ checkout is dirty")
    if not model_checkpoint.is_file():
        raise ZITSConfigurationError(
            f"ZITS++ model checkpoint is missing: {model_checkpoint}"
        )
    if not lsm_checkpoint.is_file():
        raise ZITSConfigurationError(
            f"LSM-HAWP checkpoint is missing: {lsm_checkpoint}"
        )
    if not config_path.is_file():
        raise ZITSConfigurationError(
            f"ZITS++ model config is missing: {config_path}"
        )
    if not license_path.is_file():
        raise ZITSConfigurationError(
            f"official ZITS++ license is missing: {license_path}"
        )
    model_sha = sha256_file(model_checkpoint)
    lsm_sha = sha256_file(lsm_checkpoint)
    config_sha = sha256_file(config_path)
    license_sha = sha256_file(license_path)
    if model_sha != MODEL_SHA256:
        raise ZITSConfigurationError(
            f"ZITS++ model SHA-256 differs: {model_sha}"
        )
    if lsm_sha != LSM_SHA256:
        raise ZITSConfigurationError(
            f"LSM-HAWP SHA-256 differs: {lsm_sha}"
        )
    if config_sha != CONFIG_SHA256:
        raise ZITSConfigurationError(
            f"ZITS++ model config SHA-256 differs: {config_sha}"
        )
    if license_sha != LICENSE_SHA256:
        raise ZITSConfigurationError(
            f"official ZITS++ license SHA-256 differs: {license_sha}"
        )
    if "Apache License" not in license_path.read_text(
        encoding="utf-8",
        errors="strict",
    )[:256]:
        raise ZITSConfigurationError(
            "official ZITS++ license is not the expected Apache license"
        )
    return {
        "source_commit": head,
        "model_sha256": model_sha,
        "model_size": model_checkpoint.stat().st_size,
        "lsm_sha256": lsm_sha,
        "lsm_size": lsm_checkpoint.stat().st_size,
        "config_sha256": config_sha,
        "license_sha256": license_sha,
        "license": "Apache-2.0",
    }


def _mount(source: Path, target: str, *, readonly: bool) -> str:
    value = f"type=bind,src={source.resolve()},dst={target}"
    if readonly:
        value += ",readonly"
    return value


class ZITSPlusPlusDockerInpainter:
    """One-model-per-profile CUDA process implementing the inpainter callable."""

    def __init__(
        self,
        *,
        source_root: Path,
        model_checkpoint: Path,
        lsm_checkpoint: Path,
        docker_image: str = DEFAULT_DOCKER_IMAGE,
        test_size: int = 512,
        ready_timeout_seconds: float = 240.0,
        request_timeout_seconds: float = 180.0,
    ) -> None:
        self.source_root = source_root.expanduser().resolve()
        self.model_checkpoint = model_checkpoint.expanduser().resolve()
        self.lsm_checkpoint = lsm_checkpoint.expanduser().resolve()
        self.docker_image = str(docker_image)
        self.test_size = int(test_size)
        self.ready_timeout_seconds = float(ready_timeout_seconds)
        self.request_timeout_seconds = float(request_timeout_seconds)
        self.contract = validate_external_contract(
            source_root=self.source_root,
            model_checkpoint=self.model_checkpoint,
            lsm_checkpoint=self.lsm_checkpoint,
        )
        if "@sha256:" not in self.docker_image:
            raise ZITSConfigurationError(
                "ZITS++ feasibility Docker image must be digest-pinned"
            )
        if not ADAPTER_PATH.is_file():
            raise ZITSConfigurationError(
                f"ZITS++ adapter is missing: {ADAPTER_PATH}"
            )
        self._temporary = tempfile.TemporaryDirectory(
            prefix="comic-translate-zitspp-"
        )
        self.exchange_root = Path(self._temporary.name).resolve()
        self.stderr_path = self.exchange_root / "adapter-stderr.log"
        self._stderr = self.stderr_path.open("w", encoding="utf-8")
        self.container_name = (
            f"ct-zitspp-feasibility-{os.getpid()}-{secrets.token_hex(3)}"
        )
        command = [
            "docker",
            "run",
            "--rm",
            "--interactive",
            "--name",
            self.container_name,
            "--gpus",
            "all",
            "--network",
            "none",
            "--mount",
            _mount(self.source_root, "/opt/zitspp", readonly=True),
            "--mount",
            _mount(ADAPTER_PATH, "/opt/zitspp-adapter.py", readonly=True),
            "--mount",
            _mount(
                self.model_checkpoint,
                "/weights/model_512/models/last.ckpt",
                readonly=True,
            ),
            "--mount",
            _mount(
                self.model_checkpoint.parent.parent / "config.yml",
                "/weights/model_512/config.yml",
                readonly=True,
            ),
            "--mount",
            _mount(
                self.lsm_checkpoint,
                "/weights/best_lsm_hawp.pth",
                readonly=True,
            ),
            "--mount",
            _mount(self.exchange_root, "/exchange", readonly=False),
            "--entrypoint",
            "python3",
            self.docker_image,
            "/opt/zitspp-adapter.py",
            "serve",
            "--source-root",
            "/opt/zitspp",
            "--model-checkpoint",
            "/weights/model_512/models/last.ckpt",
            "--lsm-checkpoint",
            "/weights/best_lsm_hawp.pth",
            "--exchange-root",
            "/exchange",
            "--test-size",
            str(self.test_size),
        ]
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        self._messages: queue.Queue[str | None] = queue.Queue()
        self._reader = threading.Thread(
            target=self._read_stdout,
            name="zitspp-adapter-stdout",
            daemon=True,
        )
        self._reader.start()
        try:
            ready = self._next_message(self.ready_timeout_seconds)
            if ready.get("status") != "ready":
                raise RuntimeError(
                    "ZITS++ adapter did not become ready: "
                    + json.dumps(ready, ensure_ascii=False)
                )
            if not str(ready.get("actual_device") or "").startswith("cuda"):
                raise RuntimeError("ZITS++ adapter used a non-CUDA device")
            if str(ready.get("actual_precision") or "") != "fp32":
                raise RuntimeError("ZITS++ adapter did not use FP32")
            if bool(ready.get("cpu_fallback_used")):
                raise RuntimeError("ZITS++ adapter reported CPU fallback")
            self.runtime = {
                **self.contract,
                **ready,
                "status": "ready",
                "docker_image": self.docker_image,
                "container_name": self.container_name,
                "adapter": "zitspp-isolated-jsonl-v1",
            }
        except Exception:
            self.close()
            raise
        self._request_index = 0
        self._closed = False

    @classmethod
    def from_environment(cls) -> "ZITSPlusPlusDockerInpainter":
        required = {
            "source_root": os.environ.get("CT_ZITSPP_SOURCE_ROOT", ""),
            "model_checkpoint": os.environ.get(
                "CT_ZITSPP_MODEL_CHECKPOINT",
                "",
            ),
            "lsm_checkpoint": os.environ.get(
                "CT_ZITSPP_LSM_CHECKPOINT",
                "",
            ),
        }
        missing = [
            name
            for name, value in required.items()
            if not str(value).strip()
        ]
        if missing:
            raise ZITSConfigurationError(
                "ZITS++ feasibility requires environment variables: "
                + ", ".join(
                    {
                        "source_root": "CT_ZITSPP_SOURCE_ROOT",
                        "model_checkpoint": "CT_ZITSPP_MODEL_CHECKPOINT",
                        "lsm_checkpoint": "CT_ZITSPP_LSM_CHECKPOINT",
                    }[name]
                    for name in missing
                )
            )
        return cls(
            source_root=Path(required["source_root"]),
            model_checkpoint=Path(required["model_checkpoint"]),
            lsm_checkpoint=Path(required["lsm_checkpoint"]),
            docker_image=os.environ.get(
                "CT_ZITSPP_DOCKER_IMAGE",
                DEFAULT_DOCKER_IMAGE,
            ),
            test_size=int(os.environ.get("CT_ZITSPP_TEST_SIZE", "512")),
        )

    def _read_stdout(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            self._messages.put(line)
        self._messages.put(None)

    def _stderr_tail(self) -> str:
        try:
            self._stderr.flush()
            text = self.stderr_path.read_text(
                encoding="utf-8",
                errors="replace",
            )
        except OSError:
            return ""
        return text[-4000:]

    def _next_message(self, timeout_seconds: float) -> dict[str, Any]:
        try:
            line = self._messages.get(timeout=timeout_seconds)
        except queue.Empty as exc:
            raise TimeoutError(
                "ZITS++ adapter response timed out\n" + self._stderr_tail()
            ) from exc
        if line is None:
            raise RuntimeError(
                f"ZITS++ adapter exited with {self.process.poll()}\n"
                + self._stderr_tail()
            )
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"ZITS++ adapter emitted non-JSON output: {line!r}\n"
                + self._stderr_tail()
            ) from exc
        if not isinstance(value, dict):
            raise RuntimeError("ZITS++ adapter response is not an object")
        return value

    @staticmethod
    def _write_image(path: Path, image: Any) -> None:
        import cv2
        import numpy as np

        array = np.asarray(image)
        if array.ndim == 3:
            encoded_source = cv2.cvtColor(
                array[:, :, :3],
                cv2.COLOR_RGB2BGR,
            )
        else:
            encoded_source = array
        ok, encoded = cv2.imencode(".png", encoded_source)
        if not ok:
            raise RuntimeError(f"unable to encode ZITS++ exchange: {path}")
        encoded.tofile(str(path))

    @staticmethod
    def _read_image(path: Path):
        import cv2
        import numpy as np

        image = cv2.imdecode(
            np.fromfile(str(path), dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        if image is None:
            raise RuntimeError(f"unable to decode ZITS++ output: {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def __call__(self, image: Any, mask: Any, _config: Any):
        if self._closed:
            raise RuntimeError("ZITS++ adapter is closed")
        self._request_index += 1
        request_id = f"{self._request_index:04d}"
        input_path = self.exchange_root / f"{request_id}-input.png"
        mask_path = self.exchange_root / f"{request_id}-mask.png"
        output_path = self.exchange_root / f"{request_id}-output.png"
        self._write_image(input_path, image)
        self._write_image(mask_path, mask)
        request = {
            "request_id": request_id,
            "image": input_path.name,
            "mask": mask_path.name,
            "output": output_path.name,
        }
        assert self.process.stdin is not None
        self.process.stdin.write(
            json.dumps(request, ensure_ascii=False) + "\n"
        )
        self.process.stdin.flush()
        response = self._next_message(self.request_timeout_seconds)
        if (
            response.get("status") != "completed"
            or response.get("request_id") != request_id
        ):
            raise RuntimeError(
                "ZITS++ adapter inference failed: "
                + json.dumps(response, ensure_ascii=False)
                + "\n"
                + self._stderr_tail()
            )
        self.runtime["last_request"] = response
        return self._read_image(output_path)

    def close(self) -> None:
        process = getattr(self, "process", None)
        if process is None:
            return
        if getattr(self, "_closed", False):
            return
        self._closed = True
        try:
            if process.poll() is None and process.stdin is not None:
                process.stdin.write('{"command":"shutdown"}\n')
                process.stdin.flush()
                process.wait(timeout=15)
        except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
            subprocess.run(
                ["docker", "stop", "--time", "10", self.container_name],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        finally:
            try:
                if process.stdin is not None:
                    process.stdin.close()
            except OSError:
                pass
            try:
                if process.stdout is not None:
                    process.stdout.close()
            except OSError:
                pass
            self._stderr.close()
            self._temporary.cleanup()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
