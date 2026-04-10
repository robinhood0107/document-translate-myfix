from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen

from modules.ocr.selection import is_local_ocr_engine
from modules.utils.exceptions import LocalServiceSetupError

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_LLAMA_CPP_IMAGE = "local/llama.cpp:server-cuda-b8709"
DEFAULT_HUNYUAN_N_GPU_LAYERS = "80"

_ENGINE_CONFIG = {
    "HunyuanOCR": {
        "compose_file": ROOT_DIR / "hunyuanocr_docker_files" / "docker-compose.yaml",
        "managed_url": "http://127.0.0.1:28080/v1",
        "health_url": "http://127.0.0.1:28080/health",
        "settings_page_name": "HunyuanOCR Settings",
    },
    "PaddleOCR VL": {
        "compose_file": ROOT_DIR / "paddleocr_vl_docker_files" / "docker-compose.yaml",
        "managed_url": "http://127.0.0.1:28118/layout-parsing",
        "health_url": "http://127.0.0.1:28118/docs",
        "settings_page_name": "PaddleOCR VL Settings",
    },
}


def _normalize_url(url: str) -> str:
    parsed = urlparse(str(url or "").strip())
    path = parsed.path.rstrip("/")
    if not path:
        path = "/"
    return parsed._replace(path=path, params="", query="", fragment="").geturl()


class LocalOCRRuntimeManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._compose_command: tuple[str, ...] | None = None
        self._active_engine: str | None = None

    def validate_engine(self, engine_key: str, settings_page: Any) -> None:
        if not is_local_ocr_engine(engine_key):
            return
        server_url = self._resolve_server_url(engine_key, settings_page)
        if not server_url:
            raise self._build_setup_error(
                engine_key,
                "Server URL is empty.",
            )
        if not self.should_manage_engine(engine_key, settings_page):
            return
        compose_file = self._config_for(engine_key)["compose_file"]
        if not compose_file.is_file():
            raise self._build_setup_error(
                engine_key,
                f"Bundled Docker compose file was not found: {compose_file}",
            )
        self._resolve_compose_command(engine_key)

    def should_manage_engine(self, engine_key: str, settings_page: Any) -> bool:
        if not is_local_ocr_engine(engine_key):
            return False
        config = self._config_for(engine_key)
        server_url = self._resolve_server_url(engine_key, settings_page)
        return _normalize_url(server_url) == _normalize_url(config["managed_url"])

    def ensure_engine(self, engine_key: str, settings_page: Any, *, timeout_sec: int = 240) -> None:
        with self._lock:
            if not is_local_ocr_engine(engine_key) or not self.should_manage_engine(engine_key, settings_page):
                self._deactivate_active_engine()
                return

            self.validate_engine(engine_key, settings_page)
            config = self._config_for(engine_key)
            if self._active_engine and self._active_engine != engine_key:
                self._stop_engine(self._active_engine)

            if self._wait_for_health(config["health_url"], timeout_sec=2):
                self._active_engine = engine_key
                return

            self._run_compose(engine_key, "up", "-d")
            if not self._wait_for_health(config["health_url"], timeout_sec=timeout_sec):
                raise self._build_setup_error(
                    engine_key,
                    (
                        f"Timed out while waiting for {engine_key} at {config['health_url']} "
                        f"after starting {config['compose_file'].name}."
                    ),
                )
            self._active_engine = engine_key

    def shutdown(self) -> None:
        with self._lock:
            self._deactivate_active_engine()

    def _deactivate_active_engine(self) -> None:
        if not self._active_engine:
            return
        try:
            self._stop_engine(self._active_engine)
        finally:
            self._active_engine = None

    def _stop_engine(self, engine_key: str) -> None:
        try:
            self._run_compose(engine_key, "down")
        except LocalServiceSetupError:
            logger.warning("Failed to stop managed OCR runtime for %s.", engine_key, exc_info=True)

    def _run_compose(self, engine_key: str, *compose_args: str) -> None:
        config = self._config_for(engine_key)
        compose_file = config["compose_file"]
        command = [
            *self._resolve_compose_command(engine_key),
            "-f",
            str(compose_file),
            *compose_args,
        ]
        result = subprocess.run(
            command,
            cwd=str(compose_file.parent),
            capture_output=True,
            text=True,
            env=self._build_env(engine_key),
        )
        if result.returncode == 0:
            return
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        detail = stderr or stdout or f"docker compose exited with code {result.returncode}."
        raise self._build_setup_error(engine_key, detail)

    def _resolve_compose_command(self, engine_key: str) -> tuple[str, ...]:
        if self._compose_command is not None:
            return self._compose_command

        candidates: list[tuple[str, ...]] = []
        if shutil.which("docker"):
            candidates.append(("docker", "compose"))
        if shutil.which("docker-compose"):
            candidates.append(("docker-compose",))

        for candidate in candidates:
            probe = subprocess.run(
                [*candidate, "version"],
                capture_output=True,
                text=True,
            )
            if probe.returncode == 0:
                self._compose_command = candidate
                return candidate

        raise self._build_setup_error(
            engine_key,
            "Docker Compose is not available. Install Docker Desktop or docker-compose and try again.",
        )

    def _build_env(self, engine_key: str) -> dict[str, str]:
        env = dict(os.environ)
        env.setdefault("LLAMA_CPP_IMAGE", DEFAULT_LLAMA_CPP_IMAGE)
        if engine_key == "HunyuanOCR":
            env.setdefault("LLAMA_N_GPU_LAYERS", DEFAULT_HUNYUAN_N_GPU_LAYERS)
        return env

    def _resolve_server_url(self, engine_key: str, settings_page: Any) -> str:
        if engine_key == "HunyuanOCR":
            return str(settings_page.get_hunyuan_ocr_settings().get("server_url", "")).strip()
        if engine_key == "PaddleOCR VL":
            return str(settings_page.get_paddleocr_vl_settings().get("server_url", "")).strip()
        return ""

    def _config_for(self, engine_key: str) -> dict[str, Any]:
        try:
            return _ENGINE_CONFIG[engine_key]
        except KeyError as exc:
            raise self._build_setup_error(engine_key, f"Unsupported local OCR engine: {engine_key}") from exc

    def _build_setup_error(self, engine_key: str, detail: str) -> LocalServiceSetupError:
        config = _ENGINE_CONFIG.get(engine_key, {})
        expected_url = config.get("managed_url") or ""
        compose_file = config.get("compose_file")
        settings_page_name = config.get("settings_page_name", "PaddleOCR VL Settings")
        extra = detail.strip()
        if expected_url:
            extra = f"{extra}\nRequested engine: {engine_key}\nExpected endpoint: {expected_url}"
        if compose_file:
            extra = f"{extra}\nCompose file: {compose_file}"
        return LocalServiceSetupError(
            extra.strip(),
            service_name=engine_key,
            settings_page_name=settings_page_name,
        )

    @staticmethod
    def _wait_for_health(url: str, *, timeout_sec: int) -> bool:
        deadline = time.monotonic() + max(timeout_sec, 1)
        while time.monotonic() < deadline:
            try:
                with urlopen(url, timeout=2) as response:
                    if getattr(response, "status", 200) < 500:
                        return True
            except (URLError, OSError, ValueError):
                pass
            time.sleep(1)
        return False
