from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen

from modules.translation.llm.custom_local_gemma import (
    DEFAULT_GEMMA_LOCAL_ENDPOINT,
    DEFAULT_GEMMA_LOCAL_MODEL,
)
from modules.utils.exceptions import (
    LocalServiceConnectionError,
    LocalServiceResponseError,
    LocalServiceSetupError,
)
from modules.utils.llama_cpp_runtime import (
    DEFAULT_LLAMA_CPP_IMAGE,
    inspect_llama_cpp_runtime,
    resolve_docker_compose_command,
    run_docker_command,
)

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_GEMMA_HEALTH_URL = "http://127.0.0.1:18080/health"
DEFAULT_GEMMA_MODELS_URL = "http://127.0.0.1:18080/v1/models"
DEFAULT_GEMMA_SETTINGS_PAGE = "Gemma Local Server Settings"

_RUNTIME_CONFIG = {
    "compose_file": ROOT_DIR / "docker-compose.yaml",
    "managed_url": DEFAULT_GEMMA_LOCAL_ENDPOINT,
    "health_url": DEFAULT_GEMMA_HEALTH_URL,
    "models_url": DEFAULT_GEMMA_MODELS_URL,
    "settings_page_name": DEFAULT_GEMMA_SETTINGS_PAGE,
    "container_name": "gemma-local-server",
}


def _normalize_url(url: str) -> str:
    parsed = urlparse(str(url or "").strip())
    path = parsed.path.rstrip("/")
    if not path:
        path = "/"
    return parsed._replace(path=path, params="", query="", fragment="").geturl()


def _strip_v1_path(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3] or "/"
    if not path:
        path = "/"
    return parsed._replace(path=path, params="", query="", fragment="").geturl().rstrip("/")


def _derive_probe_urls(api_base_url: str) -> list[str]:
    base = _normalize_url(api_base_url)
    stripped = _strip_v1_path(base)
    candidates = [
        f"{stripped}/health",
        f"{base}/models" if base.endswith("/v1") else f"{base}/v1/models",
    ]
    seen: set[str] = set()
    normalized: list[str] = []
    for item in candidates:
        url = item.rstrip("/")
        if url not in seen:
            seen.add(url)
            normalized.append(url)
    return normalized


class LocalGemmaRuntimeManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._compose_command: tuple[str, ...] | None = None
        self._managed_active = False

    def validate_server(self, settings_page: Any) -> None:
        api_base_url, _ = self._resolve_credentials(settings_page)
        if not api_base_url:
            raise self._build_setup_error("Endpoint URL is empty.")
        if not self.should_manage_server(settings_page):
            return
        compose_file = _RUNTIME_CONFIG["compose_file"]
        if not compose_file.is_file():
            raise self._build_setup_error(
                f"Bundled Docker compose file was not found: {compose_file}",
            )
        self._resolve_compose_command()

    def should_manage_server(self, settings_page: Any) -> bool:
        api_base_url, _ = self._resolve_credentials(settings_page)
        return _normalize_url(api_base_url) == _normalize_url(_RUNTIME_CONFIG["managed_url"])

    def ensure_server(self, settings_page: Any, *, timeout_sec: int = 240) -> None:
        with self._lock:
            api_base_url, model_name = self._resolve_credentials(settings_page)
            if not api_base_url:
                raise self._build_setup_error("Endpoint URL is empty.")

            if self.should_manage_server(settings_page):
                self.validate_server(settings_page)
                model_file = Path(ROOT_DIR / "testmodel" / Path(model_name).name)
                if not model_file.is_file():
                    raise self._build_setup_error(
                        f"Configured Gemma model file was not found: {model_file}"
                    )
                if self._managed_active and self._wait_for_any_probe(
                    [_RUNTIME_CONFIG["health_url"], _RUNTIME_CONFIG["models_url"]],
                    timeout_sec=2,
                ):
                    self._validate_loaded_model(api_base_url, model_name)
                    return
                self._run_compose("pull", "--policy", "always", step_name="pull", model_name=model_name)
                self._run_compose("up", "-d", "--force-recreate", step_name="up", model_name=model_name)
                if not self._wait_for_any_probe(
                    [_RUNTIME_CONFIG["health_url"], _RUNTIME_CONFIG["models_url"]],
                    timeout_sec=timeout_sec,
                ):
                    raise self._build_setup_error(
                        (
                            f"Timed out while waiting for Gemma at {_RUNTIME_CONFIG['health_url']} "
                            f"after pull + force-recreate of {_RUNTIME_CONFIG['compose_file'].name}."
                        ),
                    )
                self._managed_active = True
                self._validate_loaded_model(api_base_url, model_name)
                self._log_runtime_metadata()
                return

            if not self._wait_for_any_probe(_derive_probe_urls(api_base_url), timeout_sec=5):
                raise self._build_connection_error(
                    f"Unable to reach local Gemma server at {api_base_url}.",
                )
            self._validate_loaded_model(api_base_url, model_name)

    def shutdown(self) -> None:
        with self._lock:
            if not self._managed_active:
                return
            try:
                self._run_compose("down", step_name="down")
            except LocalServiceSetupError:
                logger.warning("Failed to stop managed Gemma runtime.", exc_info=True)
            finally:
                self._managed_active = False

    def _resolve_credentials(self, settings_page: Any) -> tuple[str, str]:
        creds = settings_page.get_credentials("Custom Local Server(Gemma)")
        api_base_url = str((creds or {}).get("api_url", "")).strip().rstrip("/")
        model_name = str((creds or {}).get("model", "")).strip() or DEFAULT_GEMMA_LOCAL_MODEL
        return api_base_url, model_name

    def _resolve_compose_command(self) -> tuple[str, ...]:
        if self._compose_command is not None:
            return self._compose_command
        try:
            self._compose_command = resolve_docker_compose_command()
            return self._compose_command
        except RuntimeError as exc:
            raise self._build_setup_error(
                "Docker Compose is not available. Install Docker Desktop or docker-compose and try again.",
            ) from exc

    def _build_env(self, model_name: str | None = None) -> dict[str, str]:
        env = dict(os.environ)
        env.setdefault("LLAMA_CPP_IMAGE", DEFAULT_LLAMA_CPP_IMAGE)
        model_file = Path(str(model_name or DEFAULT_GEMMA_LOCAL_MODEL)).name
        env["LLAMA_MODEL_FILE"] = model_file
        return env

    def _run_compose(self, *compose_args: str, step_name: str, model_name: str | None = None) -> None:
        compose_file = Path(_RUNTIME_CONFIG["compose_file"])
        env = self._build_env(model_name)
        command = [*self._resolve_compose_command(), "-f", str(compose_file), *compose_args]
        try:
            if step_name == "up":
                self._remove_named_container_if_needed(str(_RUNTIME_CONFIG["container_name"]), compose_file.parent, env)
            run_docker_command(command, cwd=compose_file.parent, env=env)
            return
        except RuntimeError as exc:
            detail = str(exc).strip()
        requested_image = env.get("LLAMA_CPP_IMAGE", "")
        extra = f"Docker compose {step_name} failed.\n{detail}"
        if requested_image:
            extra = f"{extra}\nRequested image: {requested_image}"
        raise self._build_setup_error(extra)

    def _remove_named_container_if_needed(self, container_name: str, cwd: Path, env: dict[str, str]) -> None:
        name = container_name.strip()
        if not name:
            return
        inspect_cmd = ["docker", "inspect", "--format", "{{.Id}}", name]
        inspect_result = run_docker_command(inspect_cmd, cwd=cwd, env=env, check=False)
        if inspect_result.returncode != 0:
            return
        run_docker_command(["docker", "rm", "-f", name], cwd=cwd, env=env, check=False)

    def _wait_for_any_probe(self, urls: list[str], *, timeout_sec: int) -> bool:
        deadline = time.monotonic() + max(timeout_sec, 1)
        while time.monotonic() < deadline:
            for url in urls:
                if self._probe_url(url):
                    return True
            time.sleep(1)
        return False

    @staticmethod
    def _probe_url(url: str) -> bool:
        try:
            with urlopen(url, timeout=2) as response:
                return getattr(response, "status", 200) < 500
        except (URLError, OSError, ValueError):
            return False

    def _validate_loaded_model(self, api_base_url: str, expected_model: str) -> None:
        base = _normalize_url(api_base_url)
        models_url = f"{base}/models" if base.endswith("/v1") else f"{base}/v1/models"
        try:
            with urlopen(models_url, timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception:
            return

        model_ids: list[str] = []
        for entry in payload.get("data", []):
            if isinstance(entry, dict):
                model_id = str(entry.get("id", "")).strip()
                if model_id:
                    model_ids.append(model_id)
        if model_ids and expected_model and expected_model not in model_ids:
            raise LocalServiceResponseError(
                (
                    "Gemma server is reachable but loaded models do not match the configured model.\n"
                    f"Configured model: {expected_model}\nAvailable models: {', '.join(model_ids)}"
                ),
                service_name="Gemma",
                settings_page_name=_RUNTIME_CONFIG["settings_page_name"],
            )

    def _log_runtime_metadata(self) -> None:
        try:
            runtime = inspect_llama_cpp_runtime(
                image_ref=self._build_env().get("LLAMA_CPP_IMAGE"),
                container_name=str(_RUNTIME_CONFIG["container_name"]),
            )
        except Exception:
            logger.warning("Failed to inspect llama.cpp runtime metadata for Gemma.", exc_info=True)
            return
        logger.info(
            "Gemma runtime ready: image=%s digest=%s version=%s",
            runtime.get("llama_cpp_image", ""),
            runtime.get("llama_cpp_digest", ""),
            runtime.get("llama_cpp_version", ""),
        )

    def _build_setup_error(self, detail: str) -> LocalServiceSetupError:
        extra = detail.strip()
        extra = f"{extra}\nRequested engine: Gemma\nExpected endpoint: {_RUNTIME_CONFIG['managed_url']}"
        extra = f"{extra}\nCompose file: {_RUNTIME_CONFIG['compose_file']}"
        return LocalServiceSetupError(
            extra.strip(),
            service_name="Gemma",
            settings_page_name=_RUNTIME_CONFIG["settings_page_name"],
        )

    def _build_connection_error(self, detail: str) -> LocalServiceConnectionError:
        return LocalServiceConnectionError(
            detail.strip(),
            service_name="Gemma",
            settings_page_name=_RUNTIME_CONFIG["settings_page_name"],
        )
