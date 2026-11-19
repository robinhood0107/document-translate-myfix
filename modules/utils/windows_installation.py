from __future__ import annotations

import json
import os
from pathlib import Path

from .paths import get_repo_root
from .download import (
    CORE_APPLICATION_MODELS,
    MIT48_AR_APPLICATION_MODELS,
    MIT48_CTC_APPLICATION_MODELS,
    OPTIONAL_APPLICATION_MODELS,
    ModelID,
)


CORE_APPLICATION_MODEL_IDS = frozenset(model.value for model in CORE_APPLICATION_MODELS)
AOT_APPLICATION_MODEL_IDS = frozenset(
    {ModelID.AOT_TORCH.value, ModelID.AOT_ONNX.value}
)
DEFAULT_OCR_APPLICATION_MODEL_IDS = frozenset(
    model.value
    for model in OPTIONAL_APPLICATION_MODELS
    if model not in {ModelID.AOT_TORCH, ModelID.AOT_ONNX}
)


def active_windows_runtime() -> str:
    value = str(os.environ.get("COMIC_WINDOWS_RUNTIME", "") or "").strip().lower()
    return value if value in {"cuda12", "cuda13"} else ""


def active_windows_install_state() -> dict:
    runtime = active_windows_runtime()
    if not runtime:
        return {}
    path = Path(get_repo_root()) / ".comic-bootstrap" / f"install-state-{runtime}.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def active_windows_install_tier() -> str:
    value = str(active_windows_install_state().get("provisioned_tier") or "").lower()
    return value if value in {"core", "full"} else ""


def active_windows_prepared_model_ids(state: dict | None = None) -> frozenset[str]:
    payload = state if isinstance(state, dict) else active_windows_install_state()
    records = payload.get("application_models")
    if not isinstance(records, list):
        return frozenset()
    return frozenset(
        str(record.get("id") or "")
        for record in records
        if isinstance(record, dict)
        and isinstance(record.get("files"), list)
        and bool(record.get("files"))
    )


def active_windows_managed_runtime_names(state: dict | None = None) -> frozenset[str]:
    payload = state if isinstance(state, dict) else active_windows_install_state()
    records = payload.get("managed_runtimes")
    if not isinstance(records, list):
        return frozenset()
    return frozenset(
        str(record.get("runtime_name") or "")
        for record in records
        if isinstance(record, dict) and str(record.get("name") or "")
    )


def _selected_mit48_application_model_ids(ocr_engine: str) -> set[str]:
    if ocr_engine == "MIT 48px AR":
        return {model.value for model in MIT48_AR_APPLICATION_MODELS}
    if ocr_engine == "MIT 48px CTC":
        return {model.value for model in MIT48_CTC_APPLICATION_MODELS}
    return set()


def _assert_selected_mit48_seal(state: dict, model_ids: set[str]) -> None:
    from scripts.windows_install_state import (
        SCHEMA_VERSION,
        InstallStateError,
        _validate_application_models,
    )

    try:
        version = state.get("schema_version")
        try:
            version_matches = (
                not isinstance(version, bool)
                and isinstance(version, (int, str))
                and int(version) == SCHEMA_VERSION
            )
        except (TypeError, ValueError, OverflowError):
            version_matches = False
        if not version_matches:
            raise InstallStateError("Windows setup state schema is invalid.")
        _validate_application_models(
            state,
            model_ids=model_ids,
            allow_digest_fallback=True,
        )
    except InstallStateError as exc:
        raise RuntimeError(
            "The selected MIT 48px OCR assets are missing or invalid. "
            "Run the matching setup_full BAT before starting this job. "
            f"({exc})"
        ) from exc


def assert_selected_mit48_assets_installed(
    settings_page,
    source_lang_english: str,
) -> None:
    if not active_windows_runtime():
        return
    from modules.ocr.selection import resolve_ocr_engine

    model_ids = _selected_mit48_application_model_ids(
        resolve_ocr_engine(
            settings_page.get_tool_selection("ocr"),
            source_lang_english,
        )
    )
    if model_ids:
        _assert_selected_mit48_seal(active_windows_install_state(), model_ids)


def assert_selected_windows_models_installed(settings_page, source_lang_english: str) -> None:
    """Require exact setup-sealed records before any page is decoded."""

    if not active_windows_runtime():
        return
    from modules.ocr.selection import resolve_ocr_engine
    from modules.utils.inpainting_runtime import normalize_inpainter_key

    state = active_windows_install_state()
    sealed_models = active_windows_prepared_model_ids(state)
    sealed_runtimes = active_windows_managed_runtime_names(state)
    ocr_engine = resolve_ocr_engine(
        settings_page.get_tool_selection("ocr"),
        source_lang_english,
    )
    selected_mit48_models = _selected_mit48_application_model_ids(ocr_engine)
    required_models = set(CORE_APPLICATION_MODEL_IDS)
    required_runtimes: set[str] = set()
    optional_required = False
    if ocr_engine == "Default":
        required_models.update(DEFAULT_OCR_APPLICATION_MODEL_IDS)
        optional_required = True
    elif ocr_engine == "HunyuanOCR":
        required_runtimes.add("HunyuanOCR-llama.cpp")
    elif ocr_engine == "PaddleOCR VL":
        required_runtimes.add("PaddleOCR-VL-llama.cpp")
    elif ocr_engine == "PaddleOCR VL Spotting":
        required_runtimes.add("PaddleOCR-VL-Spotting-llama.cpp")
        optional_required = True
    elif ocr_engine == "MangaLMM":
        required_runtimes.add("MangaLMM-llama.cpp")
        optional_required = True
    elif selected_mit48_models:
        required_models.update(selected_mit48_models)
        optional_required = True

    inpainter = normalize_inpainter_key(settings_page.get_tool_selection("inpainter"))
    if inpainter == "AOT":
        required_models.update(AOT_APPLICATION_MODEL_IDS)
        optional_required = True

    translator = str(settings_page.get_tool_selection("translator") or "").strip()
    if translator in {"Custom Local Server(Gemma)", "Custom Local Server", "gemma_local"}:
        required_runtimes.add("Gemma")

    missing_models = sorted(required_models.difference(sealed_models))
    missing_runtimes = sorted(required_runtimes.difference(sealed_runtimes))
    if not missing_models and not missing_runtimes:
        if selected_mit48_models:
            _assert_selected_mit48_seal(state, selected_mit48_models)
        return

    setup_name = "setup_full" if optional_required else "setup"
    details: list[str] = []
    if missing_models:
        details.append("models=" + ", ".join(missing_models))
    if missing_runtimes:
        details.append("runtimes=" + ", ".join(missing_runtimes))
    if optional_required:
        raise RuntimeError(
            "The selected optional runtime is not fully sealed. "
            f"Run the matching {setup_name} BAT before starting this job "
            f"({'; '.join(details)})."
        )
    raise RuntimeError(
        "The required core runtime seal is incomplete. "
        f"Run the matching {setup_name} BAT before starting this job "
        f"({'; '.join(details)})."
    )
