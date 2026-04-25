from __future__ import annotations

from dataclasses import asdict, dataclass

OCR_MODE_DEFAULT = "default"
OCR_MODE_BEST_LOCAL = "best_local"
OCR_MODE_MICROSOFT = "microsoft_ocr"
OCR_MODE_GOOGLE = "google_cloud_vision"
OCR_MODE_GEMINI = "gemini_2_0_flash"
OCR_MODE_PADDLE_VL = "paddleocr_vl"
OCR_MODE_HUNYUAN = "hunyuanocr"
OCR_MODE_MANGALMM = "mangalmm"

OCR_DEFAULT_LABEL = "Default (existing auto: MangaOCR / PPOCR / Pororo...)"
OCR_OPTIMAL_LABEL = "Optimal (HunyuanOCR / PaddleOCR VL)"
_LEGACY_OPTIMAL_PLUS_MODE = "best_local_plus"
_LEGACY_OPTIMAL_PLUS_LABEL = "Optimal+ (HunyuanOCR / MangaLMM / PaddleOCR VL)"

OCR_MODE_OPTIONS: tuple[tuple[str, str], ...] = (
    (OCR_MODE_DEFAULT, OCR_DEFAULT_LABEL),
    (OCR_MODE_BEST_LOCAL, OCR_OPTIMAL_LABEL),
    (OCR_MODE_MICROSOFT, "Microsoft OCR"),
    (OCR_MODE_GOOGLE, "Google Cloud Vision"),
    (OCR_MODE_GEMINI, "Gemini-2.0-Flash"),
    (OCR_MODE_PADDLE_VL, "PaddleOCR VL"),
    (OCR_MODE_HUNYUAN, "HunyuanOCR"),
    (OCR_MODE_MANGALMM, "MangaLMM"),
)

OCR_MODE_TO_ENGINE: dict[str, str] = {
    OCR_MODE_DEFAULT: "Default",
    OCR_MODE_MICROSOFT: "Microsoft OCR",
    OCR_MODE_GOOGLE: "Google Cloud Vision",
    OCR_MODE_GEMINI: "Gemini-2.0-Flash",
    OCR_MODE_PADDLE_VL: "PaddleOCR VL",
    OCR_MODE_HUNYUAN: "HunyuanOCR",
    OCR_MODE_MANGALMM: "MangaLMM",
}

LOCAL_OCR_ENGINES = frozenset({"PaddleOCR VL", "HunyuanOCR", "MangaLMM"})
STAGE_BATCHED_WORKFLOW_MODE = "stage_batched_pipeline"
LEGACY_PAGE_WORKFLOW_MODE = "legacy_page_pipeline"
GEMMA_TRANSLATOR_KEY = "Custom Local Server(Gemma)"
_TRANSLATOR_ALIASES: dict[str, str] = {
    GEMMA_TRANSLATOR_KEY: GEMMA_TRANSLATOR_KEY,
    "Custom Local Server": GEMMA_TRANSLATOR_KEY,
    "gemma_local": GEMMA_TRANSLATOR_KEY,
}
WORKFLOW_MODE_STAGE_BATCHED_LABEL = "Stage-Batched Pipeline (Recommended)"
WORKFLOW_MODE_LEGACY_LABEL = "Legacy Page Pipeline (Legacy)"
WORKFLOW_MODE_OPTIONS: tuple[tuple[str, str], ...] = (
    (STAGE_BATCHED_WORKFLOW_MODE, WORKFLOW_MODE_STAGE_BATCHED_LABEL),
    (LEGACY_PAGE_WORKFLOW_MODE, WORKFLOW_MODE_LEGACY_LABEL),
)

_ALIASES: dict[str, str] = {
    OCR_MODE_DEFAULT: OCR_MODE_DEFAULT,
    "Default": OCR_MODE_DEFAULT,
    OCR_DEFAULT_LABEL: OCR_MODE_DEFAULT,
    OCR_MODE_BEST_LOCAL: OCR_MODE_BEST_LOCAL,
    OCR_OPTIMAL_LABEL: OCR_MODE_BEST_LOCAL,
    _LEGACY_OPTIMAL_PLUS_MODE: OCR_MODE_BEST_LOCAL,
    _LEGACY_OPTIMAL_PLUS_LABEL: OCR_MODE_BEST_LOCAL,
    OCR_MODE_MICROSOFT: OCR_MODE_MICROSOFT,
    "Microsoft OCR": OCR_MODE_MICROSOFT,
    OCR_MODE_GOOGLE: OCR_MODE_GOOGLE,
    "Google Cloud Vision": OCR_MODE_GOOGLE,
    OCR_MODE_GEMINI: OCR_MODE_GEMINI,
    "Gemini-2.0-Flash": OCR_MODE_GEMINI,
    OCR_MODE_PADDLE_VL: OCR_MODE_PADDLE_VL,
    "PaddleOCR VL": OCR_MODE_PADDLE_VL,
    OCR_MODE_HUNYUAN: OCR_MODE_HUNYUAN,
    "HunyuanOCR": OCR_MODE_HUNYUAN,
    OCR_MODE_MANGALMM: OCR_MODE_MANGALMM,
    "MangaLMM": OCR_MODE_MANGALMM,
}
_WORKFLOW_ALIASES: dict[str, str] = {
    STAGE_BATCHED_WORKFLOW_MODE: STAGE_BATCHED_WORKFLOW_MODE,
    WORKFLOW_MODE_STAGE_BATCHED_LABEL: STAGE_BATCHED_WORKFLOW_MODE,
    LEGACY_PAGE_WORKFLOW_MODE: LEGACY_PAGE_WORKFLOW_MODE,
    WORKFLOW_MODE_LEGACY_LABEL: LEGACY_PAGE_WORKFLOW_MODE,
}


def normalize_ocr_mode(value: str | None) -> str:
    raw = str(value or "").strip()
    if not raw:
        return OCR_MODE_DEFAULT
    return _ALIASES.get(raw, OCR_MODE_DEFAULT)


def normalize_workflow_mode(value: str | None) -> str:
    raw = str(value or "").strip()
    if not raw:
        return STAGE_BATCHED_WORKFLOW_MODE
    return _WORKFLOW_ALIASES.get(raw, STAGE_BATCHED_WORKFLOW_MODE)


def is_chinese_source_language(source_lang_english: str | None) -> bool:
    normalized = str(source_lang_english or "").strip().casefold()
    return normalized in {
        "chinese",
        "simplified chinese",
        "traditional chinese",
        "chinese (simplified)",
        "chinese (traditional)",
    }


def is_japanese_source_language(source_lang_english: str | None) -> bool:
    return str(source_lang_english or "").strip().casefold() == "japanese"


def resolve_ocr_engine(mode: str | None, source_lang_english: str | None) -> str:
    normalized = normalize_ocr_mode(mode)
    if normalized == OCR_MODE_BEST_LOCAL:
        if is_chinese_source_language(source_lang_english):
            return "HunyuanOCR"
        return "PaddleOCR VL"
    return OCR_MODE_TO_ENGINE.get(normalized, "Default")


def is_local_ocr_engine(engine_key: str | None) -> bool:
    return str(engine_key or "").strip() in LOCAL_OCR_ENGINES


def normalize_translator_key(value: str | None) -> str:
    raw = str(value or "").strip()
    return _TRANSLATOR_ALIASES.get(raw, raw)


@dataclass(frozen=True)
class OCRStageRoutingPolicy:
    workflow_mode: str
    normalized_ocr_mode: str
    source_lang_english: str
    translator: str
    ocr_stage_enabled: bool
    primary_ocr_engine: str
    resident_ocr_engines: tuple[str, ...]
    requires_sidecar_collection: bool
    selector_enabled: bool
    translation_stage_runtime: str
    post_ocr_shutdown_targets: tuple[str, ...]
    stage_batched_supported: bool
    fallback_workflow_mode: str
    unsupported_reason: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def resolve_stage_batched_ocr_policy(
    workflow_mode: str | None,
    ocr_mode: str | None,
    source_lang_english: str | None,
    translator: str | None,
) -> OCRStageRoutingPolicy:
    workflow_mode_text = str(workflow_mode or "").strip() or LEGACY_PAGE_WORKFLOW_MODE
    normalized_mode = normalize_ocr_mode(ocr_mode)
    source_lang_text = str(source_lang_english or "").strip() or "Unknown"
    translator_text = normalize_translator_key(translator)

    if workflow_mode_text != STAGE_BATCHED_WORKFLOW_MODE:
        primary_engine = resolve_ocr_engine(normalized_mode, source_lang_text)
        return OCRStageRoutingPolicy(
            workflow_mode=workflow_mode_text,
            normalized_ocr_mode=normalized_mode,
            source_lang_english=source_lang_text,
            translator=translator_text,
            ocr_stage_enabled=False,
            primary_ocr_engine=primary_engine,
            resident_ocr_engines=(primary_engine,) if is_local_ocr_engine(primary_engine) else (),
            requires_sidecar_collection=False,
            selector_enabled=False,
            translation_stage_runtime="legacy_runtime_lifecycle",
            post_ocr_shutdown_targets=(),
            stage_batched_supported=False,
            fallback_workflow_mode=LEGACY_PAGE_WORKFLOW_MODE,
            unsupported_reason="workflow_mode_is_not_stage_batched_pipeline",
        )

    if translator_text != GEMMA_TRANSLATOR_KEY:
        return OCRStageRoutingPolicy(
            workflow_mode=workflow_mode_text,
            normalized_ocr_mode=normalized_mode,
            source_lang_english=source_lang_text,
            translator=translator_text,
            ocr_stage_enabled=False,
            primary_ocr_engine="",
            resident_ocr_engines=(),
            requires_sidecar_collection=False,
            selector_enabled=False,
            translation_stage_runtime="",
            post_ocr_shutdown_targets=(),
            stage_batched_supported=False,
            fallback_workflow_mode=LEGACY_PAGE_WORKFLOW_MODE,
            unsupported_reason="stage_batched_pipeline_currently_requires_gemma_translator",
        )

    primary_engine = resolve_ocr_engine(normalized_mode, source_lang_text)
    resident_engines: tuple[str, ...] = ()
    requires_sidecar_collection = False
    selector_enabled = False

    if normalized_mode == OCR_MODE_PADDLE_VL:
        resident_engines = ("PaddleOCR VL",)
        primary_engine = "PaddleOCR VL"
    elif normalized_mode == OCR_MODE_HUNYUAN:
        resident_engines = ("HunyuanOCR",)
        primary_engine = "HunyuanOCR"
    elif normalized_mode == OCR_MODE_MANGALMM:
        resident_engines = ("MangaLMM",)
        primary_engine = "MangaLMM"
    elif normalized_mode == OCR_MODE_BEST_LOCAL:
        if is_chinese_source_language(source_lang_text):
            resident_engines = ("HunyuanOCR",)
            primary_engine = "HunyuanOCR"
        else:
            resident_engines = ("PaddleOCR VL",)
            primary_engine = "PaddleOCR VL"
    else:
        return OCRStageRoutingPolicy(
            workflow_mode=workflow_mode_text,
            normalized_ocr_mode=normalized_mode,
            source_lang_english=source_lang_text,
            translator=translator_text,
            ocr_stage_enabled=False,
            primary_ocr_engine=primary_engine,
            resident_ocr_engines=(),
            requires_sidecar_collection=False,
            selector_enabled=False,
            translation_stage_runtime="",
            post_ocr_shutdown_targets=(),
            stage_batched_supported=False,
            fallback_workflow_mode=LEGACY_PAGE_WORKFLOW_MODE,
            unsupported_reason="stage_batched_pipeline_supports_only_local_ocr_modes_for_now",
        )

    return OCRStageRoutingPolicy(
        workflow_mode=workflow_mode_text,
        normalized_ocr_mode=normalized_mode,
        source_lang_english=source_lang_text,
        translator=translator_text,
        ocr_stage_enabled=True,
        primary_ocr_engine=primary_engine,
        resident_ocr_engines=resident_engines,
        requires_sidecar_collection=requires_sidecar_collection,
        selector_enabled=selector_enabled,
        translation_stage_runtime="gemma_local_server",
        post_ocr_shutdown_targets=resident_engines,
        stage_batched_supported=True,
        fallback_workflow_mode="",
        unsupported_reason="",
    )
