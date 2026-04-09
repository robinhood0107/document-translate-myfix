from __future__ import annotations

OCR_MODE_DEFAULT = "default"
OCR_MODE_BEST_LOCAL = "best_local"
OCR_MODE_MICROSOFT = "microsoft_ocr"
OCR_MODE_GOOGLE = "google_cloud_vision"
OCR_MODE_GEMINI = "gemini_2_0_flash"
OCR_MODE_PADDLE_VL = "paddleocr_vl"
OCR_MODE_HUNYUAN = "hunyuanocr"

OCR_DEFAULT_LABEL = "Default (existing auto: MangaOCR / PPOCR / Pororo...)"
OCR_OPTIMAL_LABEL = "Optimal (HunyuanOCR / PaddleOCR VL)"

OCR_MODE_OPTIONS: tuple[tuple[str, str], ...] = (
    (OCR_MODE_DEFAULT, OCR_DEFAULT_LABEL),
    (OCR_MODE_BEST_LOCAL, OCR_OPTIMAL_LABEL),
    (OCR_MODE_MICROSOFT, "Microsoft OCR"),
    (OCR_MODE_GOOGLE, "Google Cloud Vision"),
    (OCR_MODE_GEMINI, "Gemini-2.0-Flash"),
    (OCR_MODE_PADDLE_VL, "PaddleOCR VL"),
    (OCR_MODE_HUNYUAN, "HunyuanOCR"),
)

OCR_MODE_TO_ENGINE: dict[str, str] = {
    OCR_MODE_DEFAULT: "Default",
    OCR_MODE_MICROSOFT: "Microsoft OCR",
    OCR_MODE_GOOGLE: "Google Cloud Vision",
    OCR_MODE_GEMINI: "Gemini-2.0-Flash",
    OCR_MODE_PADDLE_VL: "PaddleOCR VL",
    OCR_MODE_HUNYUAN: "HunyuanOCR",
}

LOCAL_OCR_ENGINES = frozenset({"PaddleOCR VL", "HunyuanOCR"})

_ALIASES: dict[str, str] = {
    OCR_MODE_DEFAULT: OCR_MODE_DEFAULT,
    "Default": OCR_MODE_DEFAULT,
    OCR_DEFAULT_LABEL: OCR_MODE_DEFAULT,
    OCR_MODE_BEST_LOCAL: OCR_MODE_BEST_LOCAL,
    OCR_OPTIMAL_LABEL: OCR_MODE_BEST_LOCAL,
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
}


def normalize_ocr_mode(value: str | None) -> str:
    raw = str(value or "").strip()
    if not raw:
        return OCR_MODE_DEFAULT
    return _ALIASES.get(raw, OCR_MODE_DEFAULT)


def is_chinese_source_language(source_lang_english: str | None) -> bool:
    normalized = str(source_lang_english or "").strip().casefold()
    return normalized in {
        "chinese",
        "simplified chinese",
        "traditional chinese",
        "chinese (simplified)",
        "chinese (traditional)",
    }


def resolve_ocr_engine(mode: str | None, source_lang_english: str | None) -> str:
    normalized = normalize_ocr_mode(mode)
    if normalized == OCR_MODE_BEST_LOCAL:
        if is_chinese_source_language(source_lang_english):
            return "HunyuanOCR"
        return "PaddleOCR VL"
    return OCR_MODE_TO_ENGINE.get(normalized, "Default")


def is_local_ocr_engine(engine_key: str | None) -> bool:
    return str(engine_key or "").strip() in LOCAL_OCR_ENGINES
