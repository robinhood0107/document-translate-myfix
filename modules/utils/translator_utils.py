import base64
import json
import re
import jieba
import janome.tokenizer
import numpy as np
from pythainlp.tokenize import word_tokenize
from .textblock import TextBlock
import imkit as imk
from .language_utils import get_language_code, is_no_space_lang
from .text_normalization import (
    canonicalize_ellipsis_runs,
    normalize_decorative_ocr_text,
    remove_invisible_format_chars,
)

MODEL_MAP = {
    "Custom Service": "",
    "Custom Local Server": "",
    "Custom Local Server(Gemma)": "",
    "Custom": "",  
    "Deepseek-v3": "deepseek-chat", 
    "GPT-4.1": "gpt-4.1",
    "GPT-4.1-mini": "gpt-4.1-mini",
    "Claude-4.6-Sonnet": "claude-sonnet-4-6",
    "Claude-4.5-Haiku": "claude-haiku-4-5-20251001",
    "Gemini-2.0-Flash": "gemini-2.0-flash",
    "Gemini-3.0-Flash": "gemini-3-flash-preview",
    "Gemini-2.5-Pro": "gemini-2.5-pro"
}

def encode_image_array(img_array: np.ndarray):
    img_bytes = imk.encode_image(img_array, ".png")
    return base64.b64encode(img_bytes).decode('utf-8')

def get_raw_text(blk_list: list[TextBlock]):
    rw_txts_dict = {}
    for idx, blk in enumerate(blk_list):
        block_key = f"block_{idx}"
        rw_txts_dict[block_key] = blk.text
    
    raw_texts_json = json.dumps(rw_txts_dict, ensure_ascii=False, indent=4)
    
    return raw_texts_json


def _uses_mangalmm_noise_filter(ocr_engine) -> bool:
    return str(ocr_engine or "").strip() == "MangaLMM"


def normalize_text_for_translation(text, source_lang, *, ocr_engine=None) -> str:
    if _uses_mangalmm_noise_filter(ocr_engine):
        normalized = normalize_decorative_ocr_text(str(text or ""))
    else:
        normalized = remove_invisible_format_chars(str(text or ""))
        normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")

    source_lang_code = get_language_code(str(source_lang or "")) or str(source_lang or "")
    if is_no_space_lang(source_lang_code):
        normalized = re.sub(r"[\s\u3000]+", "", normalized)
    else:
        normalized = normalized.replace("\n", " ")
        normalized = re.sub(r"[ \t\f\v\u3000]+", " ", normalized).strip()

    return canonicalize_ellipsis_runs(normalized)


def _invalid_translation_response(kind: str, blocks: list[TextBlock]) -> ValueError:
    error = ValueError("Invalid translation response")
    error.block_failure_kind = kind
    error.block_ids = [
        str(block_id)
        for block in blocks
        if (block_id := getattr(block, "block_id", ""))
    ]
    return error


def build_translation_input_request(
    blk_list: list[TextBlock],
    source_lang,
    *,
    ocr_engine=None,
    include_blank: bool = True,
) -> tuple[str, tuple[int, ...]]:
    payload = {}
    request_indices = []
    for idx, blk in enumerate(blk_list):
        source_text = str(getattr(blk, "text", "") or "")
        if not include_blank and not source_text.strip():
            continue
        normalized_text = normalize_text_for_translation(
            source_text,
            source_lang,
            ocr_engine=ocr_engine,
        )
        if not include_blank and not normalized_text.strip():
            error = _invalid_translation_response(
                "translation_input_empty_after_normalization",
                [blk],
            )
            error.invalid_input_keys = [f"block_{idx}"]
            raise error
        payload[f"block_{idx}"] = normalized_text
        request_indices.append(idx)
    return (
        json.dumps(payload, ensure_ascii=False, indent=4),
        tuple(request_indices),
    )


def build_translation_input_json(
    blk_list: list[TextBlock],
    source_lang,
    *,
    ocr_engine=None,
    include_blank: bool = True,
) -> str:
    return build_translation_input_request(
        blk_list,
        source_lang,
        ocr_engine=ocr_engine,
        include_blank=include_blank,
    )[0]

def get_raw_translation(blk_list: list[TextBlock]):
    rw_translations_dict = {}
    for idx, blk in enumerate(blk_list):
        block_key = f"block_{idx}"
        rw_translations_dict[block_key] = blk.translation
    
    raw_translations_json = json.dumps(rw_translations_dict, ensure_ascii=False, indent=4)
    
    return raw_translations_json


def extract_json_object(json_string: str) -> dict:
    match = re.search(r"\{[\s\S]*\}", json_string)
    if not match:
        raise ValueError("Translator response did not contain a JSON object.")

    def reject_duplicate_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise _DuplicateJsonKeyError(key)
            result[key] = value
        return result

    translation_dict = json.loads(
        match.group(0),
        object_pairs_hook=reject_duplicate_keys,
    )
    if not isinstance(translation_dict, dict):
        raise ValueError("Translator response JSON was not an object.")

    return translation_dict


class _DuplicateJsonKeyError(ValueError):
    pass


def set_texts_from_json(
    blk_list: list[TextBlock],
    json_string: str,
    *,
    request_indices: tuple[int, ...] | None = None,
):
    if request_indices is None:
        request_indices = tuple(range(len(blk_list)))
    expected_blocks = {
        f"block_{idx}": blk_list[idx]
        for idx in request_indices
    }
    if not expected_blocks:
        return 0

    if not isinstance(json_string, str):
        raise _invalid_translation_response(
            "translation_response_invalid_content",
            list(expected_blocks.values()),
        )

    if not json_string.strip():
        raise _invalid_translation_response(
            "translation_response_empty",
            list(expected_blocks.values()),
        )

    try:
        translation_dict = extract_json_object(json_string)
    except _DuplicateJsonKeyError as exc:
        error = _invalid_translation_response(
            "translation_response_duplicate_ids",
            list(expected_blocks.values()),
        )
        error.duplicate_response_keys = [str(exc)]
        raise error from exc
    except json.JSONDecodeError as exc:
        raw_response = str(json_string or "")
        kind = (
            "translation_response_truncated"
            if raw_response.count("{") > raw_response.count("}")
            or exc.msg.startswith("Unterminated string")
            else "translation_response_invalid_json"
        )
        raise _invalid_translation_response(
            kind,
            list(expected_blocks.values()),
        ) from exc
    except ValueError as exc:
        raw_response = str(json_string or "")
        if raw_response.count("{") > raw_response.count("}"):
            kind = "translation_response_truncated"
        else:
            kind = "translation_response_invalid_json"
        raise _invalid_translation_response(
            kind,
            list(expected_blocks.values()),
        ) from exc

    if not translation_dict:
        raise _invalid_translation_response(
            "translation_response_empty",
            list(expected_blocks.values()),
        )

    expected_keys = set(expected_blocks)
    response_keys = set(translation_dict)
    missing_keys = expected_keys - response_keys
    unexpected_keys = response_keys - expected_keys
    if missing_keys or unexpected_keys:
        if missing_keys and unexpected_keys:
            kind = "translation_response_id_mismatch"
        elif missing_keys:
            kind = "translation_response_missing_ids"
        else:
            kind = "translation_response_unexpected_ids"
        error = _invalid_translation_response(
            kind,
            list(expected_blocks.values()),
        )
        error.missing_response_keys = sorted(missing_keys)
        error.unexpected_response_keys = sorted(unexpected_keys)
        raise error

    invalid_values = [
        key
        for key in expected_blocks
        if not isinstance(translation_dict[key], str)
        or not translation_dict[key].strip()
    ]
    if invalid_values:
        error = _invalid_translation_response(
            "translation_response_invalid_values",
            list(expected_blocks.values()),
        )
        error.invalid_response_keys = invalid_values
        raise error

    for idx, blk in enumerate(blk_list):
        block_key = f"block_{idx}"
        if block_key in expected_blocks:
            blk.translation = translation_dict[block_key]

    return len(expected_blocks)

def set_upper_case(blk_list: list[TextBlock], upper_case: bool):
    for blk in blk_list:
        translation = blk.translation
        if translation is None:
            continue
        if upper_case and not translation.isupper():
            blk.translation = translation.upper() 
        elif not upper_case and translation.isupper():
            blk.translation = translation.lower().capitalize()
        else:
            blk.translation = translation

def get_chinese_tokens(text):
    return list(jieba.cut(text, cut_all=False))

def get_japanese_tokens(text):
    tokenizer = janome.tokenizer.Tokenizer()
    return [token.surface for token in tokenizer.tokenize(text)]

def format_translations(blk_list: list[TextBlock], trg_lng_cd: str, upper_case: bool = True):
    for blk in blk_list:
        translation = blk.translation
        trg_lng_code_lower = trg_lng_cd.lower()
        seg_result = []

        if 'zh' in trg_lng_code_lower:
            seg_result = get_chinese_tokens(translation)

        elif 'ja' in trg_lng_code_lower:
            seg_result = get_japanese_tokens(translation)

        elif 'th' in trg_lng_code_lower:
            seg_result = word_tokenize(translation)

        if seg_result:
            blk.translation = ''.join(word if word in ['.', ','] else f' {word}' for word in seg_result).lstrip()
        else:
            # apply casing/formatting for this single block when no segmentation is done
            if translation is None:
                continue
            if upper_case and not translation.isupper():
                blk.translation = translation.upper()
            elif not upper_case and translation.isupper():
                blk.translation = translation.lower().capitalize()
            else:
                blk.translation = translation

def is_there_text(blk_list: list[TextBlock]) -> bool:
    return any(blk.text for blk in blk_list)
