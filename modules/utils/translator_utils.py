import base64
import json
import logging
import re
import jieba
import janome.tokenizer
import numpy as np
from pythainlp.tokenize import word_tokenize
from .textblock import TextBlock
import imkit as imk
from .language_utils import get_language_code, is_no_space_lang
from .text_normalization import canonicalize_ellipsis_runs, remove_invisible_format_chars

logger = logging.getLogger(__name__)


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


def normalize_text_for_translation(text, source_lang, *, ocr_engine=None) -> str:
    del ocr_engine
    normalized = remove_invisible_format_chars(str(text or ""))
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")

    source_lang_code = get_language_code(str(source_lang or "")) or str(source_lang or "")
    if is_no_space_lang(source_lang_code):
        normalized = re.sub(r"[\s\u3000]+", "", normalized)
    else:
        normalized = normalized.replace("\n", " ")
        normalized = re.sub(r"[ \t\f\v\u3000]+", " ", normalized).strip()

    return canonicalize_ellipsis_runs(normalized)


def build_translation_input_json(blk_list: list[TextBlock], source_lang, *, ocr_engine=None):
    payload = {}
    for idx, blk in enumerate(blk_list):
        block_key = f"block_{idx}"
        payload[block_key] = normalize_text_for_translation(
            getattr(blk, "text", ""),
            source_lang,
            ocr_engine=ocr_engine,
        )
    return json.dumps(payload, ensure_ascii=False, indent=4)

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

    translation_dict = json.loads(match.group(0))
    if not isinstance(translation_dict, dict):
        raise ValueError("Translator response JSON was not an object.")

    return translation_dict

def set_texts_from_json(blk_list: list[TextBlock], json_string: str):
    translation_dict = extract_json_object(json_string)

    updated_count = 0
    missing_keys = []
    for idx, blk in enumerate(blk_list):
        block_key = f"block_{idx}"
        if block_key in translation_dict:
            value = translation_dict[block_key]
            blk.translation = value if isinstance(value, str) or value is None else str(value)
            updated_count += 1
        else:
            missing_keys.append(block_key)

    if missing_keys:
        logger.warning(
            "translator response missing %d expected block key(s): %s",
            len(missing_keys),
            ", ".join(missing_keys[:10]),
        )

    if updated_count == 0:
        raise ValueError("Translator response JSON did not contain any expected block keys.")

    return updated_count

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
