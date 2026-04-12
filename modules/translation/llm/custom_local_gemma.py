from __future__ import annotations

from typing import Any
import json
import logging
import os

import numpy as np
import requests

from .base import BaseLLMTranslation
from ...utils.textblock import TextBlock
from ...utils.translator_utils import extract_json_object, get_raw_text

logger = logging.getLogger(__name__)

DEFAULT_GEMMA_LOCAL_ENDPOINT = "http://127.0.0.1:18080/v1"
DEFAULT_GEMMA_LOCAL_MODEL = "gemma-4-26B-IQ4_NL.gguf"
DEFAULT_GEMMA_CHUNK_SIZE = 6
DEFAULT_GEMMA_MAX_COMPLETION_TOKENS = 512
DEFAULT_GEMMA_REQUEST_TIMEOUT_SEC = 180
DEFAULT_GEMMA_TRANSLATION_TEMPERATURE = 0.7
DEFAULT_GEMMA_TRANSLATION_TOP_K = 64
DEFAULT_GEMMA_TRANSLATION_TOP_P = 0.95
DEFAULT_GEMMA_TRANSLATION_MIN_P = 0.0
DEFAULT_GEMMA_RESPONSE_FORMAT_MODE = "json_schema"
DEFAULT_GEMMA_RESPONSE_SCHEMA_MODE = "blocks"
DEFAULT_GEMMA_THINK_BRIEFLY_PROMPT = False
DEFAULT_GEMMA_PROMPT_PROFILE = "gemma4_balanced"
STRICT_GEMMA_PROMPT_PROFILE = "gemma4_strict_json"
GEMMA_PROMPT_PROFILES = {
    "legacy": "legacy",
    "gemma4_balanced": "gemma4_balanced",
    "gemma4_strict_json": "gemma4_strict_json",
}
GEMMA_RESPONSE_FORMAT_MODES = {"json_object", "json_schema"}
GEMMA_RESPONSE_SCHEMA_MODES = {"blocks"}


class GemmaLocalServerResponseError(RuntimeError):
    """Raised when the local Gemma server returns an unusable translation."""

    def __init__(self, message: str, *, strict_retryable: bool = False):
        super().__init__(message)
        self.strict_retryable = strict_retryable


class GemmaLocalServerTruncatedError(GemmaLocalServerResponseError):
    """Raised when the local Gemma server runs out of output budget."""


class CustomLocalGemmaTranslation(BaseLLMTranslation):
    """Translation engine specialized for the local Gemma llama.cpp server."""

    def __init__(self):
        super().__init__()
        self.api_base_url = DEFAULT_GEMMA_LOCAL_ENDPOINT
        self.model = DEFAULT_GEMMA_LOCAL_MODEL
        self.chunk_size = DEFAULT_GEMMA_CHUNK_SIZE
        self.raw_response_logging = False
        self.translation_mode_label = "Custom Local Server(Gemma)"
        self.top_k = DEFAULT_GEMMA_TRANSLATION_TOP_K
        self.min_p = DEFAULT_GEMMA_TRANSLATION_MIN_P
        self.response_format_mode = DEFAULT_GEMMA_RESPONSE_FORMAT_MODE
        self.response_schema_mode = DEFAULT_GEMMA_RESPONSE_SCHEMA_MODE
        self.think_briefly_prompt = DEFAULT_GEMMA_THINK_BRIEFLY_PROMPT
        self.prompt_profile = DEFAULT_GEMMA_PROMPT_PROFILE
        self.last_benchmark_stats = self._new_benchmark_stats()
        self._current_benchmark_stats = self._new_benchmark_stats()

    @staticmethod
    def _new_benchmark_stats() -> dict[str, int]:
        return {
            "gemma_json_retry_count": 0,
            "gemma_chunk_retry_events": 0,
            "gemma_truncated_count": 0,
            "gemma_empty_content_count": 0,
            "gemma_missing_key_count": 0,
            "gemma_reasoning_without_final_count": 0,
            "gemma_schema_validation_fail_count": 0,
        }

    @staticmethod
    def _env_or_config_float(config: dict[str, Any], key: str, env_name: str, default: float) -> float:
        raw_value = os.environ.get(env_name)
        if raw_value is None:
            raw_value = config.get(key, default)
        try:
            return float(raw_value)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _env_or_config_int(config: dict[str, Any], key: str, env_name: str, default: int) -> int:
        raw_value = os.environ.get(env_name)
        if raw_value is None:
            raw_value = config.get(key, default)
        try:
            return int(raw_value)
        except (TypeError, ValueError):
            return int(default)

    @staticmethod
    def _env_or_config_str(config: dict[str, Any], key: str, env_name: str, default: str) -> str:
        raw_value = os.environ.get(env_name)
        if raw_value is None:
            raw_value = config.get(key, default)
        if raw_value is None:
            return default
        return str(raw_value).strip() or default

    @staticmethod
    def _env_or_config_bool(config: dict[str, Any], key: str, env_name: str, default: bool) -> bool:
        raw_value = os.environ.get(env_name)
        if raw_value is None:
            raw_value = config.get(key, default)
        if isinstance(raw_value, bool):
            return raw_value
        normalized = str(raw_value).strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        return bool(default)

    @staticmethod
    def _normalize_prompt_profile(raw_value: str) -> str:
        normalized = (raw_value or "").strip().lower()
        if normalized in GEMMA_PROMPT_PROFILES:
            return normalized
        return DEFAULT_GEMMA_PROMPT_PROFILE

    @staticmethod
    def _normalize_response_format_mode(raw_value: str) -> str:
        normalized = (raw_value or "").strip().lower()
        if normalized in GEMMA_RESPONSE_FORMAT_MODES:
            return normalized
        return DEFAULT_GEMMA_RESPONSE_FORMAT_MODE

    @staticmethod
    def _normalize_response_schema_mode(raw_value: str) -> str:
        normalized = (raw_value or "").strip().lower()
        if normalized in GEMMA_RESPONSE_SCHEMA_MODES:
            return normalized
        return DEFAULT_GEMMA_RESPONSE_SCHEMA_MODE

    def initialize(
        self,
        settings: Any,
        source_lang: str,
        target_lang: str,
        translator_key: str,
        **kwargs,
    ) -> None:
        super().initialize(settings, source_lang, target_lang, **kwargs)

        credentials = settings.get_credentials(settings.ui.tr("Custom Local Server(Gemma)"))
        gemma_settings = settings.get_gemma_local_server_settings()

        self.api_base_url = credentials.get("api_url", "").strip().rstrip("/")
        self.model = credentials.get("model", "").strip()
        self.chunk_size = int(gemma_settings.get("chunk_size", DEFAULT_GEMMA_CHUNK_SIZE))
        self.max_tokens = int(
            gemma_settings.get(
                "max_completion_tokens",
                DEFAULT_GEMMA_MAX_COMPLETION_TOKENS,
            )
        )
        self.timeout = int(
            gemma_settings.get(
                "request_timeout_sec",
                DEFAULT_GEMMA_REQUEST_TIMEOUT_SEC,
            )
        )
        self.raw_response_logging = bool(gemma_settings.get("raw_response_logging", False))
        self.temperature = self._env_or_config_float(
            gemma_settings,
            "temperature",
            "CT_GEMMA_TEMPERATURE",
            DEFAULT_GEMMA_TRANSLATION_TEMPERATURE,
        )
        self.top_k = self._env_or_config_int(
            gemma_settings,
            "top_k",
            "CT_GEMMA_TOP_K",
            DEFAULT_GEMMA_TRANSLATION_TOP_K,
        )
        self.top_p = self._env_or_config_float(
            gemma_settings,
            "top_p",
            "CT_GEMMA_TOP_P",
            DEFAULT_GEMMA_TRANSLATION_TOP_P,
        )
        self.min_p = self._env_or_config_float(
            gemma_settings,
            "min_p",
            "CT_GEMMA_MIN_P",
            DEFAULT_GEMMA_TRANSLATION_MIN_P,
        )
        self.prompt_profile = self._normalize_prompt_profile(
            self._env_or_config_str(
                gemma_settings,
                "prompt_profile",
                "CT_GEMMA_PROMPT_PROFILE",
                DEFAULT_GEMMA_PROMPT_PROFILE,
            )
        )
        self.response_format_mode = self._normalize_response_format_mode(
            self._env_or_config_str(
                gemma_settings,
                "response_format_mode",
                "CT_GEMMA_RESPONSE_FORMAT_MODE",
                DEFAULT_GEMMA_RESPONSE_FORMAT_MODE,
            )
        )
        self.response_schema_mode = self._normalize_response_schema_mode(
            self._env_or_config_str(
                gemma_settings,
                "response_schema_mode",
                "CT_GEMMA_RESPONSE_SCHEMA_MODE",
                DEFAULT_GEMMA_RESPONSE_SCHEMA_MODE,
            )
        )
        self.think_briefly_prompt = self._env_or_config_bool(
            gemma_settings,
            "think_briefly_prompt",
            "CT_GEMMA_THINK_BRIEFLY_PROMPT",
            DEFAULT_GEMMA_THINK_BRIEFLY_PROMPT,
        )
        self.img_as_llm_input = False
        self.last_benchmark_stats = self._new_benchmark_stats()

    def translate(
        self,
        blk_list: list[TextBlock],
        image: np.ndarray,
        extra_context: str,
    ) -> list[TextBlock]:
        updated_blocks = 0
        if not blk_list:
            return blk_list

        working_blocks = [blk.deep_copy() for blk in blk_list]
        self._current_benchmark_stats = self._new_benchmark_stats()

        try:
            for start in range(0, len(working_blocks), self.chunk_size):
                chunk = working_blocks[start : start + self.chunk_size]
                updated_blocks += self._translate_chunk_with_retry(chunk, extra_context)

            for original_blk, translated_blk in zip(blk_list, working_blocks):
                original_blk.translation = translated_blk.translation

            logger.info(
                "translation parsed successfully (%s): updated_blocks=%d total_blocks=%d",
                self.translation_mode_label,
                updated_blocks,
                len(blk_list),
            )
            return blk_list
        finally:
            self.last_benchmark_stats = dict(self._current_benchmark_stats)

    def _translate_chunk_with_retry(
        self,
        blk_list: list[TextBlock],
        extra_context: str,
    ) -> int:
        try:
            return self._translate_chunk(blk_list, extra_context, prompt_profile=self.prompt_profile)
        except GemmaLocalServerResponseError as exc:
            if exc.strict_retryable and self.prompt_profile != STRICT_GEMMA_PROMPT_PROFILE:
                logger.warning(
                    "gemma local server chunk failed for %d block(s); retrying once with prompt_profile=%s. reason=%s",
                    len(blk_list),
                    STRICT_GEMMA_PROMPT_PROFILE,
                    exc,
                )
                try:
                    return self._translate_chunk(
                        blk_list,
                        extra_context,
                        prompt_profile=STRICT_GEMMA_PROMPT_PROFILE,
                    )
                except GemmaLocalServerResponseError as strict_exc:
                    exc = strict_exc

            if len(blk_list) <= 1:
                raise

            split_point = max(1, len(blk_list) // 2)
            self._current_benchmark_stats["gemma_chunk_retry_events"] += 1
            logger.warning(
                "gemma local server chunk failed for %d block(s); retrying as %d and %d block(s). reason=%s",
                len(blk_list),
                split_point,
                len(blk_list) - split_point,
                exc,
            )
            left = self._translate_chunk_with_retry(blk_list[:split_point], extra_context)
            right = self._translate_chunk_with_retry(blk_list[split_point:], extra_context)
            return left + right

    def _translate_chunk(
        self,
        blk_list: list[TextBlock],
        extra_context: str,
        *,
        prompt_profile: str,
    ) -> int:
        system_prompt = self._build_system_prompt(extra_context, prompt_profile=prompt_profile)
        user_prompt = get_raw_text(blk_list)
        response_data = self._request_translation(system_prompt, user_prompt)

        choice = (response_data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        finish_reason = choice.get("finish_reason")
        content = message.get("content") or ""
        reasoning_content = message.get("reasoning_content") or ""

        usage = response_data.get("usage") or {}
        logger.info(
            "gemma local response summary: blocks=%d prompt_profile=%s finish_reason=%s prompt_tokens=%s completion_tokens=%s total_tokens=%s has_content=%s has_reasoning=%s",
            len(blk_list),
            prompt_profile,
            finish_reason,
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
            usage.get("total_tokens"),
            bool(content.strip()),
            bool(reasoning_content.strip()),
        )

        if self.raw_response_logging and content:
            logger.info(
                "translation raw content (%s): %s",
                self.translation_mode_label,
                content,
            )

        if finish_reason == "length":
            self._current_benchmark_stats["gemma_truncated_count"] += 1
            raise GemmaLocalServerTruncatedError(
                "Gemma local server response was truncated before the final JSON was completed. "
                "Reduce Chunk Size or increase LLAMA_CTX_SIZE."
            )

        if not content.strip():
            self._current_benchmark_stats["gemma_empty_content_count"] += 1
            detail = "Gemma local server returned an empty message.content."
            if reasoning_content.strip():
                self._current_benchmark_stats["gemma_reasoning_without_final_count"] += 1
                detail += " The model produced reasoning output without a final JSON answer."
            raise GemmaLocalServerResponseError(
                f"{detail} Check the local server settings or reduce Chunk Size.",
                strict_retryable=True,
            )

        try:
            translation_dict = extract_json_object(content)
        except Exception as exc:
            self._current_benchmark_stats["gemma_json_retry_count"] += 1
            raise GemmaLocalServerResponseError(
                "Gemma local server did not return a valid JSON object in message.content.",
                strict_retryable=True,
            ) from exc

        expected_keys = [f"block_{index}" for index in range(len(blk_list))]
        missing_keys = [key for key in expected_keys if key not in translation_dict]
        unexpected_keys = [key for key in translation_dict if key not in expected_keys]
        if missing_keys or unexpected_keys:
            self._current_benchmark_stats["gemma_json_retry_count"] += 1
            if missing_keys:
                self._current_benchmark_stats["gemma_missing_key_count"] += len(missing_keys)
            if self.response_format_mode == "json_schema":
                self._current_benchmark_stats["gemma_schema_validation_fail_count"] += 1
            reasons: list[str] = []
            if missing_keys:
                reasons.append("missing expected block keys: " + ", ".join(missing_keys))
            if unexpected_keys:
                reasons.append("unexpected block keys: " + ", ".join(unexpected_keys))
            raise GemmaLocalServerResponseError(
                "Gemma local server JSON response was invalid: " + "; ".join(reasons),
                strict_retryable=True,
            )

        for index, blk in enumerate(blk_list):
            value = translation_dict[f"block_{index}"]
            blk.translation = value if isinstance(value, str) or value is None else str(value)

        return len(blk_list)

    def _build_system_prompt(self, extra_context: str, *, prompt_profile: str) -> str:
        if prompt_profile == "legacy":
            prompt = (
                f"Translate {self.source_lang} comic OCR text into {self.target_lang}. "
                "Return exactly one JSON object. Keep every key unchanged. "
                f"Each value must be a natural {self.target_lang} string suitable for comic dialogue. "
                "Do not add explanations, markdown, comments, or extra keys. "
                f"If text is already in {self.target_lang} or unreadable, copy it as-is."
            )
        elif prompt_profile == STRICT_GEMMA_PROMPT_PROFILE:
            prompt = (
                f"Translate the user's JSON object from {self.source_lang} to {self.target_lang}. "
                "Return only one valid JSON object and nothing before or after it. "
                "Use exactly the same keys from the input. "
                f"Each value must be a short natural {self.target_lang} comic dialogue string. "
                "Do not output markdown, comments, reasoning, analysis, channel tokens, or code fences. "
                f"If any line is unreadable or already in {self.target_lang}, copy it unchanged."
            )
        else:
            prompt = (
                f"Translate the user's JSON object of comic OCR lines from {self.source_lang} to {self.target_lang}. "
                "Return exactly one JSON object with the same keys and no extra text. "
                f"Each value must be concise natural {self.target_lang} dialogue suitable for a comic bubble. "
                "Keep key names unchanged. Do not add markdown, explanations, comments, code fences, or reasoning. "
                f"If a line is unreadable or already in {self.target_lang}, copy it as-is."
            )
        cleaned_context = (extra_context or "").strip()
        if cleaned_context:
            prompt += f" Additional comic context: {cleaned_context}"
        if self.think_briefly_prompt:
            prompt += (
                " If internal reasoning is needed, keep it brief and efficient, "
                "then return only the final JSON object."
            )
        return prompt

    def _request_translation(self, system_prompt: str, user_prompt: str) -> dict:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": user_prompt}],
                },
            ],
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "min_p": self.min_p,
            "max_completion_tokens": self.max_tokens,
            "response_format": self._build_response_format(user_prompt),
        }
        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(
                f"{self.api_base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            error_msg = f"API request failed: {exc}"
            if getattr(exc, "response", None) is not None:
                try:
                    error_msg += f" - {json.dumps(exc.response.json(), ensure_ascii=False)}"
                except Exception:
                    error_msg += f" - Status code: {exc.response.status_code}"
            raise RuntimeError(error_msg) from exc

        response_data = response.json()
        if self.raw_response_logging:
            logger.info(
                "translation raw response json (%s): %s",
                self.translation_mode_label,
                json.dumps(response_data, ensure_ascii=False),
            )
        return response_data

    def _build_response_format(self, user_prompt: str) -> dict[str, Any]:
        if self.response_format_mode != "json_schema":
            return {"type": "json_object"}

        if self.response_schema_mode != "blocks":
            return {"type": "json_object"}

        try:
            chunk_payload = extract_json_object(user_prompt)
        except Exception:
            return {"type": "json_object"}

        properties: dict[str, Any] = {}
        required: list[str] = []
        for key in chunk_payload:
            if not str(key).startswith("block_"):
                continue
            required.append(str(key))
            properties[str(key)] = {"type": ["string", "null"]}

        if not required:
            return {"type": "json_object"}

        return {
            "type": "json_schema",
            "json_schema": {
                "name": f"translation_blocks_{len(required)}",
                "schema": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
            },
        }

    def _perform_translation(self, user_prompt: str, system_prompt: str, image: np.ndarray) -> str:
        raise NotImplementedError("CustomLocalGemmaTranslation uses chunked translate().")
