from typing import Any
import logging
import numpy as np
import requests
import time

from .base import BaseLLMTranslation
from ...utils.translator_utils import MODEL_MAP

logger = logging.getLogger(__name__)


class GeminiTranslation(BaseLLMTranslation):
    """Translation engine using Google Gemini models via REST API."""
    
    def __init__(self):
        super().__init__()
        self.model_name = None
        self.api_key = None
        self.api_base_url = "https://generativelanguage.googleapis.com/v1beta/models"
    
    def initialize(self, settings: Any, source_lang: str, target_lang: str, model_name: str, **kwargs) -> None:
        """
        Initialize Gemini translation engine.
        
        Args:
            settings: Settings object with credentials
            source_lang: Source language name
            target_lang: Target language name
            model_name: Gemini model name
        """
        super().initialize(settings, source_lang, target_lang, **kwargs)
        
        self.model_name = model_name
        credentials = settings.get_credentials(settings.ui.tr('Google Gemini'))
        self.api_key = credentials.get('api_key', '')
        
        # Map friendly model name to API model name
        self.model_api_name = MODEL_MAP.get(self.model_name)

    def _get_retry_delay(self, response: requests.Response | None, attempt: int) -> float:
        """Prefer server-provided retry hints, otherwise use exponential backoff."""
        if response is not None:
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    return max(float(retry_after), 0.0)
                except ValueError:
                    pass

            try:
                error = response.json().get("error", {})
                for detail in error.get("details", []):
                    retry_delay = detail.get("retryDelay")
                    if isinstance(retry_delay, str) and retry_delay.endswith("s"):
                        return max(float(retry_delay[:-1]), 0.0)
            except Exception:
                pass

        return min(2 ** (attempt - 1), 8)

    def _should_retry_status(self, status_code: int) -> bool:
        return status_code in {408, 425, 429, 500, 502, 503, 504}

    def _post_with_retries(self, url: str, headers: dict[str, str], payload: dict[str, Any]) -> requests.Response:
        """Retry transient Gemini failures instead of skipping the page immediately."""
        max_attempts = 4
        read_timeout = max(self.timeout, 90)

        for attempt in range(1, max_attempts + 1):
            response = None
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=(10, read_timeout),
                )
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
                if attempt >= max_attempts:
                    raise

                delay = self._get_retry_delay(None, attempt)
                logger.warning(
                    "Gemini request failed with %s. Retrying in %.1fs (%d/%d)",
                    type(exc).__name__,
                    delay,
                    attempt,
                    max_attempts,
                )
                time.sleep(delay)
                continue

            if response.status_code == 200:
                return response

            if self._should_retry_status(response.status_code) and attempt < max_attempts:
                delay = self._get_retry_delay(response, attempt)
                logger.warning(
                    "Gemini request returned %s. Retrying in %.1fs (%d/%d)",
                    response.status_code,
                    delay,
                    attempt,
                    max_attempts,
                )
                time.sleep(delay)
                continue

            return response

        raise RuntimeError("Gemini retry loop exited unexpectedly")
    
    def _perform_translation(self, user_prompt: str, system_prompt: str, image: np.ndarray) -> str:
        """
        Perform translation using Gemini REST API.
        
        Args:
            user_prompt: The prompt to send to the model
            system_prompt: System instructions for the model
            image: Image data as numpy array
            
        Returns:
            Translated text from the model
        """
        # Create API endpoint URL
        url = f"{self.api_base_url}/{self.model_api_name}:generateContent?key={self.api_key}"
        
        # Setup generation config
        if self.model_name in ["Gemini-3.0-Flash"]:
            thinking_level = "minimal"
        else:
            thinking_level = "low"

        generation_config = {
            "temperature": self.temperature,
            "maxOutputTokens": self.max_tokens,
            "thinkingConfig": {
                "thinkingLevel": thinking_level
            },
        }
        
        # Setup safety settings
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        
        # Prepare parts for the request
        parts = []
        
        # Add image if needed
        if self.img_as_llm_input:
            # Base64 encode the image

            img_b64, mime_type = self.encode_image(image)
            parts.append({
                "inline_data": {
                    "mime_type": mime_type,
                    "data": img_b64
                }
            })
        
        # Add text prompt
        parts.append({"text": user_prompt})
        
        # Create the request payload
        payload = {
            "contents": [{
                "parts": parts
            }],
            "generationConfig": generation_config,
            "safetySettings": safety_settings
        }
        
        # Add system instructions if provided
        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
        
        # Send request to Gemini API
        headers = {
            "Content-Type": "application/json"
        }
        
        response = self._post_with_retries(url, headers, payload)
        
        # Handle response
        if response.status_code != 200:
            error_msg = f"API request failed with status code {response.status_code}: {response.text}"
            raise Exception(error_msg)
        
        # Extract text from response
        response_data = response.json()
        
        # Extract the generated text from the response
        candidates = response_data.get("candidates", [])
        if not candidates:
            return "No response generated"
        
        content = candidates[0].get("content", {})
        parts = content.get("parts", [])
        
        # Concatenate all text parts
        result = ""
        for part in parts:
            if "text" in part:
                result += part["text"]
        
        return result
