from typing import Any
import requests
import uuid

from .base import TraditionalTranslation
from ..utils.textblock import TextBlock


class MicrosoftTranslation(TraditionalTranslation):
    """Translation engine using Microsoft Translator API."""
    
    def __init__(self):
        self.source_lang_code = None
        self.target_lang_code = None
        self.api_key = None
        self.region = None
        
    def initialize(self, settings: Any, source_lang: str, target_lang: str) -> None:
        self.source_lang_code = self.get_language_code(source_lang)
        
        # Preprocess target language code to match Microsoft's supported formats
        target_code = self.get_language_code(target_lang)
        self.target_lang_code = self.preprocess_language_code(target_code)
        
        credentials = settings.get_credentials(settings.ui.tr("Microsoft Azure"))
        self.api_key = credentials['api_key_translator']
        self.region = credentials['region_translator']
        
    def translate(self, blk_list: list[TextBlock]) -> list[TextBlock]:
        endpoint = "https://api.cognitive.microsofttranslator.com"
        path = '/translate'
        constructed_url = endpoint + path
        
        # Set up the API request
        headers = {
            'Ocp-Apim-Subscription-Key': self.api_key,
            'Ocp-Apim-Subscription-Region': self.region,
            'Content-type': 'application/json',
            'X-ClientTraceId': str(uuid.uuid4())
        }
        
        # Set up parameters - omitting 'from' parameter for auto-detection
        params = {
            'api-version': '3.0',
            'to': self.target_lang_code
        }
        
        # Process blocks in batches to avoid request size limits
        batch_size = 25  # Adjust based on typical text length
        updates = []
        for i in range(0, len(blk_list), batch_size):
            batch = blk_list[i:i+batch_size]
            
            # Prepare the texts to translate
            body = []
            indices_to_update = []
            
            for idx, blk in enumerate(batch):
                text = self.preprocess_text(blk.text, self.source_lang_code)
                
                if not text.strip():
                    continue
                
                body.append({
                    'text': text
                })
                indices_to_update.append(i + idx)
            
            # Skip empty batches
            if not body:
                continue
            
            # Make the request
            response = requests.post(
                constructed_url, 
                headers=headers, 
                params=params, 
                json=body,
                timeout=30
            )
            response.raise_for_status()
            
            # Process the response
            translations = response.json()
            if not isinstance(translations, list) or len(translations) != len(body):
                raise ValueError(
                    "Invalid translation response"
                )
            for block_idx, translation_result in zip(indices_to_update, translations):
                try:
                    translation = translation_result["translations"][0]["text"]
                except (IndexError, KeyError, TypeError) as exc:
                    raise ValueError(
                        "Invalid translation response"
                    ) from exc
                if not isinstance(translation, str) or not translation.strip():
                    raise ValueError(
                        "Invalid translation response"
                    )
                updates.append((block_idx, translation))

        for block_idx, translation in updates:
            blk_list[block_idx].translation = translation
            
        return blk_list
    
    def preprocess_language_code(self, lang_code: str) -> str:
        """
        Preprocess language codes to match Microsoft Translator API supported formats.
        
        Handles special cases like Chinese variants and Portuguese variants.
        """
        if not lang_code:
            return lang_code
            
        # Handle Chinese variants
        if lang_code == "zh-CN":
            return "zh-Hans"
        elif lang_code == "zh-TW":
            return "zh-Hant"
        
        # Handle Portuguese variants
        if lang_code == "pt":
            return "pt-pt"
        elif lang_code.lower() == "pt-br":
            return "pt"
            
        # Return the original code for other languages
        return lang_code
