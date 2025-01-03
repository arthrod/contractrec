import re
import logging

class TextPreprocessor:
    def __init__(self, max_length: int = 128):
        self.max_length = max_length
        self.logger = logging.getLogger(__name__)

    def clean_text(self, text: str) -> str:
        if text is None or not isinstance(text, str):
            self.logger.error("Input text must be a string")
            raise ValueError("Input text must be a string")
        text = text.strip()
        if not text:
            self.logger.warning("Empty text provided")
            return ""
        # Remove special characters and extra whitespace
        text = re.sub(r"[^\w\s.,!?-]", "", text)
        text = re.sub(r"\s+", " ", text)
        self.logger.debug(f"Cleaned text: {text}")
        return text.strip()

    def normalize(self, text: str) -> str:
        if text is None or not isinstance(text, str):
            self.logger.error("Input text must be a string")
            raise ValueError("Input text must be a string")
        text = text.strip()
        if not text:
            self.logger.warning("Empty text provided")
            return ""
        # Convert to lowercase and standardize punctuation
        text = text.lower()
        text = re.sub(r"[.]+", ".", text)
        text = re.sub(r"[!]+", "!", text)
        text = re.sub(r"[?]+", "?", text)
        text = re.sub(r"[,]+", ",", text)
        self.logger.debug(f"Normalized text: {text}")
        return text.strip()

    def preprocess(self, text: str) -> str:
        if text is None or not isinstance(text, str):
            self.logger.error("Input text must be a string")
            raise ValueError("Input text must be a string")
        text = text.strip()
        if not text:
            self.logger.warning("Empty text provided")
            return ""
        # Clean and normalize text
        text = self.clean_text(text)
        text = self.normalize(text)
        # Truncate if needed
        if len(text) > self.max_length:
            self.logger.warning(f"Text truncated to {self.max_length} characters")
            text = text[:self.max_length]
        self.logger.debug(f"Preprocessed text: {text}")
        return text
