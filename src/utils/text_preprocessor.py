class TextPreprocessor:
    def preprocess(self, text: str) -> str:
        """
        Clean and normalize input text.

        Args:
            text: Input text to process

        Returns:
            Processed text string
        """
        # Basic cleaning
        text = text.strip()
        # Normalize whitespace
        text = " ".join(text.split())
        return text
