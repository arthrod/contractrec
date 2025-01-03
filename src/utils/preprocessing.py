from typing import Dict, List, Optional, Any
import re
import unicodedata
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Common legal abbreviations and their expansions
LEGAL_ABBREV = {
    "Corp.": "Corporation",
    "Inc.": "Incorporated",
    "Ltd.": "Limited",
    "LLC": "Limited Liability Company",
    "et al.": "et alia",
    "v.": "versus",
    "viz.": "videlicet",
    "etc.": "et cetera",
    "i.e.": "id est",
    "e.g.": "exempli gratia",
}

# Common legal document section identifiers
SECTION_MARKERS = [
    r"Section",
    r"Article",
    r"Clause",
    r"Paragraph",
    r"Schedule",
    r"Appendix",
    r"Exhibit",
]

# Regex patterns for common legal document formatting
LEGAL_PATTERNS = {
    "section_num": r"(?:Section|§)\s*\d+(?:\.\d+)*",
    "list_marker": r"(?:[a-z]\)|\d+\.|•|\*)",
    "date": r"\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}",
}


def clean_text(text: str) -> str:
    """Clean and normalize contract text while preserving legal formatting.

    Args:
        text: Input text to clean

    Returns:
        Cleaned text with preserved legal formatting

    Raises:
        ValueError: If input text is None or empty
    """
    if not text or not isinstance(text, str):
        raise ValueError("Input text must be a non-empty string")

    logger.debug("Cleaning text of length %d", len(text))
    
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text.strip())
    
    # Preserve section numbers
    text = re.sub(LEGAL_PATTERNS["section_num"], lambda m: f" {m.group()} ", text)
    
    # Preserve list markers
    text = re.sub(LEGAL_PATTERNS["list_marker"], lambda m: f" {m.group()} ", text)
    
    # Normalize quotes and apostrophes
    text = re.sub(r"[]", "\"", text)
    
    # Normalize dashes
    text = re.sub(r"[–—]", "-", text)
    
    logger.debug("Cleaned text to length %d", len(text))
    return text.strip()


def normalize_text(text: str) -> str:
    """Normalize contract text while preserving legal terminology.

    Args:
        text: Input text to normalize

    Returns:
        Normalized text with preserved legal terms

    Raises:
        ValueError: If input text is None or empty
    """
    if not text or not isinstance(text, str):
        raise ValueError("Input text must be a non-empty string")

    logger.debug("Normalizing text of length %d", len(text))
    
    # First clean the text
    text = clean_text(text)
    
    # Handle legal abbreviations
    for abbrev, full in LEGAL_ABBREV.items():
        text = re.sub(rf"\b{re.escape(abbrev)}\b", full, text)
    
    # Normalize dates
    text = re.sub(LEGAL_PATTERNS["date"], 
                 lambda m: datetime.strptime(m.group(), "%d %B %Y").strftime("%Y-%m-%d"),
                 text)
    
    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)
    
    logger.debug("Normalized text to length %d", len(text))
    return text.strip()


def tokenize_text(text: str) -> List[str]:
    """Tokenize contract text while preserving legal terms and structure.

    Args:
        text: Input text to tokenize

    Returns:
        List of tokens with preserved legal terms

    Raises:
        ValueError: If input text is None or empty
    """
    if not text or not isinstance(text, str):
        raise ValueError("Input text must be a non-empty string")

    logger.debug("Tokenizing text of length %d", len(text))
    
    # First normalize the text
    text = normalize_text(text)
    
    # Preserve section markers
    for marker in SECTION_MARKERS:
        text = re.sub(rf"\b{marker}\b", f" {marker} ", text)
    
    # Split on whitespace while preserving punctuation
    tokens = []
    for token in text.split():
        # Check if token is a legal term
        if any(marker in token for marker in SECTION_MARKERS):
            tokens.append(token)
        else:
            # Split on punctuation but preserve it
            parts = re.findall(r"\w+|[^\w\s]", token)
            tokens.extend(parts)
    
    logger.debug("Generated %d tokens", len(tokens))
    return tokens


def format_input(text: str, **kwargs) -> Dict[str, Any]:
    """Format text for model input with proper preprocessing.

    Args:
        text: Input text to format
        **kwargs: Additional formatting options
            - max_length: Maximum sequence length (default: None)
            - add_special_tokens: Whether to add special tokens (default: True)
            - return_tensors: Type of tensors to return (default: None)

    Returns:
        Dictionary containing formatted input

    Raises:
        ValueError: If input text is None or empty
    """
    if not text or not isinstance(text, str):
        raise ValueError("Input text must be a non-empty string")

    logger.debug("Formatting input text of length %d", len(text))
    
    # Get formatting options
    max_length = kwargs.get("max_length", None)
    add_special_tokens = kwargs.get("add_special_tokens", True)
    return_tensors = kwargs.get("return_tensors", None)
    
    # Clean and normalize text
    text = normalize_text(text)
    
    # Tokenize text
    tokens = tokenize_text(text)
    
    # Truncate if needed
    if max_length and len(tokens) > max_length:
        tokens = tokens[:max_length]
    
    # Add special tokens if requested
    if add_special_tokens:
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
    
    result = {
        "text": text,
        "tokens": tokens,
        "length": len(tokens)
    }
    
    logger.debug("Formatted input with %d tokens", len(tokens))
    return result

