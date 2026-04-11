"""Generic text preprocessing utilities.

Preprocessing levels:
    "none"  — return text as-is (only cast to str and guard against non-string input)
    "basic" — NFC normalization + URL/email removal + whitespace normalization

For backward compatibility, preprocess_article() and preprocess_title() are kept
as deprecated aliases that map to "basic" level preprocessing.
"""
import re
import unicodedata


# ---- Regex patterns ----
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
_MULTI_SPACE_RE = re.compile(r"\s+")
_QUOTE_RE = re.compile(r"[\"\"\"«»]")
_APOSTROPHE_RE = re.compile(r"[''`´]")


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _normalize_punctuation(text: str) -> str:
    text = _QUOTE_RE.sub('"', text)
    text = _APOSTROPHE_RE.sub("'", text)
    return text


def _normalize_whitespace(text: str) -> str:
    return _MULTI_SPACE_RE.sub(" ", text).strip()


def preprocess_text(text: str, level: str = "none") -> str:
    """Generic text preprocessing dispatcher.

    Args:
        text:  Input string.
        level: Preprocessing intensity.
               "none"  — return text as-is (only cast to str).
               "basic" — NFC normalization + URL/email removal +
                         punctuation normalization + whitespace normalization.

    Returns:
        Preprocessed string.
    """
    if not isinstance(text, str):
        return ""
    if level == "none":
        return text
    if level == "basic":
        text = _nfc(text)
        text = _URL_RE.sub(" ", text)
        text = _EMAIL_RE.sub(" ", text)
        text = _normalize_punctuation(text)
        text = _normalize_whitespace(text)
        return text
    raise ValueError(f"Unknown preprocessing level: {level!r}. Choose 'none' or 'basic'.")


# ---------------------------------------------------------------------------
# Backward-compatibility aliases (kept for existing scripts that import them)
# ---------------------------------------------------------------------------

def preprocess_article(text: str) -> str:
    """Deprecated: use preprocess_text(text, level='basic') instead."""
    return preprocess_text(text, level="basic")


def preprocess_title(text: str) -> str:
    """Deprecated: use preprocess_text(text, level='basic') instead."""
    return preprocess_text(text, level="basic")
