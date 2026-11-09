import html
import re


def clean_text(value):
    """Decode HTML entities (some titles are double-encoded) and tidy whitespace."""
    text = str(value)
    while html.unescape(text) != text:
        text = html.unescape(text)
    return re.sub(r'\s+', ' ', text).strip()


def normalize_title(title):
    """Key that treats titles differing only by case or punctuation as the same book."""
    return ''.join(ch for ch in clean_text(title).casefold() if ch.isalnum())
