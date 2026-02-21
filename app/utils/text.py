import re
from html import unescape


def clean_text(html_or_text):
    """Strip HTML tags and normalize whitespace."""
    text = re.sub(r'<[^>]+>', ' ', html_or_text or '')
    text = unescape(text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def truncate(text, max_words=500):
    """Truncate text to max_words."""
    words = (text or '').split()
    if len(words) <= max_words:
        return text
    return ' '.join(words[:max_words]) + '...'


def extract_lead_sentences(text, n=3):
    """Extract first n sentences for extractive fallback (degradation level 4)."""
    sentences = re.split(r'(?<=[.!?])\s+', text or '')
    return ' '.join(sentences[:n])


def word_count(text):
    """Count words in text."""
    return len((text or '').split())
