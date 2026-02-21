import pytest
from unittest.mock import patch, MagicMock
from app.integrations.extractor import extract, _extract_readability, _extract_bs4, _extract_meta
from app.utils.text import clean_text, truncate, extract_lead_sentences


class TestExtraction:
    def test_extract_meta_og_image(self, sample_article_html):
        """Should extract OG image from meta tags."""
        meta = _extract_meta(sample_article_html)
        assert meta.get('og_image_url') == 'https://example.com/images/policy.jpg'
        assert meta.get('author') == 'John Doe'

    def test_extract_readability_or_bs4(self, sample_article_html):
        """Extraction should work via readability or BS4 fallback."""
        # Readability may return None for small HTML docs; BS4 fallback handles it
        result = _extract_readability(sample_article_html, 'https://example.com/article')
        if result is None:
            # BS4 fallback should work
            result = _extract_bs4(sample_article_html, 'https://example.com/article')
        assert result is not None
        assert len(result['text']) > 50
        assert 'policy' in result['text'].lower()

    def test_extract_bs4_fallback(self, sample_article_html):
        """BS4 should extract article content as fallback."""
        result = _extract_bs4(sample_article_html, 'https://example.com/article')
        assert result is not None
        assert len(result['text']) > 50

    def test_extract_full_pipeline(self, sample_article_html):
        """Full extract function should return complete result."""
        mock_response = MagicMock()
        mock_response.text = sample_article_html
        mock_response.raise_for_status = MagicMock()

        with patch('app.integrations.extractor.requests.get', return_value=mock_response):
            result = extract('https://example.com/article')

        assert result is not None
        assert 'text' in result
        assert 'og_image_url' in result
        assert result['og_image_url'] == 'https://example.com/images/policy.jpg'
        assert len(result['text']) > 100


class TestTextUtils:
    def test_clean_text(self):
        assert clean_text('<p>Hello <b>world</b></p>') == 'Hello world'
        assert clean_text('  spaces   everywhere  ') == 'spaces everywhere'
        assert clean_text('&amp; entity') == '& entity'

    def test_truncate(self):
        text = ' '.join(['word'] * 100)
        result = truncate(text, max_words=10)
        assert result.endswith('...')
        assert len(result.split()) <= 11  # 10 words + "..."

    def test_truncate_short_text(self):
        assert truncate('short text', max_words=10) == 'short text'

    def test_extract_lead_sentences(self):
        text = 'First sentence. Second sentence. Third sentence. Fourth.'
        result = extract_lead_sentences(text, n=2)
        assert 'First sentence.' in result
        assert 'Second sentence.' in result
        assert 'Third' not in result
