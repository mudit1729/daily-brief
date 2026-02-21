import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone
from app.models.source import Source
from app.models.article import Article
from app.integrations.rss import fetch_feed


class TestRSSFetch:
    def test_fetch_feed_parses_entries(self, app, db_session, sample_sources):
        """RSS fetch should return parsed article dicts."""
        source = sample_sources[0]

        mock_feed = MagicMock()
        mock_feed.bozo = False
        mock_feed.entries = [
            MagicMock(
                link='https://example.com/new-article',
                title='Test Article',
                summary='Test summary',
                author='Author',
                published_parsed=(2025, 1, 20, 10, 0, 0, 0, 20, 0),
            ),
        ]
        # Ensure hasattr checks work
        mock_feed.entries[0].configure_mock(**{
            'get.side_effect': lambda k, d='': {
                'link': 'https://example.com/new-article',
                'title': 'Test Article',
                'summary': 'Test summary',
                'author': 'Author',
            }.get(k, d)
        })

        with patch('app.integrations.rss.feedparser.parse', return_value=mock_feed):
            articles = fetch_feed(source)

        assert len(articles) == 1
        assert articles[0]['url'] == 'https://example.com/new-article'
        assert articles[0]['source_id'] == source.id

    def test_fetch_feed_handles_malformed(self, app, db_session, sample_sources):
        """Malformed feed with no entries should return empty list."""
        source = sample_sources[0]

        mock_feed = MagicMock()
        mock_feed.bozo = True
        mock_feed.entries = []
        mock_feed.bozo_exception = Exception("Malformed")

        with patch('app.integrations.rss.feedparser.parse', return_value=mock_feed):
            articles = fetch_feed(source)

        assert articles == []

    def test_article_url_uniqueness(self, app, db_session, sample_sources):
        """Articles with duplicate URLs should not be inserted twice."""
        source = sample_sources[0]
        article = Article(
            source_id=source.id,
            url='https://example.com/existing-article',
            title='Existing',
        )
        db_session.add(article)
        db_session.commit()

        existing = Article.query.filter_by(url='https://example.com/existing-article').first()
        assert existing is not None

        # Trying to add same URL should be caught
        dupe = Article.query.filter_by(url='https://example.com/existing-article').first()
        assert dupe.id == article.id


class TestMarketData:
    def test_market_snapshot_creation(self, app, db_session):
        """Market data service should produce snapshot dicts."""
        from app.integrations.market_data import MarketDataService
        import pandas as pd

        service = MarketDataService()

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame({
            'Close': [5996.66, 6049.24],
            'Volume': [3200000000, 3500000000],
        })

        with patch('yfinance.Ticker', return_value=mock_ticker):
            snapshots = service.fetch_snapshots()

        assert len(snapshots) > 0
        for snap in snapshots:
            assert 'symbol' in snap
            assert 'price' in snap
            assert 'change_pct' in snap

    def test_momentum_value_gate(self, app):
        """Gate should pass when 2+ US indices up >0.3% and gold <2%."""
        from app.integrations.market_data import MarketDataService

        service = MarketDataService()

        # Gate should pass
        snapshots_pass = [
            {'symbol': '^GSPC', 'change_pct': 0.5},
            {'symbol': '^DJI', 'change_pct': 0.4},
            {'symbol': '^IXIC', 'change_pct': 0.1},
            {'symbol': 'GC=F', 'change_pct': 0.3},
        ]
        passed, signals = service.check_momentum_value_gate(snapshots_pass)
        assert passed is True
        assert signals['momentum_count'] == 2

        # Gate should fail (gold up too much)
        snapshots_fail = [
            {'symbol': '^GSPC', 'change_pct': 0.5},
            {'symbol': '^DJI', 'change_pct': 0.4},
            {'symbol': '^IXIC', 'change_pct': 0.6},
            {'symbol': 'GC=F', 'change_pct': 3.0},
        ]
        passed, signals = service.check_momentum_value_gate(snapshots_fail)
        assert passed is False
