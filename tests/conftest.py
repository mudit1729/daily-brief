import os
import pytest
from datetime import datetime, timezone, date

# Set test env vars before importing app
os.environ['FF_INVESTMENT_THESIS'] = 'true'
os.environ['FF_STORY_TRACKING'] = 'false'
os.environ['FF_CLAIM_LEDGER'] = 'false'
os.environ['FF_STORE_RAW_HTML'] = 'false'

from app import create_app
from app.extensions import db as _db
from app.models.source import Source
from app.models.article import Article
from config import TestConfig


@pytest.fixture(scope='session')
def app():
    """Create app with test config."""
    app = create_app(TestConfig)
    return app


@pytest.fixture(autouse=True)
def setup_db(app):
    """Create tables before each test, drop after."""
    with app.app_context():
        _db.create_all()
        yield
        _db.session.rollback()
        _db.drop_all()


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def admin_headers(app):
    return {'X-Admin-Key': app.config['ADMIN_API_KEY']}


@pytest.fixture
def db_session(app):
    with app.app_context():
        yield _db.session


@pytest.fixture
def sample_sources(db_session):
    """Insert 3 sample sources with different trust/bias."""
    sources = [
        Source(
            name='High Trust Center',
            url='https://example.com/feed1.xml',
            section='general_news',
            region='us',
            bias_label='center',
            trust_score=90,
            source_type='primary',
        ),
        Source(
            name='Medium Trust Left',
            url='https://example.com/feed2.xml',
            section='general_news',
            region='us',
            bias_label='left',
            trust_score=70,
            source_type='reporting',
        ),
        Source(
            name='Low Trust Right',
            url='https://example.com/feed3.xml',
            section='general_news',
            region='us',
            bias_label='right',
            trust_score=45,
            source_type='opinion',
        ),
    ]
    for s in sources:
        db_session.add(s)
    db_session.commit()
    return sources


@pytest.fixture
def sample_articles(db_session, sample_sources):
    """Insert sample articles across sources."""
    articles = []
    for i, source in enumerate(sample_sources):
        for j in range(3):
            article = Article(
                source_id=source.id,
                url=f'https://example.com/article-{i}-{j}',
                title=f'Test Article {i}-{j}',
                extracted_text=f'This is test article content for article {i}-{j}. '
                              f'It discusses various topics including technology and politics. '
                              f'The article provides analysis of current events.',
                published_at=datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc),
                word_count=30,
            )
            db_session.add(article)
            articles.append(article)
    db_session.commit()
    return articles


@pytest.fixture
def sample_feed_xml():
    """Load sample RSS feed XML."""
    fixtures_dir = os.path.join(os.path.dirname(__file__), 'fixtures')
    with open(os.path.join(fixtures_dir, 'sample_feed.xml')) as f:
        return f.read()


@pytest.fixture
def sample_article_html():
    """Load sample article HTML."""
    fixtures_dir = os.path.join(os.path.dirname(__file__), 'fixtures')
    with open(os.path.join(fixtures_dir, 'sample_article.html')) as f:
        return f.read()
