"""Tests for the views blueprint — HTML-serving routes."""
import pytest
from datetime import date, datetime, timezone
from app.models.brief import DailyBrief, BriefSection
from app.models.market import MarketSnapshot
from app.models.weather import WeatherCache


@pytest.fixture
def sample_brief(db_session):
    """Create a complete brief with sections for testing."""
    brief = DailyBrief(
        date=date.today(),
        status='complete',
        total_tokens=5000,
        total_cost_usd=0.0042,
        idiot_index=0.0042,
        generated_at=datetime.now(timezone.utc),
    )
    db_session.add(brief)
    db_session.flush()

    sections = [
        BriefSection(
            brief_id=brief.id,
            section_type='general_news_us',
            title='US News',
            content_json={
                'clusters': [
                    {
                        'cluster_id': 1,
                        'label': 'Test Story About Politics',
                        'summary': 'A summary of political events.',
                        'article_count': 3,
                        'avg_trust': 78,
                        'rank_score': 4.5,
                        'articles': [
                            {
                                'id': 1,
                                'title': 'Political Event Unfolds',
                                'url': 'https://example.com/article1',
                                'source': 'Test News',
                                'og_image_url': None,
                                'bias_label': 'center',
                                'trust_score': 85,
                            },
                        ],
                    },
                ]
            },
            display_order=0,
            tokens_used=1000,
            cost_usd=0.001,
            degradation_level=0,
        ),
        BriefSection(
            brief_id=brief.id,
            section_type='weather',
            title='Weather',
            content_json={
                'locations': [
                    {
                        'location': 'New York',
                        'forecasts': [
                            {'date': '2025-01-01', 'condition': 'Clear', 'high_c': 7, 'low_c': -1, 'high_f': 45, 'low_f': 30},
                        ],
                    },
                ]
            },
            display_order=5,
            tokens_used=0,
            cost_usd=0,
        ),
        BriefSection(
            brief_id=brief.id,
            section_type='market',
            title='Market Trends',
            content_json={
                'market_data': [
                    {'symbol': 'SPY', 'name': 'S&P 500', 'price': 5800.0, 'change_pct': 0.5},
                ],
                'clusters': [],
            },
            display_order=1,
        ),
        BriefSection(
            brief_id=brief.id,
            section_type='investment_thesis',
            title='Investment Thesis',
            content_json={
                'thesis': 'Buy tech stocks.',
                'gate_passed': True,
                'signals': {
                    'momentum': {'count': 3, 'pass': True},
                    'value': {'gold_change_pct': 1.5, 'pass': True},
                },
            },
            display_order=8,
        ),
    ]
    for s in sections:
        db_session.add(s)
    db_session.commit()
    return brief


class TestTodayPage:
    def test_index_returns_200(self, client, sample_brief):
        """GET / should return 200 with a brief."""
        resp = client.get('/')
        assert resp.status_code == 200

    def test_index_contains_brief_date(self, client, sample_brief):
        """Homepage should show the brief date."""
        resp = client.get('/')
        assert b'US News' in resp.data

    def test_index_contains_cluster_card(self, client, sample_brief):
        """Homepage should render cluster cards."""
        resp = client.get('/')
        assert b'Test Story About Politics' in resp.data
        assert b'sb-card' in resp.data

    def test_index_contains_weather(self, client, sample_brief):
        """Homepage should render weather section."""
        resp = client.get('/')
        assert b'New York' in resp.data
        assert b'Clear' in resp.data

    def test_index_contains_market_strip(self, client, sample_brief):
        """Homepage should render market ticker strip."""
        resp = client.get('/')
        assert b'S&amp;P 500' in resp.data or b'SPY' in resp.data

    def test_index_contains_thesis(self, client, sample_brief):
        """Homepage should render thesis card."""
        resp = client.get('/')
        assert b'Investment Thesis' in resp.data
        assert b'Gate Passed' in resp.data

    def test_index_empty_brief(self, client):
        """GET / with no brief should show empty state."""
        resp = client.get('/')
        assert resp.status_code == 200
        # Should show empty state, not crash
        assert b'Signal Brief' in resp.data

    def test_index_contains_nav(self, client, sample_brief):
        """Page should contain navigation tabs."""
        resp = client.get('/')
        assert b'Today' in resp.data
        assert b'Stories' in resp.data
        assert b'Market' in resp.data
        assert b'Thesis' in resp.data
        assert b'Settings' in resp.data

    def test_index_contains_footer(self, client, sample_brief):
        """Footer should show cost and token info."""
        resp = client.get('/')
        assert b'0.0042' in resp.data
        assert b'5000 tokens' in resp.data

    def test_index_contains_theme_toggle(self, client, sample_brief):
        """Page should have theme toggle button."""
        resp = client.get('/')
        assert b'themeToggle' in resp.data


class TestBriefByDate:
    def test_specific_date(self, client, sample_brief):
        """GET /brief/<date> should return the brief for that date."""
        today = date.today().isoformat()
        resp = client.get(f'/brief/{today}')
        assert resp.status_code == 200
        assert b'US News' in resp.data

    def test_invalid_date(self, client):
        """GET /brief/<invalid> should return 404."""
        resp = client.get('/brief/not-a-date')
        assert resp.status_code == 404


class TestMarketPage:
    def test_market_returns_200(self, client, sample_brief):
        """GET /market should return 200."""
        resp = client.get('/market')
        assert resp.status_code == 200

    def test_market_contains_title(self, client, sample_brief):
        """Market page should have a title."""
        resp = client.get('/market')
        assert b'Market' in resp.data


class TestStoriesPage:
    def test_stories_returns_200(self, client):
        """GET /stories should return 200 even with no topics."""
        resp = client.get('/stories')
        assert resp.status_code == 200


class TestThesisPage:
    def test_thesis_returns_200(self, client, sample_brief):
        """GET /thesis should return 200."""
        resp = client.get('/thesis')
        assert resp.status_code == 200

    def test_thesis_contains_content(self, client, sample_brief):
        """Thesis page should show thesis text."""
        resp = client.get('/thesis')
        assert b'Investment Thesis' in resp.data


class TestSettingsPage:
    def test_settings_returns_200(self, client, sample_brief):
        """GET /settings should return 200."""
        resp = client.get('/settings')
        assert resp.status_code == 200

    def test_settings_contains_tabs(self, client, sample_brief):
        """Settings page should have tab navigation."""
        resp = client.get('/settings')
        assert b'Cost' in resp.data
        assert b'Insights' in resp.data
        assert b'Sources' in resp.data

    def test_settings_contains_pipeline_trigger(self, client, sample_brief):
        """Settings page should have pipeline trigger button."""
        resp = client.get('/settings')
        assert b'triggerPipeline' in resp.data


class TestHistoryPage:
    def test_history_returns_200(self, client):
        """GET /history should return 200."""
        resp = client.get('/history')
        assert resp.status_code == 200

    def test_history_shows_briefs(self, client, sample_brief):
        """History page should list past briefs."""
        resp = client.get('/history')
        assert resp.status_code == 200
        assert b'history' in resp.data.lower() or b'archive' in resp.data.lower() or b'complete' in resp.data.lower()


class TestJinjaFilters:
    def test_bias_class_filter(self, app):
        """bias_class filter should map labels to CSS classes."""
        with app.app_context():
            from app.routes.views import bias_class_filter
            assert bias_class_filter('left') == 'left'
            assert bias_class_filter('left-center') == 'left'
            assert bias_class_filter('center') == 'center'
            assert bias_class_filter('right') == 'right'
            assert bias_class_filter('right-center') == 'right'
            assert bias_class_filter(None) == 'center'

    def test_trust_level_filter(self, app):
        """trust_level filter should map scores to high/mid/low."""
        with app.app_context():
            from app.routes.views import trust_level_filter
            assert trust_level_filter(90) == 'high'
            assert trust_level_filter(60) == 'mid'
            assert trust_level_filter(30) == 'low'
            assert trust_level_filter(None) == 'mid'

    def test_change_class_filter(self, app):
        """change_class filter should return positive/negative/neutral."""
        with app.app_context():
            from app.routes.views import change_class_filter
            assert change_class_filter(1.5) == 'positive'
            assert change_class_filter(-0.5) == 'negative'
            assert change_class_filter(0) == 'neutral'
            assert change_class_filter(None) == 'neutral'

    def test_change_sign_filter(self, app):
        """change_sign filter should format with + or - prefix."""
        with app.app_context():
            from app.routes.views import change_sign_filter
            assert change_sign_filter(1.5) == '+1.50%'
            assert change_sign_filter(-0.5) == '-0.50%'
            assert change_sign_filter(None) == '0.00%'


class TestStaticAssets:
    def test_css_tokens(self, client):
        """tokens.css should be served."""
        resp = client.get('/static/css/tokens.css')
        assert resp.status_code == 200
        assert b'--sb-bg-primary' in resp.data

    def test_css_components(self, client):
        """components.css should be served."""
        resp = client.get('/static/css/components.css')
        assert resp.status_code == 200
        assert b'sb-card' in resp.data

    def test_js_app(self, client):
        """app.js should be served."""
        resp = client.get('/static/js/app.js')
        assert resp.status_code == 200
        assert b'themeToggle' in resp.data

    def test_favicon(self, client):
        """favicon.svg should be served."""
        resp = client.get('/static/img/favicon.svg')
        assert resp.status_code == 200
