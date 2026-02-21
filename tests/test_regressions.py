from datetime import date, datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.extensions import db
from app.integrations.weather import WeatherService
from app.models.article import Article
from app.models.brief import DailyBrief
from app.models.cluster import Cluster, ClusterMembership
from app.models.investment import InvestmentThesis
from app.models.source import Source
from app.pipeline.orchestrator import _claim_pipeline_brief
from app.pipeline.synthesize import _build_investment_section, _build_news_section
from app.services.investment_service import InvestmentService


class DummyLLM:
    def determine_degradation_level(self, section):
        return 4


class TestRegionFiltering:
    def test_region_filter_applies_to_general_news_sections(self, db_session):
        brief = DailyBrief(date=date.today(), status='pending')
        db_session.add(brief)
        db_session.commit()

        us_source = Source(
            name='US Source',
            url='https://us.example/feed',
            section='general_news',
            region='us',
        )
        india_source = Source(
            name='India Source',
            url='https://india.example/feed',
            section='general_news',
            region='india',
        )
        db_session.add_all([us_source, india_source])
        db_session.commit()

        us_article = Article(
            source_id=us_source.id,
            url='https://us.example/a1',
            title='US Story',
            extracted_text='US sentence one. US sentence two.',
            published_at=datetime.now(timezone.utc),
        )
        india_article = Article(
            source_id=india_source.id,
            url='https://india.example/a1',
            title='India Story',
            extracted_text='India sentence one. India sentence two.',
            published_at=datetime.now(timezone.utc),
        )
        db_session.add_all([us_article, india_article])
        db_session.commit()

        us_cluster = Cluster(
            section='general_news',
            representative_article_id=us_article.id,
            article_count=1,
            rank_score=0.9,
            date=date.today(),
        )
        india_cluster = Cluster(
            section='general_news',
            representative_article_id=india_article.id,
            article_count=1,
            rank_score=0.8,
            date=date.today(),
        )
        db_session.add_all([us_cluster, india_cluster])
        db_session.flush()
        db_session.add_all([
            ClusterMembership(cluster_id=us_cluster.id, article_id=us_article.id, similarity=1.0),
            ClusterMembership(cluster_id=india_cluster.id, article_id=india_article.id, similarity=1.0),
        ])
        db_session.commit()

        us_section = _build_news_section(
            date.today(),
            brief.id,
            DummyLLM(),
            {
                'key': 'general_news_us',
                'title': 'US News',
                'cluster_section': 'general_news',
                'region_filter': 'us',
                'order': 0,
            },
        )
        india_section = _build_news_section(
            date.today(),
            brief.id,
            DummyLLM(),
            {
                'key': 'general_news_india',
                'title': 'India',
                'cluster_section': 'general_news',
                'region_filter': 'india',
                'order': 1,
            },
        )

        us_ids = [c['cluster_id'] for c in us_section.content_json['clusters']]
        india_ids = [c['cluster_id'] for c in india_section.content_json['clusters']]
        assert us_ids == [us_cluster.id]
        assert india_ids == [india_cluster.id]


class TestWeatherTargetDate:
    def test_weather_service_uses_target_date(self):
        service = WeatherService()
        target_date = date(2025, 1, 5)
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {
            'daily': {
                'time': ['2025-01-05'],
                'temperature_2m_max': [70],
                'temperature_2m_min': [55],
                'weathercode': [1],
            }
        }

        with patch('app.integrations.weather.requests.get', return_value=response):
            rows = service.fetch_weather(
                locations=[{'name': 'Boston', 'lat': 42.3601, 'lon': -71.0589}],
                target_date=target_date,
            )

        assert len(rows) == 1
        assert rows[0]['date'] == target_date


class TestInvestmentAccounting:
    def test_investment_section_includes_usage(self, db_session):
        mock_thesis = SimpleNamespace(
            thesis_text='Thesis',
            gate_passed=True,
            momentum_signal={'count': 2},
            value_signal={'pass': True},
        )
        usage = {'total_tokens': 321, 'cost_usd': 0.01234}

        with patch(
            'app.pipeline.synthesize.InvestmentService.generate_thesis',
            return_value=(mock_thesis, usage),
        ):
            section = _build_investment_section(
                target_date=date.today(),
                brief_id=1,
                llm=DummyLLM(),
                section_def={
                    'key': 'investment_thesis',
                    'title': 'Investment Thesis',
                    'order': 8,
                },
            )

        assert section.tokens_used == 321
        assert section.cost_usd == 0.01234

    def test_generate_thesis_upserts_per_date_and_brief(self, db_session):
        service = InvestmentService()
        target = date(2025, 1, 6)
        signals = {
            'momentum_count': 0,
            'momentum_pass': False,
            'gold_change_pct': 0.0,
            'value_pass': True,
        }

        with patch.object(
            service.market_service,
            'check_momentum_value_gate',
            return_value=(False, signals),
        ):
            t1, usage1 = service.generate_thesis(target, 1, [], [], llm_gateway=DummyLLM())
            t2, usage2 = service.generate_thesis(target, 1, [], [], llm_gateway=DummyLLM())

        assert t1.id == t2.id
        assert usage1['total_tokens'] == 0
        assert usage2['total_tokens'] == 0
        assert InvestmentThesis.query.filter_by(date=target, brief_id=1).count() == 1


class TestPipelineClaim:
    def test_claim_pipeline_brief_prevents_duplicate_run(self, db_session):
        target = date(2025, 1, 7)
        brief = DailyBrief(date=target, status='pending')
        db_session.add(brief)
        db_session.commit()

        claimed = _claim_pipeline_brief(target)
        assert claimed is not None
        assert claimed.status == 'running'

        duplicate = _claim_pipeline_brief(target)
        assert duplicate is None
