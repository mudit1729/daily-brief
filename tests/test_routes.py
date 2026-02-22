import pytest
import json
from datetime import date, datetime, timezone, timedelta
from app.extensions import db
from app.models.brief import DailyBrief, BriefSection
from app.models.source import Source
from app.models.user import DailyInsight


class TestHealthRoutes:
    def test_health(self, client):
        resp = client.get('/health')
        assert resp.status_code == 200
        assert resp.json['status'] == 'ok'

    def test_ready(self, client):
        resp = client.get('/ready')
        assert resp.status_code == 200
        assert resp.json['db'] is True


class TestBriefRoutes:
    def test_today_no_brief(self, client):
        resp = client.get('/api/brief/today')
        assert resp.status_code == 404

    def test_today_with_brief(self, client, db_session):
        brief = DailyBrief(
            date=date.today(),
            status='complete',
            total_tokens=500,
            total_cost_usd=0.01,
        )
        db_session.add(brief)
        db_session.commit()

        resp = client.get('/api/brief/today')
        assert resp.status_code == 200
        assert resp.json['status'] == 'complete'

    def test_brief_by_date(self, client, db_session):
        brief = DailyBrief(
            date=date(2025, 1, 20),
            status='complete',
        )
        db_session.add(brief)
        db_session.commit()

        resp = client.get('/api/brief/2025-01-20')
        assert resp.status_code == 200

    def test_brief_invalid_date(self, client):
        resp = client.get('/api/brief/not-a-date')
        assert resp.status_code == 400

    def test_brief_history(self, client, db_session):
        for i in range(3):
            brief = DailyBrief(date=date(2025, 1, i + 1), status='complete')
            db_session.add(brief)
        db_session.commit()

        resp = client.get('/api/brief/history')
        assert resp.status_code == 200
        assert len(resp.json['briefs']) == 3


class TestFeedbackRoutes:
    def test_submit_action(self, client):
        resp = client.post('/api/feedback/action', json={
            'action_type': 'upvote',
            'target_type': 'article',
            'target_id': 1,
        })
        assert resp.status_code == 201
        assert resp.json['action_type'] == 'upvote'

    def test_submit_action_invalid(self, client):
        resp = client.post('/api/feedback/action', json={
            'action_type': 'invalid',
            'target_type': 'article',
            'target_id': 1,
        })
        assert resp.status_code == 400

    def test_submit_action_missing_fields(self, client):
        resp = client.post('/api/feedback/action', json={
            'action_type': 'upvote',
        })
        assert resp.status_code == 400

    def test_submit_insight(self, client):
        resp = client.post('/api/feedback/insight', json={
            'text': 'Focus more on AI regulation news',
        })
        assert resp.status_code == 201
        assert resp.json['text'] == 'Focus more on AI regulation news'

    def test_list_insights(self, client):
        # Submit an insight first
        client.post('/api/feedback/insight', json={'text': 'test insight'})

        resp = client.get('/api/feedback/insights')
        assert resp.status_code == 200
        assert len(resp.json) >= 1

    def test_promote_insight_sets_pref_id(self, client):
        created = client.post('/api/feedback/insight', json={'text': 'long-term preference'})
        insight_id = created.json['id']

        promoted = client.post(f'/api/feedback/insight/{insight_id}/promote')
        assert promoted.status_code == 200

        insight = DailyInsight.query.filter_by(id=insight_id).first()
        assert insight.promoted_to_pref is True
        assert insight.pref_id is not None


class TestAdminRoutes:
    def test_list_sources(self, client, sample_sources, admin_headers):
        resp = client.get('/api/admin/sources', headers=admin_headers)
        assert resp.status_code == 200
        assert len(resp.json) == 3

    def test_add_source(self, client, admin_headers):
        resp = client.post('/api/admin/sources', json={
            'name': 'New Source',
            'url': 'https://newsource.com/feed.xml',
            'section': 'ai_news',
            'trust_score': 75,
        }, headers=admin_headers)
        assert resp.status_code == 201
        assert resp.json['name'] == 'New Source'

    def test_update_source(self, client, sample_sources, admin_headers):
        source_id = sample_sources[0].id
        resp = client.put(f'/api/admin/sources/{source_id}', json={
            'trust_score': 95,
        }, headers=admin_headers)
        assert resp.status_code == 200
        assert resp.json['trust_score'] == 95

    def test_delete_source(self, client, sample_sources, admin_headers):
        source_id = sample_sources[0].id
        resp = client.delete(f'/api/admin/sources/{source_id}', headers=admin_headers)
        assert resp.status_code == 200
        assert resp.json['status'] == 'deactivated'

    def test_list_flags(self, client, admin_headers):
        resp = client.get('/api/admin/flags', headers=admin_headers)
        assert resp.status_code == 200
        assert isinstance(resp.json, dict)

    def test_add_watchlist(self, client, admin_headers):
        resp = client.post('/api/admin/watchlists', json={
            'name': 'Test Topic',
            'description': 'A test tracked topic',
        }, headers=admin_headers)
        assert resp.status_code == 201

    def test_admin_requires_auth(self, client):
        resp = client.get('/api/admin/sources')
        assert resp.status_code == 401

    def test_duplicate_source_returns_409(self, client, admin_headers):
        payload = {
            'name': 'Source One',
            'url': 'https://dupe-source.com/feed.xml',
            'section': 'ai_news',
        }
        first = client.post('/api/admin/sources', json=payload, headers=admin_headers)
        assert first.status_code == 201

        second = client.post('/api/admin/sources', json=payload, headers=admin_headers)
        assert second.status_code == 409

    def test_duplicate_watchlist_returns_409(self, client, admin_headers):
        payload = {'name': 'Duplicate Topic', 'description': 'x'}
        first = client.post('/api/admin/watchlists', json=payload, headers=admin_headers)
        assert first.status_code == 201

        second = client.post('/api/admin/watchlists', json=payload, headers=admin_headers)
        assert second.status_code == 409

    def test_get_pipeline_schedule(self, client, admin_headers):
        resp = client.get('/api/admin/schedule/pipeline', headers=admin_headers)
        assert resp.status_code == 200
        assert resp.json['hour'] == 5
        assert resp.json['minute'] == 30
        assert resp.json['timezone'] == 'UTC'

    def test_update_pipeline_schedule(self, client, admin_headers):
        resp = client.put('/api/admin/schedule/pipeline', json={
            'enabled': False,
            'time': '07:45',
            'timezone': 'UTC',
        }, headers=admin_headers)
        assert resp.status_code == 200
        assert resp.json['enabled'] is False
        assert resp.json['hour'] == 7
        assert resp.json['minute'] == 45

        check = client.get('/api/admin/schedule/pipeline', headers=admin_headers)
        assert check.status_code == 200
        assert check.json['enabled'] is False
        assert check.json['hour'] == 7
        assert check.json['minute'] == 45

    def test_update_pipeline_schedule_rejects_bad_time(self, client, admin_headers):
        resp = client.put('/api/admin/schedule/pipeline', json={
            'time': '99:99',
        }, headers=admin_headers)
        assert resp.status_code == 400

    def test_source_health_endpoint(self, client, sample_sources, admin_headers, db_session):
        degraded = sample_sources[1]
        degraded.consecutive_failures = 2

        cooling = sample_sources[2]
        cooling.consecutive_failures = 4
        cooling.auto_disabled_until = datetime.now(timezone.utc) + timedelta(hours=1)
        db_session.commit()

        resp = client.get('/api/admin/sources/health', headers=admin_headers)
        assert resp.status_code == 200
        assert resp.json['summary']['healthy'] >= 1
        assert resp.json['summary']['degraded'] >= 1
        assert resp.json['summary']['cooldown'] >= 1


class TestCostRoutes:
    def test_today_cost(self, client):
        resp = client.get('/api/cost/today')
        assert resp.status_code == 200
        assert 'total_tokens' in resp.json
        assert 'idiot_index' in resp.json

    def test_dashboard(self, client):
        resp = client.get('/api/cost/dashboard')
        assert resp.status_code == 200
        assert 'summaries' in resp.json

    def test_logs(self, client):
        resp = client.get('/api/cost/logs')
        assert resp.status_code == 200
        assert 'logs' in resp.json
