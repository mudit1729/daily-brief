import pytest
import json
from datetime import date, datetime, timezone
from app.extensions import db
from app.models.brief import DailyBrief, BriefSection
from app.models.source import Source


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


class TestAdminRoutes:
    def test_list_sources(self, client, sample_sources):
        resp = client.get('/api/admin/sources')
        assert resp.status_code == 200
        assert len(resp.json) == 3

    def test_add_source(self, client):
        resp = client.post('/api/admin/sources', json={
            'name': 'New Source',
            'url': 'https://newsource.com/feed.xml',
            'section': 'ai_news',
            'trust_score': 75,
        })
        assert resp.status_code == 201
        assert resp.json['name'] == 'New Source'

    def test_update_source(self, client, sample_sources):
        source_id = sample_sources[0].id
        resp = client.put(f'/api/admin/sources/{source_id}', json={
            'trust_score': 95,
        })
        assert resp.status_code == 200
        assert resp.json['trust_score'] == 95

    def test_delete_source(self, client, sample_sources):
        source_id = sample_sources[0].id
        resp = client.delete(f'/api/admin/sources/{source_id}')
        assert resp.status_code == 200
        assert resp.json['status'] == 'deactivated'

    def test_list_flags(self, client):
        resp = client.get('/api/admin/flags')
        assert resp.status_code == 200
        assert isinstance(resp.json, dict)

    def test_add_watchlist(self, client):
        resp = client.post('/api/admin/watchlists', json={
            'name': 'Test Topic',
            'description': 'A test tracked topic',
        })
        assert resp.status_code == 201


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
