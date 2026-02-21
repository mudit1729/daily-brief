import pytest
import uuid
from datetime import datetime, timezone, timedelta
from app.extensions import db
from app.models.cluster import Cluster, ClusterMembership
from app.models.article import Article
from app.models.source import Source
from app.models.user import FeedbackAction, DailyInsight
from app.services.ranking_service import RankingService


class TestRanking:
    def _make_cluster(self, db_session, source, section='general_news',
                      published_hours_ago=1, date_val=None):
        """Helper to create a cluster with one article."""
        from datetime import date
        date_val = date_val or date.today()

        article = Article(
            source_id=source.id,
            url=f'https://example.com/rank-{uuid.uuid4().hex}',
            title=f'Article from {source.name}',
            extracted_text='Test content about technology and policy changes.',
            published_at=datetime.now(timezone.utc) - timedelta(hours=published_hours_ago),
        )
        db_session.add(article)
        db_session.flush()

        cluster = Cluster(
            section=section,
            representative_article_id=article.id,
            article_count=1,
            date=date_val,
        )
        db_session.add(cluster)
        db_session.flush()

        membership = ClusterMembership(
            cluster_id=cluster.id,
            article_id=article.id,
            similarity=1.0,
        )
        db_session.add(membership)
        db_session.commit()

        return cluster, article

    def test_high_trust_ranks_higher(self, app, db_session, sample_sources):
        """Clusters from high-trust sources should rank higher."""
        high_trust_source = sample_sources[0]  # trust_score=90
        low_trust_source = sample_sources[2]    # trust_score=45

        c_high, a_high = self._make_cluster(db_session, high_trust_source)
        c_low, a_low = self._make_cluster(db_session, low_trust_source)

        service = RankingService()
        clusters_data = [
            {'cluster': c_high, 'articles': [a_high], 'sources': [high_trust_source]},
            {'cluster': c_low, 'articles': [a_low], 'sources': [low_trust_source]},
        ]
        service.rank_clusters(clusters_data)

        assert c_high.rank_score > c_low.rank_score

    def test_recent_articles_boosted(self, app, db_session, sample_sources):
        """More recent articles should get a recency boost."""
        source = sample_sources[0]

        c_recent, a_recent = self._make_cluster(db_session, source, published_hours_ago=1)
        c_old, a_old = self._make_cluster(db_session, source, published_hours_ago=48)

        service = RankingService()
        clusters_data = [
            {'cluster': c_recent, 'articles': [a_recent], 'sources': [source]},
            {'cluster': c_old, 'articles': [a_old], 'sources': [source]},
        ]
        service.rank_clusters(clusters_data)

        assert c_recent.rank_score > c_old.rank_score

    def test_upvote_boosts_rank(self, app, db_session, sample_sources):
        """Upvoted articles should boost their cluster's rank."""
        source = sample_sources[0]
        c1, a1 = self._make_cluster(db_session, source, published_hours_ago=2)
        c2, a2 = self._make_cluster(db_session, source, published_hours_ago=2)

        # Upvote articles in c1
        for _ in range(3):
            action = FeedbackAction(
                action_type='upvote',
                target_type='article',
                target_id=a1.id,
            )
            db_session.add(action)
        db_session.commit()

        service = RankingService()
        clusters_data = [
            {'cluster': c1, 'articles': [a1], 'sources': [source]},
            {'cluster': c2, 'articles': [a2], 'sources': [source]},
        ]
        service.rank_clusters(clusters_data)

        assert c1.rank_score > c2.rank_score

    def test_expired_insight_no_boost(self, app, db_session, sample_sources):
        """Expired insights should not affect ranking."""
        source = sample_sources[0]
        c1, a1 = self._make_cluster(db_session, source)

        # Create an expired insight
        expired_insight = DailyInsight(
            text='technology policy',
            expires_at=datetime.now(timezone.utc) - timedelta(days=1),
        )
        db_session.add(expired_insight)
        db_session.commit()

        service = RankingService()
        active_insights = service._get_active_insights()
        assert len(active_insights) == 0
