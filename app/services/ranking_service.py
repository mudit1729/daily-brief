import logging
import math
from datetime import datetime, timezone
from app.extensions import db
from app.models.user import FeedbackAction, DailyInsight, UserPreference

logger = logging.getLogger(__name__)

# Ranking weight config
WEIGHT_TRUST = 0.40
WEIGHT_RECENCY = 0.25
WEIGHT_DIVERSITY = 0.15
WEIGHT_PREFERENCE = 0.20

# Recency decay half-life in hours
RECENCY_HALF_LIFE = 12.0


class RankingService:
    def rank_clusters(self, clusters_with_articles):
        """
        Rank clusters by composite score.
        clusters_with_articles: list of dicts with:
            - cluster: Cluster model
            - articles: list of Article models
            - sources: list of Source models
        Returns clusters sorted by rank_score (descending).
        """
        now = datetime.now(timezone.utc)
        active_insights = self._get_active_insights()
        preference_boosts = self._get_preference_boosts()

        for item in clusters_with_articles:
            cluster = item['cluster']
            articles = item['articles']
            sources = item['sources']

            # Trust score: weighted average of source trust scores
            trust_scores = [s.trust_score for s in sources if s.trust_score]
            avg_trust = sum(trust_scores) / len(trust_scores) if trust_scores else 50
            trust_component = (avg_trust / 100.0) * WEIGHT_TRUST

            # Recency: exponential decay based on most recent article
            pub_times = [a.published_at for a in articles if a.published_at]
            if pub_times:
                most_recent = max(pub_times)
                # Handle both naive and aware datetimes
                if most_recent.tzinfo is None:
                    most_recent = most_recent.replace(tzinfo=timezone.utc)
                hours_ago = (now - most_recent).total_seconds() / 3600
                recency_component = math.exp(-0.693 * hours_ago / RECENCY_HALF_LIFE) * WEIGHT_RECENCY
            else:
                recency_component = 0.5 * WEIGHT_RECENCY

            # Diversity: source diversity within cluster (more sources = higher)
            unique_sources = len(set(s.id for s in sources))
            diversity_component = min(unique_sources / 3.0, 1.0) * WEIGHT_DIVERSITY

            # Preference: check feedback actions and daily insights
            pref_component = self._compute_preference_score(
                cluster, articles, active_insights, preference_boosts
            ) * WEIGHT_PREFERENCE

            rank_score = trust_component + recency_component + diversity_component + pref_component

            cluster.rank_score = round(rank_score, 4)
            cluster.avg_trust_score = round(avg_trust, 1)

        clusters_with_articles.sort(key=lambda x: x['cluster'].rank_score or 0, reverse=True)
        return clusters_with_articles

    def _compute_preference_score(self, cluster, articles, insights, boosts):
        """Compute preference boost from feedback and insights."""
        score = 0.5  # neutral baseline

        # Check upvotes/downvotes for articles in this cluster
        article_ids = [a.id for a in articles]
        if article_ids:
            upvotes = FeedbackAction.query.filter(
                FeedbackAction.target_type == 'article',
                FeedbackAction.target_id.in_(article_ids),
                FeedbackAction.action_type == 'upvote',
            ).count()
            downvotes = FeedbackAction.query.filter(
                FeedbackAction.target_type == 'article',
                FeedbackAction.target_id.in_(article_ids),
                FeedbackAction.action_type == 'downvote',
            ).count()
            score += (upvotes - downvotes) * 0.1

        # Check if any insight matches cluster content
        cluster_text = (cluster.label or '') + ' ' + ' '.join(a.title or '' for a in articles)
        cluster_text_lower = cluster_text.lower()
        for insight in insights:
            if insight.text.lower() in cluster_text_lower:
                score += 0.3

        # Check muted sources
        source_ids = list(set(a.source_id for a in articles))
        muted = FeedbackAction.query.filter(
            FeedbackAction.target_type == 'source',
            FeedbackAction.target_id.in_(source_ids),
            FeedbackAction.action_type == 'mute',
        ).count()
        score -= muted * 0.2

        return max(0, min(1, score))

    def _get_active_insights(self):
        """Get non-expired daily insights."""
        now = datetime.now(timezone.utc)
        return DailyInsight.query.filter(
            DailyInsight.expires_at > now
        ).all()

    def _get_preference_boosts(self):
        """Get active user preferences for ranking boosts."""
        return UserPreference.query.filter(
            UserPreference.key.like('section_weight.%')
        ).all()
