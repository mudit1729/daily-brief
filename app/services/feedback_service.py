import logging
from datetime import datetime, timedelta, timezone
from app.extensions import db
from app.models.user import FeedbackAction, DailyInsight, UserPreference

logger = logging.getLogger(__name__)

VALID_ACTIONS = ('follow', 'mute', 'upvote', 'downvote')
VALID_TARGETS = ('cluster', 'article', 'topic', 'story', 'source')
INSIGHT_TTL_DAYS = 10


class FeedbackService:
    def record_action(self, action_type, target_type, target_id, metadata=None):
        """Record a feedback action."""
        if action_type not in VALID_ACTIONS:
            raise ValueError(f"Invalid action_type: {action_type}")
        if target_type not in VALID_TARGETS:
            raise ValueError(f"Invalid target_type: {target_type}")

        action = FeedbackAction(
            action_type=action_type,
            target_type=target_type,
            target_id=target_id,
            metadata_json=metadata,
        )
        db.session.add(action)
        db.session.commit()
        return action

    def record_insight(self, text):
        """Record a daily insight with 10-day TTL."""
        now = datetime.now(timezone.utc)
        insight = DailyInsight(
            text=text,
            expires_at=now + timedelta(days=INSIGHT_TTL_DAYS),
        )
        db.session.add(insight)
        db.session.commit()
        return insight

    def promote_insight(self, insight_id):
        """Promote a daily insight to a persistent user preference."""
        insight = DailyInsight.query.get(insight_id)
        if not insight:
            raise ValueError(f"Insight {insight_id} not found")

        pref_key = f"insight.promoted.{insight_id}"
        pref = UserPreference(
            key=pref_key,
            value_json={'text': insight.text, 'source': 'promoted_insight'},
            is_persistent=True,
        )
        db.session.add(pref)

        insight.promoted_to_pref = True
        insight.pref_id = pref.id
        db.session.commit()
        return pref

    def get_active_insights(self):
        """Get non-expired daily insights."""
        now = datetime.now(timezone.utc)
        return DailyInsight.query.filter(
            DailyInsight.expires_at > now
        ).order_by(DailyInsight.created_at.desc()).all()

    def expire_old_insights(self):
        """Delete expired insights (called by scheduler)."""
        now = datetime.now(timezone.utc)
        expired = DailyInsight.query.filter(
            DailyInsight.expires_at <= now,
            DailyInsight.promoted_to_pref == False,
        ).all()
        count = len(expired)
        for insight in expired:
            db.session.delete(insight)
        db.session.commit()
        logger.info(f"Expired {count} daily insights")
        return count
