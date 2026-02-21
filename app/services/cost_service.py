import logging
from datetime import date, datetime, timezone
from app.extensions import db
from app.models.cost import LLMCallLog, DailyCostSummary
from app.models.user import FeedbackAction

logger = logging.getLogger(__name__)


class CostService:
    def get_daily_usage(self, target_date=None):
        """Get total token usage and cost for a given date."""
        target_date = target_date or date.today()
        start = datetime(target_date.year, target_date.month, target_date.day, tzinfo=timezone.utc)
        end = datetime(target_date.year, target_date.month, target_date.day, 23, 59, 59, tzinfo=timezone.utc)

        result = db.session.query(
            db.func.coalesce(db.func.sum(LLMCallLog.total_tokens), 0),
            db.func.coalesce(db.func.sum(LLMCallLog.cost_usd), 0.0),
            db.func.count(LLMCallLog.id),
        ).filter(
            LLMCallLog.created_at >= start,
            LLMCallLog.created_at <= end,
        ).first()

        return {
            'total_tokens': result[0],
            'total_cost_usd': round(float(result[1]), 6),
            'calls_count': result[2],
        }

    def get_section_usage(self, section, target_date=None):
        """Get token usage for a specific section on a given date."""
        target_date = target_date or date.today()
        start = datetime(target_date.year, target_date.month, target_date.day, tzinfo=timezone.utc)

        result = db.session.query(
            db.func.coalesce(db.func.sum(LLMCallLog.total_tokens), 0),
            db.func.coalesce(db.func.sum(LLMCallLog.cost_usd), 0.0),
        ).filter(
            LLMCallLog.created_at >= start,
            LLMCallLog.section == section,
        ).first()

        return {
            'total_tokens': result[0],
            'total_cost_usd': round(float(result[1]), 6),
        }

    def compute_idiot_index(self, target_date=None):
        """
        idiot_index = cost_usd / max(engagement_signals, 1)
        engagement_signals = count of feedback actions today
        """
        target_date = target_date or date.today()
        usage = self.get_daily_usage(target_date)

        start = datetime(target_date.year, target_date.month, target_date.day, tzinfo=timezone.utc)
        engagement = FeedbackAction.query.filter(
            FeedbackAction.created_at >= start
        ).count()

        engagement = max(engagement, 1)
        return round(usage['total_cost_usd'] / engagement, 6)

    def create_daily_summary(self, target_date=None, budget_usd=None):
        """Create or update the daily cost summary."""
        target_date = target_date or date.today()
        usage = self.get_daily_usage(target_date)
        idiot_index = self.compute_idiot_index(target_date)

        # Per-section breakdown
        start = datetime(target_date.year, target_date.month, target_date.day, tzinfo=timezone.utc)
        section_rows = db.session.query(
            LLMCallLog.section,
            db.func.sum(LLMCallLog.total_tokens),
            db.func.sum(LLMCallLog.cost_usd),
        ).filter(
            LLMCallLog.created_at >= start,
            LLMCallLog.section.isnot(None),
        ).group_by(LLMCallLog.section).all()

        breakdown = {
            row[0]: {'tokens': row[1], 'cost_usd': round(float(row[2]), 6)}
            for row in section_rows
        }

        summary = DailyCostSummary.query.filter_by(date=target_date).first()
        if not summary:
            summary = DailyCostSummary(date=target_date)
            db.session.add(summary)

        summary.total_tokens = usage['total_tokens']
        summary.total_cost_usd = usage['total_cost_usd']
        summary.calls_count = usage['calls_count']
        summary.budget_usd = budget_usd
        summary.budget_remaining = (budget_usd - usage['total_cost_usd']) if budget_usd else None
        summary.idiot_index = idiot_index
        summary.breakdown_json = breakdown

        db.session.commit()
        return summary
