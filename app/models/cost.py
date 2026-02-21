from app.extensions import db
from sqlalchemy import func


class LLMCallLog(db.Model):
    __tablename__ = 'llm_call_logs'

    id = db.Column(db.Integer, primary_key=True)
    call_purpose = db.Column(db.String(128), nullable=False)
    model = db.Column(db.String(64), nullable=False)
    prompt_tokens = db.Column(db.Integer, nullable=False)
    completion_tokens = db.Column(db.Integer, nullable=False)
    total_tokens = db.Column(db.Integer, nullable=False)
    cost_usd = db.Column(db.Float, nullable=False)
    latency_ms = db.Column(db.Integer)
    section = db.Column(db.String(64), nullable=True)
    brief_id = db.Column(db.Integer, db.ForeignKey('daily_briefs.id'), nullable=True)
    created_at = db.Column(db.DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        db.Index('ix_llm_logs_brief', 'brief_id'),
        db.Index('ix_llm_logs_section_date', 'section', 'created_at'),
    )

    def to_dict(self):
        return {
            'id': self.id,
            'call_purpose': self.call_purpose,
            'model': self.model,
            'prompt_tokens': self.prompt_tokens,
            'completion_tokens': self.completion_tokens,
            'total_tokens': self.total_tokens,
            'cost_usd': self.cost_usd,
            'latency_ms': self.latency_ms,
            'section': self.section,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }


class DailyCostSummary(db.Model):
    __tablename__ = 'daily_cost_summaries'

    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, nullable=False, unique=True)
    total_tokens = db.Column(db.Integer, default=0)
    total_cost_usd = db.Column(db.Float, default=0.0)
    calls_count = db.Column(db.Integer, default=0)
    budget_usd = db.Column(db.Float)
    budget_remaining = db.Column(db.Float)
    idiot_index = db.Column(db.Float, nullable=True)
    breakdown_json = db.Column(db.JSON)

    def to_dict(self):
        return {
            'date': self.date.isoformat(),
            'total_tokens': self.total_tokens,
            'total_cost_usd': self.total_cost_usd,
            'calls_count': self.calls_count,
            'budget_usd': self.budget_usd,
            'budget_remaining': self.budget_remaining,
            'idiot_index': self.idiot_index,
            'breakdown': self.breakdown_json,
        }
