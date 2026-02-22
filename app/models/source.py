from app.extensions import db
from sqlalchemy import func
from datetime import datetime, timezone


class Source(db.Model):
    __tablename__ = 'sources'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(256), nullable=False)
    url = db.Column(db.String(2048), nullable=False, unique=True)
    feed_type = db.Column(db.String(32), default='rss')
    section = db.Column(db.String(64), nullable=False)
    region = db.Column(db.String(32))
    bias_label = db.Column(db.String(16), default='center')
    trust_score = db.Column(db.Integer, default=50)
    source_type = db.Column(db.String(32), default='reporting')
    is_active = db.Column(db.Boolean, default=True)
    fetch_interval_min = db.Column(db.Integer, default=60)
    last_fetched_at = db.Column(db.DateTime(timezone=True))
    last_success_at = db.Column(db.DateTime(timezone=True))
    last_failure_at = db.Column(db.DateTime(timezone=True))
    consecutive_successes = db.Column(db.Integer, default=0)
    consecutive_failures = db.Column(db.Integer, default=0)
    total_failures = db.Column(db.Integer, default=0)
    avg_latency_ms = db.Column(db.Float)
    last_error = db.Column(db.String(512))
    auto_disabled_until = db.Column(db.DateTime(timezone=True))
    created_at = db.Column(db.DateTime(timezone=True), server_default=func.now())
    updated_at = db.Column(db.DateTime(timezone=True), onupdate=func.now())

    articles = db.relationship('Article', back_populates='source', lazy='dynamic')

    __table_args__ = (
        db.Index('ix_sources_section_active', 'section', 'is_active'),
        db.Index('ix_sources_auto_disabled_until', 'auto_disabled_until'),
    )

    def health_state(self, now=None):
        now = now or datetime.now(timezone.utc)
        until = self.auto_disabled_until
        if until:
            if until.tzinfo is None:
                until = until.replace(tzinfo=timezone.utc)
            else:
                until = until.astimezone(timezone.utc)
        if not self.is_active:
            return 'inactive'
        if until and until > now:
            return 'cooldown'
        if (self.consecutive_failures or 0) >= 2:
            return 'degraded'
        return 'healthy'

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'url': self.url,
            'feed_type': self.feed_type,
            'section': self.section,
            'region': self.region,
            'bias_label': self.bias_label,
            'trust_score': self.trust_score,
            'source_type': self.source_type,
            'is_active': self.is_active,
            'fetch_interval_min': self.fetch_interval_min,
            'last_fetched_at': self.last_fetched_at.isoformat() if self.last_fetched_at else None,
            'last_success_at': self.last_success_at.isoformat() if self.last_success_at else None,
            'last_failure_at': self.last_failure_at.isoformat() if self.last_failure_at else None,
            'consecutive_successes': self.consecutive_successes or 0,
            'consecutive_failures': self.consecutive_failures or 0,
            'total_failures': self.total_failures or 0,
            'avg_latency_ms': round(self.avg_latency_ms, 1) if self.avg_latency_ms is not None else None,
            'last_error': self.last_error,
            'auto_disabled_until': self.auto_disabled_until.isoformat() if self.auto_disabled_until else None,
            'health_state': self.health_state(),
        }
