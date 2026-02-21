from app.extensions import db
from sqlalchemy import func


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
    created_at = db.Column(db.DateTime(timezone=True), server_default=func.now())
    updated_at = db.Column(db.DateTime(timezone=True), onupdate=func.now())

    articles = db.relationship('Article', back_populates='source', lazy='dynamic')

    __table_args__ = (
        db.Index('ix_sources_section_active', 'section', 'is_active'),
    )

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
        }
