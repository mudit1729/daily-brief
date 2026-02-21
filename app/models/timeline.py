from app.extensions import db
from sqlalchemy import func


class Timeline(db.Model):
    __tablename__ = 'timelines'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(256), nullable=False, unique=True)
    description = db.Column(db.Text)
    entities_json = db.Column(db.JSON)        # e.g. ["OpenAI", "Anthropic"]
    sections = db.Column(db.JSON)             # sections to pull from, e.g. ["ai_news"]
    icon = db.Column(db.String(8))            # emoji icon
    is_active = db.Column(db.Boolean, default=True)
    auto_update = db.Column(db.Boolean, default=True)  # auto-append new events from pipeline
    created_at = db.Column(db.DateTime(timezone=True), server_default=func.now())
    updated_at = db.Column(db.DateTime(timezone=True), onupdate=func.now())

    events = db.relationship(
        'TimelineEvent', back_populates='timeline',
        order_by='TimelineEvent.event_date.desc()',
        cascade='all, delete-orphan',
    )

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'entities': self.entities_json or [],
            'sections': self.sections or [],
            'icon': self.icon,
            'is_active': self.is_active,
            'auto_update': self.auto_update,
            'event_count': len(self.events) if self.events else 0,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }


class TimelineEvent(db.Model):
    __tablename__ = 'timeline_events'

    id = db.Column(db.Integer, primary_key=True)
    timeline_id = db.Column(db.Integer, db.ForeignKey('timelines.id'), nullable=False)
    event_date = db.Column(db.Date, nullable=False, index=True)
    title = db.Column(db.String(512), nullable=False)
    summary = db.Column(db.Text)
    entity = db.Column(db.String(128))          # primary entity, e.g. "OpenAI"
    event_type = db.Column(db.String(64))       # release, policy, partnership, funding, etc.
    significance = db.Column(db.Integer, default=5)  # 1-10 scale
    source_urls_json = db.Column(db.JSON)       # list of source URLs
    cluster_id = db.Column(db.Integer, db.ForeignKey('clusters.id'), nullable=True)
    article_id = db.Column(db.Integer, db.ForeignKey('articles.id'), nullable=True)
    metadata_json = db.Column(db.JSON)          # extra structured data
    created_at = db.Column(db.DateTime(timezone=True), server_default=func.now())

    timeline = db.relationship('Timeline', back_populates='events')

    __table_args__ = (
        db.Index('ix_timeline_events_date_entity', 'timeline_id', 'event_date', 'entity'),
    )

    def to_dict(self):
        return {
            'id': self.id,
            'event_date': self.event_date.isoformat() if self.event_date else None,
            'title': self.title,
            'summary': self.summary,
            'entity': self.entity,
            'event_type': self.event_type,
            'significance': self.significance,
            'source_urls': self.source_urls_json or [],
            'cluster_id': self.cluster_id,
        }
