from app.extensions import db
from sqlalchemy import func


class TrackedTopic(db.Model):
    __tablename__ = 'tracked_topics'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(256), nullable=False, unique=True)
    description = db.Column(db.Text)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime(timezone=True), server_default=func.now())

    stories = db.relationship('Story', back_populates='topic', lazy='dynamic')

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'is_active': self.is_active,
        }


class Story(db.Model):
    __tablename__ = 'stories'

    id = db.Column(db.Integer, primary_key=True)
    topic_id = db.Column(db.Integer, db.ForeignKey('tracked_topics.id'), nullable=False)
    title = db.Column(db.String(512), nullable=False)
    status = db.Column(db.String(32), default='developing')
    first_seen = db.Column(db.DateTime(timezone=True), server_default=func.now())
    last_updated = db.Column(db.DateTime(timezone=True))
    cluster_ids_json = db.Column(db.JSON)

    topic = db.relationship('TrackedTopic', back_populates='stories')
    events = db.relationship('Event', back_populates='story', order_by='Event.event_date')

    def to_dict(self):
        return {
            'id': self.id,
            'topic_id': self.topic_id,
            'title': self.title,
            'status': self.status,
            'first_seen': self.first_seen.isoformat() if self.first_seen else None,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None,
        }


class Event(db.Model):
    __tablename__ = 'events'

    id = db.Column(db.Integer, primary_key=True)
    story_id = db.Column(db.Integer, db.ForeignKey('stories.id'), nullable=False)
    cluster_id = db.Column(db.Integer, db.ForeignKey('clusters.id'), nullable=True)
    description = db.Column(db.Text, nullable=False)
    event_date = db.Column(db.DateTime(timezone=True))
    source_urls_json = db.Column(db.JSON)

    story = db.relationship('Story', back_populates='events')
