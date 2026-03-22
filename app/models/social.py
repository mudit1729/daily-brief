from datetime import datetime, timezone
from app.extensions import db


class SocialChannel(db.Model):
    """A followed social media channel / feed source."""
    __tablename__ = 'social_channels'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(256), nullable=False)
    handle = db.Column(db.String(256), nullable=True)
    platform = db.Column(db.String(32), nullable=False)  # youtube, twitter, substack, rss
    feed_url = db.Column(db.String(2048), unique=True, nullable=False)
    avatar_url = db.Column(db.String(2048), nullable=True)
    description = db.Column(db.Text, nullable=True)
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    last_fetched_at = db.Column(db.DateTime(timezone=True), nullable=True)
    last_success_at = db.Column(db.DateTime(timezone=True), nullable=True)
    consecutive_failures = db.Column(db.Integer, default=0, nullable=False)
    last_error = db.Column(db.String(512), nullable=True)
    refresh_interval_hours = db.Column(db.Integer, default=4, nullable=False)
    created_at = db.Column(
        db.DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    posts = db.relationship('SocialPost', back_populates='channel', lazy='dynamic',
                            order_by='SocialPost.published_at.desc()')

    __table_args__ = (
        db.Index('ix_social_channels_platform_active', 'platform', 'is_active'),
    )

    def to_dict(self, include_stats=False):
        d = {
            'id': self.id,
            'name': self.name,
            'handle': self.handle,
            'platform': self.platform,
            'feed_url': self.feed_url,
            'avatar_url': self.avatar_url,
            'description': self.description,
            'is_active': self.is_active,
            'last_fetched_at': self.last_fetched_at.isoformat() if self.last_fetched_at else None,
            'refresh_interval_hours': self.refresh_interval_hours or 4,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }
        if include_stats:
            d['post_count'] = getattr(self, 'post_count', 0)
            d['last_post_date'] = getattr(self, 'last_post_date', None)
            if d['last_post_date'] and hasattr(d['last_post_date'], 'isoformat'):
                d['last_post_date'] = d['last_post_date'].isoformat()
        return d


class SocialPost(db.Model):
    """A single post / video / article from a followed channel."""
    __tablename__ = 'social_posts'

    id = db.Column(db.Integer, primary_key=True)
    channel_id = db.Column(db.Integer, db.ForeignKey('social_channels.id'), nullable=False)
    url = db.Column(db.String(2048), unique=True, nullable=False)
    title = db.Column(db.String(1024), nullable=False, default='')
    author = db.Column(db.String(256), nullable=True)
    published_at = db.Column(db.DateTime(timezone=True), nullable=True)
    fetched_at = db.Column(
        db.DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    content_hint = db.Column(db.Text, nullable=True)  # RSS summary snippet
    summary_md = db.Column(db.Text, nullable=True)  # LLM-generated markdown
    summary_generated_at = db.Column(db.DateTime(timezone=True), nullable=True)
    summary_cost_usd = db.Column(db.Float, nullable=True)
    summary_tokens = db.Column(db.Integer, nullable=True)

    channel = db.relationship('SocialChannel', back_populates='posts')

    __table_args__ = (
        db.Index('ix_social_posts_channel_published', 'channel_id', 'published_at'),
        db.Index('ix_social_posts_published', 'published_at'),
    )

    def to_dict(self):
        return {
            'id': self.id,
            'channel_id': self.channel_id,
            'channel_name': self.channel.name if self.channel else None,
            'platform': self.channel.platform if self.channel else None,
            'url': self.url,
            'title': self.title,
            'author': self.author,
            'published_at': self.published_at.isoformat() if self.published_at else None,
            'content_hint': self.content_hint,
            'summary_md': self.summary_md,
            'summary_generated_at': self.summary_generated_at.isoformat() if self.summary_generated_at else None,
            'has_summary': self.summary_md is not None,
        }
