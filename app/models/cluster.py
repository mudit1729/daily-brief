from app.extensions import db
from sqlalchemy import func


class Cluster(db.Model):
    __tablename__ = 'clusters'

    id = db.Column(db.Integer, primary_key=True)
    section = db.Column(db.String(64), nullable=False)
    label = db.Column(db.String(512))
    summary = db.Column(db.Text)
    representative_article_id = db.Column(db.Integer, db.ForeignKey('articles.id'))
    article_count = db.Column(db.Integer, default=0)
    avg_trust_score = db.Column(db.Float)
    rank_score = db.Column(db.Float)
    brief_id = db.Column(db.Integer, db.ForeignKey('daily_briefs.id'), nullable=True)
    created_at = db.Column(db.DateTime(timezone=True), server_default=func.now())
    date = db.Column(db.Date, nullable=False, index=True)

    members = db.relationship('ClusterMembership', back_populates='cluster', cascade='all, delete-orphan')
    representative_article = db.relationship('Article', foreign_keys=[representative_article_id])

    __table_args__ = (
        db.Index('ix_clusters_section_date_rank', 'section', 'date', 'rank_score'),
    )

    def to_dict(self):
        return {
            'id': self.id,
            'section': self.section,
            'label': self.label,
            'summary': self.summary,
            'article_count': self.article_count,
            'avg_trust_score': self.avg_trust_score,
            'rank_score': self.rank_score,
            'date': self.date.isoformat() if self.date else None,
        }


class ClusterMembership(db.Model):
    __tablename__ = 'cluster_memberships'

    id = db.Column(db.Integer, primary_key=True)
    cluster_id = db.Column(db.Integer, db.ForeignKey('clusters.id'), nullable=False)
    article_id = db.Column(db.Integer, db.ForeignKey('articles.id'), nullable=False)
    similarity = db.Column(db.Float)

    cluster = db.relationship('Cluster', back_populates='members')
    article = db.relationship('Article', back_populates='clusters')

    __table_args__ = (
        db.UniqueConstraint('cluster_id', 'article_id', name='uq_cluster_article'),
    )
