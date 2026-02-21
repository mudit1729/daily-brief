from app.extensions import db
from sqlalchemy import func


class Article(db.Model):
    __tablename__ = 'articles'

    id = db.Column(db.Integer, primary_key=True)
    source_id = db.Column(db.Integer, db.ForeignKey('sources.id'), nullable=False, index=True)
    url = db.Column(db.String(2048), nullable=False, unique=True)
    title = db.Column(db.String(1024))
    raw_html = db.Column(db.Text)
    extracted_text = db.Column(db.Text)
    summary = db.Column(db.Text)
    og_image_url = db.Column(db.String(2048))
    author = db.Column(db.String(256))
    published_at = db.Column(db.DateTime(timezone=True))
    fetched_at = db.Column(db.DateTime(timezone=True), server_default=func.now())
    word_count = db.Column(db.Integer)
    language = db.Column(db.String(8), default='en')
    entities_json = db.Column(db.JSON)
    is_duplicate = db.Column(db.Boolean, default=False)
    duplicate_of_id = db.Column(db.Integer, db.ForeignKey('articles.id'), nullable=True)

    source = db.relationship('Source', back_populates='articles')
    embedding = db.relationship('ArticleEmbedding', uselist=False, back_populates='article')
    clusters = db.relationship('ClusterMembership', back_populates='article')

    __table_args__ = (
        db.Index('ix_articles_source_published', 'source_id', 'published_at'),
        db.Index('ix_articles_fetched', 'fetched_at'),
    )

    def to_dict(self):
        return {
            'id': self.id,
            'source_id': self.source_id,
            'url': self.url,
            'title': self.title,
            'og_image_url': self.og_image_url,
            'author': self.author,
            'published_at': self.published_at.isoformat() if self.published_at else None,
            'word_count': self.word_count,
            'is_duplicate': self.is_duplicate,
        }
