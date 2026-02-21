from app.extensions import db
from sqlalchemy import func


class ArticleEmbedding(db.Model):
    __tablename__ = 'article_embeddings'

    id = db.Column(db.Integer, primary_key=True)
    article_id = db.Column(db.Integer, db.ForeignKey('articles.id'), nullable=False, unique=True)
    simhash = db.Column(db.BigInteger, nullable=False, index=True)
    embedding_blob = db.Column(db.LargeBinary, nullable=True)
    embedding_model = db.Column(db.String(64), default='text-embedding-3-small')
    embedding_dim = db.Column(db.Integer, default=1536)
    created_at = db.Column(db.DateTime(timezone=True), server_default=func.now())

    article = db.relationship('Article', back_populates='embedding')
