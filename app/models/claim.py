from app.extensions import db
from sqlalchemy import func


class ClaimLedger(db.Model):
    __tablename__ = 'claim_ledger'

    id = db.Column(db.Integer, primary_key=True)
    story_id = db.Column(db.Integer, db.ForeignKey('stories.id'), nullable=True)
    claim_text = db.Column(db.Text, nullable=False)
    source_id = db.Column(db.Integer, db.ForeignKey('sources.id'))
    article_id = db.Column(db.Integer, db.ForeignKey('articles.id'))
    confidence = db.Column(db.Float)
    status = db.Column(db.String(32), default='unverified')
    contradicts_claim_id = db.Column(db.Integer, db.ForeignKey('claim_ledger.id'), nullable=True)
    evidence_json = db.Column(db.JSON)
    created_at = db.Column(db.DateTime(timezone=True), server_default=func.now())
    updated_at = db.Column(db.DateTime(timezone=True), onupdate=func.now())

    __table_args__ = (
        db.Index('ix_claims_story_status', 'story_id', 'status'),
        db.Index('ix_claims_article', 'article_id'),
    )
