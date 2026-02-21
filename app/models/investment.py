from app.extensions import db
from sqlalchemy import func


class InvestmentThesis(db.Model):
    __tablename__ = 'investment_theses'

    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, nullable=False)
    brief_id = db.Column(db.Integer, db.ForeignKey('daily_briefs.id'), nullable=True)
    thesis_text = db.Column(db.Text)
    momentum_signal = db.Column(db.JSON)
    value_signal = db.Column(db.JSON)
    gate_passed = db.Column(db.Boolean, default=False)
    supporting_clusters_json = db.Column(db.JSON)
    created_at = db.Column(db.DateTime(timezone=True), server_default=func.now())
