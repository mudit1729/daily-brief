from app.extensions import db
from sqlalchemy import func


class HedgeFundAnalysis(db.Model):
    """Per-ticker AI hedge fund analysis results."""
    __tablename__ = 'hedge_fund_analyses'

    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, nullable=False)
    brief_id = db.Column(db.Integer, db.ForeignKey('daily_briefs.id'), nullable=True)
    ticker = db.Column(db.String(16), nullable=False)

    # Per-analyst signals: {agent_name: {signal, confidence, reasoning}}
    analyst_signals_json = db.Column(db.JSON)

    # Consensus derived from analyst signals
    consensus_signal = db.Column(db.String(16))     # bullish / bearish / neutral
    consensus_confidence = db.Column(db.Float)       # 0–100

    # Which analysts were used
    analysts_used = db.Column(db.JSON)               # ["technicals", "valuation", ...]

    # Portfolio manager decision (if risk + portfolio agents ran)
    decision_json = db.Column(db.JSON)               # {action, quantity, confidence, reasoning}

    total_cost_usd = db.Column(db.Float, default=0.0)
    created_at = db.Column(db.DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        db.Index('ix_hf_analysis_date_ticker', 'date', 'ticker'),
        db.UniqueConstraint('date', 'ticker', name='uq_hf_analysis_date_ticker'),
    )

    def to_dict(self):
        return {
            'id': self.id,
            'date': self.date.isoformat() if self.date else None,
            'ticker': self.ticker,
            'consensus_signal': self.consensus_signal,
            'consensus_confidence': self.consensus_confidence,
            'analyst_signals': self.analyst_signals_json,
            'analysts_used': self.analysts_used,
            'decision': self.decision_json,
        }
