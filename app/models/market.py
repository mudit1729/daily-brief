from app.extensions import db
from sqlalchemy import func


class MarketSnapshot(db.Model):
    __tablename__ = 'market_snapshots'

    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(32), nullable=False)
    name = db.Column(db.String(128))
    price = db.Column(db.Float)
    change_pct = db.Column(db.Float)
    change_abs = db.Column(db.Float)
    volume = db.Column(db.BigInteger, nullable=True)
    snapshot_date = db.Column(db.Date, nullable=False)
    fetched_at = db.Column(db.DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        db.Index('ix_market_symbol_date', 'symbol', 'snapshot_date'),
    )

    def to_dict(self):
        return {
            'symbol': self.symbol,
            'name': self.name,
            'price': self.price,
            'change_pct': self.change_pct,
            'change_abs': self.change_abs,
            'volume': self.volume,
            'snapshot_date': self.snapshot_date.isoformat(),
        }


class MarketDriver(db.Model):
    __tablename__ = 'market_drivers'

    id = db.Column(db.Integer, primary_key=True)
    snapshot_id = db.Column(db.Integer, db.ForeignKey('market_snapshots.id'))
    cluster_id = db.Column(db.Integer, db.ForeignKey('clusters.id'), nullable=True)
    driver_text = db.Column(db.Text)
    confidence = db.Column(db.Float)
    date = db.Column(db.Date, nullable=False)
