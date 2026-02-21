from app.extensions import db
from sqlalchemy import func


class WeatherCache(db.Model):
    __tablename__ = 'weather_cache'

    id = db.Column(db.Integer, primary_key=True)
    location_name = db.Column(db.String(128), nullable=False)
    latitude = db.Column(db.Float, nullable=False)
    longitude = db.Column(db.Float, nullable=False)
    date = db.Column(db.Date, nullable=False)
    data_json = db.Column(db.JSON, nullable=False)
    fetched_at = db.Column(db.DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        db.UniqueConstraint('location_name', 'date', name='uq_weather_location_date'),
    )

    def to_dict(self):
        return {
            'location_name': self.location_name,
            'date': self.date.isoformat(),
            'data': self.data_json,
        }
