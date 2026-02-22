from app.extensions import db
from sqlalchemy import func


class SystemSetting(db.Model):
    __tablename__ = 'system_settings'

    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.String(128), nullable=False, unique=True)
    value_json = db.Column(db.JSON, nullable=False, default=dict)
    created_at = db.Column(db.DateTime(timezone=True), server_default=func.now())
    updated_at = db.Column(db.DateTime(timezone=True), onupdate=func.now())

    @classmethod
    def get_value(cls, key, default=None):
        row = cls.query.filter_by(key=key).first()
        if not row:
            return default
        return row.value_json

    @classmethod
    def set_value(cls, key, value):
        row = cls.query.filter_by(key=key).first()
        if not row:
            row = cls(key=key, value_json=value)
            db.session.add(row)
        else:
            row.value_json = value
        return row

