from app.extensions import db
from sqlalchemy import func


class UserPreference(db.Model):
    __tablename__ = 'user_preferences'

    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.String(128), nullable=False, unique=True)
    value_json = db.Column(db.JSON, nullable=False)
    is_persistent = db.Column(db.Boolean, default=False)
    ttl_days = db.Column(db.Integer, nullable=True)
    created_at = db.Column(db.DateTime(timezone=True), server_default=func.now())
    expires_at = db.Column(db.DateTime(timezone=True), nullable=True)

    def to_dict(self):
        return {
            'id': self.id,
            'key': self.key,
            'value': self.value_json,
            'is_persistent': self.is_persistent,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
        }


class FeedbackAction(db.Model):
    __tablename__ = 'feedback_actions'

    id = db.Column(db.Integer, primary_key=True)
    action_type = db.Column(db.String(32), nullable=False)
    target_type = db.Column(db.String(32), nullable=False)
    target_id = db.Column(db.Integer, nullable=False)
    metadata_json = db.Column(db.JSON, nullable=True)
    created_at = db.Column(db.DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        db.Index('ix_feedback_target', 'target_type', 'target_id'),
        db.Index('ix_feedback_action_date', 'action_type', 'created_at'),
    )

    def to_dict(self):
        return {
            'id': self.id,
            'action_type': self.action_type,
            'target_type': self.target_type,
            'target_id': self.target_id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }


class DailyInsight(db.Model):
    __tablename__ = 'daily_insights'

    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, nullable=False)
    promoted_to_pref = db.Column(db.Boolean, default=False)
    pref_id = db.Column(db.Integer, db.ForeignKey('user_preferences.id'), nullable=True)
    created_at = db.Column(db.DateTime(timezone=True), server_default=func.now())
    expires_at = db.Column(db.DateTime(timezone=True), nullable=False)

    def to_dict(self):
        return {
            'id': self.id,
            'text': self.text,
            'promoted_to_pref': self.promoted_to_pref,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
        }
