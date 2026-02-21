from app.extensions import db
from sqlalchemy import func


class DailyBrief(db.Model):
    __tablename__ = 'daily_briefs'

    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, nullable=False, unique=True, index=True)
    status = db.Column(db.String(32), default='pending')
    total_tokens = db.Column(db.Integer, default=0)
    total_cost_usd = db.Column(db.Float, default=0.0)
    idiot_index = db.Column(db.Float, nullable=True)
    generated_at = db.Column(db.DateTime(timezone=True))
    created_at = db.Column(db.DateTime(timezone=True), server_default=func.now())

    sections = db.relationship(
        'BriefSection', back_populates='brief',
        order_by='BriefSection.display_order',
        cascade='all, delete-orphan',
    )

    def to_dict(self):
        return {
            'id': self.id,
            'date': self.date.isoformat(),
            'status': self.status,
            'total_tokens': self.total_tokens,
            'total_cost_usd': self.total_cost_usd,
            'idiot_index': self.idiot_index,
            'generated_at': self.generated_at.isoformat() if self.generated_at else None,
            'sections': [s.to_dict() for s in self.sections],
        }


class BriefSection(db.Model):
    __tablename__ = 'brief_sections'

    id = db.Column(db.Integer, primary_key=True)
    brief_id = db.Column(db.Integer, db.ForeignKey('daily_briefs.id'), nullable=False)
    section_type = db.Column(db.String(64), nullable=False)
    title = db.Column(db.String(256))
    content_json = db.Column(db.JSON)
    content_html = db.Column(db.Text)
    display_order = db.Column(db.Integer, default=0)
    tokens_used = db.Column(db.Integer, default=0)
    cost_usd = db.Column(db.Float, default=0.0)
    degradation_level = db.Column(db.Integer, default=0)

    brief = db.relationship('DailyBrief', back_populates='sections')

    def to_dict(self):
        return {
            'id': self.id,
            'section_type': self.section_type,
            'title': self.title,
            'content_json': self.content_json,
            'display_order': self.display_order,
            'tokens_used': self.tokens_used,
            'cost_usd': self.cost_usd,
            'degradation_level': self.degradation_level,
        }
