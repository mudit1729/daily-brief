from datetime import datetime, date, timedelta, timezone
from app.extensions import db


class CalendarEvent(db.Model):
    __tablename__ = 'calendar_events'

    id = db.Column(db.Integer, primary_key=True)
    event_date = db.Column(db.Date, nullable=False, index=True)
    event_time = db.Column(db.Time, nullable=True)
    end_time = db.Column(db.Time, nullable=True)
    title = db.Column(db.String(256), nullable=False)
    description = db.Column(db.Text, nullable=True)
    color = db.Column(db.String(7), nullable=True, default='#6366f1')

    # Recurrence: none | daily | weekly | biweekly | monthly | yearly
    recurrence = db.Column(db.String(16), nullable=True, default=None)
    recurrence_end = db.Column(db.Date, nullable=True)  # null = forever

    created_at = db.Column(
        db.DateTime, default=lambda: datetime.now(timezone.utc)
    )

    def to_dict(self, override_date=None):
        return {
            'id': self.id,
            'event_date': (override_date or self.event_date).isoformat(),
            'event_time': self.event_time.strftime('%H:%M') if self.event_time else None,
            'end_time': self.end_time.strftime('%H:%M') if self.end_time else None,
            'title': self.title,
            'description': self.description or '',
            'color': self.color or '#6366f1',
            'recurrence': self.recurrence,
            'recurrence_end': self.recurrence_end.isoformat() if self.recurrence_end else None,
        }

    def occurrences_in_range(self, start_date, end_date):
        """Yield (date, self) for every occurrence within [start_date, end_date]."""
        if not self.recurrence:
            if start_date <= self.event_date <= end_date:
                yield self.event_date
            return

        rec_end = self.recurrence_end or end_date
        bound = min(rec_end, end_date)
        cur = self.event_date

        while cur <= bound:
            if cur >= start_date:
                yield cur
            cur = self._next_occurrence(cur)
            if cur is None:
                break

    def _next_occurrence(self, d):
        r = self.recurrence
        if r == 'daily':
            return d + timedelta(days=1)
        if r == 'weekly':
            return d + timedelta(weeks=1)
        if r == 'biweekly':
            return d + timedelta(weeks=2)
        if r == 'monthly':
            m = d.month + 1
            y = d.year + (m - 1) // 12
            m = (m - 1) % 12 + 1
            day = min(d.day, _days_in_month(y, m))
            return date(y, m, day)
        if r == 'yearly':
            try:
                return d.replace(year=d.year + 1)
            except ValueError:  # Feb 29
                return date(d.year + 1, 3, 1)
        return None


def _days_in_month(year, month):
    if month == 12:
        return 31
    return (date(year, month + 1, 1) - timedelta(days=1)).day
