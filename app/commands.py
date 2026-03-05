"""Flask CLI commands for seeding data."""

import click
from datetime import date, time
from flask import current_app
from flask.cli import with_appcontext

from app.extensions import db
from app.models.calendar_event import CalendarEvent

SEED_TAG = '[auto-seeded]'

# ── Colors ──────────────────────────────────────────────────────────────
COLOR_EARNINGS = '#ef4444'
COLOR_EKADASHI = '#f59e0b'
COLOR_FESTIVAL = '#ec4899'
COLOR_FARMERS  = '#22c55e'
COLOR_SD_EVENT = '#3b82f6'


# ── Event Data ──────────────────────────────────────────────────────────

def _earnings_events():
    """Top 10 stocks by market cap — next earnings dates."""
    data = [
        ('NVDA',   '2026-05-20', 'After market close. FQ1 2027 earnings.'),
        ('AAPL',   '2026-04-30', 'After market close. FQ2 2026 earnings.'),
        ('GOOGL',  '2026-04-28', 'After market close. Q1 2026 earnings.'),
        ('MSFT',   '2026-04-28', 'After market close. FQ3 2026 earnings.'),
        ('AMZN',   '2026-04-23', 'After market close. Q1 2026 earnings.'),
        ('TSM',    '2026-04-16', 'Q1 2026 earnings report.'),
        ('META',   '2026-04-29', 'After market close. Q1 2026 earnings.'),
        ('TSLA',   '2026-04-28', 'After market close. Q1 2026 earnings.'),
        ('AVGO',   '2026-06-03', 'After market close. FQ2 2026 earnings.'),
        ('Aramco', '2026-03-10', 'Full-year 2025 results.'),
    ]
    events = []
    for ticker, dt, note in data:
        events.append(CalendarEvent(
            title=f'📊 {ticker} Earnings Report',
            event_date=date.fromisoformat(dt),
            event_time=time(16, 0) if 'After market' in note else None,
            description=f'{SEED_TAG} {note}',
            color=COLOR_EARNINGS,
        ))
    return events


def _ekadashi_events():
    """All 24 Ekadashi dates for 2026."""
    data = [
        ('2026-01-14', 'Shattila Ekadashi'),
        ('2026-01-29', 'Jaya Ekadashi'),
        ('2026-02-13', 'Vijaya Ekadashi'),
        ('2026-02-27', 'Amalaki Ekadashi'),
        ('2026-03-15', 'Papamochani Ekadashi'),
        ('2026-03-29', 'Kamada Ekadashi'),
        ('2026-04-13', 'Varuthini Ekadashi'),
        ('2026-04-27', 'Mohini Ekadashi'),
        ('2026-05-13', 'Apara Ekadashi'),
        ('2026-05-27', 'Padmini Ekadashi (Adhik Maas)'),
        ('2026-06-11', 'Parama Ekadashi (Adhik Maas)'),
        ('2026-06-25', 'Nirjala Ekadashi'),
        ('2026-07-10', 'Yogini Ekadashi'),
        ('2026-07-25', 'Devshayani Ekadashi'),
        ('2026-08-09', 'Kamika Ekadashi'),
        ('2026-08-23', 'Shravana Putrada Ekadashi'),
        ('2026-09-07', 'Aja Ekadashi'),
        ('2026-09-22', 'Parsva Ekadashi'),
        ('2026-10-06', 'Indira Ekadashi'),
        ('2026-10-22', 'Papankusha Ekadashi'),
        ('2026-11-05', 'Rama Ekadashi'),
        ('2026-11-20', 'Devutthana Ekadashi'),
        ('2026-12-04', 'Utpanna Ekadashi'),
        ('2026-12-20', 'Mokshada Ekadashi'),
    ]
    events = []
    for dt, name in data:
        events.append(CalendarEvent(
            title=f'🕉 {name}',
            event_date=date.fromisoformat(dt),
            description=f'{SEED_TAG} Ekadashi fasting day. 11th day of the lunar fortnight.',
            color=COLOR_EKADASHI,
        ))
    return events


def _indian_festival_events():
    """Major Indian festivals for 2026."""
    data = [
        ('2026-01-14', 'Makar Sankranti / Pongal'),
        ('2026-01-26', 'Republic Day'),
        ('2026-02-15', 'Maha Shivaratri'),
        ('2026-03-03', 'Holika Dahan'),
        ('2026-03-04', 'Holi'),
        ('2026-03-19', 'Ugadi / Gudi Padwa'),
        ('2026-03-19', 'Chaitra Navratri Begins'),
        ('2026-03-20', 'Eid al-Fitr'),
        ('2026-03-27', 'Ram Navami'),
        ('2026-03-31', 'Mahavir Jayanti'),
        ('2026-04-14', 'Baisakhi'),
        ('2026-05-01', 'Buddha Purnima'),
        ('2026-05-27', 'Eid al-Adha'),
        ('2026-06-26', 'Muharram'),
        ('2026-07-16', 'Rath Yatra'),
        ('2026-08-15', 'Independence Day'),
        ('2026-08-26', 'Onam'),
        ('2026-08-28', 'Raksha Bandhan'),
        ('2026-09-04', 'Janmashtami'),
        ('2026-09-14', 'Ganesh Chaturthi'),
        ('2026-10-02', 'Gandhi Jayanti'),
        ('2026-10-11', 'Sharad Navratri Begins'),
        ('2026-10-20', 'Dussehra'),
        ('2026-10-29', 'Karva Chauth'),
        ('2026-11-06', 'Dhanteras'),
        ('2026-11-07', 'Chhoti Diwali'),
        ('2026-11-08', 'Diwali'),
        ('2026-11-09', 'Govardhan Puja'),
        ('2026-11-10', 'Bhai Dooj'),
        ('2026-11-24', 'Guru Nanak Jayanti'),
        ('2026-12-25', 'Christmas'),
    ]
    events = []
    for dt, name in data:
        events.append(CalendarEvent(
            title=f'🪔 {name}',
            event_date=date.fromisoformat(dt),
            description=f'{SEED_TAG} Indian festival / holiday.',
            color=COLOR_FESTIVAL,
        ))
    return events


def _sd_farmers_market_events():
    """San Diego farmer's markets — recurring weekly."""
    # (title, first_occurrence_weekday, start_date, start_time, end_time)
    # start_date = first occurrence on or after 2026-03-01 for that weekday
    markets = [
        ('Hillcrest Farmers Market',        '2026-03-08', '09:00', '14:00'),  # Sunday
        ('La Jolla Open Aire Market',       '2026-03-08', '09:00', '13:00'),  # Sunday
        ('Coronado Certified Farmers Mkt',  '2026-03-10', '14:30', '18:00'),  # Tuesday
        ('Little Italy Mercato (Wed)',       '2026-03-11', '09:30', '13:30'),  # Wednesday
        ('Ocean Beach Farmers Market',      '2026-03-11', '16:00', '19:00'),  # Wednesday
        ('North Park Thursday Market',      '2026-03-12', '15:00', '19:30'),  # Thursday
        ('Little Italy Mercato (Sat)',       '2026-03-07', '08:00', '14:00'),  # Saturday
        ('Tuna Harbor Dockside Market',     '2026-03-07', '08:00', '13:00'),  # Saturday
        ('Point Loma Farmers Market',       '2026-03-07', '10:00', '13:00'),  # Saturday
    ]
    events = []
    for name, dt, st, et in markets:
        events.append(CalendarEvent(
            title=f'🥬 {name}',
            event_date=date.fromisoformat(dt),
            event_time=time.fromisoformat(st),
            end_time=time.fromisoformat(et),
            description=f'{SEED_TAG} Weekly farmers market in San Diego.',
            color=COLOR_FARMERS,
            recurrence='weekly',
            recurrence_end=date(2026, 12, 31),
        ))
    return events


def _sd_events():
    """Major San Diego events for 2026."""
    data = [
        ('2026-02-21', None,       None,       'SD Loyal / SD FC Home Debut',
         'San Diego FC inaugural MLS season home match.'),
        ('2026-03-14', '2026-03-15', None,     'CRSSD Festival Spring',
         'Electronic music festival at Waterfront Park.'),
        ('2026-03-26', None,       '13:10',    'Padres Home Opener',
         'San Diego Padres 2026 season home opener at Petco Park.'),
        ('2026-04-25', '2026-04-26', None,     'Mission Fed ArtWalk',
         "San Diego's largest free art festival in Little Italy."),
        ('2026-05-23', '2026-05-25', None,     'Gator By The Bay',
         'Zydeco, blues, and crawfish festival at Spanish Landing Park.'),
        ('2026-06-10', None,       None,       'SD County Fair Opens',
         'San Diego County Fair at Del Mar Fairgrounds. Runs through Jul 5.'),
        ('2026-07-04', None,       '21:00',    'Big Bay Boom',
         "San Diego's 4th of July fireworks show over the bay."),
        ('2026-07-05', None,       None,       'SD County Fair Closes',
         'Last day of the San Diego County Fair.'),
        ('2026-07-18', '2026-07-19', None,     'San Diego Pride Festival',
         'Pride Parade and Festival in Hillcrest & Balboa Park.'),
        ('2026-07-22', None,       None,       'San Diego Comic-Con (Day 1)',
         'Comic-Con International at SD Convention Center. Jul 22-26.'),
        ('2026-07-26', None,       None,       'San Diego Comic-Con (Last Day)',
         'Final day of SDCC 2026.'),
        ('2026-09-19', '2026-09-20', None,     'CRSSD Festival Fall',
         'Fall edition of electronic music festival at Waterfront Park.'),
        ('2026-10-31', None,       None,       'Halloween in the Gaslamp',
         'Gaslamp Quarter Halloween block party.'),
        ('2026-11-06', '2026-11-08', None,     'SD Bay Wine + Food Festival',
         'Multi-day food & wine festival across San Diego.'),
        ('2026-12-04', '2026-12-05', None,     'December Nights',
         "Balboa Park's holiday celebration. Free admission to museums."),
    ]
    events = []
    for row in data:
        dt, end_dt, evt_time, name, desc = row
        events.append(CalendarEvent(
            title=f'🎪 {name}',
            event_date=date.fromisoformat(dt),
            event_time=time.fromisoformat(evt_time) if evt_time else None,
            description=f'{SEED_TAG} {desc}',
            color=COLOR_SD_EVENT,
        ))
        # If multi-day, add an end-date marker event
        if end_dt:
            events.append(CalendarEvent(
                title=f'🎪 {name} (ends)',
                event_date=date.fromisoformat(end_dt),
                description=f'{SEED_TAG} Last day — {desc}',
                color=COLOR_SD_EVENT,
            ))
    return events


# ── CLI Command ─────────────────────────────────────────────────────────

@click.command('seed-calendar')
@with_appcontext
def seed_calendar_cmd():
    """Populate calendar with earnings, festivals, ekadashis, and SD events."""

    # 1. Remove previously seeded events (idempotent re-run)
    deleted = CalendarEvent.query.filter(
        CalendarEvent.description.contains(SEED_TAG)
    ).delete(synchronize_session='fetch')
    if deleted:
        click.echo(f'  Removed {deleted} previously seeded events.')

    # 2. Build all events
    categories = [
        ('Earnings Reports',    _earnings_events),
        ('Ekadashi',            _ekadashi_events),
        ('Indian Festivals',    _indian_festival_events),
        ('SD Farmers Markets',  _sd_farmers_market_events),
        ('SD Events',           _sd_events),
    ]

    total = 0
    for label, builder in categories:
        events = builder()
        db.session.add_all(events)
        click.echo(f'  + {len(events):>3} {label}')
        total += len(events)

    db.session.commit()
    click.echo(f'\n  ✓ Seeded {total} calendar events.')


def register_commands(app):
    """Register all custom CLI commands with the Flask app."""
    app.cli.add_command(seed_calendar_cmd)
