#!/usr/bin/env python3
"""Load seed sources and watchlists into the database. Idempotent."""

import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from app.extensions import db
from app.models.source import Source
from app.models.topic import TrackedTopic


def seed_sources(filepath):
    """Load sources from JSON. Skip existing by URL."""
    with open(filepath) as f:
        sources = json.load(f)

    added = 0
    skipped = 0
    for s in sources:
        existing = Source.query.filter_by(url=s['url']).first()
        if existing:
            skipped += 1
            continue

        source = Source(
            name=s['name'],
            url=s['url'],
            section=s['section'],
            region=s.get('region'),
            bias_label=s.get('bias_label', 'center'),
            trust_score=s.get('trust_score', 50),
            source_type=s.get('source_type', 'reporting'),
        )
        db.session.add(source)
        added += 1

    db.session.commit()
    print(f"Sources: {added} added, {skipped} skipped (already exist)")


def seed_watchlists(filepath):
    """Load tracked topics from JSON. Skip existing by name."""
    with open(filepath) as f:
        topics = json.load(f)

    added = 0
    skipped = 0
    for t in topics:
        existing = TrackedTopic.query.filter_by(name=t['name']).first()
        if existing:
            skipped += 1
            continue

        topic = TrackedTopic(
            name=t['name'],
            description=t.get('description', ''),
        )
        db.session.add(topic)
        added += 1

    db.session.commit()
    print(f"Topics: {added} added, {skipped} skipped (already exist)")


if __name__ == '__main__':
    app = create_app()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    with app.app_context():
        print("Seeding database...")
        seed_sources(os.path.join(project_root, 'seed_sources.json'))
        seed_watchlists(os.path.join(project_root, 'seed_watchlists.json'))
        print("Done.")
