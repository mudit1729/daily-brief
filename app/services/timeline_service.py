import json
import logging
from datetime import date, datetime, timezone
from flask import current_app
from app.extensions import db
from app.models.timeline import Timeline, TimelineEvent
from app.integrations.llm_gateway import LLMGateway

logger = logging.getLogger(__name__)


class TimelineService:
    """Service for creating and populating timelines."""

    def create_timeline(self, name, description, entities=None, sections=None, icon=None):
        """Create a new timeline."""
        timeline = Timeline(
            name=name,
            description=description,
            entities_json=entities or [],
            sections=sections or [],
            icon=icon or '📅',
        )
        db.session.add(timeline)
        db.session.commit()
        return timeline

    def add_event(self, timeline_id, event_date, title, summary=None,
                  entity=None, event_type=None, significance=5,
                  source_urls=None, cluster_id=None, article_id=None,
                  metadata=None):
        """Add a single event to a timeline."""
        event = TimelineEvent(
            timeline_id=timeline_id,
            event_date=event_date,
            title=title,
            summary=summary,
            entity=entity,
            event_type=event_type,
            significance=significance,
            source_urls_json=source_urls or [],
            cluster_id=cluster_id,
            article_id=article_id,
            metadata_json=metadata,
        )
        db.session.add(event)
        db.session.commit()
        return event

    def generate_timeline_with_llm(self, timeline_id, topic_prompt, brief_id=None):
        """Use LLM to generate historical timeline events for a topic.

        This is used for initial seeding of timelines with historical context.
        The LLM generates structured event data based on its knowledge.
        """
        config = current_app.config
        llm = LLMGateway(config)

        timeline = Timeline.query.get(timeline_id)
        if not timeline:
            raise ValueError(f"Timeline {timeline_id} not found")

        system_prompt = """You are a precise timeline generator. Given a topic, generate a chronological list of key events.

Output ONLY a JSON array of events. Each event must have:
- "date": ISO date string (YYYY-MM-DD)
- "title": Short event title (max 100 chars)
- "summary": 1-2 sentence description
- "entity": Primary entity/actor involved
- "event_type": One of: release, policy, partnership, funding, conflict, diplomacy, regulation, announcement, research, milestone
- "significance": 1-10 (10 = most significant)
- "source_urls": Array of 0-2 relevant URLs (use real, official URLs when possible)

Return ONLY the JSON array, no other text. Generate 15-25 events covering the most important developments."""

        user_prompt = f"""Topic: {topic_prompt}

Entities to track: {', '.join(timeline.entities_json or [])}

Generate a comprehensive timeline of the most important events. Focus on:
- Major product launches and releases
- Key policy decisions and announcements
- Significant partnerships or conflicts
- Funding rounds or financial milestones
- Regulatory actions
- Research breakthroughs

Be precise with dates. If you're unsure of the exact date, use the closest known date."""

        try:
            result = llm.call(
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt},
                ],
                purpose=f'timeline.generate.{timeline.id}',
                section='ai_news',
                brief_id=brief_id,
                max_tokens=4000,
            )

            content = result['content'].strip()
            # Handle markdown code blocks
            if content.startswith('```'):
                content = content.split('\n', 1)[1]
                if content.endswith('```'):
                    content = content[:-3]
                content = content.strip()

            events_data = json.loads(content)

            events_added = 0
            for ev in events_data:
                try:
                    event_date = date.fromisoformat(ev['date'])
                    event = TimelineEvent(
                        timeline_id=timeline_id,
                        event_date=event_date,
                        title=ev['title'][:512],
                        summary=ev.get('summary', ''),
                        entity=ev.get('entity', ''),
                        event_type=ev.get('event_type', 'announcement'),
                        significance=min(max(ev.get('significance', 5), 1), 10),
                        source_urls_json=ev.get('source_urls', []),
                    )
                    db.session.add(event)
                    events_added += 1
                except (ValueError, KeyError) as e:
                    logger.warning(f"Skipping malformed event: {e}")
                    continue

            db.session.commit()
            logger.info(f"Generated {events_added} events for timeline '{timeline.name}'")
            return {
                'events_added': events_added,
                'tokens': result['total_tokens'],
                'cost': result['cost_usd'],
            }

        except Exception as e:
            logger.error(f"Failed to generate timeline events: {e}")
            raise

    def get_timeline_events_grouped(self, timeline_id):
        """Get timeline events grouped by month for display."""
        events = TimelineEvent.query.filter_by(
            timeline_id=timeline_id
        ).order_by(TimelineEvent.event_date.desc()).all()

        months = {}
        for ev in events:
            key = ev.event_date.strftime('%Y-%m')
            label = ev.event_date.strftime('%B %Y')
            if key not in months:
                months[key] = {'label': label, 'events': []}
            months[key]['events'].append(ev)

        return list(months.values())

    def get_entity_colors(self, timeline):
        """Assign consistent colors to entities in a timeline."""
        palette = [
            '#3b82f6',  # blue
            '#ef4444',  # red
            '#22c55e',  # green
            '#eab308',  # yellow
            '#a855f7',  # purple
            '#f97316',  # orange
            '#06b6d4',  # cyan
            '#ec4899',  # pink
        ]
        entities = timeline.entities_json or []
        return {entity: palette[i % len(palette)] for i, entity in enumerate(entities)}
