import json
import logging
from datetime import date, datetime, timezone
from flask import current_app
from app.extensions import db
from app.models.timeline import Timeline, TimelineEvent
from app.models.cluster import Cluster, ClusterMembership
from app.models.article import Article
from app.integrations.llm_gateway import LLMGateway, BudgetExhaustedError

logger = logging.getLogger(__name__)

MAX_AUTO_TIMELINES = 8  # limit auto-created timelines to prevent unbounded growth


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
        """Use dual-LLM approach to generate rich, citation-filled timeline events.

        Phase 1 — XAI/Grok (real-time web access): generates events with real
        citations, URLs, and up-to-date facts.
        Phase 2 — OpenAI (best model): enriches/verifies events, adds deeper
        context, fills gaps, and ensures citation quality.

        Falls back to single-provider if the other is unavailable.
        """
        config = current_app.config
        llm = LLMGateway(config)

        timeline = Timeline.query.get(timeline_id)
        if not timeline:
            raise ValueError(f"Timeline {timeline_id} not found")

        entities_str = ', '.join(timeline.entities_json or [])
        total_tokens = 0
        total_cost = 0.0

        # ── Phase 1: XAI/Grok for real-time events + citations ────────
        grok_events = []
        if llm.xai_available:
            grok_events = self._generate_events_xai(
                llm, timeline, topic_prompt, entities_str, brief_id
            )
            total_tokens += grok_events.pop('_tokens', 0) if isinstance(grok_events, dict) else 0

        # ── Phase 2: OpenAI for deep enrichment ───────────────────────
        openai_events = self._generate_events_openai(
            llm, timeline, topic_prompt, entities_str, grok_events, brief_id
        )

        # Merge: deduplicate by date+title similarity, prefer richer version
        merged = self._merge_events(grok_events, openai_events)

        # ── Persist ───────────────────────────────────────────────────
        events_added = 0
        for ev in merged:
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
                    source_urls_json=ev.get('source_urls', [])[:3],
                    metadata_json=ev.get('metadata'),
                )
                db.session.add(event)
                events_added += 1
            except (ValueError, KeyError) as e:
                logger.warning(f"Skipping malformed event: {e}")
                continue

        db.session.commit()
        logger.info(f"Generated {events_added} events for timeline '{timeline.name}' (dual-LLM)")

        # ── Pick a topic-appropriate emoji icon ────────────────────────
        if timeline.icon == '📅':
            try:
                icon_result = llm.call(
                    messages=[
                        {'role': 'system', 'content': 'Return ONLY a single emoji that best represents the given topic. No text, no explanation, just one emoji character.'},
                        {'role': 'user', 'content': topic_prompt},
                    ],
                    purpose=f'timeline.icon.{timeline.id}',
                    section='timelines',
                    brief_id=brief_id,
                    max_tokens=10,
                    model='gpt-4.1-nano',
                )
                icon = icon_result['content'].strip()[:4]
                if icon:
                    timeline.icon = icon
                    db.session.commit()
                    logger.info(f"Set icon '{icon}' for timeline '{timeline.name}'")
            except Exception as e:
                logger.warning(f"Icon selection failed for timeline {timeline.id}: {e}")

        return {
            'events_added': events_added,
        }

    def _generate_events_xai(self, llm, timeline, topic_prompt, entities_str, brief_id):
        """Phase 1: Use Grok for real-time, citation-rich event generation."""
        system_prompt = """You are an expert timeline researcher with live web search access. Given a topic, search the web for real events and generate a detailed chronological timeline.

Use your web search to find real, verifiable information. For each event, provide:
- Precise dates (YYYY-MM-DD)
- Detailed summaries with specific numbers, names, and facts
- Real source URLs from official announcements, major news outlets, or press releases

Output ONLY a JSON array. Each event:
- "date": ISO date (YYYY-MM-DD)
- "title": Descriptive title (max 120 chars)
- "summary": 2-3 sentence detailed description with specific facts, numbers, and quotes where relevant
- "entity": Primary entity/actor
- "event_type": One of: release, policy, partnership, funding, conflict, diplomacy, regulation, announcement, research, milestone
- "significance": 1-10
- "source_urls": Array of 1-3 REAL URLs (official blogs, Reuters, Bloomberg, AP, etc.)

Generate 15-25 events. Prioritize recency and accuracy. Include the most recent developments."""

        user_prompt = f"""Topic: {topic_prompt}
Entities: {entities_str}

Generate a comprehensive, citation-rich timeline. Include:
- Major product launches, releases, and version updates
- Key policy decisions, executive statements, leadership changes
- Significant partnerships, acquisitions, or competitive moves
- Funding rounds, revenue milestones, market cap changes
- Regulatory actions, lawsuits, investigations
- Research breakthroughs, patents, technical achievements
- Recent developments from the last 3-6 months

Be specific: include dollar amounts, user counts, percentages, and direct quotes where possible."""

        try:
            result = llm.call(
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt},
                ],
                purpose=f'timeline.generate.xai.{timeline.id}',
                section='timelines',
                brief_id=brief_id,
                max_tokens=5000,
                provider='xai',
                search=True,
            )
            content = result['content'].strip()
            events = _parse_json_response(content)
            logger.info(f"XAI generated {len(events)} events for '{timeline.name}'")
            return events
        except Exception as e:
            logger.warning(f"XAI timeline generation failed for '{timeline.name}': {e}")
            return []

    def _generate_events_openai(self, llm, timeline, topic_prompt, entities_str,
                                 existing_events, brief_id):
        """Phase 2: Use OpenAI to generate/enrich events with deep analysis."""
        # If we already have Grok events, ask OpenAI to verify + fill gaps
        if existing_events:
            existing_summary = json.dumps(
                [{'date': e.get('date'), 'title': e.get('title')} for e in existing_events[:20]],
                indent=None,
            )
            system_prompt = f"""You are a meticulous timeline analyst. You have been given a draft timeline and must:

1. VERIFY events: flag any with incorrect dates or facts
2. FILL GAPS: add 5-10 important events that are missing
3. ENRICH: for any event you add, provide detailed summaries with specific facts
4. CITE: include real, verifiable source URLs

The draft timeline already covers these events (do NOT duplicate them):
{existing_summary}

Output ONLY a JSON array of NEW events to add (not duplicates of the above). Each event:
- "date": ISO date (YYYY-MM-DD)
- "title": Descriptive title (max 120 chars)
- "summary": 2-3 sentence description with specific facts and context
- "entity": Primary entity/actor
- "event_type": One of: release, policy, partnership, funding, conflict, diplomacy, regulation, announcement, research, milestone
- "significance": 1-10
- "source_urls": Array of 0-2 relevant URLs

Return ONLY the JSON array. Generate 5-10 gap-filling events."""
        else:
            system_prompt = """You are an expert timeline researcher. Generate a detailed, citation-rich chronological timeline.

Output ONLY a JSON array. Each event:
- "date": ISO date (YYYY-MM-DD)
- "title": Descriptive title (max 120 chars)
- "summary": 2-3 sentence description with specific facts, numbers, dollar amounts, and context
- "entity": Primary entity/actor
- "event_type": One of: release, policy, partnership, funding, conflict, diplomacy, regulation, announcement, research, milestone
- "significance": 1-10
- "source_urls": Array of 0-2 relevant URLs (use official sources where possible)

Generate 15-25 events covering the full history of major developments."""

        user_prompt = f"""Topic: {topic_prompt}
Entities: {entities_str}

Generate a comprehensive timeline focusing on:
- Major product launches and releases with version details
- Key leadership decisions and organizational changes
- Significant partnerships, acquisitions, or competitive dynamics
- Funding rounds and financial milestones with dollar amounts
- Regulatory actions and legal developments
- Research breakthroughs and technical achievements

Be precise with dates and include specific numbers wherever possible."""

        try:
            result = llm.call(
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt},
                ],
                purpose=f'timeline.generate.openai.{timeline.id}',
                section='timelines',
                brief_id=brief_id,
                max_tokens=5000,
            )
            content = result['content'].strip()
            events = _parse_json_response(content)
            logger.info(f"OpenAI generated {len(events)} events for '{timeline.name}'")
            return events
        except Exception as e:
            logger.warning(f"OpenAI timeline generation failed for '{timeline.name}': {e}")
            return []

    def _merge_events(self, grok_events, openai_events):
        """Merge events from both providers, deduplicating by date + title similarity."""
        if not grok_events:
            return openai_events or []
        if not openai_events:
            return grok_events or []

        merged = list(grok_events)  # Grok events are primary (real-time)
        existing_keys = set()
        for ev in merged:
            key = (ev.get('date', ''), ev.get('title', '').lower()[:40])
            existing_keys.add(key)

        for ev in openai_events:
            key = (ev.get('date', ''), ev.get('title', '').lower()[:40])
            # Check for date + rough title match
            is_dup = key in existing_keys
            if not is_dup:
                # Also check same-date events with similar titles
                ev_date = ev.get('date', '')
                ev_title = ev.get('title', '').lower()
                for ekey in existing_keys:
                    if ekey[0] == ev_date and (
                        ekey[1] in ev_title or ev_title[:30] in ekey[1]
                    ):
                        is_dup = True
                        break
            if not is_dup:
                merged.append(ev)
                existing_keys.add(key)

        # Sort by date descending
        merged.sort(key=lambda e: e.get('date', ''), reverse=True)
        return merged[:35]  # Cap at 35 events

    # ── Auto-update: append today's matching clusters as events ──────

    def auto_update_timelines(self, target_date, brief_id=None):
        """For each active timeline with auto_update=True, find today's
        matching clusters and use LLM to convert them into timeline events.

        Returns dict with summary stats.
        """
        timelines = Timeline.query.filter_by(is_active=True, auto_update=True).all()
        if not timelines:
            logger.info("[Timelines] No auto-update timelines found")
            return {'timelines_updated': 0, 'events_added': 0}

        config = current_app.config
        llm = LLMGateway(config)

        total_events = 0
        total_tokens = 0
        updated = 0

        for timeline in timelines:
            try:
                events = self._update_timeline_from_clusters(
                    timeline, target_date, llm, brief_id
                )
                total_events += events
                if events > 0:
                    updated += 1
            except BudgetExhaustedError:
                logger.warning("[Timelines] Budget exhausted, stopping auto-update")
                break
            except Exception as e:
                logger.error(f"[Timelines] Failed to update '{timeline.name}': {e}")

        logger.info(
            f"[Timelines] Auto-update complete: {updated} timelines, "
            f"{total_events} events added"
        )
        return {'timelines_updated': updated, 'events_added': total_events}

    def _update_timeline_from_clusters(self, timeline, target_date, llm, brief_id):
        """Find matching clusters for a timeline and create events via LLM."""
        # Build query: clusters from today matching timeline's sections
        sections = timeline.sections or []
        if not sections:
            # If no sections configured, search across all
            sections = None

        query = Cluster.query.filter_by(date=target_date)
        if sections:
            query = query.filter(Cluster.section.in_(sections))
        query = query.order_by(Cluster.rank_score.desc())
        clusters = query.limit(20).all()

        if not clusters:
            return 0

        # Filter clusters by entity match (if timeline has tracked entities)
        entities = [e.lower() for e in (timeline.entities_json or [])]
        if entities:
            matching_clusters = []
            for cluster in clusters:
                headline = (cluster.label or '').lower()
                # Check if any tracked entity appears in cluster headline
                if any(ent in headline for ent in entities):
                    matching_clusters.append(cluster)
            # Also include top-ranked clusters from matching sections even without entity match
            if len(matching_clusters) < 3:
                for cluster in clusters:
                    if cluster not in matching_clusters:
                        matching_clusters.append(cluster)
                    if len(matching_clusters) >= 5:
                        break
            clusters = matching_clusters

        if not clusters:
            return 0

        # Check for duplicate events (don't add events for clusters already tracked)
        existing_cluster_ids = set(
            row[0] for row in
            db.session.query(TimelineEvent.cluster_id)
            .filter(
                TimelineEvent.timeline_id == timeline.id,
                TimelineEvent.cluster_id.isnot(None),
            ).all()
        )
        new_clusters = [c for c in clusters if c.id not in existing_cluster_ids]
        if not new_clusters:
            return 0

        # Build cluster summaries for LLM prompt
        cluster_texts = []
        for c in new_clusters[:8]:  # Max 8 clusters per timeline update
            articles = (
                Article.query
                .join(ClusterMembership, ClusterMembership.article_id == Article.id)
                .filter(ClusterMembership.cluster_id == c.id)
                .limit(3).all()
            )
            titles = [a.title for a in articles if a.title]
            source_urls = [a.url for a in articles if a.url][:2]
            cluster_texts.append({
                'cluster_id': c.id,
                'headline': c.label or titles[0] if titles else 'Unknown',
                'titles': titles[:3],
                'section': c.section,
                'source_urls': source_urls,
            })

        if not cluster_texts:
            return 0

        # Ask LLM to convert clusters into timeline events
        system_prompt = """You convert news clusters into timeline events for an existing timeline.

Output ONLY a JSON array. Each event must have:
- "cluster_index": The index (0-based) of the cluster this event is derived from
- "title": Short event title (max 100 chars)
- "summary": 1-2 sentence description of what happened
- "entity": Primary entity/actor involved
- "event_type": One of: release, policy, partnership, funding, conflict, diplomacy, regulation, announcement, research, milestone
- "significance": 1-10 (10 = most significant)

Return ONLY the JSON array. Only include events that are genuinely relevant to the timeline topic. Skip clusters that aren't related. If none are relevant, return an empty array []."""

        cluster_summary = '\n'.join(
            f"[{i}] {ct['headline']} (section: {ct['section']})\n    Articles: {', '.join(ct['titles'][:2])}"
            for i, ct in enumerate(cluster_texts)
        )

        user_prompt = f"""Timeline: {timeline.name}
Description: {timeline.description}
Tracked entities: {', '.join(timeline.entities_json or [])}

Today's news clusters:
{cluster_summary}

Convert relevant clusters into timeline events for this timeline."""

        try:
            result = llm.call(
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt},
                ],
                purpose=f'timeline.auto_update.{timeline.id}',
                section='timelines',
                brief_id=brief_id,
                max_tokens=2000,
            )

            content = result['content'].strip()
            events_data = _parse_json_response(content)

            events_added = 0
            for ev in events_data:
                try:
                    idx = ev.get('cluster_index', 0)
                    if idx < 0 or idx >= len(cluster_texts):
                        continue

                    cluster_info = cluster_texts[idx]
                    event = TimelineEvent(
                        timeline_id=timeline.id,
                        event_date=target_date,
                        title=ev['title'][:512],
                        summary=ev.get('summary', ''),
                        entity=ev.get('entity', ''),
                        event_type=ev.get('event_type', 'announcement'),
                        significance=min(max(ev.get('significance', 5), 1), 10),
                        source_urls_json=cluster_info.get('source_urls', []),
                        cluster_id=cluster_info['cluster_id'],
                    )
                    db.session.add(event)
                    events_added += 1
                except (ValueError, KeyError) as e:
                    logger.warning(f"Skipping malformed auto-update event: {e}")

            db.session.commit()
            if events_added:
                logger.info(
                    f"[Timelines] Added {events_added} events to '{timeline.name}'"
                )
            return events_added

        except BudgetExhaustedError:
            raise
        except Exception as e:
            logger.error(f"[Timelines] LLM auto-update failed for '{timeline.name}': {e}")
            return 0

    # ── Auto-discover: create new timelines from trending topics ──────

    def auto_discover_timelines(self, target_date, brief_id=None):
        """Analyze today's top clusters and suggest new timelines for ongoing stories.

        Only creates timelines if they don't duplicate existing ones.
        Returns dict with stats.
        """
        # Check capacity — don't create unbounded timelines
        existing_count = Timeline.query.filter_by(is_active=True).count()
        if existing_count >= MAX_AUTO_TIMELINES:
            logger.info(
                f"[Timelines] Already have {existing_count} active timelines "
                f"(max {MAX_AUTO_TIMELINES}), skipping auto-discover"
            )
            return {'timelines_created': 0}

        config = current_app.config
        llm = LLMGateway(config)

        # Get today's top clusters across all sections
        clusters = (
            Cluster.query
            .filter_by(date=target_date)
            .order_by(Cluster.rank_score.desc())
            .limit(20)
            .all()
        )

        if len(clusters) < 3:
            return {'timelines_created': 0}

        # Get existing timeline names + entities for dedup
        existing_timelines = Timeline.query.filter_by(is_active=True).all()
        existing_info = [
            {'name': t.name, 'entities': t.entities_json or []}
            for t in existing_timelines
        ]

        # Build cluster overview
        cluster_overview = '\n'.join(
            f"- [{c.section}] {c.label or 'Untitled'} (score: {c.rank_score:.1f})"
            for c in clusters
        )

        slots = MAX_AUTO_TIMELINES - existing_count
        existing_names = ', '.join(t.name for t in existing_timelines) or 'None'

        system_prompt = """You suggest new timelines to track ongoing news stories.

Output ONLY a JSON array. Each suggestion must have:
- "name": Short timeline name (max 40 chars)
- "description": 1-sentence description of what the timeline tracks
- "icon": A single emoji that represents the topic
- "entities": Array of 3-6 key entities/actors to track
- "sections": Array of relevant news sections (from: general_news_us, general_news_india, general_news_geopolitics, ai_news, market, science, health)

Return ONLY the JSON array. Suggest timelines for MAJOR ongoing stories that would benefit from day-over-day tracking. Do NOT duplicate existing timelines. Return an empty array [] if no good candidates exist."""

        user_prompt = f"""Existing timelines (DO NOT duplicate): {existing_names}

Today's top news clusters:
{cluster_overview}

Suggest up to {min(slots, 3)} new timelines for major ongoing stories worth tracking over multiple days."""

        try:
            result = llm.call(
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt},
                ],
                purpose='timeline.auto_discover',
                section='timelines',
                brief_id=brief_id,
                max_tokens=1500,
            )

            content = result['content'].strip()
            suggestions = _parse_json_response(content)

            created = 0
            for suggestion in suggestions:
                try:
                    name = suggestion.get('name', '').strip()
                    if not name:
                        continue

                    # Dedup check: skip if similar name exists
                    if any(name.lower() in t.name.lower() or t.name.lower() in name.lower()
                           for t in existing_timelines):
                        logger.debug(f"[Timelines] Skipping duplicate suggestion: {name}")
                        continue

                    timeline = Timeline(
                        name=name[:80],
                        description=suggestion.get('description', '')[:255],
                        entities_json=suggestion.get('entities', [])[:8],
                        sections=suggestion.get('sections', []),
                        icon=suggestion.get('icon', '📰')[:4],
                        is_active=True,
                        auto_update=True,
                    )
                    db.session.add(timeline)
                    db.session.flush()  # get ID

                    # Seed with historical events
                    try:
                        seed_result = self.generate_timeline_with_llm(
                            timeline.id,
                            f"{name}: {suggestion.get('description', '')}",
                            brief_id=brief_id,
                        )
                        logger.info(
                            f"[Timelines] Created '{name}' with "
                            f"{seed_result.get('events_added', 0)} seed events"
                        )
                    except Exception as seed_err:
                        logger.warning(
                            f"[Timelines] Created '{name}' but seed failed: {seed_err}"
                        )

                    created += 1
                    if created >= slots:
                        break

                except Exception as e:
                    logger.warning(f"[Timelines] Failed to create suggested timeline: {e}")

            db.session.commit()
            return {'timelines_created': created}

        except BudgetExhaustedError:
            logger.warning("[Timelines] Budget exhausted during auto-discover")
            return {'timelines_created': 0}
        except Exception as e:
            logger.error(f"[Timelines] Auto-discover failed: {e}")
            return {'timelines_created': 0}

    # ── Display helpers ──────────────────────────────────────────

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


def _parse_json_response(content):
    """Parse LLM JSON response, handling markdown code blocks."""
    text = content.strip()
    if text.startswith('```'):
        text = text.split('\n', 1)[1]
        if text.endswith('```'):
            text = text[:-3]
        text = text.strip()
    return json.loads(text)
