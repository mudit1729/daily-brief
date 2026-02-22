import logging
from datetime import datetime, timezone, date
from flask import current_app
from app.extensions import db
from app.models.cluster import Cluster, ClusterMembership
from app.models.article import Article
from app.models.source import Source
from app.models.brief import DailyBrief, BriefSection
from app.models.market import MarketSnapshot
from app.models.weather import WeatherCache
from app.integrations.llm_gateway import LLMGateway, BudgetExhaustedError
from app.integrations.weather import WeatherService
from app.services.investment_service import InvestmentService
from app.services.hedge_fund_service import HedgeFundService
from app.services.cost_service import CostService
from app.utils.text import extract_lead_sentences, truncate

logger = logging.getLogger(__name__)

# Section definitions — cluster_section now matches compress.py section keys directly
SECTIONS = [
    {'key': 'general_news_us', 'title': 'US News', 'cluster_section': 'general_news_us', 'order': 0},
    {'key': 'market', 'title': 'Market Trends', 'cluster_section': 'market', 'order': 1},
    {'key': 'ai_news', 'title': 'AI & Tech', 'cluster_section': 'ai_news', 'order': 2},
    {'key': 'general_news_india', 'title': 'India', 'cluster_section': 'general_news_india', 'order': 3},
    {'key': 'general_news_geopolitics', 'title': 'Geopolitics', 'cluster_section': 'general_news_geopolitics', 'order': 4},
    {'key': 'weather', 'title': 'Weather', 'order': 5},
    {'key': 'science', 'title': 'Science', 'cluster_section': 'science', 'order': 6},
    {'key': 'health', 'title': 'Health', 'cluster_section': 'health', 'order': 7},
    {'key': 'feel_good', 'title': 'Feel Good', 'cluster_section': 'feel_good', 'order': 8},
    {'key': 'investment_thesis', 'title': 'Investment Thesis', 'order': 9},
]


def run(target_date, brief_id):
    """Step 5: Synthesize daily brief from ranked clusters."""
    logger.info(f"[Synthesize] Starting for {target_date}")

    config = current_app.config
    model = config.get('LLM_MODEL', 'unknown')
    budget = config.get('LLM_DAILY_BUDGET_USD', 0)
    logger.info(f"[Synthesize] Config: model={model}, budget=${budget}, sections={len(SECTIONS)}")
    llm = LLMGateway(config)
    cost_service = CostService()

    brief = DailyBrief.query.get(brief_id)
    brief.status = 'generating'
    db.session.commit()

    # Idempotent rerun safety: regenerate sections from scratch.
    BriefSection.query.filter_by(brief_id=brief_id).delete(synchronize_session=False)
    db.session.commit()

    total_tokens = 0
    total_cost = 0.0

    for section_def in SECTIONS:
        try:
            section_key = section_def['key']

            if section_key == 'weather':
                section = _build_weather_section(target_date, brief_id, section_def)
            elif section_key == 'investment_thesis':
                section = _build_investment_section(target_date, brief_id, llm, section_def)
            elif section_key == 'market':
                section = _build_market_section(target_date, brief_id, llm, section_def)
            else:
                section = _build_news_section(target_date, brief_id, llm, section_def)

            if section:
                db.session.add(section)
                total_tokens += section.tokens_used or 0
                total_cost += section.cost_usd or 0.0

        except Exception as e:
            logger.error(f"[Synthesize] Failed to build section {section_def['key']}: {e}")
            # Create empty section with error note
            section = BriefSection(
                brief_id=brief_id,
                section_type=section_def['key'],
                title=section_def['title'],
                content_json={'error': str(e), 'clusters': []},
                display_order=section_def['order'],
            )
            db.session.add(section)

    # ── Story tracking: link today's clusters to tracked topics ──
    try:
        from app.services.story_tracker import StoryTracker
        from app.models.cluster import Cluster, ClusterMembership
        tracker = StoryTracker()
        if tracker.is_enabled():
            today_clusters = Cluster.query.filter_by(date=target_date).all()
            stories_linked = 0
            for cluster in today_clusters:
                if not cluster.label:
                    continue
                # Get articles in this cluster
                memberships = ClusterMembership.query.filter_by(cluster_id=cluster.id).all()
                articles = [Article.query.get(m.article_id) for m in memberships]
                articles = [a for a in articles if a]
                story = tracker.link_cluster_to_story(cluster, articles)
                if story:
                    stories_linked += 1
            db.session.commit()
            logger.info(f"[Synthesize] Story tracking: {stories_linked} clusters linked to tracked stories")
        else:
            logger.info("[Synthesize] Story tracking disabled (FF_STORY_TRACKING)")
    except Exception as e:
        logger.error(f"[Synthesize] Story tracking failed (non-fatal): {e}")

    # ── Grok analysis pass (secondary LLM enrichment) ────────
    try:
        grok_result = _grok_analysis_pass(brief_id, llm)
        if grok_result:
            total_tokens += grok_result.get('tokens', 0)
            total_cost += grok_result.get('cost', 0)
    except Exception as e:
        logger.error(f"[Synthesize] Grok analysis pass failed (non-fatal): {e}")

    # ── Timeline auto-update + auto-discover ──────────────────
    try:
        from app.services.timeline_service import TimelineService
        tl_service = TimelineService()

        # Auto-update existing timelines with today's clusters
        update_result = tl_service.auto_update_timelines(target_date, brief_id)
        logger.info(f"[Synthesize] Timelines auto-update: {update_result}")

        # Auto-discover new timelines from trending topics
        discover_result = tl_service.auto_discover_timelines(target_date, brief_id)
        logger.info(f"[Synthesize] Timelines auto-discover: {discover_result}")

    except Exception as e:
        logger.error(f"[Synthesize] Timeline step failed: {e}")

    # ── Grok enrichment for timelines ─────────────────────────
    try:
        grok_tl = _grok_timeline_enrichment(brief_id, llm, target_date)
        if grok_tl:
            total_tokens += grok_tl.get('tokens', 0)
            total_cost += grok_tl.get('cost', 0)
    except Exception as e:
        logger.error(f"[Synthesize] Grok timeline enrichment failed (non-fatal): {e}")

    # ── Grok enrichment for tracked stories ───────────────────
    try:
        grok_st = _grok_stories_enrichment(brief_id, llm, target_date)
        if grok_st:
            total_tokens += grok_st.get('tokens', 0)
            total_cost += grok_st.get('cost', 0)
    except Exception as e:
        logger.error(f"[Synthesize] Grok stories enrichment failed (non-fatal): {e}")

    # ── Grok stock fundamentals for investment thesis ──────────
    try:
        grok_fund = _grok_stock_fundamentals(brief_id, llm)
        if grok_fund:
            total_tokens += grok_fund.get('tokens', 0)
            total_cost += grok_fund.get('cost', 0)
    except Exception as e:
        logger.error(f"[Synthesize] Grok stock fundamentals failed (non-fatal): {e}")

    brief.total_tokens = total_tokens
    brief.total_cost_usd = round(total_cost, 6)
    brief.status = 'complete'
    brief.generated_at = datetime.now(timezone.utc)

    # Compute idiot index
    brief.idiot_index = cost_service.compute_idiot_index(target_date)

    db.session.commit()
    logger.info(f"[Synthesize] Complete: {total_tokens} tokens, ${total_cost:.4f}")
    return {'total_tokens': total_tokens, 'total_cost': total_cost}


def _build_news_section(target_date, brief_id, llm, section_def):
    """Build a news section from ranked clusters."""
    section_key = section_def['key']
    cluster_section = section_def.get('cluster_section', section_key)

    clusters = _query_clusters(
        target_date=target_date,
        section=cluster_section,
        region_filter=section_def.get('region_filter'),
    )

    if not clusters:
        return BriefSection(
            brief_id=brief_id,
            section_type=section_key,
            title=section_def['title'],
            content_json={'clusters': [], 'note': 'No stories found'},
            display_order=section_def['order'],
        )

    degradation = llm.determine_degradation_level(section_key)
    max_clusters = _max_clusters_for_degradation(degradation)
    top_clusters = clusters[:max_clusters]

    cluster_summaries = []
    section_tokens = 0
    section_cost = 0.0

    for cluster in top_clusters:
        articles = _get_cluster_articles(cluster)
        article_texts = [a.extracted_text for a in articles if a.extracted_text]

        # Always set cluster label from article title if not already set
        if not cluster.label and articles:
            titles = [a.title for a in articles if a.title]
            if titles:
                cluster.label = titles[0][:200]

        if degradation >= 4:
            # Extractive: use lead sentences, no LLM
            summary_text = _extractive_summary(articles)
            cluster_summaries.append(_format_cluster(cluster, articles, summary_text))
            continue

        try:
            summary_text = _llm_summarize_cluster(
                llm, cluster, articles, article_texts,
                degradation, section_key, brief_id
            )
            if summary_text:
                section_tokens += summary_text.get('tokens', 0)
                section_cost += summary_text.get('cost', 0)
                cluster_summaries.append(
                    _format_cluster(cluster, articles, summary_text['content'])
                )
                # Store summary on cluster
                cluster.summary = summary_text['content']
        except BudgetExhaustedError:
            logger.warning(f"Budget exhausted for {section_key}, falling back to extractive")
            summary_text = _extractive_summary(articles)
            cluster_summaries.append(_format_cluster(cluster, articles, summary_text))
            degradation = 4

    db.session.flush()

    return BriefSection(
        brief_id=brief_id,
        section_type=section_key,
        title=section_def['title'],
        content_json={'clusters': cluster_summaries},
        display_order=section_def['order'],
        tokens_used=section_tokens,
        cost_usd=round(section_cost, 6),
        degradation_level=degradation,
    )


def _build_market_section(target_date, brief_id, llm, section_def):
    """Build market section with prices + news driver attribution."""
    snapshots = MarketSnapshot.query.filter_by(snapshot_date=target_date).all()
    market_data = [s.to_dict() for s in snapshots]

    # Get market-related clusters
    clusters = Cluster.query.filter_by(
        date=target_date,
        section='market',
    ).order_by(Cluster.rank_score.desc()).limit(10).all()

    cluster_summaries = []
    section_tokens = 0
    section_cost = 0.0

    for cluster in clusters:
        articles = _get_cluster_articles(cluster)
        degradation = llm.determine_degradation_level('market')

        # Always set cluster label from article title if not already set
        if not cluster.label and articles:
            titles = [a.title for a in articles if a.title]
            if titles:
                cluster.label = titles[0][:200]

        if degradation >= 4:
            summary_text = _extractive_summary(articles)
            cluster_summaries.append(_format_cluster(cluster, articles, summary_text))
        else:
            try:
                result = _llm_summarize_cluster(
                    llm, cluster, articles,
                    [a.extracted_text for a in articles if a.extracted_text],
                    degradation, 'market', brief_id
                )
                if result:
                    section_tokens += result.get('tokens', 0)
                    section_cost += result.get('cost', 0)
                    cluster.summary = result['content']
                    cluster_summaries.append(_format_cluster(cluster, articles, result['content']))
            except BudgetExhaustedError:
                summary_text = _extractive_summary(articles)
                cluster_summaries.append(_format_cluster(cluster, articles, summary_text))

    return BriefSection(
        brief_id=brief_id,
        section_type='market',
        title=section_def['title'],
        content_json={
            'market_data': market_data,
            'clusters': cluster_summaries,
        },
        display_order=section_def['order'],
        tokens_used=section_tokens,
        cost_usd=round(section_cost, 6),
    )


def _build_weather_section(target_date, brief_id, section_def):
    """Build weather section from cached data. No LLM needed."""
    entries = WeatherCache.query.filter_by(date=target_date).all()
    weather_service = WeatherService()
    formatted = weather_service.format_weather_section(
        [{'location_name': e.location_name, 'data_json': e.data_json} for e in entries]
    )

    return BriefSection(
        brief_id=brief_id,
        section_type='weather',
        title=section_def['title'],
        content_json={'locations': formatted},
        display_order=section_def['order'],
        tokens_used=0,
        cost_usd=0.0,
    )


def _build_investment_section(target_date, brief_id, llm, section_def):
    """Build investment thesis section, optionally enriched with hedge fund signals."""
    snapshots = MarketSnapshot.query.filter_by(snapshot_date=target_date).all()
    snapshot_dicts = [
        {'symbol': s.symbol, 'name': s.name, 'price': s.price,
         'change_pct': s.change_pct, 'change_abs': s.change_abs}
        for s in snapshots
    ]

    top_clusters = Cluster.query.filter_by(date=target_date).order_by(
        Cluster.rank_score.desc()
    ).limit(5).all()

    # ── Run hedge fund analysis first (if enabled) ──
    hf_signals = []
    try:
        hf_service = HedgeFundService()
        hf_analyses, hf_usage = hf_service.run_analysis(target_date, brief_id)
        for a in hf_analyses:
            hf_signals.append({
                'ticker': a.ticker,
                'consensus': a.consensus_signal,
                'confidence': a.consensus_confidence,
                'analysts': a.analyst_signals_json or {},
                'decision': a.decision_json,
            })
    except Exception as e:
        logger.error(f"[Synthesize] Hedge fund analysis failed (non-fatal): {e}")

    # ── Generate thesis (now with HF context) ──
    investment_service = InvestmentService()
    thesis, usage = investment_service.generate_thesis(
        target_date, brief_id, snapshot_dicts, top_clusters, llm,
        hedge_fund_signals=hf_signals,
    )

    content = {
        'thesis': thesis.thesis_text if thesis else 'Investment thesis feature disabled',
        'gate_passed': thesis.gate_passed if thesis else False,
        'signals': {
            'momentum': thesis.momentum_signal if thesis else None,
            'value': thesis.value_signal if thesis else None,
        },
        'hedge_fund_signals': hf_signals,
    }

    return BriefSection(
        brief_id=brief_id,
        section_type='investment_thesis',
        title=section_def['title'],
        content_json=content,
        display_order=section_def['order'],
        tokens_used=usage.get('total_tokens', 0),
        cost_usd=round(float(usage.get('cost_usd', 0.0)), 6),
    )


def _get_cluster_articles(cluster):
    """Get all articles in a cluster."""
    memberships = ClusterMembership.query.filter_by(cluster_id=cluster.id).all()
    article_ids = [m.article_id for m in memberships]
    return Article.query.filter(Article.id.in_(article_ids)).all() if article_ids else []


def _query_clusters(target_date, section, region_filter=None):
    """Query clusters with optional source-region filtering."""
    query = Cluster.query.filter_by(
        date=target_date,
        section=section,
    )

    if region_filter:
        query = query.join(
            ClusterMembership, ClusterMembership.cluster_id == Cluster.id
        ).join(
            Article, Article.id == ClusterMembership.article_id
        ).join(
            Source, Source.id == Article.source_id
        ).filter(
            db.func.lower(Source.region) == region_filter.lower()
        ).distinct()

    return query.order_by(Cluster.rank_score.desc()).all()


def _extractive_summary(articles):
    """Generate extractive summary from lead sentences (no LLM)."""
    sentences = []
    for article in articles[:3]:
        if article.extracted_text:
            lead = extract_lead_sentences(article.extracted_text, n=2)
            sentences.append(lead)
    return ' '.join(sentences) if sentences else 'No content available.'


def _llm_summarize_cluster(llm, cluster, articles, texts, degradation, section, brief_id):
    """Summarize a cluster using LLM with degradation-aware prompting."""
    if not texts:
        return None

    combined = '\n---\n'.join(truncate(t, max_words=300) for t in texts[:5])

    if degradation == 0:
        system_prompt = (
            "Summarize this news cluster in 3-5 sentences. Include key claims, "
            "note the framing or perspective differences between sources, and "
            "highlight any contradictions."
        )
    elif degradation == 1:
        system_prompt = (
            "Summarize this news cluster in 2-3 sentences. Focus on the key facts."
        )
    elif degradation == 2:
        system_prompt = "Summarize this news cluster in 1-2 sentences."
    else:
        system_prompt = "Summarize in one sentence."

    max_tokens = {0: 400, 1: 250, 2: 150, 3: 80}.get(degradation, 80)

    titles = [a.title for a in articles if a.title]
    title_list = '\n'.join(f"- {t}" for t in titles[:5])

    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': f"Headlines:\n{title_list}\n\nArticle excerpts:\n{combined}"},
    ]

    result = llm.call(
        messages=messages,
        purpose=f'synthesize.{section}',
        section=section,
        brief_id=brief_id,
        max_tokens=max_tokens,
    )

    # Also generate cluster label if not set
    if not cluster.label and titles:
        cluster.label = titles[0][:200]

    return {
        'content': result['content'],
        'tokens': result['total_tokens'],
        'cost': result['cost_usd'],
    }


def _format_cluster(cluster, articles, summary):
    """Format cluster data for brief section JSON."""
    return {
        'cluster_id': cluster.id,
        'label': cluster.label,
        'summary': summary,
        'article_count': len(articles),
        'avg_trust': cluster.avg_trust_score,
        'rank_score': cluster.rank_score,
        'articles': [
            {
                'id': a.id,
                'title': a.title,
                'url': a.url,
                'source': a.source.name if a.source else None,
                'og_image_url': a.og_image_url,
            }
            for a in articles[:5]
        ],
    }


def _max_clusters_for_degradation(level):
    """Max clusters to process at each degradation level."""
    return {0: 15, 1: 12, 2: 8, 3: 5, 4: 6}.get(level, 6)


def _grok_analysis_pass(brief_id, llm):
    """Enrich top stories with Grok's real-time perspective.

    Iterates over news sections, picks the top 3 clusters by rank_score,
    and asks Grok for a concise alternative take. Stores the result as
    ``grok_take`` inside each cluster's summary JSON.
    """
    if not llm.xai_available:
        logger.info("[Grok] Skipping — XAI_API_KEY not configured")
        return None

    sections = BriefSection.query.filter_by(brief_id=brief_id).all()
    news_sections = [
        s for s in sections
        if s.section_type not in ('weather', 'investment_thesis')
        and s.content_json
        and s.content_json.get('clusters')
    ]

    # Collect top clusters across all news sections
    candidates = []
    for sec in news_sections:
        for cluster_data in sec.content_json.get('clusters', []):
            candidates.append((sec, cluster_data))

    # Sort by rank_score descending, take top 5
    candidates.sort(key=lambda x: x[1].get('rank_score', 0), reverse=True)
    top = candidates[:5]

    if not top:
        return None

    total_tokens = 0
    total_cost = 0.0

    for sec, cluster_data in top:
        label = cluster_data.get('label', '')
        summary = cluster_data.get('summary', '')
        if not summary:
            continue

        messages = [
            {
                'role': 'system',
                'content': (
                    'You are Grok, an analyst with access to real-time X/Twitter posts and '
                    'breaking developments. Given a news story, provide:\n'
                    '1. A sharp 2-3 sentence analysis adding the LATEST real-time context, '
                    'social media reaction, or developments that traditional news may not '
                    'have caught yet.\n'
                    '2. Cite specific sources where possible — X/Twitter accounts, posts, '
                    'or recent reports that back up your take.\n'
                    'Format: Start with your analysis, then add "Sources: ..." at the end '
                    'with any relevant @handles, links, or report names.'
                ),
            },
            {
                'role': 'user',
                'content': f'Story: {label}\n\nSummary: {summary}\n\nWhat is the latest '
                           f'real-time context and what are people saying about this?',
            },
        ]

        try:
            result = llm.call(
                messages=messages,
                purpose=f'grok_analysis.{sec.section_type}',
                section='grok_analysis',
                brief_id=brief_id,
                max_tokens=250,
                provider='xai',
            )
            cluster_data['grok_take'] = result['content']
            total_tokens += result['total_tokens']
            total_cost += result['cost_usd']
        except (BudgetExhaustedError, Exception) as e:
            logger.warning(f"[Grok] Failed for '{label[:50]}': {e}")
            continue

    # Persist updated content_json with grok_take fields
    for sec in news_sections:
        db.session.execute(
            db.update(BriefSection)
            .where(BriefSection.id == sec.id)
            .values(content_json=sec.content_json)
        )
    db.session.commit()

    logger.info(
        f"[Grok] Enriched {sum(1 for _, c in top if c.get('grok_take'))} "
        f"stories | {total_tokens} tokens | ${total_cost:.4f}"
    )
    return {'tokens': total_tokens, 'cost': total_cost}


def _grok_timeline_enrichment(brief_id, llm, target_date):
    """Use Grok's real-time data to add latest developments to active timelines.

    For each active timeline, asks Grok what the latest breaking developments
    are, with sources. Adds new events to timelines with source URLs.
    """
    if not llm.xai_available:
        logger.info("[Grok] Skipping timeline enrichment — XAI_API_KEY not configured")
        return None

    from app.models.timeline import Timeline, TimelineEvent
    import json

    timelines = Timeline.query.filter_by(is_active=True).all()
    if not timelines:
        return None

    total_tokens = 0
    total_cost = 0.0
    events_added = 0

    for timeline in timelines:
        entities = ', '.join(timeline.entities_json or [])
        if not entities:
            continue

        messages = [
            {
                'role': 'system',
                'content': (
                    'You are Grok with access to real-time X/Twitter data and breaking news. '
                    'Given a topic and key entities, report the LATEST developments from '
                    'the last 24 hours.\n\n'
                    'Output ONLY a JSON array of events (or empty array [] if nothing new). '
                    'Each event:\n'
                    '- "title": Short event title (max 100 chars)\n'
                    '- "summary": 2-3 sentences with specific details and context\n'
                    '- "entity": Primary entity involved\n'
                    '- "event_type": One of: release, policy, partnership, funding, '
                    'conflict, diplomacy, regulation, announcement, research, milestone\n'
                    '- "significance": 1-10\n'
                    '- "sources": Array of source descriptions (e.g. "@elonmusk post", '
                    '"Reuters report", "SEC filing")\n\n'
                    'Return ONLY the JSON array. Focus on real breaking developments, '
                    'not speculation. Include specific sources.'
                ),
            },
            {
                'role': 'user',
                'content': (
                    f'Timeline: {timeline.name}\n'
                    f'Description: {timeline.description}\n'
                    f'Key entities: {entities}\n\n'
                    f'What are the latest real-time developments (last 24 hours) '
                    f'for these entities? Check X/Twitter posts, breaking news, '
                    f'and official announcements.'
                ),
            },
        ]

        try:
            result = llm.call(
                messages=messages,
                purpose=f'grok_analysis.timeline.{timeline.id}',
                section='grok_analysis',
                brief_id=brief_id,
                max_tokens=800,
                provider='xai',
            )
            total_tokens += result['total_tokens']
            total_cost += result['cost_usd']

            content = result['content'].strip()
            # Parse JSON response
            text = content
            if text.startswith('```'):
                text = text.split('\n', 1)[1]
                if text.endswith('```'):
                    text = text[:-3]
                text = text.strip()

            events_data = json.loads(text)
            for ev in events_data:
                try:
                    sources = ev.get('sources', [])
                    event = TimelineEvent(
                        timeline_id=timeline.id,
                        event_date=target_date,
                        title=ev['title'][:512],
                        summary=ev.get('summary', ''),
                        entity=ev.get('entity', ''),
                        event_type=ev.get('event_type', 'announcement'),
                        significance=min(max(ev.get('significance', 5), 1), 10),
                        source_urls_json=sources,
                        metadata_json={'source': 'grok', 'raw_sources': sources},
                    )
                    db.session.add(event)
                    events_added += 1
                except (ValueError, KeyError) as e:
                    logger.warning(f"[Grok] Skipping malformed timeline event: {e}")

            db.session.commit()

        except (BudgetExhaustedError, Exception) as e:
            logger.warning(f"[Grok] Timeline enrichment failed for '{timeline.name}': {e}")
            continue

    logger.info(
        f"[Grok] Timeline enrichment: {events_added} events added to "
        f"{len(timelines)} timelines | {total_tokens} tokens | ${total_cost:.4f}"
    )
    return {'tokens': total_tokens, 'cost': total_cost, 'events_added': events_added}


def _grok_stories_enrichment(brief_id, llm, target_date):
    """Use Grok to add latest real-time context to tracked stories.

    For each active story with 'developing' or 'ongoing' status, asks Grok
    for the latest developments and adds events with source citations.
    """
    if not llm.xai_available:
        logger.info("[Grok] Skipping stories enrichment — XAI_API_KEY not configured")
        return None

    from app.models.topic import TrackedTopic, Story, Event
    import json

    active_stories = (
        Story.query
        .filter(Story.status.in_(['developing', 'ongoing']))
        .join(TrackedTopic)
        .filter(TrackedTopic.is_active == True)
        .limit(10)
        .all()
    )

    if not active_stories:
        return None

    total_tokens = 0
    total_cost = 0.0
    events_added = 0

    for story in active_stories:
        # Get recent events for context
        recent_events = (
            Event.query
            .filter_by(story_id=story.id)
            .order_by(Event.event_date.desc())
            .limit(3)
            .all()
        )
        recent_context = '\n'.join(
            f"- {e.event_date.strftime('%b %d') if e.event_date else '?'}: {e.description}"
            for e in recent_events
        ) if recent_events else 'No recent events tracked.'

        messages = [
            {
                'role': 'system',
                'content': (
                    'You are Grok with access to real-time X/Twitter data and breaking news. '
                    'Given a tracked news story and its recent events, report any NEW '
                    'developments from the last 24-48 hours.\n\n'
                    'Output ONLY a JSON array of new events (or empty array [] if nothing new). '
                    'Each event:\n'
                    '- "description": 2-3 sentences describing the development with specifics\n'
                    '- "sources": Array of source citations (e.g. "@username on X", '
                    '"AP report", "official press release URL")\n\n'
                    'Return ONLY the JSON array. Only include genuine new developments, '
                    'not rehashes of existing events. Cite specific sources.'
                ),
            },
            {
                'role': 'user',
                'content': (
                    f'Story: {story.title}\n'
                    f'Topic: {story.topic.name}\n'
                    f'Status: {story.status}\n\n'
                    f'Recent events:\n{recent_context}\n\n'
                    f'What are the latest real-time developments on this story? '
                    f'Check X/Twitter, breaking news, and official sources.'
                ),
            },
        ]

        try:
            result = llm.call(
                messages=messages,
                purpose=f'grok_analysis.story.{story.id}',
                section='grok_analysis',
                brief_id=brief_id,
                max_tokens=500,
                provider='xai',
            )
            total_tokens += result['total_tokens']
            total_cost += result['cost_usd']

            content = result['content'].strip()
            text = content
            if text.startswith('```'):
                text = text.split('\n', 1)[1]
                if text.endswith('```'):
                    text = text[:-3]
                text = text.strip()

            events_data = json.loads(text)
            for ev in events_data:
                try:
                    sources = ev.get('sources', [])
                    source_text = ' | '.join(sources) if sources else ''
                    description = ev.get('description', '')
                    if source_text:
                        description = f"{description}\n[Sources: {source_text}]"

                    event = Event(
                        story_id=story.id,
                        description=description,
                        event_date=datetime.now(timezone.utc),
                        source_urls_json=sources,
                    )
                    db.session.add(event)
                    events_added += 1
                except (ValueError, KeyError) as e:
                    logger.warning(f"[Grok] Skipping malformed story event: {e}")

            # Update story's last_updated
            if events_data:
                story.last_updated = datetime.now(timezone.utc)

            db.session.commit()

        except (BudgetExhaustedError, Exception) as e:
            logger.warning(f"[Grok] Stories enrichment failed for '{story.title[:50]}': {e}")
            continue

    logger.info(
        f"[Grok] Stories enrichment: {events_added} events added to "
        f"{len(active_stories)} stories | {total_tokens} tokens | ${total_cost:.4f}"
    )
    return {'tokens': total_tokens, 'cost': total_cost, 'events_added': events_added}


def _grok_stock_fundamentals(brief_id, llm):
    """Use Grok to generate stock fundamentals overview for tracked tickers.

    Adds quarterly performance, revenue, EPS, PE, ROCE, bull/bear cases
    to the investment thesis section's content_json.
    """
    if not llm.xai_available:
        logger.info("[Grok] Skipping stock fundamentals — XAI_API_KEY not configured")
        return None

    import json
    from flask import current_app

    config = current_app.config
    raw_tickers = config.get('HEDGE_FUND_TICKERS', 'AAPL,MSFT,NVDA,GOOGL,AMZN')
    if isinstance(raw_tickers, str):
        tickers = [t.strip() for t in raw_tickers.split(',') if t.strip()]
    else:
        tickers = list(raw_tickers)

    if not tickers:
        return None

    ticker_list = ', '.join(tickers)

    messages = [
        {
            'role': 'system',
            'content': (
                'You are Grok, a financial analyst with access to real-time market data. '
                'Given a list of stock tickers, provide a structured fundamentals overview.\n\n'
                'Output ONLY a JSON array, one object per ticker. Each object:\n'
                '- "ticker": The stock symbol\n'
                '- "company": Full company name\n'
                '- "current_price": Current/latest price (number)\n'
                '- "pe_ratio": Current P/E ratio (number or null)\n'
                '- "eps_ttm": Trailing twelve months EPS (number)\n'
                '- "roce_pct": Return on Capital Employed % (number or null)\n'
                '- "revenue_trend": Brief description of last 2-3 quarters revenue trend\n'
                '- "quarterly_summary": One sentence on recent quarterly performance\n'
                '- "focus_areas": Array of 2-3 key strategic focus areas\n'
                '- "bull_case": 2-3 reasons why the stock could go up\n'
                '- "bear_case": 2-3 reasons why the stock could go down\n'
                '- "sources": Array of source citations\n\n'
                'Use the latest available data. Return ONLY the JSON array.'
            ),
        },
        {
            'role': 'user',
            'content': (
                f'Tickers: {ticker_list}\n\n'
                f'Provide a comprehensive fundamentals overview for each ticker. '
                f'Include their latest quarterly performance, revenue trends, '
                f'EPS numbers, P/E ratio, ROCE, strategic focus areas, '
                f'and specific bull/bear cases with real-time context.'
            ),
        },
    ]

    try:
        result = llm.call(
            messages=messages,
            purpose='grok_analysis.stock_fundamentals',
            section='grok_analysis',
            brief_id=brief_id,
            max_tokens=2000,
            provider='xai',
        )

        content = result['content'].strip()
        text = content
        if text.startswith('```'):
            text = text.split('\n', 1)[1]
            if text.endswith('```'):
                text = text[:-3]
            text = text.strip()

        fundamentals = json.loads(text)

        # Store in the investment thesis section
        thesis_section = BriefSection.query.filter_by(
            brief_id=brief_id, section_type='investment_thesis'
        ).first()

        if thesis_section and thesis_section.content_json:
            updated_content = dict(thesis_section.content_json)
            updated_content['stock_fundamentals'] = fundamentals
            db.session.execute(
                db.update(BriefSection)
                .where(BriefSection.id == thesis_section.id)
                .values(content_json=updated_content)
            )
            db.session.commit()

        logger.info(
            f"[Grok] Stock fundamentals: {len(fundamentals)} tickers | "
            f"{result['total_tokens']} tokens | ${result['cost_usd']:.4f}"
        )
        return {'tokens': result['total_tokens'], 'cost': result['cost_usd']}

    except Exception as e:
        logger.error(f"[Grok] Stock fundamentals failed: {e}")
        return None
