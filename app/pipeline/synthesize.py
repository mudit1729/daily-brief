import logging
from datetime import datetime, timezone, date
from flask import current_app
from app.extensions import db
from app.models.cluster import Cluster, ClusterMembership
from app.models.article import Article
from app.models.brief import DailyBrief, BriefSection
from app.models.market import MarketSnapshot
from app.models.weather import WeatherCache
from app.integrations.llm_gateway import LLMGateway, BudgetExhaustedError
from app.integrations.weather import WeatherService
from app.services.investment_service import InvestmentService
from app.services.cost_service import CostService
from app.utils.text import extract_lead_sentences, truncate

logger = logging.getLogger(__name__)

# Section definitions with priority order and display config
SECTIONS = [
    {'key': 'general_news_us', 'title': 'US News', 'cluster_section': 'general_news', 'region_filter': 'us', 'order': 0},
    {'key': 'market', 'title': 'Market Trends', 'cluster_section': 'market', 'order': 1},
    {'key': 'ai_news', 'title': 'AI & Tech', 'cluster_section': 'ai_news', 'order': 2},
    {'key': 'general_news_india', 'title': 'India', 'cluster_section': 'general_news', 'region_filter': 'india', 'order': 3},
    {'key': 'general_news_geopolitics', 'title': 'Geopolitics', 'cluster_section': 'general_news', 'region_filter': 'global', 'order': 4},
    {'key': 'weather', 'title': 'Weather', 'order': 5},
    {'key': 'science', 'title': 'Science', 'cluster_section': 'science', 'order': 6},
    {'key': 'health', 'title': 'Health', 'cluster_section': 'health', 'order': 7},
    {'key': 'investment_thesis', 'title': 'Investment Thesis', 'order': 8},
]


def run(target_date, brief_id):
    """Step 5: Synthesize daily brief from ranked clusters."""
    logger.info(f"[Synthesize] Starting for {target_date}")

    config = current_app.config
    llm = LLMGateway(config)
    cost_service = CostService()

    brief = DailyBrief.query.get(brief_id)
    brief.status = 'generating'
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
                total_cost += section.cost_usd or 0

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

    clusters = Cluster.query.filter_by(
        date=target_date,
        section=cluster_section,
    ).order_by(Cluster.rank_score.desc()).all()

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
    ).order_by(Cluster.rank_score.desc()).limit(5).all()

    cluster_summaries = []
    section_tokens = 0
    section_cost = 0.0

    for cluster in clusters:
        articles = _get_cluster_articles(cluster)
        degradation = llm.determine_degradation_level('market')

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
    """Build investment thesis section."""
    snapshots = MarketSnapshot.query.filter_by(snapshot_date=target_date).all()
    snapshot_dicts = [
        {'symbol': s.symbol, 'name': s.name, 'price': s.price,
         'change_pct': s.change_pct, 'change_abs': s.change_abs}
        for s in snapshots
    ]

    top_clusters = Cluster.query.filter_by(date=target_date).order_by(
        Cluster.rank_score.desc()
    ).limit(5).all()

    investment_service = InvestmentService()
    thesis = investment_service.generate_thesis(
        target_date, brief_id, snapshot_dicts, top_clusters, llm
    )

    content = {
        'thesis': thesis.thesis_text if thesis else 'Investment thesis feature disabled',
        'gate_passed': thesis.gate_passed if thesis else False,
        'signals': {
            'momentum': thesis.momentum_signal if thesis else None,
            'value': thesis.value_signal if thesis else None,
        },
    }

    return BriefSection(
        brief_id=brief_id,
        section_type='investment_thesis',
        title=section_def['title'],
        content_json=content,
        display_order=section_def['order'],
    )


def _get_cluster_articles(cluster):
    """Get all articles in a cluster."""
    memberships = ClusterMembership.query.filter_by(cluster_id=cluster.id).all()
    article_ids = [m.article_id for m in memberships]
    return Article.query.filter(Article.id.in_(article_ids)).all() if article_ids else []


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
                'bias_label': a.source.bias_label if a.source else 'center',
                'trust_score': a.source.trust_score if a.source else 50,
            }
            for a in articles[:5]
        ],
    }


def _max_clusters_for_degradation(level):
    """Max clusters to process at each degradation level."""
    return {0: 10, 1: 8, 2: 6, 3: 3, 4: 5}.get(level, 5)
