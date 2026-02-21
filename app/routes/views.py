"""
Views blueprint — serves HTML pages via Jinja2.
Queries the database directly (same models as API routes).
"""
import logging
from datetime import date, timedelta
from flask import Blueprint, render_template, request, abort

from app.extensions import db
from app.models.brief import DailyBrief, BriefSection
from app.models.market import MarketSnapshot
from app.models.topic import TrackedTopic, Story, Event
from app.models.investment import InvestmentThesis
from app.models.source import Source
from app.models.cost import DailyCostSummary
from app.models.user import DailyInsight, FeedbackAction
from app.services.cost_service import CostService

logger = logging.getLogger(__name__)

views_bp = Blueprint('views', __name__)


# ── Jinja filters ──────────────────────────────────────

@views_bp.app_template_filter('bias_class')
def bias_class_filter(label):
    """Return CSS class suffix for bias label."""
    if not label:
        return 'center'
    mapping = {
        'left': 'left', 'left-center': 'left',
        'center': 'center',
        'right-center': 'right', 'right': 'right',
    }
    return mapping.get(label.lower(), 'center')


@views_bp.app_template_filter('unique_sources')
def unique_sources_filter(articles):
    """Deduplicate articles by source name, keeping first of each."""
    seen = set()
    result = []
    for a in (articles or []):
        name = a.get('source', 'Unknown')
        if name not in seen:
            seen.add(name)
            result.append(a)
    return result


@views_bp.app_template_filter('trust_level')
def trust_level_filter(score):
    """Return trust level: high/mid/low."""
    if score is None:
        return 'mid'
    if score > 75:
        return 'high'
    if score > 50:
        return 'mid'
    return 'low'


@views_bp.app_template_filter('change_class')
def change_class_filter(pct):
    """Return positive/negative/neutral for market change %."""
    if pct is None or pct == 0:
        return 'neutral'
    return 'positive' if pct > 0 else 'negative'


@views_bp.app_template_filter('change_sign')
def change_sign_filter(pct):
    """Format change with + or - sign."""
    if pct is None:
        return '0.00%'
    sign = '+' if pct > 0 else ''
    return f'{sign}{pct:.2f}%'


@views_bp.app_context_processor
def inject_common():
    """Inject common context into all templates."""
    return {
        'today_date': date.today(),
        'section_labels': {
            'general_news_us': 'US News',
            'ai_news': 'AI & Tech',
            'general_news_india': 'India',
            'general_news_geopolitics': 'Geopolitics',
            'market': 'Market',
            'science': 'Science',
            'health': 'Health',
            'weather': 'Weather',
            'investment_thesis': 'Thesis',
        },
        'news_sections': [
            'general_news_us', 'ai_news', 'general_news_india',
            'general_news_geopolitics', 'science', 'health',
        ],
    }


# ── Helpers ────────────────────────────────────────────

def _get_brief(target_date=None):
    """Get brief for date, or latest complete brief."""
    if target_date:
        brief = DailyBrief.query.filter_by(date=target_date).first()
    else:
        brief = DailyBrief.query.filter_by(date=date.today()).first()
    if not brief or brief.status != 'complete':
        brief = DailyBrief.query.filter_by(status='complete').order_by(
            DailyBrief.date.desc()
        ).first()
    return brief


def _sections_dict(brief):
    """Convert brief sections list to a dict keyed by section_type."""
    if not brief:
        return {}
    result = {}
    for s in sorted(brief.sections, key=lambda x: x.display_order or 0):
        result[s.section_type] = {
            'id': s.id,
            'title': s.title,
            'content': s.content_json or {},
            'degradation': s.degradation_level,
            'tokens': s.tokens_used or 0,
            'cost': s.cost_usd or 0,
        }
    return result


def _compute_overview(sections):
    """Compute overview stats for the brief header tile."""
    total_clusters = 0
    total_sources = 0
    top_stories = []

    for key, sec in sections.items():
        content = sec.get('content', {})
        clusters = content.get('clusters', [])
        total_clusters += len(clusters)
        for cl in clusters:
            total_sources += cl.get('article_count', 0)
            if cl.get('rank_score', 0) > 0.5 and len(top_stories) < 5:
                top_stories.append({
                    'label': cl.get('label', '')[:80],
                    'section': key,
                    'score': cl.get('rank_score', 0),
                })

    # Sort by rank_score, take top 3
    top_stories.sort(key=lambda x: x.get('score', 0), reverse=True)
    top_stories = top_stories[:3]

    # Market highlight
    market = sections.get('market', {}).get('content', {})
    market_data = market.get('market_data', [])
    market_highlight = None
    for m in market_data:
        if m.get('name') and m.get('change_pct') is not None:
            market_highlight = m
            break

    return {
        'total_clusters': total_clusters,
        'total_sources': total_sources,
        'section_count': len([k for k in sections if sections[k].get('content', {}).get('clusters')]),
        'top_stories': top_stories,
        'market_highlight': market_highlight,
    }


# ── Routes ─────────────────────────────────────────────

@views_bp.route('/')
def index():
    """Today's brief — main page."""
    brief = _get_brief()
    sections = _sections_dict(brief)

    # Compute overview stats
    overview = _compute_overview(sections)

    return render_template(
        'pages/today.html',
        active_tab='today',
        brief=brief,
        sections=sections,
        overview=overview,
    )


@views_bp.route('/brief/<brief_date>')
def brief_by_date(brief_date):
    """View brief for a specific date."""
    try:
        from datetime import datetime
        target = datetime.strptime(brief_date, '%Y-%m-%d').date()
    except ValueError:
        abort(404)

    brief = _get_brief(target)
    if not brief:
        abort(404)

    sections = _sections_dict(brief)
    return render_template(
        'pages/today.html',
        active_tab='today',
        brief=brief,
        sections=sections,
    )


@views_bp.route('/market')
def market_page():
    """Full market page with indices and clusters."""
    brief = _get_brief()
    sections = _sections_dict(brief)
    market_section = sections.get('market', {})

    return render_template(
        'pages/market.html',
        active_tab='market',
        brief=brief,
        market=market_section,
    )


@views_bp.route('/stories')
def stories_page():
    """Tracked stories and developing threads."""
    topics = TrackedTopic.query.filter_by(is_active=True).order_by(
        TrackedTopic.name
    ).all()

    topics_data = []
    for topic in topics:
        stories = Story.query.filter_by(topic_id=topic.id).order_by(
            Story.last_updated.desc()
        ).limit(10).all()

        stories_with_events = []
        for story in stories:
            events = Event.query.filter_by(story_id=story.id).order_by(
                Event.event_date.desc()
            ).limit(20).all()
            stories_with_events.append({
                'story': story,
                'events': events,
            })

        topics_data.append({
            'topic': topic,
            'stories': stories_with_events,
        })

    return render_template(
        'pages/stories.html',
        active_tab='stories',
        topics_data=topics_data,
        brief=_get_brief(),
    )


@views_bp.route('/thesis')
def thesis_page():
    """Investment thesis detail page."""
    brief = _get_brief()
    sections = _sections_dict(brief)
    thesis_section = sections.get('investment_thesis', {})

    # Get the actual thesis model for extra detail
    thesis_model = None
    if brief:
        thesis_model = InvestmentThesis.query.filter_by(
            date=brief.date
        ).first()

    return render_template(
        'pages/thesis.html',
        active_tab='thesis',
        brief=brief,
        thesis_section=thesis_section,
        thesis=thesis_model,
    )


@views_bp.route('/settings')
def settings_page():
    """Settings — insights, cost dashboard, sources."""
    from datetime import datetime, timezone

    # Active insights
    now = datetime.now(timezone.utc)
    insights = DailyInsight.query.filter(
        (DailyInsight.expires_at > now) | (DailyInsight.expires_at.is_(None))
    ).order_by(DailyInsight.created_at.desc()).all()

    # Cost data
    cost_service = CostService()
    cost_today = cost_service.get_daily_usage()

    summaries = DailyCostSummary.query.filter(
        DailyCostSummary.date >= date.today() - timedelta(days=30)
    ).order_by(DailyCostSummary.date.desc()).all()

    # Sources
    sources = Source.query.filter_by(is_active=True).order_by(
        Source.section, Source.name
    ).all()

    # Group sources by section
    sources_by_section = {}
    for s in sources:
        sources_by_section.setdefault(s.section, []).append(s)

    return render_template(
        'pages/settings.html',
        active_tab='settings',
        brief=_get_brief(),
        insights=insights,
        cost_today=cost_today,
        cost_history=summaries,
        sources_by_section=sources_by_section,
    )


@views_bp.route('/history')
def history_page():
    """Paginated brief archive."""
    page = request.args.get('page', 1, type=int)
    per_page = 20

    pagination = DailyBrief.query.order_by(
        DailyBrief.date.desc()
    ).paginate(page=page, per_page=per_page, error_out=False)

    return render_template(
        'pages/history.html',
        active_tab='settings',
        brief=_get_brief(),
        briefs=pagination.items,
        pagination=pagination,
    )
