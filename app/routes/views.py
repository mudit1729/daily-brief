"""
Views blueprint — serves HTML pages via Jinja2.
Queries the database directly (same models as API routes).
"""
import hmac
import logging
from datetime import date, timedelta, datetime, timezone
from flask import Blueprint, render_template, request, abort, jsonify, current_app, session, redirect, url_for

from app.extensions import db, scheduler
from app.models.brief import DailyBrief, BriefSection
from app.models.market import MarketSnapshot
from app.models.topic import TrackedTopic, Story, Event
from app.models.investment import InvestmentThesis
from app.models.source import Source
from app.models.cost import DailyCostSummary
from app.models.user import DailyInsight, FeedbackAction
from app.models.timeline import Timeline, TimelineEvent
from app.models.cluster import Cluster, ClusterMembership
from app.models.article import Article
from app.models.calendar_event import CalendarEvent
from app.services.cost_service import CostService
from app.services.timeline_service import TimelineService
from app.services.scheduler_config_service import SchedulerConfigService

logger = logging.getLogger(__name__)

views_bp = Blueprint('views', __name__)


# ── Site-wide password gate ───────────────────────────

@views_bp.before_request
def require_site_password():
    """If SITE_PASSWORD is set, require login before any page."""
    password = current_app.config.get('SITE_PASSWORD')
    if not password:
        return  # no password configured — site is open
    if request.endpoint in ('views.login', 'static'):
        return  # allow the login page and static assets through
    if session.get('site_authenticated'):
        return  # already logged in
    return redirect(url_for('views.login', next=request.path))


@views_bp.route('/login', methods=['GET', 'POST'])
def login():
    """Simple password gate."""
    password = current_app.config.get('SITE_PASSWORD')
    if not password:
        return redirect(url_for('views.index'))

    error = None
    if request.method == 'POST':
        entered = request.form.get('password', '')
        logger.info(f"Login attempt: entered={entered!r}, expected={password!r}")
        if entered == password:
            session.permanent = False  # Expire when browser closes
            session['site_authenticated'] = True
            next_url = request.args.get('next', '/')
            return redirect(next_url)
        error = 'Wrong password.'

    return render_template('pages/login.html', error=error)


@views_bp.route('/logout')
def logout():
    session.pop('site_authenticated', None)
    return redirect(url_for('views.login'))


@views_bp.route('/debug-auth')
def debug_auth():
    """Temporary debug endpoint — remove after fixing login."""
    import os
    return jsonify({
        'SITE_PASSWORD_from_config': bool(current_app.config.get('SITE_PASSWORD')),
        'SITE_PASSWORD_len': len(current_app.config.get('SITE_PASSWORD') or ''),
        'PASSWORD_env': bool(os.getenv('PASSWORD')),
        'SITE_PASSWORD_env': bool(os.getenv('SITE_PASSWORD')),
        'session_authenticated': session.get('site_authenticated', False),
    })


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


def _fmt_number(val, prefix='', suffix='', decimals=1):
    """Format a number with optional prefix/suffix, handling large values."""
    if val is None:
        return 'N/A'
    try:
        v = float(val)
    except (TypeError, ValueError):
        return str(val)
    if abs(v) >= 1e12:
        return f'{prefix}{v/1e12:.{decimals}f}T{suffix}'
    if abs(v) >= 1e9:
        return f'{prefix}{v/1e9:.{decimals}f}B{suffix}'
    if abs(v) >= 1e6:
        return f'{prefix}{v/1e6:.{decimals}f}M{suffix}'
    if abs(v) >= 1e3:
        return f'{prefix}{v/1e3:.{decimals}f}K{suffix}'
    return f'{prefix}{v:.{decimals}f}{suffix}'


def _fmt_pct(val, decimals=1):
    """Format a value as percentage."""
    if val is None:
        return 'N/A'
    try:
        v = float(val)
        return f'{v:.{decimals}f}%'
    except (TypeError, ValueError):
        return str(val)


def _signal_span(signal):
    """Wrap a signal word in a colored span."""
    s = str(signal).lower() if signal else 'neutral'
    return f'<span class="sb-hf-signal--{s}">{s}</span>'


def _format_technical_reasoning(data):
    """Format Technical Analyst structured reasoning."""
    lines = []
    strategy_labels = {
        'trend_following': 'Trend',
        'mean_reversion': 'Mean Rev',
        'momentum': 'Momentum',
        'volatility': 'Volatility',
        'statistical_arbitrage': 'Stat Arb',
    }
    for key, label in strategy_labels.items():
        strat = data.get(key)
        if not strat or not isinstance(strat, dict):
            continue
        sig = strat.get('signal', 'neutral')
        conf = strat.get('confidence', 0)
        metrics = strat.get('metrics', {})

        # Pick the most meaningful metric(s) to show
        detail_parts = []
        if key == 'trend_following':
            if 'adx' in metrics:
                detail_parts.append(f'ADX: {float(metrics["adx"]):.1f}')
        elif key == 'mean_reversion':
            if 'rsi_14' in metrics:
                detail_parts.append(f'RSI-14: {float(metrics["rsi_14"]):.1f}')
            if 'z_score' in metrics:
                detail_parts.append(f'Z: {float(metrics["z_score"]):.2f}')
        elif key == 'momentum':
            for m_key, m_label in [('momentum_1m', '1M'), ('momentum_3m', '3M')]:
                if m_key in metrics:
                    v = float(metrics[m_key]) * 100
                    detail_parts.append(f'{m_label}: {v:+.1f}%')
        elif key == 'volatility':
            if 'volatility_regime' in metrics:
                detail_parts.append(f'{metrics["volatility_regime"]}')
            if 'historical_volatility' in metrics:
                detail_parts.append(f'HV: {float(metrics["historical_volatility"]):.1%}')
        elif key == 'statistical_arbitrage':
            if 'hurst_exponent' in metrics:
                detail_parts.append(f'Hurst: {float(metrics["hurst_exponent"]):.2f}')

        detail = ' · '.join(detail_parts) if detail_parts else ''
        line = f'<div class="sb-hf-reasoning-line">'
        line += f'<span class="sb-hf-reasoning-label">{label}:</span> '
        line += f'{_signal_span(sig)} ({conf}%)'
        if detail:
            line += f' <span class="sb-hf-reasoning-detail">{detail}</span>'
        line += '</div>'
        lines.append(line)
    return '\n'.join(lines) if lines else _fallback_format(data)


def _format_sentiment_reasoning(data):
    """Format Sentiment Analyst structured reasoning."""
    lines = []
    # Insider trading
    insider = data.get('insider_trading')
    if isinstance(insider, dict):
        sig = insider.get('signal', 'neutral')
        conf = insider.get('confidence', 0)
        metrics = insider.get('metrics', {})
        total = metrics.get('total_trades', 0)
        bearish = metrics.get('bearish_trades', 0)
        bullish = metrics.get('bullish_trades', 0)
        detail = f'{total} trades ({bullish} bullish, {bearish} bearish)' if total else ''
        line = f'<div class="sb-hf-reasoning-line">'
        line += f'<span class="sb-hf-reasoning-label">Insider:</span> '
        line += f'{_signal_span(sig)} ({conf}%)'
        if detail:
            line += f' <span class="sb-hf-reasoning-detail">{detail}</span>'
        line += '</div>'
        lines.append(line)

    # News sentiment
    news = data.get('news_sentiment')
    if isinstance(news, dict):
        sig = news.get('signal', 'neutral')
        conf = news.get('confidence', 0)
        metrics = news.get('metrics', {})
        total = metrics.get('total_articles', 0)
        detail = f'{total} articles' if total else ''
        line = f'<div class="sb-hf-reasoning-line">'
        line += f'<span class="sb-hf-reasoning-label">News:</span> '
        line += f'{_signal_span(sig)} ({conf}%)'
        if detail:
            line += f' <span class="sb-hf-reasoning-detail">{detail}</span>'
        line += '</div>'
        lines.append(line)

    # Combined analysis
    combined = data.get('combined_analysis')
    if isinstance(combined, dict):
        determination = combined.get('signal_determination', '')
        if determination:
            lines.append(
                f'<div class="sb-hf-reasoning-line sb-hf-reasoning-line--summary">'
                f'<span class="sb-hf-reasoning-detail">{determination}</span>'
                f'</div>'
            )
    return '\n'.join(lines) if lines else _fallback_format(data)


def _format_valuation_reasoning(data):
    """Format Valuation Analyst structured reasoning."""
    lines = []
    method_labels = {
        'dcf_analysis': 'DCF',
        'owner_earnings_analysis': 'Owner Earnings',
        'ev_ebitda_analysis': 'EV/EBITDA',
        'residual_income_analysis': 'Residual Income',
    }
    for key, label in method_labels.items():
        method = data.get(key)
        if not method or not isinstance(method, dict):
            continue
        sig = method.get('signal', 'neutral')
        details_str = method.get('details', '')
        # Extract gap percentage from details string
        gap_part = ''
        if isinstance(details_str, str) and 'Gap:' in details_str:
            import re
            gap_match = re.search(r'Gap:\s*([-+]?[\d.]+%)', details_str)
            if gap_match:
                gap_part = f'Gap: {gap_match.group(1)}'
        line = f'<div class="sb-hf-reasoning-line">'
        line += f'<span class="sb-hf-reasoning-label">{label}:</span> '
        line += f'{_signal_span(sig)}'
        if gap_part:
            line += f' <span class="sb-hf-reasoning-detail">{gap_part}</span>'
        line += '</div>'
        lines.append(line)

    # DCF scenarios
    scenarios = data.get('dcf_scenario_analysis')
    if isinstance(scenarios, dict):
        bear = scenarios.get('bear_case', '')
        base = scenarios.get('base_case', '')
        bull = scenarios.get('bull_case', '')
        if bear or base or bull:
            lines.append(
                f'<div class="sb-hf-reasoning-line sb-hf-reasoning-line--summary">'
                f'<span class="sb-hf-reasoning-detail">'
                f'Bear: {bear} · Base: {base} · Bull: {bull}'
                f'</span></div>'
            )
    return '\n'.join(lines) if lines else _fallback_format(data)


def _format_risk_reasoning(data):
    """Format Risk Management structured reasoning."""
    reasoning = data.get('reasoning', data)
    if not isinstance(reasoning, dict):
        return str(reasoning)[:200] if reasoning else ''

    parts = []
    if 'combined_position_limit_pct' in reasoning:
        parts.append(f'Position limit: {float(reasoning["combined_position_limit_pct"]):.1%}')
    if 'current_position_value' in reasoning:
        parts.append(f'Exposure: {_fmt_number(reasoning["current_position_value"], prefix="$")}')
    if 'portfolio_value' in reasoning or 'available_cash' in reasoning:
        val = reasoning.get('portfolio_value') or reasoning.get('available_cash')
        parts.append(f'Portfolio: {_fmt_number(val, prefix="$")}')

    if parts:
        return (
            f'<div class="sb-hf-reasoning-line">'
            f'<span class="sb-hf-reasoning-detail">{" · ".join(parts)}</span>'
            f'</div>'
        )
    return _fallback_format(reasoning)


def _fallback_format(data):
    """Fallback: extract any signal/confidence sub-dicts, or show truncated string."""
    if isinstance(data, str):
        return data[:200]
    if not isinstance(data, dict):
        return str(data)[:200]

    # Try to find sub-signal dicts
    lines = []
    for key, val in list(data.items())[:6]:
        label = key.replace('_', ' ').title()
        if isinstance(val, dict):
            sig = val.get('signal')
            conf = val.get('confidence')
            if sig:
                line = f'<div class="sb-hf-reasoning-line">'
                line += f'<span class="sb-hf-reasoning-label">{label}:</span> '
                line += f'{_signal_span(sig)}'
                if conf is not None:
                    line += f' ({conf}%)'
                line += '</div>'
                lines.append(line)
                continue
        # Simple key-value
        val_str = str(val)[:80]
        lines.append(
            f'<div class="sb-hf-reasoning-line">'
            f'<span class="sb-hf-reasoning-label">{label}:</span> '
            f'<span class="sb-hf-reasoning-detail">{val_str}</span>'
            f'</div>'
        )
    return '\n'.join(lines) if lines else str(data)[:200]


@views_bp.app_template_filter('render_thesis')
def render_thesis_filter(text):
    """Convert thesis text into structured HTML with bold rendering and paragraph breaks."""
    import re
    from markupsafe import Markup

    if not text:
        return ''

    # Convert **bold** to <strong>
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)

    # Split into paragraphs at key transition phrases
    sentences = re.split(
        r'(?<=\.) (?=However,|Watch for|Key risks|In India,|In [A-Z][a-z]+,)',
        text,
    )

    if len(sentences) <= 1:
        return Markup(f'<p>{text}</p>')

    paragraphs = [f'<p>{s.strip()}</p>' for s in sentences if s.strip()]
    return Markup('\n'.join(paragraphs))


@views_bp.app_template_filter('format_hf_reasoning')
def format_hf_reasoning_filter(reasoning, agent_name=''):
    """Format hedge fund analyst reasoning into readable HTML.

    Handles both string reasoning (Warren Buffett, etc.) and structured
    dict reasoning (Technical, Sentiment, Valuation, Risk Management).
    """
    from markupsafe import Markup

    if not reasoning:
        return ''

    # String reasoning — already human-readable
    if isinstance(reasoning, str):
        return reasoning

    if not isinstance(reasoning, dict):
        return str(reasoning)[:200]

    # Route to agent-specific formatter based on agent name
    name = agent_name.lower().replace(' ', '_')
    if 'technical' in name:
        html = _format_technical_reasoning(reasoning)
    elif 'sentiment' in name:
        html = _format_sentiment_reasoning(reasoning)
    elif 'valuation' in name:
        html = _format_valuation_reasoning(reasoning)
    elif 'risk' in name:
        html = _format_risk_reasoning(reasoning)
    else:
        html = _fallback_format(reasoning)

    return Markup(html)


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
            'feel_good': 'Feel Good',
            'weather': 'Weather',
            'investment_thesis': 'Thesis',
        },
        'news_sections': [
            'general_news_us', 'ai_news', 'general_news_india',
            'general_news_geopolitics', 'science', 'health', 'feel_good',
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
                    'label': (cl.get('label') or '')[:80],
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
    """Full market page with live indices and brief clusters."""
    brief = _get_brief()
    sections = _sections_dict(brief)
    market_section = sections.get('market', {})

    # Fetch live market data (with multi-period performance)
    try:
        from app.integrations.market_data import MarketDataService
        live_snapshots = MarketDataService().fetch_snapshots()
    except Exception as e:
        logger.warning(f"Failed to fetch live market data: {e}")
        live_snapshots = None

    # Use live data if available, fall back to brief data
    if live_snapshots:
        content = dict(market_section.get('content', {}))
        content['market_data'] = live_snapshots
        market_section = dict(market_section, content=content)

    return render_template(
        'pages/market.html',
        active_tab='market',
        brief=brief,
        market=market_section,
    )


@views_bp.route('/stories')
def stories_page():
    """Tracked stories and developing threads."""
    now = datetime.now(timezone.utc)

    def _to_utc(dt_val):
        if not dt_val:
            return None
        if dt_val.tzinfo is None:
            return dt_val.replace(tzinfo=timezone.utc)
        return dt_val.astimezone(timezone.utc)

    topics = TrackedTopic.query.filter_by(is_active=True).order_by(
        TrackedTopic.name
    ).all()

    topics_data = []
    totals = {
        'stories': 0,
        'events': 0,
        'active': 0,
        'resolved': 0,
    }

    for topic in topics:
        stories = Story.query.filter_by(topic_id=topic.id).order_by(
            Story.last_updated.desc()
        ).limit(10).all()

        stories_with_events = []
        for story in stories:
            events = Event.query.filter_by(story_id=story.id).order_by(
                Event.event_date.desc()
            ).limit(20).all()

            source_urls = set()
            source_labels = set()
            for event in events:
                for src in (event.source_urls_json or []):
                    if not src:
                        continue
                    s = str(src).strip()
                    if s.startswith('http://') or s.startswith('https://'):
                        source_urls.add(s)
                    else:
                        source_labels.add(s)

            last_activity = _to_utc(story.last_updated or story.first_seen)
            quiet_days = 0
            age_days = 0
            if last_activity:
                quiet_days = max(0, (now - last_activity).days)
            first_seen = _to_utc(story.first_seen)
            if first_seen:
                age_days = max(0, (now - first_seen).days)

            status = (story.status or '').lower()
            totals['stories'] += 1
            totals['events'] += len(events)
            if status == 'resolved':
                totals['resolved'] += 1
            else:
                totals['active'] += 1

            stories_with_events.append({
                'story': story,
                'events': events,
                'event_count': len(events),
                'source_count': len(source_urls) + len(source_labels),
                'source_urls': sorted(source_urls)[:3],
                'source_labels': sorted(source_labels)[:5],
                'last_activity': last_activity,
                'quiet_days': quiet_days,
                'age_days': age_days,
            })

        topics_data.append({
            'topic': topic,
            'stories': stories_with_events,
            'event_count': sum(sd['event_count'] for sd in stories_with_events),
        })

    return render_template(
        'pages/stories.html',
        active_tab='stories',
        topics_data=topics_data,
        story_totals=totals,
        brief=_get_brief(),
    )


@views_bp.route('/thesis')
def thesis_page():
    """Investment thesis detail page with AI analyst signals."""
    brief = _get_brief()
    sections = _sections_dict(brief)
    thesis_section = sections.get('investment_thesis', {})

    # Get the actual thesis model for extra detail
    thesis_model = None
    if brief:
        thesis_model = InvestmentThesis.query.filter_by(
            date=brief.date
        ).first()

    # Hedge fund signals (from BriefSection content or direct DB query)
    hf_signals = []
    thesis_content = thesis_section.get('content', {})
    if thesis_content.get('hedge_fund_signals'):
        hf_signals = thesis_content['hedge_fund_signals']
    elif brief:
        from app.models.hedge_fund import HedgeFundAnalysis
        hf_analyses = HedgeFundAnalysis.query.filter_by(
            date=brief.date
        ).order_by(HedgeFundAnalysis.ticker).all()
        hf_signals = [a.to_dict() for a in hf_analyses]

    return render_template(
        'pages/thesis.html',
        active_tab='thesis',
        brief=brief,
        thesis_section=thesis_section,
        thesis=thesis_model,
        hf_signals=hf_signals,
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
    source_health_summary = {
        'healthy': 0,
        'degraded': 0,
        'cooldown': 0,
        'inactive': 0,
    }
    for s in sources:
        sources_by_section.setdefault(s.section, []).append(s)
        state = s.health_state()
        source_health_summary[state] = source_health_summary.get(state, 0) + 1

    pipeline_schedule = SchedulerConfigService().get_pipeline_schedule()
    next_run_at = None
    try:
        job = scheduler.get_job('daily_pipeline')
        if job and job.next_run_time:
            next_run_at = job.next_run_time
    except Exception:
        pass

    return render_template(
        'pages/settings.html',
        active_tab='settings',
        brief=_get_brief(),
        insights=insights,
        cost_today=cost_today,
        cost_history=summaries,
        sources_by_section=sources_by_section,
        source_health_summary=source_health_summary,
        pipeline_schedule=pipeline_schedule,
        pipeline_next_run_at=next_run_at,
    )


@views_bp.route('/timelines')
def timelines_page():
    """Timelines — curated chronological event views."""
    timelines = Timeline.query.filter_by(is_active=True).order_by(
        Timeline.created_at.desc()
    ).all()

    # Enrich with event counts and latest event
    timelines_data = []
    for t in timelines:
        events = TimelineEvent.query.filter_by(timeline_id=t.id).order_by(
            TimelineEvent.event_date.desc()
        ).all()

        # Date range
        first_date = events[-1].event_date if events else None
        last_date = events[0].event_date if events else None

        # Entity color assignments
        ts = TimelineService()
        entity_colors = ts.get_entity_colors(t)

        timelines_data.append({
            'timeline': t,
            'event_count': len(events),
            'first_date': first_date,
            'last_date': last_date,
            'entity_colors': entity_colors,
        })

    return render_template(
        'pages/timelines.html',
        active_tab='timelines',
        timelines_data=timelines_data,
        brief=_get_brief(),
    )


@views_bp.route('/timelines/<int:timeline_id>')
def timeline_detail(timeline_id):
    """Detailed view of a single timeline."""
    timeline = Timeline.query.get_or_404(timeline_id)

    ts = TimelineService()
    months = ts.get_timeline_events_grouped(timeline_id)
    entity_colors = ts.get_entity_colors(timeline)

    # Entity filter
    entity_filter = request.args.get('entity')
    if entity_filter:
        for month in months:
            month['events'] = [
                e for e in month['events'] if e.entity == entity_filter
            ]
        months = [m for m in months if m['events']]

    return render_template(
        'pages/timeline_detail.html',
        active_tab='timelines',
        timeline=timeline,
        months=months,
        entity_colors=entity_colors,
        entity_filter=entity_filter,
        brief=_get_brief(),
    )


@views_bp.route('/timelines/<int:timeline_id>/delete', methods=['POST'])
def delete_timeline(timeline_id):
    """Soft-delete a timeline (set is_active=False)."""
    timeline = Timeline.query.get_or_404(timeline_id)
    timeline.is_active = False
    db.session.commit()
    return jsonify({'status': 'ok', 'id': timeline_id})


@views_bp.route('/stories/<int:story_id>/delete', methods=['POST'])
def delete_story(story_id):
    """Soft-delete a story (set status to 'removed')."""
    story = Story.query.get_or_404(story_id)
    story.status = 'removed'
    db.session.commit()
    return jsonify({'status': 'ok', 'id': story_id})


@views_bp.route('/topics/<int:topic_id>/delete', methods=['POST'])
def delete_topic(topic_id):
    """Soft-delete a tracked topic (set is_active=False)."""
    topic = TrackedTopic.query.get_or_404(topic_id)
    topic.is_active = False
    db.session.commit()
    return jsonify({'status': 'ok', 'id': topic_id})


@views_bp.route('/prep')
def prep_page():
    """Prep — algorithm visualizer link + categorized markdown notes viewer."""
    import os
    notes_dir = current_app.config.get('PREP_NOTES_DIR', 'notes')
    # Resolve relative to project root
    if not os.path.isabs(notes_dir):
        notes_dir = os.path.join(current_app.root_path, '..', notes_dir)
    notes_dir = os.path.abspath(notes_dir)

    all_notes = []
    if os.path.isdir(notes_dir):
        for fname in sorted(os.listdir(notes_dir)):
            if fname.lower().endswith('.md'):
                all_notes.append(fname)

    # Categorize notes into sections
    # Each category: { 'label': str, 'icon': str, 'notes': [filename, ...] }
    CATEGORIES = [
        # Prefix-based categories first (more specific → less specific)
        {'label': 'Ilya 30', 'icon': 'paper', 'match': ['Ilya30-']},
        {'label': 'ML Paper Implementation', 'icon': 'cpu', 'match': ['MLPaper-']},
        {'label': 'ML Theory Interview', 'icon': 'book', 'match': ['MLTheory-']},
        {'label': 'BEV Seminal Papers', 'icon': 'eye', 'match': ['BEV-']},
        {'label': 'Paper Summaries', 'icon': 'paper', 'match': ['Paper-']},
        {'label': 'Async Processing', 'icon': 'async', 'match': ['Async-']},
        {'label': 'Planner Papers', 'icon': 'cpu', 'match': ['Planner-']},
        {'label': 'VLA Seminal Papers', 'icon': 'eye', 'match': ['VLA-']},
        {'label': 'ML Coursework', 'icon': 'book', 'match': ['CMU-']},
        # Substring-match categories after prefix categories
        {'label': 'Coding', 'icon': 'code', 'match': ['Amazon-150', 'Blind-75', 'Graph-Problems']},
        {'label': 'ML Coding', 'icon': 'cpu', 'match': ['DeepML', 'Transformers-ML150', 'Vision-Transformer', 'Attention-Is-All-You-Need']},
        {'label': 'ML Theory', 'icon': 'book', 'match': ['ML-Comprehensive-Guide']},
    ]

    categorized = []
    categorized_files = set()
    for cat in CATEGORIES:
        cat_notes = []
        for note in all_notes:
            if note in categorized_files:
                continue
            name_no_ext = note.rsplit('.', 1)[0]
            if any(m.lower() in name_no_ext.lower() for m in cat['match']):
                cat_notes.append(note)
                categorized_files.add(note)
        if cat_notes:
            categorized.append({
                'label': cat['label'],
                'icon': cat['icon'],
                'notes': cat_notes,
            })

    # Uncategorized notes go into "Other"
    other = [n for n in all_notes if n not in categorized_files and n != 'README.md']
    if other:
        categorized.append({
            'label': 'Other',
            'icon': 'file',
            'notes': other,
        })

    return render_template(
        'pages/prep.html',
        active_tab='prep',
        categories=categorized,
        has_notes=bool(all_notes),
        brief=_get_brief(),
    )


@views_bp.route('/api/prep/notes/<path:filename>')
def prep_note(filename):
    """Render a markdown note to HTML."""
    import os
    import re
    import markdown2

    notes_dir = current_app.config.get('PREP_NOTES_DIR', 'notes')
    if not os.path.isabs(notes_dir):
        notes_dir = os.path.join(current_app.root_path, '..', notes_dir)
    notes_dir = os.path.abspath(notes_dir)

    # Prevent path traversal
    safe_path = os.path.normpath(os.path.join(notes_dir, filename))
    if not safe_path.startswith(notes_dir):
        abort(403)
    if not os.path.isfile(safe_path):
        abort(404)

    with open(safe_path, 'r', encoding='utf-8') as f:
        raw = f.read()

    html = markdown2.markdown(
        raw,
        extras=['fenced-code-blocks', 'tables', 'code-friendly', 'header-ids', 'cuddled-lists'],
    )

    # Strip Pygments' inline-styled spans so highlight.js can re-highlight client-side.
    # Converts <div class="codehilite"><pre ...><code><span style="...">text</span></code></pre></div>
    # into clean <pre><code class="language-X">text</code></pre>
    def _clean_pygments(m):
        block = m.group(0)
        # Extract language from codehilite class or data attribute if present
        import re as _re
        # Strip all <span ...> and </span> tags, keeping text content
        clean = _re.sub(r'</?span[^>]*>', '', block)
        # Remove codehilite wrapper div
        clean = _re.sub(r'<div[^>]*class="codehilite"[^>]*>\s*', '', clean)
        clean = _re.sub(r'\s*</div>', '', clean)
        # Remove inline styles from pre tag
        clean = _re.sub(r'<pre[^>]*>', '<pre>', clean)
        # Decode HTML entities back for highlight.js
        clean = clean.replace('&#39;', "'").replace('&amp;', '&')
        return clean
    html = re.sub(r'<div[^>]*class="codehilite"[^>]*>.*?</div>', _clean_pygments, html, flags=re.DOTALL)

    # Rewrite relative image paths to point to the notes image route
    html = re.sub(
        r'<img\s+([^>]*?)src="(?!https?://|/)([^"]+)"',
        r'<img \1src="/api/prep/images/\2"',
        html,
    )

    return jsonify({'filename': filename, 'html': html})


@views_bp.route('/api/prep/images/<path:filepath>')
def prep_image(filepath):
    """Serve images from the notes directory."""
    import os
    from flask import send_file

    notes_dir = current_app.config.get('PREP_NOTES_DIR', 'notes')
    if not os.path.isabs(notes_dir):
        notes_dir = os.path.join(current_app.root_path, '..', notes_dir)
    notes_dir = os.path.abspath(notes_dir)

    safe_path = os.path.normpath(os.path.join(notes_dir, filepath))
    if not safe_path.startswith(notes_dir):
        abort(403)
    if not os.path.isfile(safe_path):
        abort(404)

    return send_file(safe_path)


@views_bp.route('/api/prep/notes/rename', methods=['POST'])
def prep_rename_note():
    """Rename a markdown note file.

    Request JSON:
        old_filename: str — current filename (e.g. "Blind-75.md")
        new_title: str    — new display title (will be converted to a safe filename)
    """
    import os
    import re as _re

    data = request.get_json(silent=True) or {}
    old_filename = data.get('old_filename', '').strip()
    new_title = data.get('new_title', '').strip()

    if not old_filename or not new_title:
        return jsonify({'error': 'old_filename and new_title required'}), 400

    notes_dir = current_app.config.get('PREP_NOTES_DIR', 'notes')
    if not os.path.isabs(notes_dir):
        notes_dir = os.path.join(current_app.root_path, '..', notes_dir)
    notes_dir = os.path.abspath(notes_dir)

    old_path = os.path.normpath(os.path.join(notes_dir, old_filename))
    if not old_path.startswith(notes_dir) or not os.path.isfile(old_path):
        return jsonify({'error': 'File not found'}), 404

    # Preserve the category prefix (e.g. "Ilya30-", "BEV-", "MLPaper-")
    old_base = old_filename.rsplit('.', 1)[0]  # Remove .md
    prefix = ''
    KNOWN_PREFIXES = [
        'Ilya30-', 'MLPaper-', 'MLTheory-', 'BEV-', 'Paper-', 'Async-', 'Planner-',
    ]
    for p in KNOWN_PREFIXES:
        if old_base.startswith(p):
            prefix = p
            break

    # Convert title to safe filename: replace spaces with hyphens, keep alphanumeric
    safe_title = _re.sub(r'[^\w\s-]', '', new_title).strip()
    safe_title = _re.sub(r'[\s]+', '-', safe_title)
    if not safe_title:
        return jsonify({'error': 'Invalid title'}), 400

    new_filename = prefix + safe_title + '.md'
    new_path = os.path.normpath(os.path.join(notes_dir, new_filename))

    if not new_path.startswith(notes_dir):
        return jsonify({'error': 'Invalid path'}), 400

    if new_path == old_path:
        return jsonify({'filename': new_filename, 'unchanged': True})

    if os.path.exists(new_path):
        return jsonify({'error': 'A file with that name already exists'}), 409

    try:
        os.rename(old_path, new_path)
    except OSError as e:
        logger.error(f"Failed to rename {old_filename} → {new_filename}: {e}")
        return jsonify({'error': 'Rename failed'}), 500

    logger.info(f"[Prep] Renamed note: {old_filename} → {new_filename}")
    return jsonify({'filename': new_filename, 'old_filename': old_filename})


@views_bp.route('/api/prep/chat', methods=['POST'])
def prep_chat():
    """Stream a Claude response about the currently viewed markdown note.

    Uses Anthropic prompt caching so the document context is cached across
    messages in a conversation, reducing latency and cost.

    Request JSON:
        filename: str   — markdown file being viewed
        message: str    — user's question
        history: list   — prior [{role, content}, ...] messages (optional)
        section: str|null — selected text to narrow context (optional)
    """
    import os
    import json as _json
    import anthropic

    data = request.get_json(silent=True) or {}
    filename = data.get('filename')
    user_message = (data.get('message') or '').strip()
    history = data.get('history') or []
    section = data.get('section')

    if not filename or not user_message:
        return jsonify({'error': 'filename and message required'}), 400

    api_key = current_app.config.get('ANTHROPIC_API_KEY')
    if not api_key:
        return jsonify({'error': 'ANTHROPIC_API_KEY not configured'}), 503

    # ── Resolve & validate file path ──
    notes_dir = current_app.config.get('PREP_NOTES_DIR', 'notes')
    if not os.path.isabs(notes_dir):
        notes_dir = os.path.join(current_app.root_path, '..', notes_dir)
    notes_dir = os.path.abspath(notes_dir)

    safe_path = os.path.normpath(os.path.join(notes_dir, filename))
    if not safe_path.startswith(notes_dir) or not os.path.isfile(safe_path):
        return jsonify({'error': 'File not found'}), 404

    with open(safe_path, 'r', encoding='utf-8') as f:
        markdown_text = f.read()

    # ── Build system message with prompt caching ──
    system_content = [
        {
            "type": "text",
            "text": (
                "You are a concise study assistant. The user is reading a document "
                "and will ask questions about it. Answer clearly and concisely using "
                "the document as your primary source. Use markdown formatting in your "
                "responses. Keep answers focused and to the point."
            ),
        },
        {
            "type": "text",
            "text": f"DOCUMENT:\n\n{markdown_text}",
            "cache_control": {"type": "ephemeral"},
        },
    ]

    if section:
        system_content.append({
            "type": "text",
            "text": (
                f"FOCUSED SECTION — The user has selected this specific section. "
                f"Answer questions primarily based on this excerpt:\n\n{section}"
            ),
        })

    # ── Build conversation messages ──
    chat_messages = []
    for msg in history[-10:]:  # Cap history to last 10 messages
        if msg.get('role') in ('user', 'assistant') and msg.get('content'):
            chat_messages.append({
                'role': msg['role'],
                'content': msg['content'],
            })
    chat_messages.append({'role': 'user', 'content': user_message})

    # ── Stream response via SSE ──
    model = current_app.config.get('ANTHROPIC_MODEL', 'claude-sonnet-4-6')
    client = anthropic.Anthropic(api_key=api_key)

    def generate():
        try:
            with client.messages.stream(
                model=model,
                max_tokens=2048,
                system=system_content,
                messages=chat_messages,
            ) as stream:
                for text in stream.text_stream:
                    yield f"data: {_json.dumps({'text': text})}\n\n"

            # Final event with usage info
            resp = stream.get_final_message()
            usage_info = {}
            if resp and resp.usage:
                usage_info = {
                    'input_tokens': resp.usage.input_tokens,
                    'output_tokens': resp.usage.output_tokens,
                }
                if hasattr(resp.usage, 'cache_read_input_tokens'):
                    usage_info['cache_read_tokens'] = resp.usage.cache_read_input_tokens
                if hasattr(resp.usage, 'cache_creation_input_tokens'):
                    usage_info['cache_creation_tokens'] = resp.usage.cache_creation_input_tokens
            yield f"data: {_json.dumps({'done': True, 'usage': usage_info})}\n\n"
        except anthropic.APIError as e:
            logger.error(f"Anthropic API error in prep chat: {e}")
            yield f"data: {_json.dumps({'error': str(e)})}\n\n"
        except Exception as e:
            logger.error(f"Prep chat error: {e}")
            yield f"data: {_json.dumps({'error': 'An error occurred'})}\n\n"

    from flask import Response
    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
        },
    )


@views_bp.route('/calendar')
def calendar_page():
    """Shared calendar page."""
    return render_template(
        'pages/calendar.html',
        active_tab='calendar',
        brief=_get_brief(),
    )


@views_bp.route('/api/calendar/events')
def calendar_events():
    """List calendar events for a date range.

    Query params:
      start  – YYYY-MM-DD  (required)
      end    – YYYY-MM-DD  (required)
      month  – YYYY-MM     (legacy shorthand; start/end take precedence)
    """
    start_str = request.args.get('start')
    end_str = request.args.get('end')
    if start_str and end_str:
        try:
            range_start = date.fromisoformat(start_str)
            range_end = date.fromisoformat(end_str)
        except ValueError:
            range_start = date.today().replace(day=1)
            range_end = range_start + timedelta(days=31)
    else:
        month_str = request.args.get('month')
        if month_str:
            try:
                year, month = month_str.split('-')
                range_start = date(int(year), int(month), 1)
            except (ValueError, TypeError):
                range_start = date.today().replace(day=1)
        else:
            range_start = date.today().replace(day=1)
        if range_start.month == 12:
            range_end = date(range_start.year + 1, 1, 1) - timedelta(days=1)
        else:
            range_end = date(range_start.year, range_start.month + 1, 1) - timedelta(days=1)

    # Fetch all events that *could* have occurrences in this range:
    # non-recurring in range  OR  recurring that started on or before range_end
    candidates = CalendarEvent.query.filter(
        db.or_(
            db.and_(CalendarEvent.recurrence.is_(None), CalendarEvent.event_date >= range_start, CalendarEvent.event_date <= range_end),
            db.and_(CalendarEvent.recurrence.isnot(None), CalendarEvent.event_date <= range_end),
        )
    ).all()

    results = []
    for evt in candidates:
        for occ_date in evt.occurrences_in_range(range_start, range_end):
            results.append(evt.to_dict(override_date=occ_date))
    results.sort(key=lambda e: (e['event_date'], e['event_time'] or ''))
    return jsonify(results)


@views_bp.route('/api/calendar/events', methods=['POST'])
def calendar_event_create():
    """Create a calendar event."""
    data = request.get_json(force=True)
    if not data.get('title') or not data.get('event_date'):
        return jsonify({'error': 'title and event_date required'}), 400

    from datetime import time as time_type
    evt = CalendarEvent(
        title=data['title'],
        event_date=date.fromisoformat(data['event_date']),
        event_time=(
            time_type.fromisoformat(data['event_time'])
            if data.get('event_time') else None
        ),
        end_time=(
            time_type.fromisoformat(data['end_time'])
            if data.get('end_time') else None
        ),
        description=data.get('description', ''),
        color=data.get('color', '#6366f1'),
        recurrence=data.get('recurrence') or None,
        recurrence_end=(
            date.fromisoformat(data['recurrence_end'])
            if data.get('recurrence_end') else None
        ),
    )
    db.session.add(evt)
    db.session.commit()
    return jsonify(evt.to_dict()), 201


@views_bp.route('/api/calendar/events/<int:event_id>', methods=['PUT'])
def calendar_event_update(event_id):
    """Update a calendar event."""
    evt = CalendarEvent.query.get_or_404(event_id)
    data = request.get_json(force=True)

    from datetime import time as time_type
    if 'title' in data:
        evt.title = data['title']
    if 'event_date' in data:
        evt.event_date = date.fromisoformat(data['event_date'])
    if 'event_time' in data:
        evt.event_time = (
            time_type.fromisoformat(data['event_time'])
            if data['event_time'] else None
        )
    if 'end_time' in data:
        evt.end_time = (
            time_type.fromisoformat(data['end_time'])
            if data['end_time'] else None
        )
    if 'description' in data:
        evt.description = data['description']
    if 'color' in data:
        evt.color = data['color']
    if 'recurrence' in data:
        evt.recurrence = data['recurrence'] or None
    if 'recurrence_end' in data:
        evt.recurrence_end = (
            date.fromisoformat(data['recurrence_end'])
            if data['recurrence_end'] else None
        )

    db.session.commit()
    return jsonify(evt.to_dict())


@views_bp.route('/api/calendar/events/<int:event_id>', methods=['DELETE'])
def calendar_event_delete(event_id):
    """Delete a calendar event."""
    evt = CalendarEvent.query.get_or_404(event_id)
    db.session.delete(evt)
    db.session.commit()
    return jsonify({'status': 'ok', 'id': event_id})


@views_bp.route('/api/calendar/seed', methods=['POST'])
def calendar_seed():
    """Seed calendar with earnings, festivals, ekadashis, and SD events."""
    from app.commands import (
        SEED_TAG, _earnings_events, _ekadashi_events,
        _indian_festival_events, _sd_farmers_market_events, _sd_events,
    )
    # Remove previously seeded
    deleted = CalendarEvent.query.filter(
        CalendarEvent.description.contains(SEED_TAG)
    ).delete(synchronize_session='fetch')

    categories = [
        ('Earnings Reports',   _earnings_events),
        ('Ekadashi',           _ekadashi_events),
        ('Indian Festivals',   _indian_festival_events),
        ('SD Farmers Markets', _sd_farmers_market_events),
        ('SD Events',          _sd_events),
    ]
    total = 0
    breakdown = {}
    for label, builder in categories:
        events = builder()
        db.session.add_all(events)
        breakdown[label] = len(events)
        total += len(events)

    db.session.commit()
    return jsonify({
        'status': 'ok',
        'deleted': deleted,
        'seeded': total,
        'breakdown': breakdown,
    })


@views_bp.route('/voice')
def voice_page():
    """Voice coach — sargam keyboard and pitch detector."""
    return render_template(
        'pages/voice.html',
        active_tab='voice',
        brief=_get_brief(),
    )


@views_bp.route('/reader')
def reader_page():
    """PDF reader with highlighting and fullscreen support."""
    import os
    pdfs_dir = os.path.join(current_app.root_path, '..', 'pdfs')
    pdfs_dir = os.path.abspath(pdfs_dir)
    pdfs = []
    if os.path.isdir(pdfs_dir):
        for f in sorted(os.listdir(pdfs_dir)):
            if f.lower().endswith('.pdf'):
                display = f.replace('.pdf', '').replace('-', ' ').replace('_', ' ')
                pdfs.append({'filename': f, 'display_name': display.title()})
    return render_template(
        'pages/reader.html',
        active_tab='reader',
        brief=_get_brief(),
        pdfs=pdfs,
    )


@views_bp.route('/api/reader/file/<path:filename>')
def reader_file(filename):
    """Serve a PDF file from the pdfs directory with range request support."""
    import os
    from flask import send_file
    pdfs_dir = os.path.join(current_app.root_path, '..', 'pdfs')
    pdfs_dir = os.path.abspath(pdfs_dir)
    safe_path = os.path.normpath(os.path.join(pdfs_dir, filename))
    if not safe_path.startswith(pdfs_dir):
        abort(403)
    if not os.path.isfile(safe_path):
        abort(404)
    return send_file(safe_path, conditional=True)


@views_bp.route('/api/reader/gdrive/<file_id>')
def reader_gdrive(file_id):
    """Proxy-download a PDF from Google Drive and cache it locally.

    Streams the file back with Content-Length so the client can show
    download progress.
    """
    import os
    import re
    import requests as _req
    from flask import send_file, Response

    # Validate file_id (alphanumeric + dash/underscore only)
    if not re.match(r'^[\w-]+$', file_id):
        abort(400)

    cache_dir = os.path.abspath(
        os.path.join(current_app.root_path, '..', 'pdfs', '.gdrive_cache'))
    os.makedirs(cache_dir, exist_ok=True)
    cached = os.path.join(cache_dir, file_id + '.pdf')

    if not os.path.isfile(cached) or os.path.getsize(cached) < 10000:
        # Remove stale/invalid cache (e.g. HTML error pages)
        if os.path.isfile(cached):
            os.remove(cached)

        # Google Drive direct download URL (with confirm=t to skip virus scan)
        url = (f'https://drive.usercontent.google.com/download'
               f'?id={file_id}&export=download&confirm=t')
        session = _req.Session()
        resp = session.get(url, stream=True, timeout=(15, 300))

        if resp.status_code != 200:
            abort(502)

        # Verify we got a PDF, not an HTML error/confirmation page
        content_type = resp.headers.get('Content-Type', '')
        if 'text/html' in content_type:
            abort(502)

        # Stream from GDrive → save to cache AND stream to client
        total = resp.headers.get('Content-Length', '')

        def generate():
            with open(cached + '.tmp', 'wb') as f:
                for chunk in resp.iter_content(65536):
                    f.write(chunk)
                    yield chunk
            # Atomically move tmp → final
            os.replace(cached + '.tmp', cached)

        headers = {'Content-Type': 'application/pdf'}
        if total:
            headers['Content-Length'] = total
        return Response(generate(), headers=headers)

    return send_file(cached, mimetype='application/pdf', conditional=True)


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


@views_bp.route('/api/deep-dive/<int:cluster_id>', methods=['POST'])
def deep_dive(cluster_id):
    """Generate a deep-dive analysis of a cluster using LLM."""
    from app.integrations.llm_gateway import LLMGateway

    cluster = Cluster.query.get_or_404(cluster_id)

    # Get all articles in this cluster
    memberships = ClusterMembership.query.filter_by(cluster_id=cluster_id).all()
    article_ids = [m.article_id for m in memberships]
    articles = Article.query.filter(Article.id.in_(article_ids)).all() if article_ids else []

    # Build context from articles
    article_texts = []
    for a in articles[:5]:
        text = a.extracted_text or ''
        if len(text) > 1500:
            text = text[:1500] + '...'
        source_name = a.source.name if a.source else 'Unknown'
        article_texts.append(
            f"Source: {source_name}\nTitle: {a.title}\nURL: {a.url}\n\n{text}"
        )

    combined = '\n\n---\n\n'.join(article_texts)
    label = cluster.headline or cluster.label or 'this story'

    config = current_app.config
    llm = LLMGateway(config)

    # Try Grok first for real-time context, fall back to OpenAI
    provider = 'xai' if llm.xai_available else 'openai'

    messages = [
        {
            'role': 'system',
            'content': (
                'You are an expert news analyst providing a comprehensive deep-dive '
                'on a news story. Given multiple source articles, provide:\n\n'
                '1. **Summary**: A clear 3-4 sentence overview of what is happening\n'
                '2. **Key Facts**: Bullet points of the most important facts\n'
                '3. **Context**: Historical context and why this matters\n'
                '4. **Different Perspectives**: How different sources frame this story\n'
                '5. **What to Watch**: What developments to expect next\n'
                '6. **Sources**: Key sources and their reliability\n\n'
                'Be thorough but concise. Use markdown formatting.'
            ),
        },
        {
            'role': 'user',
            'content': f'Story: {label}\n\nArticles:\n{combined}\n\nProvide a deep-dive analysis.',
        },
    ]

    try:
        result = llm.call(
            messages=messages,
            purpose=f'deep_dive.{cluster_id}',
            section='grok_analysis' if provider == 'xai' else 'general_news_us',
            max_tokens=1000,
            provider=provider,
        )
        return jsonify({
            'content': result['content'],
            'provider': provider,
            'model': result.get('model', ''),
            'tokens': result.get('total_tokens', 0),
        })
    except Exception as e:
        logger.error(f"Deep dive failed for cluster {cluster_id}: {e}")
        return jsonify({'error': str(e)}), 500
