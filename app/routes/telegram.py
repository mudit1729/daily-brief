"""
Telegram Bot webhook + command handlers.
Blueprint at /api/telegram — only registered if TELEGRAM_BOT_TOKEN is set.
"""
import logging
import threading
from datetime import date, datetime, timezone
from flask import Blueprint, jsonify, request, current_app
from app.extensions import db

logger = logging.getLogger(__name__)
telegram_bp = Blueprint('telegram', __name__)

_research_threads = {}  # chat_id -> thread, prevent duplicate research runs


def _get_bot():
    from app.integrations.telegram_bot import TelegramBot
    token = current_app.config.get('TELEGRAM_BOT_TOKEN')
    if not token:
        return None
    return TelegramBot(token)


def _is_authorized(user_id):
    allowed = current_app.config.get('TELEGRAM_ALLOWED_USERS', [])
    return not allowed or user_id in allowed


# --------------- Webhook endpoint ---------------

@telegram_bp.route('/webhook', methods=['POST'])
def webhook():
    from app.integrations.telegram_bot import TelegramBot
    data = request.get_json(silent=True) or {}
    chat_id, user_id, text, username = TelegramBot.parse_update(data)

    if not chat_id or not text:
        return jsonify({'ok': True})

    if not _is_authorized(user_id):
        bot = _get_bot()
        if bot:
            bot.send_message(chat_id, 'Unauthorized. Your user ID is not in the allowed list.')
        return jsonify({'ok': True})

    # Route to command handler
    text = text.strip()
    if text.startswith('/'):
        parts = text.split(None, 1)
        command = parts[0].lower().split('@')[0]  # strip @botname
        args = parts[1] if len(parts) > 1 else ''
        _handle_command(chat_id, command, args)
    else:
        bot = _get_bot()
        if bot:
            bot.send_message(chat_id, 'Send /help for available commands.')

    return jsonify({'ok': True})


# --------------- Webhook management ---------------

@telegram_bp.route('/setup-webhook', methods=['POST'])
def setup_webhook():
    """Admin endpoint to register webhook URL with Telegram."""
    from app.routes.admin import require_admin_key
    # Manual admin key check since we can't use decorator on this route easily
    import hmac
    configured_key = current_app.config.get('ADMIN_API_KEY')
    if not configured_key:
        return jsonify({'error': 'Admin API disabled'}), 503

    auth = request.headers.get('Authorization', '')
    token = auth.split(' ', 1)[1].strip() if auth.lower().startswith('bearer ') else ''
    if not token:
        token = (request.headers.get('X-Admin-Key') or '').strip()
    if not token or not hmac.compare_digest(token, configured_key):
        return jsonify({'error': 'Unauthorized'}), 401

    data = request.get_json(silent=True) or {}
    base_url = data.get('base_url', '').rstrip('/')
    if not base_url:
        return jsonify({'error': 'base_url required'}), 400

    bot = _get_bot()
    if not bot:
        return jsonify({'error': 'TELEGRAM_BOT_TOKEN not configured'}), 503

    webhook_url = f'{base_url}/api/telegram/webhook'
    result = bot.set_webhook(webhook_url)
    return jsonify({'webhook_url': webhook_url, 'telegram_response': result})


@telegram_bp.route('/webhook-info')
def webhook_info():
    bot = _get_bot()
    if not bot:
        return jsonify({'error': 'TELEGRAM_BOT_TOKEN not configured'}), 503
    return jsonify(bot.get_webhook_info())


# --------------- Command router ---------------

def _handle_command(chat_id, command, args):
    handlers = {
        '/start': _cmd_start,
        '/help': _cmd_help,
        '/addtopic': _cmd_addtopic,
        '/addtimeline': _cmd_addtimeline,
        '/research': _cmd_research,
        '/brief': _cmd_brief,
        '/market': _cmd_market,
        '/stories': _cmd_stories,
        '/topics': _cmd_topics,
        '/timelines': _cmd_timelines,
        '/pipeline': _cmd_pipeline,
        '/status': _cmd_status,
    }
    handler = handlers.get(command)
    if handler:
        try:
            handler(chat_id, args)
        except Exception as e:
            logger.error(f"Command {command} failed: {e}", exc_info=True)
            bot = _get_bot()
            if bot:
                bot.send_message(chat_id, f'Error: {e}')
    else:
        bot = _get_bot()
        if bot:
            bot.send_message(chat_id, f'Unknown command: {command}\nSend /help for available commands.')


# --------------- Command handlers ---------------

def _cmd_start(chat_id, args):
    bot = _get_bot()
    bot.send_message(chat_id, (
        '*Welcome to Pulse* 📡\n\n'
        'Your daily intelligence brief, right here in Telegram.\n\n'
        'Send /help for available commands.'
    ))


def _cmd_help(chat_id, args):
    bot = _get_bot()
    bot.send_message(chat_id, (
        '*Available Commands*\n\n'
        '/brief — Today\'s brief summary\n'
        '/market — Market snapshot\n'
        '/research TICKER — In-depth stock research\n'
        '/topics — List tracked topics\n'
        '/stories — Tracked stories by topic\n'
        '/timelines — Active timelines\n'
        '/addtopic name | description — Add tracked topic\n'
        '/addtimeline name | desc | entities — Add timeline\n'
        '/pipeline — Trigger pipeline run\n'
        '/status — System status\n'
    ))


def _cmd_addtopic(chat_id, args):
    """Add a tracked topic. Format: /addtopic name | description"""
    bot = _get_bot()
    if not args:
        bot.send_message(chat_id, 'Usage: `/addtopic name | description`')
        return

    from app.models.topic import TrackedTopic

    parts = [p.strip() for p in args.split('|', 1)]
    name = parts[0]
    desc = parts[1] if len(parts) > 1 else ''

    existing = TrackedTopic.query.filter_by(name=name).first()
    if existing:
        bot.send_message(chat_id, f'Topic "{name}" already exists (id={existing.id}).')
        return

    topic = TrackedTopic(name=name, description=desc)
    db.session.add(topic)
    db.session.commit()
    bot.send_message(chat_id, f'Topic created: *{name}* (id={topic.id})\nStories will auto-track on next pipeline run.')


def _cmd_addtimeline(chat_id, args):
    """Add a timeline. Format: /addtimeline name | desc | entity1, entity2"""
    bot = _get_bot()
    if not args:
        bot.send_message(chat_id, 'Usage: `/addtimeline name | description | entity1, entity2`')
        return

    from app.services.timeline_service import TimelineService

    parts = [p.strip() for p in args.split('|')]
    name = parts[0]
    desc = parts[1] if len(parts) > 1 else ''
    entities = [e.strip() for e in parts[2].split(',')] if len(parts) > 2 else []

    svc = TimelineService()
    try:
        timeline = svc.create_timeline(name, desc, entities=entities)
    except Exception as e:
        bot.send_message(chat_id, f'Error creating timeline: {e}')
        return

    msg = f'Timeline created: *{name}* (id={timeline.id})'
    if entities:
        msg += f'\nEntities: {", ".join(entities)}'
    bot.send_message(chat_id, msg)

    # Seed with LLM if entities are provided
    if entities:
        bot.send_message(chat_id, 'Seeding historical events with AI...')
        app = current_app._get_current_object()

        def seed():
            with app.app_context():
                try:
                    svc2 = TimelineService()
                    svc2.generate_timeline_with_llm(timeline.id, f'{name}: {desc}')
                    from app.integrations.telegram_bot import TelegramBot
                    tb = TelegramBot(app.config['TELEGRAM_BOT_TOKEN'])
                    count = len(timeline.events)
                    tb.send_message(chat_id, f'Timeline *{name}* seeded with events.')
                except Exception as e:
                    logger.error(f"Timeline seeding failed: {e}", exc_info=True)
                    from app.integrations.telegram_bot import TelegramBot
                    tb = TelegramBot(app.config['TELEGRAM_BOT_TOKEN'])
                    tb.send_message(chat_id, f'Timeline seeding failed: {e}')

        threading.Thread(target=seed, daemon=True).start()


def _cmd_research(chat_id, args):
    """In-depth stock research. Format: /research TICKER"""
    bot = _get_bot()
    ticker = args.strip().upper()
    if not ticker or not ticker.isalpha():
        bot.send_message(chat_id, 'Usage: `/research AAPL`')
        return

    # Prevent duplicate runs
    existing = _research_threads.get(chat_id)
    if existing and existing.is_alive():
        bot.send_message(chat_id, 'A research is already running. Please wait.')
        return

    app = current_app._get_current_object()

    def run():
        with app.app_context():
            from app.services.stock_research_service import StockResearchService
            from app.integrations.telegram_bot import TelegramBot
            tb = TelegramBot(app.config['TELEGRAM_BOT_TOKEN'])
            svc = StockResearchService(tb, app)
            svc.run_research(ticker, chat_id)

    t = threading.Thread(target=run, daemon=True)
    _research_threads[chat_id] = t
    t.start()


def _cmd_brief(chat_id, args):
    """Show today's brief summary."""
    bot = _get_bot()
    from app.models.brief import DailyBrief

    brief = DailyBrief.query.filter_by(date=date.today()).first()
    if not brief or brief.status != 'complete':
        status = brief.status if brief else 'not generated'
        bot.send_message(chat_id, f"Today's brief: {status}\nRun /pipeline to generate.")
        return

    lines = [f"*Pulse — {date.today().strftime('%b %d, %Y')}*\n"]

    for section in brief.sections:
        content = section.content_json
        if not content:
            continue

        title = section.title or section.section_type.replace('_', ' ').title()
        lines.append(f'\n*{title}*')

        clusters = content.get('clusters', [])
        for c in clusters[:3]:
            label = c.get('label', c.get('title', ''))
            summary = c.get('summary', c.get('one_liner', ''))
            if label:
                text = f'• {label}'
                if summary:
                    text += f' — {summary[:120]}'
                lines.append(text)

        # For non-cluster sections (market, etc.)
        if not clusters:
            items = content.get('items', content.get('indices', []))
            for item in items[:5]:
                if isinstance(item, dict):
                    name = item.get('name', item.get('symbol', ''))
                    val = item.get('summary', item.get('change_pct', ''))
                    if name:
                        lines.append(f'• {name}: {val}')
                elif isinstance(item, str):
                    lines.append(f'• {item}')

    bot.send_message(chat_id, '\n'.join(lines))


def _cmd_market(chat_id, args):
    """Show latest market snapshot."""
    bot = _get_bot()
    from app.models.market import MarketSnapshot

    snapshots = MarketSnapshot.query.filter_by(
        snapshot_date=date.today()
    ).order_by(MarketSnapshot.symbol).all()

    if not snapshots:
        # Try yesterday
        from datetime import timedelta
        yesterday = date.today() - timedelta(days=1)
        snapshots = MarketSnapshot.query.filter_by(
            snapshot_date=yesterday
        ).order_by(MarketSnapshot.symbol).all()

    if not snapshots:
        bot.send_message(chat_id, 'No market data available.')
        return

    lines = [f"*Market Snapshot* ({snapshots[0].snapshot_date})\n"]
    for s in snapshots:
        emoji = '🟢' if (s.change_pct or 0) >= 0 else '🔴'
        pct = f'{s.change_pct:+.2f}%' if s.change_pct else 'N/A'
        price = f'${s.price:,.2f}' if s.price else 'N/A'
        name = s.name or s.symbol
        lines.append(f'{emoji} *{s.symbol}* ({name}): {price} ({pct})')

    bot.send_message(chat_id, '\n'.join(lines))


def _cmd_stories(chat_id, args):
    """List tracked stories grouped by topic."""
    bot = _get_bot()
    from app.models.topic import TrackedTopic, Story

    topics = TrackedTopic.query.filter_by(is_active=True).all()
    if not topics:
        bot.send_message(chat_id, 'No tracked topics. Use /addtopic to create one.')
        return

    lines = ['*Tracked Stories*\n']
    for topic in topics:
        stories = Story.query.filter_by(
            topic_id=topic.id
        ).filter(Story.status.in_(['developing', 'ongoing'])).all()

        lines.append(f'*{topic.name}*')
        if stories:
            for s in stories:
                events_count = len(s.events) if s.events else 0
                lines.append(f'  • {s.title} [{s.status}] ({events_count} events)')
        else:
            lines.append('  No active stories')
        lines.append('')

    bot.send_message(chat_id, '\n'.join(lines))


def _cmd_topics(chat_id, args):
    """List tracked topics."""
    bot = _get_bot()
    from app.models.topic import TrackedTopic

    topics = TrackedTopic.query.order_by(TrackedTopic.name).all()
    if not topics:
        bot.send_message(chat_id, 'No tracked topics. Use /addtopic to create one.')
        return

    lines = ['*Tracked Topics*\n']
    for t in topics:
        status = '✅' if t.is_active else '❌'
        lines.append(f'{status} {t.name}')
        if t.description:
            lines.append(f'   {t.description[:100]}')
    bot.send_message(chat_id, '\n'.join(lines))


def _cmd_timelines(chat_id, args):
    """List active timelines."""
    bot = _get_bot()
    from app.models.timeline import Timeline

    timelines = Timeline.query.filter_by(is_active=True).order_by(Timeline.name).all()
    if not timelines:
        bot.send_message(chat_id, 'No active timelines. Use /addtimeline to create one.')
        return

    lines = ['*Active Timelines*\n']
    for tl in timelines:
        icon = tl.icon or '📅'
        count = len(tl.events) if tl.events else 0
        lines.append(f'{icon} *{tl.name}* — {count} events')
        if tl.entities_json:
            lines.append(f'   Entities: {", ".join(tl.entities_json[:5])}')
    bot.send_message(chat_id, '\n'.join(lines))


def _cmd_pipeline(chat_id, args):
    """Trigger pipeline run."""
    bot = _get_bot()
    from app.pipeline.orchestrator import run_daily_pipeline

    app = current_app._get_current_object()
    target = date.today()
    bot.send_message(chat_id, f'Triggering pipeline for {target}...')

    def run():
        with app.app_context():
            try:
                brief = run_daily_pipeline(target, force=True)
                from app.integrations.telegram_bot import TelegramBot
                tb = TelegramBot(app.config['TELEGRAM_BOT_TOKEN'])
                if brief and brief.status == 'complete':
                    tb.send_message(chat_id, f'Pipeline complete for {target}. Send /brief to view.')
                else:
                    status = brief.status if brief else 'unknown'
                    tb.send_message(chat_id, f'Pipeline finished with status: {status}')
            except Exception as e:
                logger.error(f"Pipeline trigger failed: {e}", exc_info=True)
                from app.integrations.telegram_bot import TelegramBot
                tb = TelegramBot(app.config['TELEGRAM_BOT_TOKEN'])
                tb.send_message(chat_id, f'Pipeline failed: {e}')

    threading.Thread(target=run, daemon=True).start()


def _cmd_status(chat_id, args):
    """System status."""
    bot = _get_bot()
    from app.models.brief import DailyBrief
    from app.models.source import Source
    from app.services.cost_service import CostService

    today = date.today()
    brief = DailyBrief.query.filter_by(date=today).first()
    sources_total = Source.query.count()
    sources_active = Source.query.filter_by(is_active=True).count()

    cost_svc = CostService()
    usage = cost_svc.get_daily_usage(today)
    budget = current_app.config.get('LLM_DAILY_BUDGET_USD', 1.00)

    lines = [
        '*System Status*\n',
        f"Date: {today.strftime('%b %d, %Y')}",
        f"Brief: {brief.status if brief else 'not generated'}",
        f'Sources: {sources_active}/{sources_total} active',
        f'LLM Calls: {usage["calls_count"]}',
        f'Tokens: {usage["total_tokens"]:,}',
        f'Cost: ${usage["total_cost_usd"]:.4f} / ${budget:.2f}',
    ]
    bot.send_message(chat_id, '\n'.join(lines))


# --------------- Utility: notify after pipeline ---------------

def notify_pipeline_complete(app, target_date, brief):
    """Send pipeline completion notification to all allowed Telegram users."""
    token = app.config.get('TELEGRAM_BOT_TOKEN')
    allowed = app.config.get('TELEGRAM_ALLOWED_USERS', [])
    if not token or not allowed:
        return

    from app.integrations.telegram_bot import TelegramBot
    bot = TelegramBot(token)

    status = brief.status if brief else 'unknown'
    sections = len(brief.sections) if brief and brief.sections else 0
    msg = (
        f'*Pipeline Complete* — {target_date}\n'
        f'Status: {status}\n'
        f'Sections: {sections}\n'
        f'Send /brief to view.'
    )

    for user_id in allowed:
        try:
            bot.send_message(user_id, msg)
        except Exception as e:
            logger.warning(f"Failed to notify Telegram user {user_id}: {e}")
