import threading
import hmac
from datetime import date
from functools import wraps
from flask import Blueprint, jsonify, request, current_app
from sqlalchemy.exc import IntegrityError
from app.extensions import db
from app.models.source import Source
from app.models.topic import TrackedTopic
from app import feature_flags

admin_bp = Blueprint('admin', __name__)
_pipeline_thread = None
_pipeline_trigger_lock = threading.Lock()


def _extract_admin_token():
    auth_header = request.headers.get('Authorization', '')
    if auth_header.lower().startswith('bearer '):
        return auth_header.split(' ', 1)[1].strip()
    return (request.headers.get('X-Admin-Key') or '').strip()


def require_admin_key(func):
    """Require ADMIN_API_KEY for all admin endpoints."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        configured_key = current_app.config.get('ADMIN_API_KEY')
        if not configured_key:
            return jsonify({'error': 'Admin API is disabled: ADMIN_API_KEY is not configured'}), 503

        presented_key = _extract_admin_token()
        if not presented_key or not hmac.compare_digest(presented_key, configured_key):
            return jsonify({'error': 'Unauthorized'}), 401

        return func(*args, **kwargs)

    return wrapper


@admin_bp.route('/sources')
@require_admin_key
def list_sources():
    """List all sources."""
    sources = Source.query.order_by(Source.section, Source.name).all()
    return jsonify([s.to_dict() for s in sources])


@admin_bp.route('/sources', methods=['POST'])
@require_admin_key
def add_source():
    """Add a new source."""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'JSON body required'}), 400

    required = ['name', 'url', 'section']
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({'error': f'Missing fields: {missing}'}), 400

    existing = Source.query.filter_by(url=data['url']).first()
    if existing:
        return jsonify({'error': 'Source with this URL already exists', 'id': existing.id}), 409

    source = Source(
        name=data['name'],
        url=data['url'],
        section=data['section'],
        region=data.get('region'),
        bias_label=data.get('bias_label', 'center'),
        trust_score=data.get('trust_score', 50),
        source_type=data.get('source_type', 'reporting'),
        feed_type=data.get('feed_type', 'rss'),
        fetch_interval_min=data.get('fetch_interval_min', 60),
    )
    db.session.add(source)
    try:
        db.session.commit()
    except IntegrityError:
        db.session.rollback()
        return jsonify({'error': 'Source with this URL already exists'}), 409

    return jsonify(source.to_dict()), 201


@admin_bp.route('/sources/<int:source_id>', methods=['PUT'])
@require_admin_key
def update_source(source_id):
    """Update a source."""
    source = Source.query.get_or_404(source_id)
    data = request.get_json()
    if not data:
        return jsonify({'error': 'JSON body required'}), 400

    if 'url' in data and data['url'] != source.url:
        existing = Source.query.filter_by(url=data['url']).first()
        if existing:
            return jsonify({'error': 'Source with this URL already exists', 'id': existing.id}), 409

    for field in ['name', 'url', 'section', 'region', 'bias_label',
                  'trust_score', 'source_type', 'is_active', 'fetch_interval_min']:
        if field in data:
            setattr(source, field, data[field])

    try:
        db.session.commit()
    except IntegrityError:
        db.session.rollback()
        return jsonify({'error': 'Source with this URL already exists'}), 409

    return jsonify(source.to_dict())


@admin_bp.route('/sources/<int:source_id>', methods=['DELETE'])
@require_admin_key
def delete_source(source_id):
    """Soft-delete a source (set is_active=False)."""
    source = Source.query.get_or_404(source_id)
    source.is_active = False
    db.session.commit()
    return jsonify({'status': 'deactivated', 'id': source_id})


@admin_bp.route('/watchlists')
@require_admin_key
def list_watchlists():
    """List tracked topics."""
    topics = TrackedTopic.query.order_by(TrackedTopic.name).all()
    return jsonify([t.to_dict() for t in topics])


@admin_bp.route('/watchlists', methods=['POST'])
@require_admin_key
def add_watchlist():
    """Add a tracked topic."""
    data = request.get_json()
    if not data or 'name' not in data:
        return jsonify({'error': 'JSON body with "name" required'}), 400

    existing = TrackedTopic.query.filter_by(name=data['name']).first()
    if existing:
        return jsonify({'error': 'Watchlist with this name already exists', 'id': existing.id}), 409

    topic = TrackedTopic(
        name=data['name'],
        description=data.get('description', ''),
    )
    db.session.add(topic)
    try:
        db.session.commit()
    except IntegrityError:
        db.session.rollback()
        return jsonify({'error': 'Watchlist with this name already exists'}), 409

    return jsonify(topic.to_dict()), 201


@admin_bp.route('/pipeline/trigger', methods=['POST'])
@require_admin_key
def trigger_pipeline():
    """Manually trigger a pipeline run."""
    from app.pipeline.orchestrator import run_daily_pipeline

    global _pipeline_thread

    target = date.today()
    app = current_app._get_current_object()

    def run_in_thread():
        with app.app_context():
            run_daily_pipeline(target)

    with _pipeline_trigger_lock:
        if _pipeline_thread and _pipeline_thread.is_alive():
            return jsonify({'error': 'Pipeline already running'}), 409

        _pipeline_thread = threading.Thread(target=run_in_thread, daemon=True)
        _pipeline_thread.start()

    return jsonify({'status': 'triggered', 'date': target.isoformat()})


@admin_bp.route('/flags')
@require_admin_key
def list_flags():
    """List all feature flags."""
    return jsonify(feature_flags.all_flags())


@admin_bp.route('/flags/<key>', methods=['PUT'])
@require_admin_key
def toggle_flag(key):
    """Toggle a feature flag at runtime."""
    data = request.get_json()
    if data is None or 'value' not in data:
        return jsonify({'error': 'JSON body with "value" (bool) required'}), 400

    if not isinstance(data['value'], bool):
        return jsonify({'error': '"value" must be a boolean'}), 400

    feature_flags.set_flag(key, data['value'])
    return jsonify({'flag': key, 'value': feature_flags.is_enabled(key)})
