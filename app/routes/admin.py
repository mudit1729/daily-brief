import threading
from datetime import date
from flask import Blueprint, jsonify, request, current_app
from app.extensions import db
from app.models.source import Source
from app.models.topic import TrackedTopic
from app import feature_flags

admin_bp = Blueprint('admin', __name__)


@admin_bp.route('/sources')
def list_sources():
    """List all sources."""
    sources = Source.query.order_by(Source.section, Source.name).all()
    return jsonify([s.to_dict() for s in sources])


@admin_bp.route('/sources', methods=['POST'])
def add_source():
    """Add a new source."""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'JSON body required'}), 400

    required = ['name', 'url', 'section']
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({'error': f'Missing fields: {missing}'}), 400

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
    db.session.commit()
    return jsonify(source.to_dict()), 201


@admin_bp.route('/sources/<int:source_id>', methods=['PUT'])
def update_source(source_id):
    """Update a source."""
    source = Source.query.get_or_404(source_id)
    data = request.get_json()
    if not data:
        return jsonify({'error': 'JSON body required'}), 400

    for field in ['name', 'url', 'section', 'region', 'bias_label',
                  'trust_score', 'source_type', 'is_active', 'fetch_interval_min']:
        if field in data:
            setattr(source, field, data[field])

    db.session.commit()
    return jsonify(source.to_dict())


@admin_bp.route('/sources/<int:source_id>', methods=['DELETE'])
def delete_source(source_id):
    """Soft-delete a source (set is_active=False)."""
    source = Source.query.get_or_404(source_id)
    source.is_active = False
    db.session.commit()
    return jsonify({'status': 'deactivated', 'id': source_id})


@admin_bp.route('/watchlists')
def list_watchlists():
    """List tracked topics."""
    topics = TrackedTopic.query.order_by(TrackedTopic.name).all()
    return jsonify([t.to_dict() for t in topics])


@admin_bp.route('/watchlists', methods=['POST'])
def add_watchlist():
    """Add a tracked topic."""
    data = request.get_json()
    if not data or 'name' not in data:
        return jsonify({'error': 'JSON body with "name" required'}), 400

    topic = TrackedTopic(
        name=data['name'],
        description=data.get('description', ''),
    )
    db.session.add(topic)
    db.session.commit()
    return jsonify(topic.to_dict()), 201


@admin_bp.route('/pipeline/trigger', methods=['POST'])
def trigger_pipeline():
    """Manually trigger a pipeline run."""
    from app.pipeline.orchestrator import run_daily_pipeline

    target = date.today()
    app = current_app._get_current_object()

    def run_in_thread():
        with app.app_context():
            run_daily_pipeline(target)

    thread = threading.Thread(target=run_in_thread)
    thread.start()

    return jsonify({'status': 'triggered', 'date': target.isoformat()})


@admin_bp.route('/flags')
def list_flags():
    """List all feature flags."""
    return jsonify(feature_flags.all_flags())


@admin_bp.route('/flags/<key>', methods=['PUT'])
def toggle_flag(key):
    """Toggle a feature flag at runtime."""
    data = request.get_json()
    if data is None or 'value' not in data:
        return jsonify({'error': 'JSON body with "value" (bool) required'}), 400

    feature_flags.set_flag(key, bool(data['value']))
    return jsonify({'flag': key, 'value': feature_flags.is_enabled(key)})
