import logging
from flask import Blueprint, jsonify, request
from app.services.feedback_service import FeedbackService

logger = logging.getLogger(__name__)

feedback_bp = Blueprint('feedback', __name__)
feedback_service = FeedbackService()


@feedback_bp.route('/action', methods=['POST'])
def submit_action():
    """Submit a feedback action (follow/mute/upvote/downvote)."""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'JSON body required'}), 400

    required = ['action_type', 'target_type', 'target_id']
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({'error': f'Missing fields: {missing}'}), 400

    try:
        action = feedback_service.record_action(
            action_type=data['action_type'],
            target_type=data['target_type'],
            target_id=data['target_id'],
            metadata=data.get('metadata'),
        )
        response_data = action.to_dict()

        # Follow a cluster → auto-create a tracked story
        if data['action_type'] == 'follow' and data['target_type'] == 'cluster':
            try:
                from app.services.story_tracker import StoryTracker
                tracker = StoryTracker()
                if tracker.is_enabled():
                    topic, story = tracker.create_story_from_cluster(int(data['target_id']))
                    if topic and story:
                        response_data['story_created'] = {
                            'topic_name': topic.name,
                            'story_title': story.title,
                        }
            except Exception as e:
                logger.error(f"Follow-to-story failed for cluster {data['target_id']}: {e}")

        return jsonify(response_data), 201
    except ValueError as e:
        return jsonify({'error': str(e)}), 400


@feedback_bp.route('/insight', methods=['POST'])
def submit_insight():
    """Submit a daily insight (10-day TTL boost)."""
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'JSON body with "text" field required'}), 400

    insight = feedback_service.record_insight(data['text'])
    return jsonify(insight.to_dict()), 201


@feedback_bp.route('/insight/<int:insight_id>/promote', methods=['POST'])
def promote_insight(insight_id):
    """Promote a daily insight to a persistent preference."""
    try:
        pref = feedback_service.promote_insight(insight_id)
        return jsonify(pref.to_dict())
    except ValueError as e:
        return jsonify({'error': str(e)}), 404


@feedback_bp.route('/insights')
def list_insights():
    """List active (non-expired) daily insights."""
    insights = feedback_service.get_active_insights()
    return jsonify([i.to_dict() for i in insights])


@feedback_bp.route('/actions')
def list_actions():
    """List recent feedback actions."""
    from app.models.user import FeedbackAction
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)

    pagination = FeedbackAction.query.order_by(
        FeedbackAction.created_at.desc()
    ).paginate(page=page, per_page=per_page, error_out=False)

    return jsonify({
        'actions': [a.to_dict() for a in pagination.items],
        'total': pagination.total,
        'page': page,
    })
