from datetime import date, datetime
from flask import Blueprint, jsonify, request
from app.models.brief import DailyBrief, BriefSection

brief_bp = Blueprint('brief', __name__)


@brief_bp.route('/today')
def get_today():
    """Get today's brief or the latest complete one."""
    brief = DailyBrief.query.filter_by(date=date.today()).first()
    if not brief:
        brief = DailyBrief.query.filter_by(status='complete').order_by(
            DailyBrief.date.desc()
        ).first()

    if not brief:
        return jsonify({'error': 'No brief available'}), 404

    return jsonify(brief.to_dict())


@brief_bp.route('/<brief_date>')
def get_by_date(brief_date):
    """Get brief for a specific date (YYYY-MM-DD)."""
    try:
        target = datetime.strptime(brief_date, '%Y-%m-%d').date()
    except ValueError:
        return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD'}), 400

    brief = DailyBrief.query.filter_by(date=target).first()
    if not brief:
        return jsonify({'error': f'No brief for {brief_date}'}), 404

    return jsonify(brief.to_dict())


@brief_bp.route('/<int:brief_id>/section/<section_type>')
def get_section(brief_id, section_type):
    """Get a specific section from a brief."""
    section = BriefSection.query.filter_by(
        brief_id=brief_id, section_type=section_type
    ).first()
    if not section:
        return jsonify({'error': 'Section not found'}), 404

    return jsonify(section.to_dict())


@brief_bp.route('/history')
def get_history():
    """List past briefs (paginated)."""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)

    pagination = DailyBrief.query.order_by(
        DailyBrief.date.desc()
    ).paginate(page=page, per_page=per_page, error_out=False)

    return jsonify({
        'briefs': [
            {
                'id': b.id,
                'date': b.date.isoformat(),
                'status': b.status,
                'total_tokens': b.total_tokens,
                'total_cost_usd': b.total_cost_usd,
                'idiot_index': b.idiot_index,
            }
            for b in pagination.items
        ],
        'total': pagination.total,
        'page': page,
        'pages': pagination.pages,
    })
