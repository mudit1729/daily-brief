from datetime import date, timedelta
from flask import Blueprint, jsonify, request
from app.services.cost_service import CostService
from app.models.cost import DailyCostSummary, LLMCallLog

cost_bp = Blueprint('cost', __name__)
cost_service = CostService()


@cost_bp.route('/today')
def today():
    """Get today's cost summary and idiot index."""
    usage = cost_service.get_daily_usage()
    idiot_index = cost_service.compute_idiot_index()
    return jsonify({
        **usage,
        'idiot_index': idiot_index,
        'date': date.today().isoformat(),
    })


@cost_bp.route('/dashboard')
def dashboard():
    """Historical cost data (last 30 days)."""
    days = request.args.get('days', 30, type=int)
    start_date = date.today() - timedelta(days=days)

    summaries = DailyCostSummary.query.filter(
        DailyCostSummary.date >= start_date
    ).order_by(DailyCostSummary.date.desc()).all()

    return jsonify({
        'summaries': [s.to_dict() for s in summaries],
        'period_days': days,
    })


@cost_bp.route('/logs')
def logs():
    """Raw LLM call logs (paginated, filterable)."""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 50, type=int)
    section = request.args.get('section')

    query = LLMCallLog.query
    if section:
        query = query.filter_by(section=section)

    pagination = query.order_by(
        LLMCallLog.created_at.desc()
    ).paginate(page=page, per_page=per_page, error_out=False)

    return jsonify({
        'logs': [l.to_dict() for l in pagination.items],
        'total': pagination.total,
        'page': page,
    })
