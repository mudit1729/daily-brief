from flask import Blueprint, jsonify
from sqlalchemy import text
from app.extensions import db

health_bp = Blueprint('health', __name__)


@health_bp.route('/health')
def health():
    return jsonify({'status': 'ok'})


@health_bp.route('/ready')
def ready():
    try:
        db.session.execute(text('SELECT 1'))
        db_ok = True
    except Exception:
        db_ok = False

    status = 'ready' if db_ok else 'not_ready'
    code = 200 if db_ok else 503
    return jsonify({'status': status, 'db': db_ok}), code
