import logging
from datetime import date, datetime, timezone
from flask import current_app
from app.extensions import db
from app.models.brief import DailyBrief
from app.pipeline import acquire, normalize, compress, rank, synthesize
from app.services.cost_service import CostService

logger = logging.getLogger(__name__)


def run_daily_pipeline(target_date=None):
    """
    Master pipeline runner. Executes steps 1-5 sequentially.
    Creates DailyBrief, handles partial failures, computes final metrics.
    """
    target_date = target_date or date.today()
    logger.info(f"=== Pipeline starting for {target_date} ===")

    # Create or get today's brief
    brief = DailyBrief.query.filter_by(date=target_date).first()
    if brief and brief.status == 'complete':
        logger.info(f"Brief for {target_date} already complete, skipping")
        return brief

    if not brief:
        brief = DailyBrief(date=target_date, status='pending')
        db.session.add(brief)
        db.session.commit()

    results = {}

    # Step 1: Acquire
    try:
        logger.info("--- Step 1: Acquire ---")
        results['acquire'] = acquire.run(target_date)
    except Exception as e:
        logger.error(f"Step 1 (Acquire) failed: {e}", exc_info=True)
        results['acquire'] = {'error': str(e)}

    # Step 2: Normalize
    try:
        logger.info("--- Step 2: Normalize ---")
        results['normalize'] = normalize.run(target_date)
    except Exception as e:
        logger.error(f"Step 2 (Normalize) failed: {e}", exc_info=True)
        results['normalize'] = {'error': str(e)}

    # Step 3: Compress
    try:
        logger.info("--- Step 3: Compress ---")
        results['compress'] = compress.run(target_date)
    except Exception as e:
        logger.error(f"Step 3 (Compress) failed: {e}", exc_info=True)
        results['compress'] = {'error': str(e)}
        db.session.rollback()

    # Step 4: Rank
    try:
        logger.info("--- Step 4: Rank ---")
        results['rank'] = rank.run(target_date)
    except Exception as e:
        logger.error(f"Step 4 (Rank) failed: {e}", exc_info=True)
        results['rank'] = {'error': str(e)}

    # Step 5: Synthesize
    try:
        logger.info("--- Step 5: Synthesize ---")
        results['synthesize'] = synthesize.run(target_date, brief.id)
    except Exception as e:
        logger.error(f"Step 5 (Synthesize) failed: {e}", exc_info=True)
        results['synthesize'] = {'error': str(e)}
        try:
            db.session.rollback()
            brief = DailyBrief.query.filter_by(date=target_date).first()
            if brief:
                brief.status = 'failed'
                db.session.commit()
        except Exception:
            db.session.rollback()

    # Final cost summary
    try:
        cost_service = CostService()
        budget_usd = current_app.config.get('LLM_DAILY_BUDGET_USD', 1.00)
        cost_service.create_daily_summary(target_date, budget_usd)
    except Exception as e:
        logger.error(f"Cost summary failed: {e}")
        db.session.rollback()

    logger.info(f"=== Pipeline complete for {target_date} ===")
    logger.info(f"Results: {results}")
    return brief
