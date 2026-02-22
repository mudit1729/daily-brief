import logging
from datetime import date
from sqlalchemy.exc import IntegrityError
from app.extensions import db
from app.models.brief import DailyBrief
from app.pipeline import acquire, normalize, compress, rank, synthesize
from app.services.cost_service import CostService
from flask import current_app

logger = logging.getLogger(__name__)


def _claim_pipeline_brief(target_date, force=False):
    """
    Claim a brief for execution using status transitions.
    Returns DailyBrief if this worker should run pipeline, else None.
    """
    brief = DailyBrief.query.filter_by(date=target_date).first()
    if brief and brief.status == 'complete':
        if force:
            logger.info(f"Force re-running pipeline for {target_date}")
            brief.status = 'running'
            db.session.commit()
            return brief
        logger.info(f"Brief for {target_date} already complete, skipping")
        return None

    if not brief:
        brief = DailyBrief(date=target_date, status='running')
        db.session.add(brief)
        try:
            db.session.commit()
            return brief
        except IntegrityError:
            db.session.rollback()
            brief = DailyBrief.query.filter_by(date=target_date).first()

    if brief.status in ('running', 'generating'):
        logger.info(f"Pipeline already running for {target_date}, skipping duplicate run")
        return None

    claimed = DailyBrief.query.filter(
        DailyBrief.id == brief.id,
        DailyBrief.status.in_(('pending', 'failed')),
    ).update({'status': 'running'}, synchronize_session=False)
    db.session.commit()

    if claimed == 0:
        latest = DailyBrief.query.filter_by(date=target_date).first()
        logger.info(
            f"Could not claim brief for {target_date}; current status is "
            f"{latest.status if latest else 'missing'}"
        )
        return None

    return DailyBrief.query.filter_by(id=brief.id).first()


def run_daily_pipeline(target_date=None, force=False):
    """
    Master pipeline runner. Executes steps 1-5 sequentially.
    Creates DailyBrief, handles partial failures, computes final metrics.
    """
    target_date = target_date or date.today()
    logger.info(f"=== Pipeline starting for {target_date} ===")

    brief = _claim_pipeline_brief(target_date, force=force)
    if not brief:
        return DailyBrief.query.filter_by(date=target_date).first()

    results = {}

    # Step 1: Acquire
    try:
        logger.info("--- Step 1: Acquire ---")
        results['acquire'] = acquire.run(target_date)
        logger.info(f"--- Step 1 Result: {results['acquire']} ---")
    except Exception as e:
        logger.error(f"Step 1 (Acquire) failed: {e}", exc_info=True)
        db.session.rollback()
        results['acquire'] = {'error': str(e)}

    # Step 2: Normalize
    try:
        logger.info("--- Step 2: Normalize ---")
        results['normalize'] = normalize.run(target_date)
        logger.info(f"--- Step 2 Result: {results['normalize']} ---")
    except Exception as e:
        logger.error(f"Step 2 (Normalize) failed: {e}", exc_info=True)
        db.session.rollback()
        results['normalize'] = {'error': str(e)}

    # Step 3: Compress
    try:
        logger.info("--- Step 3: Compress ---")
        results['compress'] = compress.run(target_date)
        logger.info(f"--- Step 3 Result: {results['compress']} ---")
    except Exception as e:
        logger.error(f"Step 3 (Compress) failed: {e}", exc_info=True)
        db.session.rollback()
        results['compress'] = {'error': str(e)}

    # Step 4: Rank
    try:
        logger.info("--- Step 4: Rank ---")
        results['rank'] = rank.run(target_date)
        logger.info(f"--- Step 4 Result: {results['rank']} ---")
    except Exception as e:
        logger.error(f"Step 4 (Rank) failed: {e}", exc_info=True)
        db.session.rollback()
        results['rank'] = {'error': str(e)}

    # Step 5: Synthesize
    try:
        logger.info("--- Step 5: Synthesize ---")
        results['synthesize'] = synthesize.run(target_date, brief.id)
        logger.info(f"--- Step 5 Result: {results['synthesize']} ---")
    except Exception as e:
        logger.error(f"Step 5 (Synthesize) failed: {e}", exc_info=True)
        db.session.rollback()
        results['synthesize'] = {'error': str(e)}
        brief = DailyBrief.query.filter_by(date=target_date).first()
        if brief:
            brief.status = 'failed'
            db.session.commit()

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
    return DailyBrief.query.filter_by(date=target_date).first()
