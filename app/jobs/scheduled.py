import logging
from datetime import date
from app.services.scheduler_config_service import SchedulerConfigService

logger = logging.getLogger(__name__)


def _daily_pipeline_job(app):
    with app.app_context():
        logger.info("[Job] Starting daily pipeline")
        from app.pipeline.orchestrator import run_daily_pipeline
        run_daily_pipeline(date.today())


def _rss_refresh_job(app):
    with app.app_context():
        logger.info("[Job] RSS refresh")
        from app.pipeline.acquire import _fetch_all_rss
        count = _fetch_all_rss()
        logger.info(f"[Job] RSS refresh: {count} new articles")


def _market_refresh_job(app):
    with app.app_context():
        logger.info("[Job] Market data refresh")
        from app.pipeline.acquire import _fetch_market_data
        count = _fetch_market_data(date.today())
        logger.info(f"[Job] Market refresh: {count} snapshots")


def _weather_refresh_job(app):
    with app.app_context():
        logger.info("[Job] Weather refresh")
        from app.pipeline.acquire import _fetch_weather
        count = _fetch_weather(date.today())
        logger.info(f"[Job] Weather refresh: {count} entries")


def _expire_insights_job(app):
    with app.app_context():
        logger.info("[Job] Expiring old insights")
        from app.services.feedback_service import FeedbackService
        service = FeedbackService()
        count = service.expire_old_insights()
        logger.info(f"[Job] Expired {count} insights")


def _cost_rollup_job(app):
    with app.app_context():
        logger.info("[Job] Cost rollup")
        from app.services.cost_service import CostService
        from flask import current_app
        service = CostService()
        budget = current_app.config.get('LLM_DAILY_BUDGET_USD', 1.00)
        service.create_daily_summary(date.today(), budget)
        logger.info("[Job] Cost rollup complete")


def _upsert_job(scheduler, **kwargs):
    scheduler.add_job(replace_existing=True, **kwargs)


def refresh_daily_pipeline_job(scheduler, app, schedule=None):
    """(Re)register daily pipeline job using persisted schedule config."""
    try:
        config = schedule or SchedulerConfigService().get_pipeline_schedule()
    except Exception as e:
        logger.error("Failed to load pipeline schedule config: %s", e)
        return None

    if not config.get('enabled', True):
        try:
            scheduler.remove_job('daily_pipeline')
        except Exception:
            pass
        logger.info("Daily pipeline schedule disabled")
        return None

    try:
        _upsert_job(
            scheduler,
            id='daily_pipeline',
            func=_daily_pipeline_job,
            trigger='cron',
            args=[app],
            hour=config['hour'],
            minute=config['minute'],
            timezone=config['timezone'],
            misfire_grace_time=3600,
            coalesce=True,
            max_instances=1,
        )
        job = scheduler.get_job('daily_pipeline')
        logger.info(
            "Daily pipeline scheduled at %02d:%02d %s",
            config['hour'],
            config['minute'],
            config['timezone'],
        )
        return job
    except Exception as e:
        logger.error("Failed to register daily pipeline schedule: %s", e)
        return None


def register_jobs(scheduler, app):
    """Register all scheduled jobs."""
    refresh_daily_pipeline_job(scheduler, app)

    _upsert_job(
        scheduler,
        id='rss_refresh',
        func=_rss_refresh_job,
        trigger='cron',
        args=[app],
        hour='6-22/2',
        misfire_grace_time=1800,
        coalesce=True,
        max_instances=1,
    )
    _upsert_job(
        scheduler,
        id='market_refresh',
        func=_market_refresh_job,
        trigger='cron',
        args=[app],
        hour='13-21',
        minute='0,30',
        misfire_grace_time=900,
        coalesce=True,
        max_instances=1,
    )
    _upsert_job(
        scheduler,
        id='weather_refresh',
        func=_weather_refresh_job,
        trigger='cron',
        args=[app],
        hour='6,18',
        misfire_grace_time=3600,
        coalesce=True,
        max_instances=1,
    )
    _upsert_job(
        scheduler,
        id='expire_insights',
        func=_expire_insights_job,
        trigger='cron',
        args=[app],
        hour=0,
        misfire_grace_time=3600,
        coalesce=True,
        max_instances=1,
    )
    _upsert_job(
        scheduler,
        id='cost_rollup',
        func=_cost_rollup_job,
        trigger='cron',
        args=[app],
        hour=23,
        minute=55,
        misfire_grace_time=3600,
        coalesce=True,
        max_instances=1,
    )

    logger.info("All scheduled jobs registered")

