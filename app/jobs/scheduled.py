import logging
from datetime import date

logger = logging.getLogger(__name__)


def register_jobs(scheduler, app):
    """Register all scheduled jobs."""

    @scheduler.task('cron', id='daily_pipeline', hour=5, minute=30,
                    misfire_grace_time=3600, coalesce=True, max_instances=1)
    def daily_pipeline_job():
        """Full daily pipeline run at 5:30 AM UTC."""
        with app.app_context():
            logger.info("[Job] Starting daily pipeline")
            from app.pipeline.orchestrator import run_daily_pipeline
            run_daily_pipeline(date.today())

    @scheduler.task('cron', id='rss_refresh', hour='6-22/2',
                    misfire_grace_time=1800, coalesce=True, max_instances=1)
    def rss_refresh_job():
        """Refresh RSS feeds every 2 hours (6AM-10PM UTC)."""
        with app.app_context():
            logger.info("[Job] RSS refresh")
            from app.pipeline.acquire import _fetch_all_rss
            count = _fetch_all_rss()
            logger.info(f"[Job] RSS refresh: {count} new articles")

    @scheduler.task('cron', id='market_refresh', hour='13-21', minute='0,30',
                    misfire_grace_time=900, coalesce=True, max_instances=1)
    def market_refresh_job():
        """Refresh market data every 30min during US market hours."""
        with app.app_context():
            logger.info("[Job] Market data refresh")
            from app.pipeline.acquire import _fetch_market_data
            count = _fetch_market_data(date.today())
            logger.info(f"[Job] Market refresh: {count} snapshots")

    @scheduler.task('cron', id='weather_refresh', hour='6,18',
                    misfire_grace_time=3600, coalesce=True, max_instances=1)
    def weather_refresh_job():
        """Refresh weather data twice daily."""
        with app.app_context():
            logger.info("[Job] Weather refresh")
            from app.pipeline.acquire import _fetch_weather
            count = _fetch_weather(date.today())
            logger.info(f"[Job] Weather refresh: {count} entries")

    @scheduler.task('cron', id='expire_insights', hour=0,
                    misfire_grace_time=3600, coalesce=True, max_instances=1)
    def expire_insights_job():
        """Expire old daily insights at midnight UTC."""
        with app.app_context():
            logger.info("[Job] Expiring old insights")
            from app.services.feedback_service import FeedbackService
            service = FeedbackService()
            count = service.expire_old_insights()
            logger.info(f"[Job] Expired {count} insights")

    @scheduler.task('cron', id='cost_rollup', hour=23, minute=55,
                    misfire_grace_time=3600, coalesce=True, max_instances=1)
    def cost_rollup_job():
        """Daily cost summary rollup."""
        with app.app_context():
            logger.info("[Job] Cost rollup")
            from app.services.cost_service import CostService
            from flask import current_app
            service = CostService()
            budget = current_app.config.get('LLM_DAILY_BUDGET_USD', 1.00)
            service.create_daily_summary(date.today(), budget)
            logger.info("[Job] Cost rollup complete")

    logger.info("All scheduled jobs registered")
