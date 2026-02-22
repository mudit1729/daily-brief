import logging
from datetime import datetime, timezone, timedelta
from time import perf_counter
from sqlalchemy.exc import IntegrityError
from flask import current_app, has_app_context
from app.extensions import db
from app.models.source import Source
from app.models.article import Article
from app.models.market import MarketSnapshot
from app.models.weather import WeatherCache
from app.integrations.rss import fetch_feed
from app.integrations.market_data import MarketDataService
from app.integrations.weather import WeatherService

logger = logging.getLogger(__name__)


def run(target_date):
    """Step 1: Acquire data from all sources."""
    logger.info(f"[Acquire] Starting for {target_date}")

    articles_added = _fetch_all_rss()
    snapshots_added = _fetch_market_data(target_date)
    weather_added = _fetch_weather(target_date)

    logger.info(
        f"[Acquire] Complete: {articles_added} articles, "
        f"{snapshots_added} market snapshots, {weather_added} weather entries"
    )
    return {
        'articles_added': articles_added,
        'snapshots_added': snapshots_added,
        'weather_added': weather_added,
    }


def _fetch_all_rss():
    """Fetch RSS feeds from all active sources."""
    sources = Source.query.filter_by(is_active=True).all()
    total_added = 0
    now = datetime.now(timezone.utc)

    for source in sources:
        cooldown_until = source.auto_disabled_until
        if cooldown_until and cooldown_until.tzinfo is None:
            cooldown_until = cooldown_until.replace(tzinfo=timezone.utc)
        if cooldown_until and cooldown_until > now:
            logger.info(
                "Skipping source %s: in cooldown until %s",
                source.name,
                cooldown_until.isoformat(),
            )
            continue

        started_at = datetime.now(timezone.utc)
        t0 = perf_counter()
        try:
            entries, meta = fetch_feed(source, include_meta=True)
            latency_ms = (perf_counter() - t0) * 1000.0
            if not meta.get('ok', True):
                raise RuntimeError(meta.get('error') or 'Feed fetch failed')

            added = 0
            for entry in entries:
                # Check if article already exists by URL
                existing = Article.query.filter_by(url=entry['url']).first()
                if existing:
                    continue

                article = Article(
                    source_id=entry['source_id'],
                    url=entry['url'],
                    title=entry.get('title', ''),
                    author=entry.get('author'),
                    published_at=entry.get('published_at'),
                )
                db.session.add(article)
                added += 1

            _mark_source_fetch_success(source, started_at, latency_ms)
            db.session.commit()
            total_added += added
            logger.debug(f"Source {source.name}: {added} new articles from {len(entries)} entries")

        except IntegrityError:
            db.session.rollback()
            logger.warning(f"Integrity error for source {source.name}, skipping duplicates")
            try:
                latency_ms = (perf_counter() - t0) * 1000.0
                _mark_source_fetch_success(source, started_at, latency_ms)
                db.session.commit()
            except Exception:
                db.session.rollback()
        except Exception as e:
            db.session.rollback()
            logger.error(f"Failed to process source {source.name}: {e}")
            _mark_source_fetch_failure(source, str(e), started_at)

    return total_added


def _mark_source_fetch_success(source, started_at, latency_ms):
    source.last_fetched_at = started_at
    source.last_success_at = started_at
    source.consecutive_failures = 0
    source.consecutive_successes = (source.consecutive_successes or 0) + 1
    source.last_error = None
    source.auto_disabled_until = None

    alpha = _source_latency_alpha()
    if source.avg_latency_ms is None:
        source.avg_latency_ms = latency_ms
    else:
        source.avg_latency_ms = (source.avg_latency_ms * (1.0 - alpha)) + (latency_ms * alpha)


def _mark_source_fetch_failure(source, error, started_at):
    source.last_failure_at = started_at
    source.consecutive_successes = 0
    source.consecutive_failures = (source.consecutive_failures or 0) + 1
    source.total_failures = (source.total_failures or 0) + 1
    source.last_error = (error or 'unknown error')[:512]

    threshold = max(_source_failure_threshold(), 1)
    if source.consecutive_failures >= threshold:
        disable_minutes = max(_source_auto_disable_minutes(), 1)
        source.auto_disabled_until = started_at + timedelta(minutes=disable_minutes)
        logger.warning(
            "Source %s entered cooldown for %sm after %s consecutive failures",
            source.name,
            disable_minutes,
            source.consecutive_failures,
        )

    try:
        db.session.commit()
    except Exception:
        db.session.rollback()


def _source_failure_threshold():
    if has_app_context():
        return int(current_app.config.get('SOURCE_FAILURE_THRESHOLD', 3))
    return 3


def _source_auto_disable_minutes():
    if has_app_context():
        return int(current_app.config.get('SOURCE_AUTO_DISABLE_MINUTES', 180))
    return 180


def _source_latency_alpha():
    if has_app_context():
        raw = current_app.config.get('SOURCE_LATENCY_ALPHA', 0.3)
        try:
            value = float(raw)
            return max(0.0, min(1.0, value))
        except (TypeError, ValueError):
            pass
    return 0.3


def _fetch_market_data(target_date):
    """Fetch market snapshots."""
    try:
        service = MarketDataService()
        snapshots = service.fetch_snapshots(target_date)
        added = 0
        for snap_data in snapshots:
            # Avoid duplicates for same symbol+date
            existing = MarketSnapshot.query.filter_by(
                symbol=snap_data['symbol'],
                snapshot_date=snap_data['snapshot_date'],
            ).first()
            if existing:
                # Update price
                existing.price = snap_data['price']
                existing.change_pct = snap_data['change_pct']
                existing.change_abs = snap_data['change_abs']
                existing.volume = snap_data.get('volume')
            else:
                snapshot = MarketSnapshot(**snap_data)
                db.session.add(snapshot)
                added += 1

        db.session.commit()
        return added
    except Exception as e:
        db.session.rollback()
        logger.error(f"Market data fetch failed: {e}")
        return 0


def _fetch_weather(target_date):
    """Fetch and cache weather data."""
    try:
        service = WeatherService()
        entries = service.fetch_weather(target_date=target_date)
        added = 0
        for entry_data in entries:
            existing = WeatherCache.query.filter_by(
                location_name=entry_data['location_name'],
                date=entry_data['date'],
            ).first()
            if not existing:
                entry = WeatherCache(**entry_data)
                db.session.add(entry)
                added += 1

        db.session.commit()
        return added
    except Exception as e:
        db.session.rollback()
        logger.error(f"Weather fetch failed: {e}")
        return 0
