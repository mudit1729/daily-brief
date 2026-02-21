import logging
from datetime import datetime, timezone
from sqlalchemy.exc import IntegrityError
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

    for source in sources:
        try:
            entries = fetch_feed(source)
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

            source.last_fetched_at = datetime.now(timezone.utc)
            db.session.commit()
            total_added += added
            logger.debug(f"Source {source.name}: {added} new articles from {len(entries)} entries")

        except IntegrityError:
            db.session.rollback()
            logger.warning(f"Integrity error for source {source.name}, skipping duplicates")
        except Exception as e:
            db.session.rollback()
            logger.error(f"Failed to process source {source.name}: {e}")

    return total_added


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
