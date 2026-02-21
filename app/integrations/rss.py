import logging
from datetime import datetime, timezone
import feedparser
from time import mktime

logger = logging.getLogger(__name__)


def fetch_feed(source):
    """
    Fetch and parse an RSS feed for a given Source object.
    Returns list of dicts ready to become Article rows.
    """
    try:
        feed = feedparser.parse(source.url)
    except Exception as e:
        logger.error(f"Failed to parse feed {source.url}: {e}")
        return []

    if feed.bozo and not feed.entries:
        logger.warning(f"Malformed feed {source.url}: {feed.bozo_exception}")
        return []

    articles = []
    for entry in feed.entries:
        link = entry.get('link')
        if not link:
            continue

        published_at = None
        if hasattr(entry, 'published_parsed') and entry.published_parsed:
            try:
                published_at = datetime.fromtimestamp(
                    mktime(entry.published_parsed), tz=timezone.utc
                )
            except (ValueError, OverflowError):
                pass
        elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
            try:
                published_at = datetime.fromtimestamp(
                    mktime(entry.updated_parsed), tz=timezone.utc
                )
            except (ValueError, OverflowError):
                pass

        articles.append({
            'url': link,
            'title': entry.get('title', ''),
            'summary_hint': entry.get('summary', ''),
            'author': entry.get('author', ''),
            'published_at': published_at,
            'source_id': source.id,
        })

    logger.info(f"Fetched {len(articles)} entries from {source.name}")
    return articles
