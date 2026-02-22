import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from sqlalchemy import func
from app.extensions import db
from app.models.article import Article
from app.models.source import Source
from app.integrations.extractor import extract
from app.utils.text import clean_text, word_count
from app import feature_flags

logger = logging.getLogger(__name__)

MAX_ARTICLES_PER_RUN = 500
FETCH_WORKERS = 20
MIN_PER_SECTION = 50   # guarantee every section gets at least this many slots


def _fetch_one(article_id, url):
    """Fetch and extract a single article (runs in thread pool)."""
    try:
        result = extract(url)
        return article_id, result, None
    except Exception as e:
        return article_id, None, str(e)


def _section_balanced_query(cutoff, max_total):
    """Query unextracted articles with balanced allocation across source sections.

    Without this, whichever section was ingested first fills the entire limit,
    starving other sections (e.g. market, science, health get 0 extractions).
    """
    # Find distinct source sections that have pending articles
    section_subq = (
        db.session.query(Source.section)
        .join(Article, Article.source_id == Source.id)
        .filter(Article.fetched_at >= cutoff, Article.extracted_text.is_(None))
        .distinct()
        .all()
    )
    sections = [row[0] for row in section_subq if row[0]]

    if not sections:
        return []

    # Allocate slots: guarantee MIN_PER_SECTION per section, distribute remainder
    per_section = max(max_total // len(sections), MIN_PER_SECTION)

    articles = []
    for section in sections:
        batch = (
            Article.query
            .join(Source, Article.source_id == Source.id)
            .filter(
                Article.fetched_at >= cutoff,
                Article.extracted_text.is_(None),
                Source.section == section,
            )
            .limit(per_section)
            .all()
        )
        articles.extend(batch)
        logger.debug(f"[Normalize] Section '{section}': {len(batch)} articles queued")

    # Trim to max if combined exceeds limit
    if len(articles) > max_total:
        articles = articles[:max_total]

    return articles


def run(target_date):
    """Step 2: Normalize articles - extract content, entities, metadata."""
    logger.info(f"[Normalize] Starting for {target_date}")

    # Get articles from today using section-balanced sampling
    cutoff = datetime(target_date.year, target_date.month, target_date.day, tzinfo=timezone.utc)
    articles = _section_balanced_query(cutoff, MAX_ARTICLES_PER_RUN)

    logger.info(f"[Normalize] {len(articles)} articles to process (capped at {MAX_ARTICLES_PER_RUN})")
    processed = 0
    failed = 0

    # Build lookup for articles by id
    article_map = {a.id: a for a in articles}

    # Fetch in parallel using thread pool
    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as executor:
        futures = {
            executor.submit(_fetch_one, a.id, a.url): a.id
            for a in articles
        }
        for future in as_completed(futures):
            article_id, result, error = future.result()
            article = article_map[article_id]

            if error:
                logger.error(f"Failed to normalize article {article_id} ({article.url}): {error}")
                failed += 1
                continue

            if not result:
                failed += 1
                continue

            try:
                article.extracted_text = clean_text(result.get('text', ''))
                article.word_count = word_count(article.extracted_text)

                if result.get('title') and not article.title:
                    article.title = result['title']
                if result.get('og_image_url'):
                    article.og_image_url = result['og_image_url']
                if result.get('author') and not article.author:
                    article.author = result['author']

                if feature_flags.is_enabled('store_raw_html') and result.get('raw_html'):
                    article.raw_html = result['raw_html']

                article.entities_json = _extract_entities(article.extracted_text)
                processed += 1
            except Exception as e:
                logger.error(f"Failed to normalize article {article_id}: {e}")
                failed += 1

    db.session.commit()
    logger.info(f"[Normalize] Complete: {processed} processed, {failed} failed")
    return {'processed': processed, 'failed': failed}


def _extract_entities(text):
    """Simple regex-based entity extraction. Returns list of entity dicts."""
    if not text:
        return []

    entities = []
    seen = set()

    # Extract capitalized multi-word phrases (likely proper nouns)
    # Pattern: 2+ consecutive capitalized words
    pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'
    matches = re.findall(pattern, text)
    for match in matches:
        normalized = match.strip()
        if normalized.lower() not in seen and len(normalized) > 3:
            seen.add(normalized.lower())
            entities.append({
                'name': normalized,
                'type': 'ENTITY',
                'count': text.count(normalized),
            })

    # Deduplicate and sort by frequency
    entities.sort(key=lambda e: e['count'], reverse=True)
    return entities[:20]  # Keep top 20
