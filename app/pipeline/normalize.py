import logging
import re
from datetime import datetime, timezone, timedelta
from sqlalchemy import func
from app.extensions import db
from app.models.article import Article
from app.models.source import Source
from app.integrations.extractor import extract
from app.utils.text import clean_text, word_count
from app import feature_flags

logger = logging.getLogger(__name__)

MAX_ARTICLES_PER_RUN = 200
BATCH_SIZE = 30          # commit after each batch to free memory
MIN_PER_SECTION = 30     # guarantee every section gets at least this many slots


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
    import gc
    logger.info(f"[Normalize] Starting for {target_date}")

    # Get articles from today using section-balanced sampling
    cutoff = datetime(target_date.year, target_date.month, target_date.day, tzinfo=timezone.utc)
    articles = _section_balanced_query(cutoff, MAX_ARTICLES_PER_RUN)

    logger.info(f"[Normalize] {len(articles)} articles to process (capped at {MAX_ARTICLES_PER_RUN})")
    processed = 0
    failed = 0

    # Process in batches to limit peak memory usage
    for batch_start in range(0, len(articles), BATCH_SIZE):
        batch = articles[batch_start:batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (len(articles) + BATCH_SIZE - 1) // BATCH_SIZE
        logger.info(f"[Normalize] Processing batch {batch_num}/{total_batches} ({len(batch)} articles)")

        # Process sequentially - lxml/readability are NOT thread-safe
        # Using threads causes free()/SIGABRT crashes from memory corruption
        for article in batch:
            article_id, result, error = _fetch_one(article.id, article.url)

            if error:
                logger.debug(f"Failed to normalize article {article_id}: {error}")
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

        # Commit each batch and free memory
        db.session.commit()
        gc.collect()
        logger.info(f"[Normalize] Batch {batch_num} committed: {processed} processed so far")

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
