import logging
import numpy as np
from datetime import datetime, timezone
from flask import current_app
from app.extensions import db
from app.models.article import Article
from app.models.embedding import ArticleEmbedding
from app.models.cluster import Cluster, ClusterMembership
from app.models.source import Source
from app.services.embedding_service import EmbeddingService
from app.services.clustering_service import ClusteringService
from app.utils.hashing import hamming_distance
from app.utils.serialization import bytes_to_embedding

logger = logging.getLogger(__name__)

DEDUP_HAMMING_THRESHOLD = 3

SECTION_MAPPING = {
    'ai_news': {'source_sections': ['ai_news']},
    'market': {'source_sections': ['market']},
    'science': {'source_sections': ['science']},
    'health': {'source_sections': ['health']},
    # general_news split by region for proper segregation
    'general_news_us': {'source_sections': ['general_news'], 'region': 'us'},
    'general_news_india': {'source_sections': ['general_news'], 'region': 'india'},
    'general_news_geopolitics': {'source_sections': ['general_news'], 'region': 'global'},
    'feel_good': {'source_sections': ['feel_good']},
}


def run(target_date):
    """Step 3: Compute embeddings, dedup, and cluster articles."""
    logger.info(f"[Compress] Starting for {target_date}")

    config = current_app.config
    embedding_service = EmbeddingService(
        provider=config.get('EMBEDDING_PROVIDER', 'openai'),
        model=config.get('EMBEDDING_MODEL', 'text-embedding-3-small'),
    )
    clustering_service = ClusteringService()

    # Get today's normalized articles without embeddings
    cutoff = datetime(target_date.year, target_date.month, target_date.day, tzinfo=timezone.utc)
    articles = Article.query.filter(
        Article.fetched_at >= cutoff,
        Article.extracted_text.isnot(None),
        Article.extracted_text != '',
        Article.is_duplicate == False,
    ).all()

    # Filter out articles that already have embeddings
    articles_needing_embeddings = [
        a for a in articles
        if not ArticleEmbedding.query.filter_by(article_id=a.id).first()
    ]

    # Also filter out empty/whitespace-only extracted_text
    articles_needing_embeddings = [
        a for a in articles_needing_embeddings
        if a.extracted_text and a.extracted_text.strip()
    ]

    # Compute embeddings in batch
    if articles_needing_embeddings:
        logger.info(f"[Compress] Computing embeddings for {len(articles_needing_embeddings)} articles")
        texts = [a.extracted_text[:8000] for a in articles_needing_embeddings]
        article_ids = [a.id for a in articles_needing_embeddings]
        embedding_service.store_embeddings_batch(article_ids, texts)
        db.session.commit()

    # Dedup via SimHash
    _dedup_articles(cutoff)

    # Idempotent rerun safety: replace all clusters for this date.
    _delete_clusters_for_date(target_date)

    # Cluster per section (with region filtering for general_news)
    total_clusters = 0
    for section_key, section_config in SECTION_MAPPING.items():
        source_sections = section_config['source_sections']
        region = section_config.get('region')

        query = Source.query.filter(
            Source.section.in_(source_sections),
            Source.is_active == True,
        )
        if region:
            query = query.filter(Source.region == region)

        source_ids = [s.id for s in query.all()]
        if not source_ids:
            continue

        section_articles = Article.query.filter(
            Article.fetched_at >= cutoff,
            Article.source_id.in_(source_ids),
            Article.is_duplicate == False,
            Article.extracted_text.isnot(None),
        ).all()

        if not section_articles:
            continue

        clusters_created = _cluster_section(section_key, section_articles, target_date, clustering_service)
        total_clusters += clusters_created

    logger.info(f"[Compress] Complete: {total_clusters} clusters created")
    return {'clusters_created': total_clusters}


def _delete_clusters_for_date(target_date):
    """Delete existing clusters for date before re-clustering."""
    existing = Cluster.query.filter_by(date=target_date).all()
    if not existing:
        return

    for cluster in existing:
        db.session.delete(cluster)

    db.session.commit()
    logger.info(f"[Compress] Cleared {len(existing)} existing clusters for idempotent rerun")


def _dedup_articles(cutoff):
    """Flag near-duplicate articles using SimHash hamming distance."""
    embeddings = ArticleEmbedding.query.join(Article).filter(
        Article.fetched_at >= cutoff,
        Article.is_duplicate == False,
    ).all()

    dedup_count = 0
    checked = set()

    for i, emb_a in enumerate(embeddings):
        if emb_a.article_id in checked:
            continue
        for j in range(i + 1, len(embeddings)):
            emb_b = embeddings[j]
            if emb_b.article_id in checked:
                continue

            dist = hamming_distance(emb_a.simhash, emb_b.simhash)
            if dist <= DEDUP_HAMMING_THRESHOLD:
                # Mark the newer article as duplicate
                article_b = Article.query.get(emb_b.article_id)
                if article_b:
                    article_b.is_duplicate = True
                    article_b.duplicate_of_id = emb_a.article_id
                    checked.add(emb_b.article_id)
                    dedup_count += 1

    db.session.commit()
    if dedup_count:
        logger.info(f"[Compress] Flagged {dedup_count} duplicates")


def _cluster_section(section, articles, target_date, clustering_service):
    """Cluster articles within a section."""
    article_ids = []
    embeddings_list = []

    for article in articles:
        emb = ArticleEmbedding.query.filter_by(article_id=article.id).first()
        if emb and emb.embedding_blob:
            vec = bytes_to_embedding(emb.embedding_blob, emb.embedding_dim)
            article_ids.append(article.id)
            embeddings_list.append(vec)

    if not article_ids:
        return 0

    embeddings_array = np.array(embeddings_list)
    cluster_results = clustering_service.cluster_articles(article_ids, embeddings_array)

    clusters_created = 0
    for cluster_members in cluster_results:
        # Find representative article (highest similarity to centroid)
        cluster_members.sort(key=lambda x: x[1], reverse=True)
        rep_id = cluster_members[0][0]

        # Get source trust scores for avg
        member_articles = Article.query.filter(
            Article.id.in_([m[0] for m in cluster_members])
        ).all()
        trust_scores = []
        for a in member_articles:
            src = Source.query.get(a.source_id)
            if src:
                trust_scores.append(src.trust_score)

        cluster = Cluster(
            section=section,
            representative_article_id=rep_id,
            article_count=len(cluster_members),
            avg_trust_score=sum(trust_scores) / len(trust_scores) if trust_scores else 50,
            date=target_date,
        )
        db.session.add(cluster)
        db.session.flush()

        for article_id, similarity in cluster_members:
            membership = ClusterMembership(
                cluster_id=cluster.id,
                article_id=article_id,
                similarity=similarity,
            )
            db.session.add(membership)

        clusters_created += 1

    db.session.commit()
    logger.info(f"[Compress] Section '{section}': {clusters_created} clusters from {len(article_ids)} articles")
    return clusters_created
