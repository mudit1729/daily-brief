import logging
from app.extensions import db
from app.models.cluster import Cluster, ClusterMembership
from app.models.article import Article
from app.models.source import Source
from app.services.ranking_service import RankingService

logger = logging.getLogger(__name__)


def run(target_date):
    """Step 4: Rank clusters by trust, recency, diversity, and preferences."""
    logger.info(f"[Rank] Starting for {target_date}")

    clusters = Cluster.query.filter_by(date=target_date).all()
    if not clusters:
        logger.info("[Rank] No clusters to rank")
        return {'clusters_ranked': 0}

    ranking_service = RankingService()

    # Build cluster data with articles and sources
    clusters_with_data = []
    for cluster in clusters:
        memberships = ClusterMembership.query.filter_by(cluster_id=cluster.id).all()
        article_ids = [m.article_id for m in memberships]
        articles = Article.query.filter(Article.id.in_(article_ids)).all()
        source_ids = list(set(a.source_id for a in articles))
        sources = Source.query.filter(Source.id.in_(source_ids)).all()

        clusters_with_data.append({
            'cluster': cluster,
            'articles': articles,
            'sources': sources,
        })

    ranking_service.rank_clusters(clusters_with_data)
    db.session.commit()

    logger.info(f"[Rank] Complete: {len(clusters)} clusters ranked")
    return {'clusters_ranked': len(clusters)}
