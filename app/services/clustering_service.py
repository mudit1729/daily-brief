import logging
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class ClusteringService:
    def __init__(self, distance_threshold=0.35, min_cluster_size=2, max_clusters=15):
        self.distance_threshold = distance_threshold
        self.min_cluster_size = min_cluster_size
        self.max_clusters = max_clusters

    def cluster_articles(self, article_ids, embeddings):
        """
        Cluster articles by embedding similarity.
        embeddings: np.ndarray of shape (N, dim)
        Returns: list of clusters, each = list of (article_id, similarity_to_centroid)
        """
        if len(article_ids) == 0:
            return []

        if len(article_ids) == 1:
            return [[(article_ids[0], 1.0)]]

        embeddings = np.array(embeddings)

        # Cosine distance matrix
        sim_matrix = cosine_similarity(embeddings)
        distance_matrix = 1.0 - sim_matrix
        np.fill_diagonal(distance_matrix, 0)
        distance_matrix = np.clip(distance_matrix, 0, 2)

        try:
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=self.distance_threshold,
                metric='precomputed',
                linkage='average',
            )
            labels = clustering.fit_predict(distance_matrix)
        except Exception as e:
            logger.warning(f"Clustering failed: {e}. Treating each article as its own cluster.")
            return [[(aid, 1.0)] for aid in article_ids]

        # Group articles by label
        groups = {}
        for idx, label in enumerate(labels):
            groups.setdefault(label, []).append(idx)

        # Sort groups by size (largest first)
        sorted_groups = sorted(groups.values(), key=len, reverse=True)

        result = []
        for indices in sorted_groups:
            if len(result) >= self.max_clusters:
                break

            # Compute centroid and per-article similarity
            cluster_embeddings = embeddings[indices]
            centroid = cluster_embeddings.mean(axis=0)
            centroid_sim = cosine_similarity(
                cluster_embeddings, centroid.reshape(1, -1)
            ).flatten()

            cluster = [
                (article_ids[indices[j]], float(centroid_sim[j]))
                for j in range(len(indices))
            ]
            result.append(cluster)

        return result
