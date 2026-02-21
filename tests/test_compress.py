import pytest
import numpy as np
from app.utils.hashing import simhash, hamming_distance
from app.utils.serialization import embedding_to_bytes, bytes_to_embedding
from app.services.clustering_service import ClusteringService


class TestSimHash:
    def test_identical_text_same_hash(self):
        """Identical texts should produce the same SimHash."""
        text = "The quick brown fox jumps over the lazy dog"
        h1 = simhash(text)
        h2 = simhash(text)
        assert h1 == h2

    def test_similar_text_low_distance(self):
        """Similar texts should have low hamming distance."""
        text1 = "The quick brown fox jumps over the lazy dog near the river"
        text2 = "The quick brown fox jumps over the lazy dog by the river"
        h1 = simhash(text1)
        h2 = simhash(text2)
        dist = hamming_distance(h1, h2)
        assert dist <= 10  # Very similar texts

    def test_different_text_high_distance(self):
        """Very different texts should have higher hamming distance."""
        text1 = "The stock market rose sharply today on strong earnings reports"
        text2 = "Scientists discover new species of deep sea creatures in Pacific Ocean"
        h1 = simhash(text1)
        h2 = simhash(text2)
        dist = hamming_distance(h1, h2)
        assert dist > 5  # Different topics should diverge

    def test_empty_text(self):
        """Empty text should return 0."""
        assert simhash('') == 0
        assert simhash('  ') == 0

    def test_hamming_distance_zero(self):
        assert hamming_distance(0, 0) == 0
        assert hamming_distance(42, 42) == 0

    def test_hamming_distance_known(self):
        assert hamming_distance(0b1010, 0b1001) == 2


class TestSerialization:
    def test_roundtrip(self):
        """Embedding should survive bytes roundtrip."""
        original = np.random.randn(1536).astype(np.float32)
        blob = embedding_to_bytes(original)
        recovered = bytes_to_embedding(blob, dim=1536)
        np.testing.assert_array_almost_equal(original, recovered)

    def test_blob_size(self):
        """Blob should be exactly dim * 4 bytes (float32)."""
        vec = np.zeros(1536, dtype=np.float32)
        blob = embedding_to_bytes(vec)
        assert len(blob) == 1536 * 4


class TestClustering:
    def test_single_article(self):
        """Single article should be its own cluster."""
        service = ClusteringService()
        result = service.cluster_articles([1], np.random.randn(1, 128))
        assert len(result) == 1
        assert result[0][0][0] == 1

    def test_identical_embeddings_cluster_together(self):
        """Identical embeddings should form one cluster."""
        service = ClusteringService(distance_threshold=0.5)
        vec = np.random.randn(128).astype(np.float32)
        embeddings = np.array([vec, vec, vec])
        ids = [1, 2, 3]

        result = service.cluster_articles(ids, embeddings)
        assert len(result) == 1
        assert len(result[0]) == 3

    def test_different_embeddings_separate(self):
        """Orthogonal embeddings should form separate clusters."""
        service = ClusteringService(distance_threshold=0.3, min_cluster_size=1)
        np.random.seed(42)

        # Create clearly different embeddings
        embeddings = np.eye(5, 128, dtype=np.float32)
        ids = [1, 2, 3, 4, 5]

        result = service.cluster_articles(ids, embeddings)
        assert len(result) >= 3  # Should be mostly separate

    def test_max_clusters_cap(self):
        """Should not exceed max_clusters."""
        service = ClusteringService(distance_threshold=0.1, max_clusters=3)
        embeddings = np.eye(10, 128, dtype=np.float32)
        ids = list(range(10))

        result = service.cluster_articles(ids, embeddings)
        assert len(result) <= 3
