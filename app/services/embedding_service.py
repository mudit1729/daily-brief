import logging
import numpy as np
from app.extensions import db
from app.models.embedding import ArticleEmbedding
from app.utils.hashing import simhash
from app.utils.serialization import embedding_to_bytes

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self, provider='openai', model='text-embedding-3-small'):
        self.provider = provider
        self.model = model
        self.dim = 1536

    def embed_texts(self, texts, api_key=None):
        """Batch embed texts. Returns list of numpy arrays."""
        if not texts:
            return []

        if self.provider == 'openai':
            return self._embed_openai(texts, api_key)
        else:
            raise ValueError(f"Unknown embedding provider: {self.provider}")

    def _embed_openai(self, texts, api_key=None):
        """Embed via OpenAI API."""
        import openai
        from flask import current_app

        client = openai.OpenAI(api_key=api_key or current_app.config.get('OPENAI_API_KEY'))

        # Batch in groups of 100 (API limit is 2048 but keep batches reasonable)
        all_embeddings = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            # Truncate very long texts to avoid token limits
            batch = [t[:8000] if len(t) > 8000 else t for t in batch]

            response = client.embeddings.create(
                model=self.model,
                input=batch,
            )
            for item in response.data:
                all_embeddings.append(np.array(item.embedding, dtype=np.float32))

        return all_embeddings

    def store_embedding(self, article_id, text):
        """Compute and store embedding + simhash for one article."""
        embeddings = self.embed_texts([text])
        if not embeddings:
            return None

        vec = embeddings[0]
        sh = simhash(text)
        blob = embedding_to_bytes(vec)

        emb = ArticleEmbedding(
            article_id=article_id,
            simhash=sh,
            embedding_blob=blob,
            embedding_model=self.model,
            embedding_dim=self.dim,
        )
        db.session.add(emb)
        return emb

    def store_embeddings_batch(self, article_ids, texts):
        """Compute and store embeddings for multiple articles."""
        embeddings = self.embed_texts(texts)
        results = []

        for article_id, text, vec in zip(article_ids, texts, embeddings):
            sh = simhash(text)
            blob = embedding_to_bytes(vec)

            emb = ArticleEmbedding(
                article_id=article_id,
                simhash=sh,
                embedding_blob=blob,
                embedding_model=self.model,
                embedding_dim=self.dim,
            )
            db.session.add(emb)
            results.append(emb)

        return results
