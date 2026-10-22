"""Ressources partagées : modèle d'embedding et client Qdrant (chargés une seule fois)."""
from functools import lru_cache

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

import config


@lru_cache(maxsize=1)
def get_embedder():
    return SentenceTransformer(config.EMBED_MODEL)


@lru_cache(maxsize=1)
def get_client():
    if config.QDRANT_URL:
        return QdrantClient(url=config.QDRANT_URL)
    return QdrantClient(path=config.QDRANT_PATH)

def embed_passages(texts):
    # Le modèle e5 attend le préfixe "passage: " pour les documents
    return get_embedder().encode(
        [f"passage: {t}" for t in texts],
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def embed_query(text):
    # ... et "query: " pour les questions
    return get_embedder().encode(f"query: {text}", normalize_embeddings=True).tolist()
