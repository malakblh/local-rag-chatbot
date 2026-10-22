"""Indexation d'un PDF dans Qdrant.

Usage : python ingest.py data/Ai.pdf
"""
import sys
import uuid
import re
from pathlib import Path

from pypdf import PdfReader
from qdrant_client.models import (
    Distance, FieldCondition, Filter, FilterSelector, MatchValue, PointStruct, VectorParams,
)

import config
from core import embed_passages, get_client


def chunk_pages(pages, size, overlap):
    """Découpe en passages qui peuvent traverser les pages (page = celle du premier mot)."""
    words, word_pages = [], []
    for page_no, text in pages:
        for w in text.split():
            words.append(w)
            word_pages.append(page_no)

    step = max(size - overlap, 1)
    chunks = []
    for i in range(0, len(words), step):
        piece = words[i:i + size]
        if len(piece) < 30 and chunks:  
            break
        chunks.append({
            "text": " ".join(piece),
              "page": word_pages[i],
              "page_end": word_pages[i + len(piece) - 1],
                })
    return chunks

def extract_pages(pdf_source):
    reader = PdfReader(pdf_source)
    pages = []
    for n, page in enumerate(reader.pages, start=1):
        raw = (page.extract_text() or "").strip()
        if len(re.findall(r"(?:\.\s*){5,}", raw)) >= 5:   # page de table des matières
            continue
        pages.append((n, raw))
    return pages

def ensure_collection(client, size):
    if not client.collection_exists(config.COLLECTION_NAME):
        client.create_collection(
            collection_name=config.COLLECTION_NAME,
            vectors_config=VectorParams(size=size, distance=Distance.COSINE),
        )


def reset_collection():
    client = get_client()
    if client.collection_exists(config.COLLECTION_NAME):
        client.delete_collection(config.COLLECTION_NAME)


def ingest_pdf(pdf_source, source_name):
    """Indexe un PDF (chemin ou fichier) et retourne le nombre de passages créés."""
    client = get_client()

    pages = extract_pages(pdf_source)
    chunks = [
        {**c, "source": source_name}
        for c in chunk_pages(pages, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    ]

    if not chunks:
        raise ValueError(
            "Aucun texte extrait. Le PDF est peut-être scanné (images) : il faudrait un OCR."
        )

    vectors = embed_passages([c["text"] for c in chunks])
    ensure_collection(client, vectors.shape[1])

    # Évite les doublons si on ré-indexe le même fichier
    client.delete(
        collection_name=config.COLLECTION_NAME,
        points_selector=FilterSelector(
            filter=Filter(must=[FieldCondition(key="source", match=MatchValue(value=source_name))])
        ),
    )

    points = [
        PointStruct(id=str(uuid.uuid4()), vector=v.tolist(), payload=c)
        for c, v in zip(chunks, vectors)
    ]
    client.upsert(collection_name=config.COLLECTION_NAME, points=points)
    return len(chunks)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else config.DEFAULT_PDF
    n = ingest_pdf(path, Path(path).name)
    print(f"✅ {n} passages indexés depuis {Path(path).name}")
