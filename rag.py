"""Recherche dans Qdrant + génération de la réponse avec Ollama.

Usage : python rag.py "Explique le deep learning"
"""
import sys

import ollama

import config
from core import embed_query, get_client

NOT_FOUND = "Je ne trouve pas la réponse dans le document."

SYSTEM_PROMPT = f"""Tu es un assistant qui répond en français, uniquement à partir du contexte fourni.
Règles strictes :
- Utilise uniquement les informations présentes dans le contexte.
- N'introduis aucun terme, sigle, date ou exemple qui n'apparaît pas dans le contexte.
- Reprends autant que possible les formulations du contexte.
- Si la réponse ne se trouve pas dans le contexte, réponds exactement : "{NOT_FOUND}"
"""

def retrieve(question, top_k=config.TOP_K, min_score=config.MIN_SCORE):
    """Retourne les passages les plus proches (avec source, pages et score)."""
    client = get_client()
    if not client.collection_exists(config.COLLECTION_NAME):
        return []
    res = client.query_points(
        collection_name=config.COLLECTION_NAME,
        query=embed_query(question),
        limit=top_k,
        score_threshold=min_score,
        with_payload=True,
    )
    return [
        {
            "text": p.payload["text"],
            "source": p.payload.get("source", "?"),
            "page": p.payload.get("page"),
            "page_end": p.payload.get("page_end", p.payload.get("page")),
            "score": p.score,
        }
        for p in res.points
    ]
def page_label(s):
    return f"p. {s['page']}" if s["page"] == s["page_end"] else f"p. {s['page']}–{s['page_end']}"


def build_messages(question, sources):
    context = "\n\n".join(
        f"[Source {i} – {s['source']}, p. {s['page']}]\n{s['text']}"
        for i, s in enumerate(sources, start=1)
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Contexte :\n{context}\n\nQuestion : {question}"},
    ]


def rag_stream(question):
    """Retourne (sources, générateur de texte). Répond NOT_FOUND si rien de pertinent."""
    sources = retrieve(question)
    if not sources:
        return [], iter([NOT_FOUND])

    stream = ollama.chat(
        model=config.LLM_MODEL,
        messages=build_messages(question, sources),
        options={"temperature": 0, "num_ctx": 2048},
        stream=True,
    )

    def generator():
        for chunk in stream:
            yield chunk["message"]["content"]

    return sources, generator()


def rag_query(question):
    """Version non-streaming : retourne (réponse, sources)."""
    sources, gen = rag_stream(question)
    return "".join(gen), sources


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Explique le deep learning"
    answer, srcs = rag_query(q)
    print("\n🤖 Réponse :\n", answer)
    for s in srcs:
        print(f"  - {s['source']}, {page_label(s)} (score {s['score']:.2f})")
