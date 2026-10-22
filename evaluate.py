"""Mesure la qualité de la recherche : une bonne page est-elle dans le top-k ?

Usage : python evaluate.py [questions.json]
Format : [{"question": "...", "source": "Ai.pdf", "expected_pages": [[debut, fin], ...]}, ...]
(ancien format accepté : "expected_page": 12)
"""
import json
import sys

import config
from rag import page_label, retrieve

path = sys.argv[1] if len(sys.argv) > 1 else "questions.json"
with open(path, encoding="utf-8") as f:
    questions = json.load(f)


def expected_ranges(q):
    if "expected_pages" in q:
        return q["expected_pages"]
    return [[q["expected_page"], q["expected_page"]]]


def is_hit(result, q):
    if "source" in q and q["source"] != result["source"]:
        return False
    # le passage couvre les pages [page, page_end] : il suffit qu'il touche une plage attendue
    return any(result["page"] <= end and result["page_end"] >= start
               for start, end in expected_ranges(q))


hits = hits_top1 = 0
for q in questions:
    results = retrieve(q["question"], top_k=config.TOP_K, min_score=None)
    ok = any(is_hit(r, q) for r in results)
    hits += ok
    hits_top1 += bool(results) and is_hit(results[0], q)
    pages = ", ".join(page_label(r).replace("p. ", "") for r in results)
    print(f"{'✅' if ok else '❌'} {q['question']}\n     pages trouvées : {pages}")

rate = hits / len(questions)
print(f"Taux de réussite top-1 : {hits_top1}/{len(questions)} = {hits_top1 / len(questions):.0%}")