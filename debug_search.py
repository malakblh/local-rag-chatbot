from rag import retrieve

for r in retrieve("Qui a inventé le machine learning ?", top_k=5, min_score=None):
    print(f"p.{r['page']}-{r['page_end']}  score={r['score']:.3f}  contient 'Samuel': {'Samuel' in r['text']}")