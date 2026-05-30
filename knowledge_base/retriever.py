import lancedb
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict

LANCEDB_PATH = "lancedb_store"
TABLE_NAME = "clinical_knowledge"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_model = None
_table = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model

def get_table():
    global _table
    if _table is None:
        db = lancedb.connect(LANCEDB_PATH)
        _table = db.open_table(TABLE_NAME)
    return _table

def reciprocal_rank_fusion(vector_results: List[Dict], fts_results: List[Dict], k: int = 60) -> List[Dict]:
    scores = {}
    
    for rank, result in enumerate(vector_results):
        key = result["text"][:100]
        if key not in scores:
            scores[key] = {"score": 0, "data": result}
        scores[key]["score"] += 1 / (k + rank + 1)
    
    for rank, result in enumerate(fts_results):
        key = result["text"][:100]
        if key not in scores:
            scores[key] = {"score": 0, "data": result}
        scores[key]["score"] += 1 / (k + rank + 1)
    
    sorted_results = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
    return [r["data"] for r in sorted_results]

def retrieve(query: str, top_k: int = 5) -> List[Dict]:
    model = get_model()
    table = get_table()
    
    query_embedding = model.encode([query])[0]
    vector_results = table.search(query_embedding.tolist()).limit(top_k * 2).to_list()
    
    try:
        fts_results = table.search(query, query_type="fts").limit(top_k * 2).to_list()
    except Exception:
        fts_results = []
    
    if fts_results:
        combined = reciprocal_rank_fusion(vector_results, fts_results)
    else:
        combined = vector_results
    
    formatted = []
    for r in combined[:top_k]:
        formatted.append({
            "text": r["text"],
            "source": r["source"],
            "type": r["type"],
            "drug_name": r.get("drug_name", "")
        })
    
    return formatted

if __name__ == "__main__":
    print("Testing hybrid retriever...\n")
    
    test_queries = [
        "patient has fever joint pain red spots dengue",
        "chest pain shortness of breath emergency",
        "aspirin contraindications warnings",
        "malaria treatment India"
    ]
    
    for query in test_queries:
        print(f"Query: {query}")
        results = retrieve(query, top_k=2)
        for r in results:
            print(f"  [{r['source']}] {r['text'][:120]}...")
        print()