import lancedb
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

LANCEDB_PATH = "lancedb_store"
TABLE_NAME = "clinical_knowledge"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def build_index():
    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    chunks_path = Path("data/processed/all_chunks.json")
    with open(chunks_path) as f:
        chunks = json.load(f)
    
    print(f"Indexing {len(chunks)} chunks...")
    
    db = lancedb.connect(LANCEDB_PATH)
    
    if TABLE_NAME in db.table_names():
        db.drop_table(TABLE_NAME)
        print("Dropped existing table")
    
    records = []
    batch_size = 32
    
    for i in tqdm(range(0, len(chunks), batch_size)):
        batch = chunks[i:i+batch_size]
        texts = [c["text"] for c in batch]
        embeddings = model.encode(texts)
        
        for j, chunk in enumerate(batch):
            records.append({
                "text": chunk["text"],
                "source": chunk.get("source", "unknown"),
                "type": chunk.get("type", "general"),
                "drug_name": chunk.get("drug_name", ""),
                "vector": embeddings[j].tolist()
            })
    
    table = db.create_table(TABLE_NAME, data=records)
    
    try:
        table.create_fts_index("text", replace=True)
        print("Full-text search index created")
    except Exception as e:
        print(f"FTS index note: {e}")
    
    print(f"\nIndexed {len(records)} chunks into LanceDB")
    print(f"LanceDB stored at: {LANCEDB_PATH}/")
    return table

def test_retrieval():
    print("\nTesting retrieval...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    db = lancedb.connect(LANCEDB_PATH)
    table = db.open_table(TABLE_NAME)
    
    query = "dengue fever symptoms treatment"
    query_embedding = model.encode([query])[0]
    
    results = table.search(query_embedding.tolist()).limit(3).to_list()
    print(f"\nTop 3 results for '{query}':")
    for i, r in enumerate(results):
        print(f"\n{i+1}. Source: {r['source']}")
        print(f"   Text: {r['text'][:200]}...")

if __name__ == "__main__":
    build_index()
    test_retrieval()
    print("\nLanceDB index built successfully.")