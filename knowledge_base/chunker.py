import json
import re
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import List, Dict

print("Loading embedding model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("Embedding model loaded")

def split_into_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 30]
    return sentences

def semantic_chunk(sentences: List[str], similarity_threshold: float = 0.75, max_chunk_size: int = 5) -> List[str]:
    if not sentences:
        return []
    
    embeddings = model.encode(sentences, show_progress_bar=False)
    
    chunks = []
    current_chunk = [sentences[0]]
    current_embeddings = [embeddings[0]]
    
    for i in range(1, len(sentences)):
        current_mean = np.mean(current_embeddings, axis=0)
        similarity = np.dot(current_mean, embeddings[i]) / (
            np.linalg.norm(current_mean) * np.linalg.norm(embeddings[i])
        )
        
        if similarity >= similarity_threshold and len(current_chunk) < max_chunk_size:
            current_chunk.append(sentences[i])
            current_embeddings.append(embeddings[i])
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
            current_embeddings = [embeddings[i]]
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    chunks = [c for c in chunks if len(c) > 100]
    return chunks

def chunk_drug_documents(drug_data: List[Dict]) -> List[Dict]:
    chunks = []
    for drug in drug_data:
        drug_name = drug.get("drug_name", "unknown")
        for field, content in drug.items():
            if field == "drug_name" or content == "Not available":
                continue
            if len(content) > 100:
                chunks.append({
                    "text": f"Drug: {drug_name}\n{field.replace('_', ' ').title()}: {content[:1000]}",
                    "source": f"OpenFDA-{drug_name}",
                    "type": "drug",
                    "drug_name": drug_name,
                    "field": field
                })
    return chunks

def process_all_documents():
    all_chunks = []
    
    guidelines_dir = Path("data/raw/guidelines")
    for txt_file in guidelines_dir.glob("*.txt"):
        print(f"Processing: {txt_file.name}")
        text = txt_file.read_text(encoding="utf-8")
        sentences = split_into_sentences(text)
        chunks = semantic_chunk(sentences)
        
        for chunk in chunks:
            all_chunks.append({
                "text": chunk,
                "source": txt_file.name,
                "type": "guideline"
            })
        print(f"  Created {len(chunks)} chunks from {txt_file.name}")
    
    drug_path = Path("data/raw/drugs/drug_database.json")
    if drug_path.exists():
        with open(drug_path) as f:
            drug_data = json.load(f)
        drug_chunks = chunk_drug_documents(drug_data)
        all_chunks.extend(drug_chunks)
        print(f"Created {len(drug_chunks)} drug chunks")
    
    output_path = Path("data/processed/all_chunks.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_chunks, f, indent=2)
    
    print(f"\nTotal chunks created: {len(all_chunks)}")
    print(f"Saved to {output_path}")
    return all_chunks

if __name__ == "__main__":
    process_all_documents()