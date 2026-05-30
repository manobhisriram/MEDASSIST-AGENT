from pipeline.schemas import RetrievedChunk
from knowledge_base.retriever import retrieve

def clinical_retriever_node(state: dict) -> dict:
    print("Node 3: Retrieving clinical guidelines...")
    
    symptoms = state["parsed_symptoms"]
    query = f"{symptoms.primary_complaint} {' '.join(symptoms.associated_symptoms[:3])}"
    
    raw_results = retrieve(query, top_k=5)
    
    chunks = []
    for r in raw_results:
        chunks.append(RetrievedChunk(
            text=r["text"],
            source=r["source"],
            chunk_type=r["type"]
        ))
    
    state["retrieved_chunks"] = chunks
    print(f"Retrieved {len(chunks)} relevant chunks")
    return state