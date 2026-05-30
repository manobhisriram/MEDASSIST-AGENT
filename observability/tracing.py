import os
from dotenv import load_dotenv
load_dotenv()

def setup_langfuse():
    from langfuse import Langfuse
    
    client = Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    )
    print("Langfuse connected successfully")
    return client

def trace_agent_run(client, run_id: str, input_text: str, result: dict):
    trace = client.trace(
        name="medassist-triage",
        id=run_id,
        input={"symptoms": input_text},
        metadata={"model": "medassist"}
    )
    
    decision = result.get("triage_decision")
    if decision:
        trace.update(
            output={
                "severity": decision.severity_classification,
                "next_steps": decision.recommended_next_steps,
                "sources": decision.sources
            }
        )
    
    if result.get("parsed_symptoms"):
        trace.span(
            name="symptom_parser",
            input={"raw_input": input_text},
            output={"primary_complaint": result["parsed_symptoms"].primary_complaint}
        )
    
    if result.get("severity_assessment"):
        trace.span(
            name="severity_assessor",
            input={"symptoms": result["parsed_symptoms"].primary_complaint if result.get("parsed_symptoms") else ""},
            output={
                "severity": result["severity_assessment"].severity_level,
                "confidence": result["severity_assessment"].confidence_score
            }
        )
    
    if result.get("retrieved_chunks"):
        trace.span(
            name="clinical_retriever",
            input={"query": input_text},
            output={"chunks_retrieved": len(result["retrieved_chunks"]),
                   "sources": [c.source for c in result["retrieved_chunks"]]}
        )
    
    client.flush()
    print(f"Trace sent to Langfuse: {run_id[:8]}")
    return trace

if __name__ == "__main__":
    client = setup_langfuse()
    
    from pipeline.agent import run_triage
    
    result = run_triage("I have fever for 3 days, joint pain, and red spots on my skin.")
    trace_agent_run(client, result["run_id"], result["original_input"], result)
    
    print("\nCheck cloud.langfuse.com to see the trace")