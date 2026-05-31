from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from pipeline.agent import run_triage
from pipeline.cost_tracker import get_stats
from pipeline.schemas import SeverityLevel

app = FastAPI(
    title="MedAssist API",
    description="Production-grade medical triage agent",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

class TriageRequest(BaseModel):
    symptoms: str

@app.get("/")
def root():
    return {"message": "MedAssist API is running", "status": "healthy"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/triage")
def triage_endpoint(request: TriageRequest):
    if len(request.symptoms) < 10:
        raise HTTPException(status_code=400, detail="Please provide more detail about your symptoms")
    if len(request.symptoms) > 2000:
        raise HTTPException(status_code=400, detail="Symptom description too long")
    
    try:
        result = run_triage(request.symptoms)
        decision = result.get("triage_decision")
        
        if not decision:
            raise HTTPException(status_code=500, detail="Failed to generate triage decision")
        
        severity = decision.severity_classification
        if isinstance(severity, SeverityLevel):
            severity = severity.value
        
        return {
            "severity": severity,
            "recommended_next_steps": decision.recommended_next_steps,
            "clinical_reasoning": decision.clinical_reasoning,
            "sources": decision.sources,
            "red_flag_warnings": decision.red_flag_warnings,
            "drug_interaction_warnings": decision.drug_interaction_warnings,
            "disclaimer": decision.disclaimer,
            "uncertainty_note": decision.uncertainty_note,
            "errors": result.get("error_log", [])
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def metrics_endpoint():
    scores_path = Path("evaluation/ragas_scores.json")
    ragas_scores = {}
    if scores_path.exists():
        with open(scores_path) as f:
            ragas_scores = json.load(f)
    
    stats = get_stats()
    return {
        "ragas_scores": ragas_scores,
        "total_runs": stats["total_runs"],
        "avg_tokens": stats["avg_tokens"]
    }