from pipeline.schemas import TriageDecision, SeverityLevel
from pipeline.llm_caller import call_llm_structured

def triage_decision_node(state: dict) -> dict:
    print("Node 5: Generating triage decision...")
    
    symptoms = state["parsed_symptoms"]
    severity = state["severity_assessment"]
    chunks = state["retrieved_chunks"]
    drugs = state["drug_info"]
    
    context_text = "\n".join([f"- [{c.source}]: {c.text[:300]}" for c in chunks[:4]])
    sources = list(set([c.source for c in chunks]))
    
    drug_text = ""
    if drugs:
        drug_text = "\n".join([
            f"- {d.drug_name}: {d.warnings[:200] if d.warnings else 'no warnings available'}"
            for d in drugs
        ])
    
    prompt = f"""Generate a complete medical triage decision for this patient.

PATIENT SYMPTOMS:
Primary complaint: {symptoms.primary_complaint}
Duration: {symptoms.duration}
Associated symptoms: {', '.join(symptoms.associated_symptoms) if symptoms.associated_symptoms else 'none'}

SEVERITY ASSESSMENT: {severity.severity_level}
Red flags identified: {', '.join(severity.red_flag_indicators) if severity.red_flag_indicators else 'none'}
Clinical reasoning: {severity.reasoning}

CLINICAL GUIDELINES RETRIEVED:
{context_text if context_text else 'No specific guidelines retrieved'}

MEDICATION INFORMATION:
{drug_text if drug_text else 'No medications mentioned'}

Return a JSON object with these exact fields:
- severity_classification: one of "LOW", "MEDIUM", "HIGH", "EMERGENCY"
- recommended_next_steps: array of 2-4 strings, each a clear actionable step
- clinical_reasoning: string, 2-3 sentences explaining the decision
- sources: array of strings listing the source documents used
- red_flag_warnings: array of strings listing symptoms that would require immediate emergency care
- drug_interaction_warnings: array of strings listing drug warnings, empty array if no medications
- disclaimer: string with medical disclaimer
- uncertainty_note: null or string if confidence was low"""

    try:
        decision = call_llm_structured(prompt, TriageDecision)
        if not decision.sources:
            decision.sources = sources
        if severity.confidence_score < 0.6:
            decision.uncertainty_note = "Note: Confidence in this assessment is low. Please seek medical advice promptly."
        state["triage_decision"] = decision
        print(f"Decision: {decision.severity_classification}")
    except Exception as e:
        print(f"TriageDecision error: {e}")
        state["error_log"].append(f"TriageDecision error: {e}")
        state["triage_decision"] = TriageDecision(
            severity_classification=severity.severity_level,
            recommended_next_steps=["Please consult a qualified healthcare provider for proper assessment"],
            clinical_reasoning=severity.reasoning,
            sources=sources,
            red_flag_warnings=severity.red_flag_indicators,
            drug_interaction_warnings=[]
        )
    
    return state