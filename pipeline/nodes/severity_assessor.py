from pipeline.schemas import SeverityAssessment, SeverityLevel
from pipeline.llm_caller import call_llm_structured

def severity_assessor_node(state: dict) -> dict:
    print("Node 2: Assessing severity...")
    
    symptoms = state["parsed_symptoms"]
    
    prompt = f"""Assess the medical severity of this patient's condition.

Primary complaint: {symptoms.primary_complaint}
Duration: {symptoms.duration}
Severity description: {symptoms.severity_descriptor}
Associated symptoms: {', '.join(symptoms.associated_symptoms) if symptoms.associated_symptoms else 'none'}
Patient context: {symptoms.patient_context or 'not provided'}

Return a JSON object with these exact fields:
- severity_level: one of exactly "LOW", "MEDIUM", "HIGH", or "EMERGENCY"
- confidence_score: a number between 0.0 and 1.0
- red_flag_indicators: array of strings listing concerning symptoms, empty array if none
- reasoning: string explaining your clinical reasoning in 2-3 sentences

Severity guide:
EMERGENCY: life-threatening, needs ambulance now (chest pain + breathlessness, stroke, severe bleeding)
HIGH: serious, needs same-day hospital (high fever + rash, severe abdominal pain, dengue warning signs)
MEDIUM: needs doctor within 24-48 hours (moderate symptoms, possible infection)
LOW: can manage at home, monitor symptoms (mild cold, minor aches)"""

    try:
        assessment = call_llm_structured(prompt, SeverityAssessment)
        state["severity_assessment"] = assessment
        print(f"Severity: {assessment.severity_level} (confidence: {assessment.confidence_score})")
    except Exception as e:
        print(f"SeverityAssessor error: {e}")
        state["error_log"].append(f"SeverityAssessor error: {e}")
        state["severity_assessment"] = SeverityAssessment(
            severity_level=SeverityLevel.MEDIUM,
            confidence_score=0.5,
            red_flag_indicators=[],
            reasoning="Unable to fully assess. Defaulting to MEDIUM severity. Please consult a doctor."
        )
    
    return state