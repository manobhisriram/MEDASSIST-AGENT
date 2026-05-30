from pipeline.schemas import ParsedSymptoms
from pipeline.llm_caller import call_llm_structured

def symptom_parser_node(state: dict) -> dict:
    print("Node 1: Parsing symptoms...")
    
    user_input = state["original_input"]
    
    prompt = f"""Extract structured symptom information from this patient description.

Patient says: "{user_input}"

Extract the following as a flat JSON object:
- primary_complaint: string - the main symptom or health concern
- duration: string - how long they have had the symptoms, say "not specified" if not mentioned
- severity_descriptor: string - how they describe the severity, say "not specified" if not mentioned
- associated_symptoms: array of strings - other symptoms mentioned, empty array if none
- medications_mentioned: array of strings - medications they mentioned, empty array if none
- patient_context: string or null - age, existing conditions as a plain string, null if not mentioned"""

    try:
        parsed = call_llm_structured(prompt, ParsedSymptoms)
        state["parsed_symptoms"] = parsed
        print(f"Parsed: {parsed.primary_complaint}")
    except Exception as e:
        print(f"SymptomParser error: {e}")
        state["error_log"].append(f"SymptomParser error: {e}")
        state["parsed_symptoms"] = ParsedSymptoms(
            primary_complaint=user_input,
            duration="not specified",
            severity_descriptor="not specified",
            associated_symptoms=[],
            medications_mentioned=[],
            patient_context=None
        )
    
    return state