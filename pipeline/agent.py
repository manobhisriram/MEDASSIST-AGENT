import uuid
from langgraph.graph import StateGraph, END
from pipeline.schemas import SeverityLevel
from pipeline.nodes.symptom_parser import symptom_parser_node
from pipeline.nodes.severity_assessor import severity_assessor_node
from pipeline.nodes.clinical_retriever import clinical_retriever_node
from pipeline.nodes.drug_checker import drug_checker_node
from pipeline.nodes.triage_decision import triage_decision_node
from pipeline.cost_tracker import log_run

def should_check_drugs(state: dict) -> str:
    severity = state.get("severity_assessment")
    symptoms = state.get("parsed_symptoms")
    if severity and severity.severity_level in [SeverityLevel.HIGH, SeverityLevel.EMERGENCY]:
        return "drug_checker"
    if symptoms and symptoms.medications_mentioned:
        return "drug_checker"
    return "triage_decision"

def build_agent():
    workflow = StateGraph(dict)
    workflow.add_node("symptom_parser", symptom_parser_node)
    workflow.add_node("severity_assessor", severity_assessor_node)
    workflow.add_node("clinical_retriever", clinical_retriever_node)
    workflow.add_node("drug_checker", drug_checker_node)
    workflow.add_node("triage_decision", triage_decision_node)
    workflow.set_entry_point("symptom_parser")
    workflow.add_edge("symptom_parser", "severity_assessor")
    workflow.add_edge("severity_assessor", "clinical_retriever")
    workflow.add_conditional_edges(
        "clinical_retriever",
        should_check_drugs,
        {
            "drug_checker": "drug_checker",
            "triage_decision": "triage_decision"
        }
    )
    workflow.add_edge("drug_checker", "triage_decision")
    workflow.add_edge("triage_decision", END)
    return workflow.compile()

def run_triage(symptom_text: str) -> dict:
    run_id = str(uuid.uuid4())
    initial_state = {
        "original_input": symptom_text,
        "parsed_symptoms": None,
        "severity_assessment": None,
        "retrieved_chunks": [],
        "drug_info": [],
        "triage_decision": None,
        "error_log": [],
        "run_id": run_id
    }
    agent = build_agent()
    print(f"\n{'='*50}")
    print(f"MedAssist Triage | Run: {run_id[:8]}")
    print(f"{'='*50}")
    final_state = agent.invoke(initial_state)
    decision = final_state.get("triage_decision")
    severity = decision.severity_classification if decision else "UNKNOWN"
    log_run(run_id=run_id, input_text=symptom_text, severity=severity,
            total_tokens=0, model="medassist", success=decision is not None)
    return final_state

if __name__ == "__main__":
    test_cases = [
        "I have had fever for 3 days, joint pain in my knees, and red spots on my arms today.",
        "I have chest pain and shortness of breath since this morning. I am 52 years old.",
        "My child has mild cold and runny nose. No fever. She is 5 years old.",
    ]
    
    for i, case in enumerate(test_cases):
        print(f"\n\nTEST CASE {i+1}: {case}")
        result = run_triage(case)
        decision = result.get("triage_decision")
        if decision:
            print(f"\n{'='*50}")
            print(f"SEVERITY: {decision.severity_classification}")
            print(f"\nNEXT STEPS:")
            for step in decision.recommended_next_steps:
                print(f"  - {step}")
            print(f"\nREASONING: {decision.clinical_reasoning}")
            print(f"\nSOURCES: {', '.join(decision.sources)}")
            if decision.red_flag_warnings:
                print(f"\nRED FLAGS:")
                for w in decision.red_flag_warnings:
                    print(f"  - {w}")
            print(f"\nDISCLAIMER: {decision.disclaimer}")
        if result.get("error_log"):
            print(f"\nErrors: {result['error_log']}")