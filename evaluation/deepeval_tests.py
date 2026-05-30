import json
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from pipeline.schemas import TriageDecision, SeverityLevel
from pipeline.agent import run_triage

def run_test(name: str, symptom_text: str, expected_severities: list, check_sources: bool = True):
    print(f"\nRunning: {name}")
    result = run_triage(symptom_text)
    decision = result.get("triage_decision")
    
    assert decision is not None, f"FAIL {name}: No triage decision returned"
    
    severity = decision.severity_classification
    if isinstance(severity, SeverityLevel):
        severity = severity.value
    
    assert severity in expected_severities, \
        f"FAIL {name}: Expected {expected_severities}, got {severity}"
    
    assert len(decision.recommended_next_steps) >= 1, \
        f"FAIL {name}: No next steps returned"
    
    assert len(decision.clinical_reasoning) > 20, \
        f"FAIL {name}: Clinical reasoning too short"
    
    print(f"PASS {name}: severity={severity}, steps={len(decision.recommended_next_steps)}")
    return True

def test_schema_validation():
    print("\nRunning: test_schema_validation")
    decision = TriageDecision(
        severity_classification=SeverityLevel.HIGH,
        recommended_next_steps=["Go to doctor today"],
        clinical_reasoning="Test reasoning for validation",
        sources=["test_source"],
        red_flag_warnings=[],
        drug_interaction_warnings=[]
    )
    assert decision.severity_classification == SeverityLevel.HIGH
    assert len(decision.recommended_next_steps) == 1
    assert decision.disclaimer is not None
    print("PASS test_schema_validation")
    return True

def test_ragas_scores():
    print("\nRunning: test_ragas_scores")
    scores_path = Path("evaluation/ragas_scores.json")
    if not scores_path.exists():
        print("SKIP test_ragas_scores: no scores file yet")
        return True
    with open(scores_path) as f:
        scores = json.load(f)
    assert scores.get("faithfulness", 0) >= 0.60, \
        f"FAIL test_ragas_scores: faithfulness {scores.get('faithfulness')} below 0.60"
    print(f"PASS test_ragas_scores: faithfulness={scores.get('faithfulness'):.3f}")
    return True

if __name__ == "__main__":
    print("="*50)
    print("MEDASSIST DEEPEVAL TESTS")
    print("="*50)
    
    passed = 0
    failed = 0
    
    # Schema tests (no LLM needed)
    schema_tests = [test_schema_validation, test_ragas_scores]
    for test in schema_tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"{e}")
            failed += 1
        except Exception as e:
            print(f"ERROR: {e}")
            failed += 1
    
    # Agent tests (need Ollama running)
    agent_tests = [
        ("chest_pain_emergency", "I have severe chest pain and cannot breathe since 30 minutes", ["HIGH", "EMERGENCY"]),
        ("mild_cold_low", "I have mild runny nose and slight cough for 1 day. No fever.", ["LOW", "MEDIUM"]),
        ("dengue_high", "fever for 3 days, severe joint pain, red spots on skin, feeling weak", ["HIGH", "EMERGENCY"]),
    ]
    
    for name, symptoms, expected in agent_tests:
        try:
            run_test(name, symptoms, expected)
            passed += 1
        except AssertionError as e:
            print(f"{e}")
            failed += 1
        except Exception as e:
            print(f"ERROR in {name}: {e}")
            failed += 1
    
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)