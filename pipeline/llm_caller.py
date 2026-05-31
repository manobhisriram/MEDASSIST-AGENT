import requests
import json
import re
from typing import Type, TypeVar
from pydantic import BaseModel
import time

T = TypeVar("T", bound=BaseModel)

def call_ollama_raw(prompt: str, model: str = "medassist", system_prompt: str = "You are MedAssist, a clinical triage assistant.") -> str:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": f"{system_prompt}\n\n{prompt}",
            "stream": False
        },
        timeout=180
    )
    return response.json()["response"]

def extract_json_from_text(text: str) -> str:
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        return json_match.group()
    return text

def call_llm_structured(
    prompt: str,
    response_model: Type[T],
    model: str = "medassist",
    max_retries: int = 3,
    system_prompt: str = "You are MedAssist, a clinical triage assistant. Always respond with valid JSON only. No explanation, no markdown, just the JSON object."
) -> T:
    
    schema = response_model.model_json_schema()
    
    full_prompt = f"""{prompt}

Respond with a JSON object that matches this exact schema:
{json.dumps(schema, indent=2)}

Return only the JSON object. No explanation. No markdown code blocks. Just the raw JSON."""

    for attempt in range(max_retries):
        try:
            raw_response = call_ollama_raw(full_prompt, model=model, system_prompt=system_prompt)
            cleaned = extract_json_from_text(raw_response)
            parsed_json = json.loads(cleaned)
            result = response_model(**parsed_json)
            return result
        except json.JSONDecodeError as e:
            print(f"Attempt {attempt + 1}: JSON parse error — {e}")
            if attempt < max_retries - 1:
                full_prompt += f"\n\nPrevious response was invalid JSON. Return ONLY a valid JSON object."
                time.sleep(1)
        except Exception as e:
            print(f"Attempt {attempt + 1}: Validation error — {e}")
            if attempt < max_retries - 1:
                full_prompt += f"\n\nPrevious response had validation error: {e}. Fix and return valid JSON."
                time.sleep(1)
    
    raise Exception(f"Failed to get valid structured output after {max_retries} attempts")

if __name__ == "__main__":
    from pipeline.schemas import ParsedSymptoms
    
    print("Testing structured output...")
    
    test_prompt = """Extract structured symptom information from this patient description:

Patient says: "I have had fever for 3 days, my joints are aching badly, and I noticed red spots on my arms today. I am 35 years old. I took paracetamol yesterday."

Extract:
- primary_complaint: main symptom
- duration: how long
- severity_descriptor: how severe they describe it
- associated_symptoms: list of other symptoms
- medications_mentioned: list of medications
- patient_context: age or other context"""

    result = call_llm_structured(test_prompt, ParsedSymptoms)
    print(f"Primary complaint: {result.primary_complaint}")
    print(f"Duration: {result.duration}")
    print(f"Associated symptoms: {result.associated_symptoms}")
    print(f"Medications: {result.medications_mentioned}")
    print("\nStructured output validation working correctly.")