import requests
from pipeline.schemas import DrugInfo

def fetch_drug_info(drug_name: str) -> dict:
    try:
        url = f"https://api.fda.gov/drug/label.json?search=openfda.generic_name:{drug_name}&limit=1"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get("results"):
                result = data["results"][0]
                return {
                    "drug_name": drug_name,
                    "indication": result.get("indications_and_usage", [""])[0][:300] if result.get("indications_and_usage") else None,
                    "contraindications": result.get("contraindications", [""])[0][:300] if result.get("contraindications") else None,
                    "warnings": result.get("warnings", [""])[0][:300] if result.get("warnings") else None,
                    "interaction_alerts": []
                }
    except Exception as e:
        print(f"Could not fetch drug info for {drug_name}: {e}")
    return {"drug_name": drug_name, "indication": None, "contraindications": None, "warnings": None, "interaction_alerts": []}

def drug_checker_node(state: dict) -> dict:
    print("Node 4: Checking drug information...")
    
    symptoms = state["parsed_symptoms"]
    medications = symptoms.medications_mentioned
    
    if not medications:
        print("No medications mentioned, skipping drug check")
        state["drug_info"] = []
        return state
    
    drug_infos = []
    for med in medications[:3]:
        print(f"Fetching info for: {med}")
        info = fetch_drug_info(med.lower())
        drug_infos.append(DrugInfo(**info))
    
    state["drug_info"] = drug_infos
    print(f"Retrieved info for {len(drug_infos)} medications")
    return state