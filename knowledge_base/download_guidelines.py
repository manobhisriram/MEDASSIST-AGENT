import requests
import json
from pathlib import Path

def fetch_openfda_drugs():
    drugs = [
        "aspirin", "paracetamol", "ibuprofen", "amoxicillin",
        "metformin", "amlodipine", "atorvastatin", "omeprazole",
        "ciprofloxacin", "azithromycin", "metronidazole", "doxycycline",
        "cetirizine", "montelukast", "salbutamol", "prednisolone"
    ]
    
    output_dir = Path("data/raw/drugs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_drug_docs = []
    
    for drug in drugs:
        print(f"Fetching FDA data for {drug}...")
        try:
            url = f"https://api.fda.gov/drug/label.json?search=openfda.generic_name:{drug}&limit=1"
            response = requests.get(url, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("results"):
                    result = data["results"][0]
                    doc = {
                        "drug_name": drug,
                        "indications": result.get("indications_and_usage", ["Not available"])[0] if result.get("indications_and_usage") else "Not available",
                        "contraindications": result.get("contraindications", ["Not available"])[0] if result.get("contraindications") else "Not available",
                        "warnings": result.get("warnings", ["Not available"])[0] if result.get("warnings") else "Not available",
                        "adverse_reactions": result.get("adverse_reactions", ["Not available"])[0] if result.get("adverse_reactions") else "Not available",
                        "dosage": result.get("dosage_and_administration", ["Not available"])[0] if result.get("dosage_and_administration") else "Not available"
                    }
                    all_drug_docs.append(doc)
                    print(f"Got data for {drug}")
                else:
                    print(f"No FDA data found for {drug}")
            else:
                print(f"Failed {drug}: status {response.status_code}")
        except Exception as e:
            print(f"Error fetching {drug}: {e}")
    
    output_path = output_dir / "drug_database.json"
    with open(output_path, "w") as f:
        json.dump(all_drug_docs, f, indent=2)
    
    print(f"\nSaved drug data for {len(all_drug_docs)} drugs to {output_path}")
    return all_drug_docs

def create_clinical_guidelines():
    guidelines_dir = Path("data/raw/guidelines")
    guidelines_dir.mkdir(parents=True, exist_ok=True)
    
    guidelines = [
        {
            "filename": "dengue_triage.txt",
            "content": """WHO DENGUE TRIAGE GUIDELINES

SEVERITY CLASSIFICATION:
1. Dengue without warning signs (Group A): Fever, nausea, vomiting, rash, aches and pains, positive tourniquet test. Home treatment possible with adequate hydration.

2. Dengue with warning signs (Group B): Abdominal pain or tenderness, persistent vomiting, clinical fluid accumulation, mucosal bleeding, lethargy, liver enlargement, increase in haematocrit with rapid decrease in platelet count. Requires hospital admission and IV fluid therapy.

3. Severe dengue (Group C): Severe plasma leakage leading to shock, severe bleeding, severe organ impairment. Requires emergency treatment.

WARNING SIGNS requiring immediate hospital admission:
- Abdominal pain or tenderness
- Persistent vomiting (3 or more times in 24 hours)
- Clinical fluid accumulation (ascites, pleural effusion)
- Mucosal bleeding (gum bleeding, nosebleed)
- Lethargy or restlessness
- Liver enlargement greater than 2cm
- Laboratory: increase in haematocrit with rapid decrease in platelet count

RED FLAGS for severe dengue:
- Severe abdominal pain
- Persistent vomiting
- Rapid breathing
- Bleeding gums
- Fatigue, restlessness
- Blood in vomit or stool
- Extreme thirst, pale and cold skin
- Feeling weak

TRIAGE DECISION:
- No warning signs: Oral rehydration, rest, paracetamol for fever. Return if warning signs develop.
- Warning signs present: Immediate hospital referral
- Severe dengue: Emergency hospital admission, IV fluids, close monitoring

TREATMENT:
- No specific antiviral treatment exists for dengue
- Supportive care is the mainstay of treatment
- Paracetamol for fever and pain (NOT aspirin or ibuprofen - risk of bleeding)
- Oral rehydration with ORS or coconut water
- Monitor platelet count daily
- Blood transfusion only if severe bleeding with haemodynamic compromise"""
        },
        {
            "filename": "fever_triage.txt",
            "content": """CLINICAL GUIDELINES FOR FEVER TRIAGE

DEFINITION:
Fever is defined as body temperature above 38.0°C (100.4°F). High fever is above 39.5°C (103.1°F).

SEVERITY CLASSIFICATION BY AGE:

Infants under 3 months:
- Any fever above 38.0°C is a medical emergency
- Immediate emergency department evaluation required
- Risk of serious bacterial infection is high

Children 3 months to 3 years:
- Fever above 39.0°C without source: evaluate same day
- Fever with rash: evaluate same day
- Fever with stiff neck: emergency - possible meningitis
- Fever lasting more than 5 days: medical evaluation required

Adults:
- Fever above 38.5°C lasting more than 3 days: medical evaluation
- Fever above 40.0°C: urgent medical evaluation
- Fever with stiff neck, severe headache, light sensitivity: emergency
- Fever with difficulty breathing: emergency
- Fever with confusion: emergency

RED FLAG SYMPTOMS WITH FEVER:
- Stiff neck
- Severe headache
- Photophobia (light sensitivity)
- Petechial or purpuric rash (does not blanch with pressure)
- Difficulty breathing
- Severe abdominal pain
- Confusion or altered consciousness
- Inability to stand or walk
- Persistent vomiting

COMMON CAUSES BY DURATION:
- Less than 7 days: viral infection (most common), bacterial infection
- 7-14 days: typhoid fever, malaria, dengue, bacterial infection
- More than 14 days: tuberculosis, infective endocarditis, malignancy, autoimmune

MANAGEMENT:
- Paracetamol 500-1000mg every 6 hours for adults (NOT aspirin for children)
- Adequate hydration: 2-3 litres water per day for adults
- Rest
- Remove excess clothing
- Seek medical attention if fever persists more than 3 days or any red flag present"""
        },
        {
            "filename": "chest_pain_triage.txt",
            "content": """EMERGENCY TRIAGE GUIDELINES FOR CHEST PAIN

IMMEDIATE EMERGENCY - CALL AMBULANCE:
Any chest pain with the following features requires immediate emergency response:
- Crushing or pressure-like chest pain
- Pain radiating to left arm, jaw, neck, or back
- Chest pain with shortness of breath
- Chest pain with sweating
- Chest pain with nausea or vomiting
- Chest pain lasting more than 15 minutes
- Chest pain at rest

POSSIBLE CAUSES BY PRESENTATION:

Cardiac (Emergency):
- Acute myocardial infarction: crushing chest pain, radiation to left arm or jaw, sweating, nausea
- Unstable angina: chest pain at rest or with minimal exertion
- Aortic dissection: sudden severe tearing chest pain radiating to back

Pulmonary (Urgent to Emergency):
- Pulmonary embolism: pleuritic chest pain, shortness of breath, tachycardia, risk factors (immobility, DVT)
- Pneumothorax: sudden onset pleuritic chest pain, absent breath sounds

Non-cardiac (Less urgent):
- Gastroesophageal reflux: burning, worse after meals, relieved by antacids
- Musculoskeletal: pain reproduced by palpation or movement
- Anxiety/panic attack: associated with palpitations, tingling

TRIAGE PRIORITY:
- Any cardiac features: EMERGENCY - do not let patient drive themselves
- Pleuritic chest pain with breathlessness: URGENT - same day evaluation
- Reproducible musculoskeletal pain: can be seen within 24-48 hours

INITIAL MANAGEMENT:
- Emergency cases: Call ambulance immediately, aspirin 300mg if not allergic and cardiac cause suspected, patient should rest
- Do not give aspirin if aortic dissection suspected
- Oxygen if available and SpO2 below 94%"""
        },
        {
            "filename": "malaria_guidelines.txt", 
            "content": """MALARIA TRIAGE AND TREATMENT GUIDELINES - INDIA

CLINICAL PRESENTATION:
Uncomplicated malaria:
- Fever (may be cyclical - every 48 or 72 hours)
- Chills and rigors
- Headache
- Myalgia (muscle pain)
- Nausea and vomiting
- Fatigue

Severe malaria (medical emergency):
- Cerebral malaria: altered consciousness, seizures, coma
- Severe anaemia: haemoglobin below 7 g/dL
- Respiratory distress
- Hypoglycaemia
- Renal failure
- Shock

SPECIES IN INDIA:
- Plasmodium vivax: most common, can relapse
- Plasmodium falciparum: more dangerous, drug resistance possible

DIAGNOSIS:
- Rapid Diagnostic Test (RDT): available at primary health centres
- Blood smear microscopy: gold standard
- All fever cases in endemic areas should be tested

TREATMENT (Government of India Guidelines):
Vivax malaria:
- Chloroquine 25mg/kg over 3 days
- Primaquine 0.25mg/kg daily for 14 days (test for G6PD deficiency first)

Falciparum malaria:
- Artemisinin-based Combination Therapy (ACT): Artesunate + Sulfadoxine-Pyrimethamine
- Do not use chloroquine for falciparum (resistance)

Severe malaria:
- IV Artesunate is first line
- Immediate hospital admission required

PREVENTION:
- Insecticide-treated bed nets
- Indoor residual spraying
- Eliminate breeding sites
- Repellents

TRIAGE DECISION:
- Suspected uncomplicated malaria: Test immediately, treat same day
- Any severe features: Emergency hospital admission
- Pregnant women with malaria: Always refer to hospital"""
        },
        {
            "filename": "respiratory_guidelines.txt",
            "content": """RESPIRATORY ILLNESS TRIAGE GUIDELINES

SEVERITY CLASSIFICATION:

EMERGENCY - Call ambulance immediately:
- Severe difficulty breathing (cannot complete sentences)
- Cyanosis (blue lips or fingertips)
- Altered consciousness with breathing difficulty
- Silent chest (no breath sounds despite respiratory effort)
- Respiratory rate above 30 per minute in adults

URGENT - Same day hospital evaluation:
- Moderate difficulty breathing
- Respiratory rate 20-30 per minute
- Oxygen saturation below 94% if measurable
- High fever with productive cough
- Coughing blood (haemoptysis)
- Chest pain with breathing (pleuritic)

STANDARD - See doctor within 48 hours:
- Persistent cough more than 3 weeks
- Mild wheeze with known asthma, controlled with inhaler
- Cough with fever in adults (possible pneumonia)

HOME MANAGEMENT - Monitor symptoms:
- Common cold: clear or white nasal discharge, mild cough, no fever
- Mild viral upper respiratory infection: symptom relief, adequate hydration

RED FLAGS for serious respiratory illness:
- Coughing blood
- Night sweats with cough (possible tuberculosis)
- Weight loss with cough
- Cough lasting more than 3 weeks

ASTHMA ACUTE ATTACK MANAGEMENT:
- Mild: 2-4 puffs salbutamol inhaler, repeat every 20 minutes up to 3 times
- Moderate: 4-8 puffs salbutamol, seek medical attention
- Severe: Ambulance immediately, salbutamol continuously

TUBERCULOSIS SCREENING:
Refer for TB testing if:
- Cough more than 2 weeks
- Blood in sputum
- Night sweats
- Unexplained weight loss
- Close contact with known TB patient"""
        }
    ]
    
    for guideline in guidelines:
        filepath = guidelines_dir / guideline["filename"]
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(guideline["content"])
        print(f"Created: {guideline['filename']}")
    
    print(f"\nCreated {len(guidelines)} clinical guideline files")
    return guidelines

if __name__ == "__main__":
    print("=== Creating Clinical Guidelines ===")
    create_clinical_guidelines()
    print("\n=== Fetching Drug Data from OpenFDA ===")
    fetch_openfda_drugs()
    print("\nData collection complete.")