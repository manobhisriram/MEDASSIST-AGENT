from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Union
from enum import Enum

class SeverityLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    EMERGENCY = "EMERGENCY"

class ParsedSymptoms(BaseModel):
    primary_complaint: str = Field(description="The main symptom or complaint")
    duration: str = Field(description="How long the symptoms have been present")
    severity_descriptor: str = Field(description="How the patient describes the severity")
    associated_symptoms: List[str] = Field(description="Other symptoms mentioned")
    medications_mentioned: List[str] = Field(default=[], description="Any medications mentioned")
    patient_context: Optional[str] = Field(default=None, description="Age, existing conditions, other context as a plain string")

    @field_validator("patient_context", mode="before")
    @classmethod
    def flatten_patient_context(cls, v):
        if isinstance(v, dict):
            return " ".join(f"{k}: {val}" for k, val in v.items())
        return v

    @field_validator("associated_symptoms", "medications_mentioned", mode="before")
    @classmethod
    def ensure_string_list(cls, v):
        if isinstance(v, list):
            return [str(item) if not isinstance(item, str) else item for item in v]
        return v

class SeverityAssessment(BaseModel):
    severity_level: SeverityLevel = Field(description="Triage severity classification")
    confidence_score: float = Field(description="Confidence in assessment between 0 and 1")
    red_flag_indicators: List[str] = Field(default=[], description="Specific red flag symptoms present")
    reasoning: str = Field(description="Clinical reasoning for this severity classification")

    @field_validator("confidence_score", mode="before")
    @classmethod
    def validate_confidence(cls, v):
        try:
            v = float(v)
        except Exception:
            return 0.5
        if not 0.0 <= v <= 1.0:
            return 0.5
        return round(v, 2)

    @field_validator("severity_level", mode="before")
    @classmethod
    def validate_severity(cls, v):
        if isinstance(v, str):
            v = v.upper().strip()
            if v not in ["LOW", "MEDIUM", "HIGH", "EMERGENCY"]:
                return "MEDIUM"
        return v

class RetrievedChunk(BaseModel):
    text: str
    source: str
    chunk_type: str

class DrugInfo(BaseModel):
    drug_name: str
    indication: Optional[str] = None
    contraindications: Optional[str] = None
    warnings: Optional[str] = None
    interaction_alerts: List[str] = Field(default=[])

class TriageDecision(BaseModel):
    severity_classification: SeverityLevel = Field(description="Final severity level")
    recommended_next_steps: List[str] = Field(description="Clear actionable steps for the patient")
    clinical_reasoning: str = Field(description="Explanation of the triage decision")
    sources: List[str] = Field(default=[], description="Sources cited")
    red_flag_warnings: List[str] = Field(default=[], description="Critical warning signs")
    drug_interaction_warnings: List[str] = Field(default=[], description="Drug interaction alerts")
    disclaimer: str = Field(default="This assessment is for informational purposes only and does not replace professional medical advice. Please consult a qualified healthcare provider.")
    uncertainty_note: Optional[str] = Field(default=None)

    @field_validator("severity_classification", mode="before")
    @classmethod
    def validate_severity(cls, v):
        if isinstance(v, str):
            v = v.upper().strip()
            if v not in ["LOW", "MEDIUM", "HIGH", "EMERGENCY"]:
                return "MEDIUM"
        return v

    @field_validator("recommended_next_steps", "red_flag_warnings", "drug_interaction_warnings", "sources", mode="before")
    @classmethod
    def ensure_list(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return v