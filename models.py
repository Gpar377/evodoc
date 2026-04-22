"""
models.py — Pydantic v2 schemas for EvoDoc Clinical Drug Safety Engine
"""

from __future__ import annotations
from typing import List, Optional, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
import re


# ─────────────────────────────────────────────
# REQUEST MODELS
# ─────────────────────────────────────────────

class PatientHistory(BaseModel):
    current_medications: List[str] = Field(default_factory=list)
    known_allergies: List[str] = Field(default_factory=list)
    conditions: List[str] = Field(default_factory=list)
    age: Optional[int] = Field(default=None, ge=0, le=130)
    weight_kg: Optional[float] = Field(default=None, ge=0.5, le=500)

    @field_validator("current_medications", "known_allergies", "conditions", mode="before")
    @classmethod
    def clean_string_list(cls, v):
        if not isinstance(v, list):
            raise ValueError("Must be a list")
        cleaned = []
        for item in v:
            if not isinstance(item, str):
                raise ValueError(f"List items must be strings, got {type(item)}")
            item = item.strip()
            if item:
                cleaned.append(item)
        return cleaned

    @field_validator("age", mode="before")
    @classmethod
    def validate_age(cls, v):
        if v is None:
            return v
        if isinstance(v, (int, float)) and v < 0:
            raise ValueError("Age cannot be negative")
        return int(v)


class SafetyCheckRequest(BaseModel):
    proposed_medicines: List[str] = Field(..., min_length=1)
    patient_history: PatientHistory = Field(default_factory=PatientHistory)

    @field_validator("proposed_medicines", mode="before")
    @classmethod
    def validate_medicines(cls, v):
        if not isinstance(v, list):
            raise ValueError("proposed_medicines must be a list")
        if len(v) == 0:
            raise ValueError("At least one medicine must be provided")

        cleaned = []
        seen = set()
        for med in v:
            if not isinstance(med, str):
                raise ValueError("Medicine names must be strings")
            med = med.strip()
            if not med:
                continue
            # Warn but don't reject slightly odd names — just normalise
            key = med.lower()
            if key not in seen:
                seen.add(key)
                cleaned.append(med)
            # duplicates are silently deduplicated
        if not cleaned:
            raise ValueError("No valid medicine names after cleaning")
        return cleaned


# ─────────────────────────────────────────────
# RESPONSE MODELS
# ─────────────────────────────────────────────

SeverityLevel = Literal["high", "medium", "low"]
RiskLevel = Literal["high", "medium", "low", "minimal"]
SourceType = Literal["llm", "fallback", "rules"]


class DrugInteraction(BaseModel):
    drug_a: str
    drug_b: str
    severity: SeverityLevel
    mechanism: str
    clinical_recommendation: str
    source_confidence: Literal["high", "medium", "low"] = "medium"


class AllergyAlert(BaseModel):
    medicine: str
    reason: str
    severity: Literal["critical", "high", "medium", "low"] = "critical"


class ContraindicationAlert(BaseModel):
    medicine: str
    condition: str
    reason: str
    severity: SeverityLevel = "high"


class RiskScoreBreakdown(BaseModel):
    interaction_score: float = Field(ge=0, le=40, description="Weighted interaction risk (0–40)")
    allergy_score: float = Field(ge=0, le=30, description="Allergy risk contribution (0–30)")
    condition_score: float = Field(ge=0, le=20, description="Condition contraindication risk (0–20)")
    age_weight_modifier: float = Field(ge=0, le=10, description="Age/weight adjustment (0–10)")
    total: float = Field(ge=0, le=100)


class SafetyCheckResponse(BaseModel):
    interactions: List[DrugInteraction] = Field(default_factory=list)
    allergy_alerts: List[AllergyAlert] = Field(default_factory=list)
    contraindication_alerts: List[ContraindicationAlert] = Field(default_factory=list)
    safe_to_prescribe: bool
    overall_risk_level: RiskLevel
    requires_doctor_review: bool
    source: SourceType
    cache_hit: bool
    processing_time_ms: int
    patient_risk_score: Optional[float] = Field(default=None, ge=0, le=100)
    risk_score_breakdown: Optional[RiskScoreBreakdown] = None
    warnings: List[str] = Field(default_factory=list)
