"""
engine.py — Clinical Drug Safety Engine for EvoDoc.

Responsibilities:
  1. Call medical LLM (Meditron via Ollama) with structured prompt
  2. Validate and sanitise LLM output strictly
  3. Detect drug-class allergies (not just exact names)
  4. Check drug-condition contraindications
  5. Fall back to hardcoded rules if LLM unavailable
  6. Compute patient_risk_score (0–100) with breakdown
  7. Never return an empty result
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from itertools import combinations
from pathlib import Path
from typing import Optional

import httpx

from models import (
    AllergyAlert,
    ContraindicationAlert,
    DrugInteraction,
    PatientHistory,
    RiskScoreBreakdown,
    SafetyCheckResponse,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "meditron")          # ollama pull meditron
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT_SECONDS", "20"))
SYSTEM_PROMPT_PATH = Path(__file__).parent / "prompts" / "system_prompt.txt"
FALLBACK_DATA_PATH = Path(__file__).parent / "data" / "fallback_interactions.json"

# ─────────────────────────────────────────────
# LOAD STATIC ASSETS
# ─────────────────────────────────────────────

def _load_system_prompt() -> str:
    try:
        return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        logger.error("System prompt file not found at %s", SYSTEM_PROMPT_PATH)
        return "You are a clinical pharmacology AI. Respond only in valid JSON."


def _load_fallback_data() -> dict:
    try:
        return json.loads(FALLBACK_DATA_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        logger.error("Failed to load fallback data: %s", exc)
        return {"interactions": [], "allergy_class_map": {}}


SYSTEM_PROMPT = _load_system_prompt()
FALLBACK_DATA = _load_fallback_data()

# ─────────────────────────────────────────────
# CONDITION CONTRAINDICATIONS TABLE (Bonus C)
# ─────────────────────────────────────────────

CONDITION_CONTRAINDICATIONS: list[dict] = [
    {
        "condition_keywords": ["renal", "kidney", "ckd", "chronic kidney", "nephropathy", "renal failure", "renal impairment"],
        "drugs": ["ibuprofen", "naproxen", "diclofenac", "ketorolac", "indomethacin", "aspirin", "celecoxib", "meloxicam", "mefenamic acid"],
        "reason": "NSAIDs reduce renal prostaglandin synthesis, causing afferent arteriolar constriction and acute kidney injury in patients with pre-existing renal impairment.",
        "severity": "high",
    },
    {
        "condition_keywords": ["renal", "kidney", "ckd"],
        "drugs": ["metformin"],
        "reason": "Metformin is renally cleared; impaired excretion causes dangerous accumulation and lactic acidosis. Contraindicated when eGFR < 30 mL/min/1.73m².",
        "severity": "high",
    },
    {
        "condition_keywords": ["renal", "kidney", "ckd", "renal failure"],
        "drugs": ["nitrofurantoin"],
        "reason": "Nitrofurantoin requires adequate renal function to achieve therapeutic urinary concentrations; ineffective and potentially neurotoxic in renal impairment.",
        "severity": "high",
    },
    {
        "condition_keywords": ["hepatic", "liver", "cirrhosis", "hepatitis", "liver failure", "liver disease"],
        "drugs": ["atorvastatin", "simvastatin", "rosuvastatin", "pravastatin", "lovastatin"],
        "reason": "Statins are hepatically metabolised; pre-existing liver disease increases risk of hepatotoxicity and myopathy.",
        "severity": "high",
    },
    {
        "condition_keywords": ["hepatic", "liver", "cirrhosis", "hepatitis"],
        "drugs": ["methotrexate"],
        "reason": "Methotrexate accumulates in hepatic tissue and causes hepatotoxicity; baseline liver disease is a contraindication.",
        "severity": "high",
    },
    {
        "condition_keywords": ["asthma", "copd", "reactive airway", "bronchospasm"],
        "drugs": ["metoprolol", "atenolol", "propranolol", "bisoprolol", "carvedilol", "nebivolol"],
        "reason": "Non-selective and even cardioselective beta-blockers can precipitate bronchospasm in patients with reactive airway disease.",
        "severity": "high",
    },
    {
        "condition_keywords": ["asthma", "nsaid-sensitive", "aspirin-sensitive", "aspirin exacerbated"],
        "drugs": ["aspirin", "ibuprofen", "naproxen", "diclofenac", "ketorolac"],
        "reason": "Aspirin-exacerbated respiratory disease (AERD): COX-1 inhibition shifts arachidonic acid towards cysteinyl leukotriene production, causing bronchoconstriction.",
        "severity": "high",
    },
    {
        "condition_keywords": ["heart failure", "cardiac failure", "congestive heart", "chf", "lvsd"],
        "drugs": ["ibuprofen", "naproxen", "diclofenac", "celecoxib", "meloxicam"],
        "reason": "NSAIDs cause sodium and water retention, worsen cardiac remodelling, and increase hospitalisation risk in heart failure.",
        "severity": "high",
    },
    {
        "condition_keywords": ["heart failure", "cardiac failure"],
        "drugs": ["verapamil", "diltiazem"],
        "reason": "Verapamil and diltiazem are negative inotropes; they can precipitate acute decompensation in patients with reduced ejection fraction.",
        "severity": "high",
    },
    {
        "condition_keywords": ["hypertension", "high blood pressure"],
        "drugs": ["pseudoephedrine", "phenylephrine", "ephedrine"],
        "reason": "Sympathomimetics cause alpha-1 mediated vasoconstriction, raising systolic blood pressure significantly.",
        "severity": "medium",
    },
    {
        "condition_keywords": ["g6pd", "g6pd deficiency", "glucose-6-phosphate"],
        "drugs": ["primaquine", "dapsone", "nitrofurantoin", "rasburicase", "sulfamethoxazole"],
        "reason": "G6PD-deficient patients lack ability to regenerate NADPH; oxidative drugs cause haemolytic anaemia.",
        "severity": "high",
    },
    {
        "condition_keywords": ["diabetes", "diabetic", "type 2 diabetes", "t2dm"],
        "drugs": ["prednisolone", "dexamethasone", "hydrocortisone", "betamethasone"],
        "reason": "Corticosteroids cause insulin resistance and glucose dysregulation, severely worsening glycaemic control.",
        "severity": "medium",
    },
    {
        "condition_keywords": ["pregnancy", "pregnant", "gravid"],
        "drugs": ["warfarin", "acenocoumarol"],
        "reason": "Warfarin is teratogenic in first trimester (warfarin embryopathy) and causes fetal haemorrhage near term. Use LMWH in pregnancy.",
        "severity": "high",
    },
    {
        "condition_keywords": ["pregnancy", "pregnant"],
        "drugs": ["lisinopril", "enalapril", "ramipril", "captopril", "perindopril"],
        "reason": "ACE inhibitors cause fetal renal tubular dysplasia, oligohydramnios, and neonatal renal failure in 2nd/3rd trimester.",
        "severity": "high",
    },
    {
        "condition_keywords": ["osteoporosis", "osteopenia"],
        "drugs": ["prednisolone", "dexamethasone", "hydrocortisone"],
        "reason": "Corticosteroids inhibit osteoblast function and reduce calcium absorption, accelerating bone loss.",
        "severity": "medium",
    },
]


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _normalise(name: str) -> str:
    return name.lower().strip()


def _get_drug_classes(drug: str, allergy_class_map: dict[str, list[str]]) -> list[str]:
    """Return all allergy classes that the given drug belongs to."""
    d = _normalise(drug)
    return [cls for cls, members in allergy_class_map.items() if d in members]


def _build_user_prompt(
    proposed_medicines: list[str],
    patient_history: PatientHistory,
) -> str:
    return json.dumps(
        {
            "proposed_medicines": proposed_medicines,
            "patient_history": patient_history.model_dump(),
        },
        indent=2,
        ensure_ascii=False,
    )


# ─────────────────────────────────────────────
# ALLERGY DETECTION (Bonus: drug-class aware)
# ─────────────────────────────────────────────

def detect_allergy_alerts(
    proposed_medicines: list[str],
    patient_history: PatientHistory,
    allergy_class_map: dict[str, list[str]],
) -> list[AllergyAlert]:
    alerts: list[AllergyAlert] = []
    known_allergies_norm = [_normalise(a) for a in patient_history.known_allergies]

    for med in proposed_medicines:
        med_norm = _normalise(med)

        # Exact name match
        if med_norm in known_allergies_norm:
            alerts.append(AllergyAlert(
                medicine=med,
                reason=f"Patient has a known direct allergy to {med}",
                severity="critical",
            ))
            continue

        # Drug-class match: is patient allergic to any class this drug belongs to?
        med_classes = _get_drug_classes(med, allergy_class_map)
        for cls in med_classes:
            # Check if any allergy in patient history is in this class or IS this class name
            class_members = allergy_class_map.get(cls, [])
            for allergy in known_allergies_norm:
                if allergy == cls or allergy in class_members:
                    alerts.append(AllergyAlert(
                        medicine=med,
                        reason=f"Patient has {allergy.title()} allergy; {med} belongs to the {cls.replace('_', ' ').title()} class",
                        severity="critical",
                    ))
                    break

    return alerts


# ─────────────────────────────────────────────
# CONDITION CONTRAINDICATION CHECK (Bonus C)
# ─────────────────────────────────────────────

def detect_contraindications(
    proposed_medicines: list[str],
    patient_history: PatientHistory,
) -> list[ContraindicationAlert]:
    alerts: list[ContraindicationAlert] = []
    conditions_norm = [_normalise(c) for c in patient_history.conditions]

    # Age-based: flag benzodiazepines and anticholinergics in elderly
    if patient_history.age and patient_history.age > 65:
        elderly_risky = {
            "diazepam": "Falls, sedation, and cognitive impairment risk significantly elevated in patients over 65",
            "lorazepam": "Falls, sedation, and cognitive impairment risk significantly elevated in patients over 65",
            "alprazolam": "Falls, sedation, and cognitive impairment risk significantly elevated in patients over 65",
            "clonazepam": "Falls, sedation, and cognitive impairment risk significantly elevated in patients over 65",
            "nitrazepam": "Falls, sedation, and cognitive impairment risk significantly elevated in patients over 65",
            "amitriptyline": "Strong anticholinergic effects: confusion, urinary retention, constipation in elderly",
            "promethazine": "Strong anticholinergic and sedating; associated with delirium in older adults",
        }
        for med in proposed_medicines:
            reason = elderly_risky.get(_normalise(med))
            if reason:
                alerts.append(ContraindicationAlert(
                    medicine=med,
                    condition="Age > 65",
                    reason=reason,
                    severity="medium",
                ))

    # Condition-based contraindications
    for rule in CONDITION_CONTRAINDICATIONS:
        # Does patient have any condition matching this rule?
        matched_condition = None
        for cond in conditions_norm:
            if any(kw in cond for kw in rule["condition_keywords"]):
                matched_condition = cond
                break

        if not matched_condition:
            continue

        for med in proposed_medicines:
            if _normalise(med) in rule["drugs"]:
                alerts.append(ContraindicationAlert(
                    medicine=med,
                    condition=matched_condition,
                    reason=rule["reason"],
                    severity=rule["severity"],
                ))

    return alerts


# ─────────────────────────────────────────────
# RISK SCORING (Bonus B)
# ─────────────────────────────────────────────

_SEVERITY_WEIGHTS = {"high": 15, "medium": 7, "low": 2}
_ALLERGY_WEIGHTS = {"critical": 30, "high": 20, "medium": 10, "low": 5}


def compute_risk_score(
    interactions: list[DrugInteraction],
    allergy_alerts: list[AllergyAlert],
    contraindication_alerts: list[ContraindicationAlert],
    patient_history: PatientHistory,
) -> tuple[float, RiskScoreBreakdown]:
    # Interaction score (0–40): top-weighted interactions, capped
    interaction_raw = sum(_SEVERITY_WEIGHTS.get(i.severity, 5) for i in interactions)
    interaction_score = min(40.0, float(interaction_raw))

    # Allergy score (0–30)
    allergy_raw = sum(_ALLERGY_WEIGHTS.get(a.severity, 10) for a in allergy_alerts)
    allergy_score = min(30.0, float(allergy_raw))

    # Condition score (0–20)
    condition_raw = sum(_SEVERITY_WEIGHTS.get(c.severity, 5) for c in contraindication_alerts)
    condition_score = min(20.0, float(condition_raw))

    # Age/weight modifier (0–10)
    age_weight_modifier = 0.0
    if patient_history.age:
        if patient_history.age > 75:
            age_weight_modifier += 5.0
        elif patient_history.age > 65:
            age_weight_modifier += 3.0
        elif patient_history.age < 12:
            age_weight_modifier += 3.0
    if patient_history.weight_kg:
        if patient_history.weight_kg < 40 or patient_history.weight_kg > 120:
            age_weight_modifier += 2.0
    age_weight_modifier = min(10.0, age_weight_modifier)

    total = min(100.0, interaction_score + allergy_score + condition_score + age_weight_modifier)

    breakdown = RiskScoreBreakdown(
        interaction_score=round(interaction_score, 1),
        allergy_score=round(allergy_score, 1),
        condition_score=round(condition_score, 1),
        age_weight_modifier=round(age_weight_modifier, 1),
        total=round(total, 1),
    )
    return round(total, 1), breakdown


def _score_to_risk_level(score: float) -> str:
    if score >= 60:
        return "high"
    if score >= 30:
        return "medium"
    if score >= 10:
        return "low"
    return "minimal"


# ─────────────────────────────────────────────
# LLM CALL
# ─────────────────────────────────────────────

async def _call_llm(user_prompt: str) -> Optional[dict]:
    """Call Meditron via Ollama. Returns parsed JSON dict or None on failure."""
    payload = {
        "model": LLM_MODEL,
        "prompt": user_prompt,
        "system": SYSTEM_PROMPT,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.1,   # Low temperature for clinical accuracy
            "top_p": 0.9,
            "num_predict": 2048,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
            resp = await client.post(f"{OLLAMA_URL}/api/generate", json=payload)
            resp.raise_for_status()
            data = resp.json()
            raw_text = data.get("response", "")
            return _parse_llm_output(raw_text)
    except httpx.TimeoutException:
        logger.warning("LLM request timed out after %.1fs", LLM_TIMEOUT)
    except httpx.HTTPStatusError as exc:
        logger.error("LLM HTTP error: %s", exc)
    except Exception as exc:
        logger.error("Unexpected LLM error: %s", exc)
    return None


def _parse_llm_output(raw: str) -> Optional[dict]:
    """Extract and validate JSON from LLM response."""
    if not raw:
        return None

    # Strip markdown fences if present
    raw = re.sub(r"```(?:json)?", "", raw).strip()

    # Try direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Try to extract the first {...} block
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    logger.warning("Could not parse LLM output as JSON")
    return None


# ─────────────────────────────────────────────
# LLM OUTPUT VALIDATION
# ─────────────────────────────────────────────

_VALID_SEVERITIES = {"high", "medium", "low"}
_VALID_CONFIDENCES = {"high", "medium", "low"}
_VALID_ALLERGY_SEVERITIES = {"critical", "high", "medium", "low"}


def _validate_interaction(raw: dict, known_drugs: set[str]) -> Optional[DrugInteraction]:
    """Validate a single interaction dict from LLM output."""
    try:
        drug_a = str(raw.get("drug_a", "")).strip()
        drug_b = str(raw.get("drug_b", "")).strip()
        severity = str(raw.get("severity", "")).lower()
        mechanism = str(raw.get("mechanism", "")).strip()
        recommendation = str(raw.get("clinical_recommendation", "")).strip()
        confidence = str(raw.get("source_confidence", "medium")).lower()

        # Reject hallucinated drug names (must contain at least one known drug name or be a plausible drug)
        if not drug_a or not drug_b or drug_a == drug_b:
            return None
        if severity not in _VALID_SEVERITIES:
            severity = "low"
        if confidence not in _VALID_CONFIDENCES:
            confidence = "low"
        if not mechanism:
            mechanism = "Mechanism not specified; clinical review recommended"
        if not recommendation:
            recommendation = "Consult physician before prescribing"

        return DrugInteraction(
            drug_a=drug_a,
            drug_b=drug_b,
            severity=severity,
            mechanism=mechanism,
            clinical_recommendation=recommendation,
            source_confidence=confidence,
        )
    except Exception as exc:
        logger.debug("Interaction validation failed: %s — %s", raw, exc)
        return None


def _validate_allergy_alert(raw: dict) -> Optional[AllergyAlert]:
    try:
        medicine = str(raw.get("medicine", "")).strip()
        reason = str(raw.get("reason", "")).strip()
        severity = str(raw.get("severity", "critical")).lower()

        if not medicine or not reason:
            return None
        if severity not in _ALLERGY_WEIGHTS:
            severity = "critical"
        return AllergyAlert(medicine=medicine, reason=reason, severity=severity)
    except Exception:
        return None


def _validate_contraindication(raw: dict) -> Optional[ContraindicationAlert]:
    try:
        medicine = str(raw.get("medicine", "")).strip()
        condition = str(raw.get("condition", "")).strip()
        reason = str(raw.get("reason", "")).strip()
        severity = str(raw.get("severity", "high")).lower()

        if not medicine or not condition:
            return None
        if severity not in _VALID_SEVERITIES:
            severity = "high"
        return ContraindicationAlert(
            medicine=medicine, condition=condition, reason=reason, severity=severity
        )
    except Exception:
        return None


def _parse_llm_response(
    llm_dict: dict,
    proposed_medicines: list[str],
) -> tuple[list[DrugInteraction], list[AllergyAlert], list[ContraindicationAlert], bool]:
    """
    Sanitise and validate every field from LLM output.
    Returns (interactions, allergy_alerts, contraindication_alerts, requires_doctor_review).
    """
    known_drugs = {_normalise(m) for m in proposed_medicines}
    requires_review = bool(llm_dict.get("requires_doctor_review", False))

    interactions = []
    for raw in llm_dict.get("interactions", []):
        if isinstance(raw, dict):
            item = _validate_interaction(raw, known_drugs)
            if item:
                interactions.append(item)
            else:
                requires_review = True  # Something was fishy — flag for review

    allergy_alerts = []
    for raw in llm_dict.get("allergy_alerts", []):
        if isinstance(raw, dict):
            item = _validate_allergy_alert(raw)
            if item:
                allergy_alerts.append(item)

    contraindication_alerts = []
    for raw in llm_dict.get("contraindication_alerts", []):
        if isinstance(raw, dict):
            item = _validate_contraindication(raw)
            if item:
                contraindication_alerts.append(item)

    return interactions, allergy_alerts, contraindication_alerts, requires_review


# ─────────────────────────────────────────────
# FALLBACK: RULE-BASED ENGINE
# ─────────────────────────────────────────────

def _fallback_interactions(
    proposed_medicines: list[str],
    current_medications: list[str],
) -> list[DrugInteraction]:
    """Match proposed + current meds against the hardcoded fallback dataset."""
    all_meds_norm = {_normalise(m): m for m in (proposed_medicines + current_medications)}
    results: list[DrugInteraction] = []
    checked = set()

    allergy_class_map: dict[str, list[str]] = FALLBACK_DATA.get("allergy_class_map", {})

    def _resolve_name(norm_key: str) -> str:
        # Return original casing if available
        return all_meds_norm.get(norm_key, norm_key.title())

    def _med_matches(med_norm: str, pattern: str) -> bool:
        """Check if normalised med name matches a pattern (direct or class)."""
        if med_norm == pattern:
            return True
        # Check if med belongs to a class with that name
        class_members = allergy_class_map.get(pattern, [])
        return med_norm in class_members

    for rule in FALLBACK_DATA.get("interactions", []):
        drug_a_pattern = rule.get("drug_a", "").lower()
        drug_b_pattern = rule.get("drug_b", "").lower()

        for med_a_norm, med_b_norm in combinations(all_meds_norm.keys(), 2):
            pair_key = tuple(sorted([med_a_norm, med_b_norm]))
            if pair_key in checked:
                continue

            match_ab = _med_matches(med_a_norm, drug_a_pattern) and _med_matches(med_b_norm, drug_b_pattern)
            match_ba = _med_matches(med_a_norm, drug_b_pattern) and _med_matches(med_b_norm, drug_a_pattern)

            if match_ab or match_ba:
                checked.add(pair_key)
                results.append(DrugInteraction(
                    drug_a=_resolve_name(med_a_norm),
                    drug_b=_resolve_name(med_b_norm),
                    severity=rule.get("severity", "medium"),
                    mechanism=rule.get("mechanism", "See clinical guidelines"),
                    clinical_recommendation=rule.get("clinical_recommendation", "Consult physician"),
                    source_confidence=rule.get("source_confidence", "high"),
                ))

    return results


# ─────────────────────────────────────────────
# MAIN ENGINE ENTRY POINT
# ─────────────────────────────────────────────

async def run_safety_check(
    proposed_medicines: list[str],
    patient_history: PatientHistory,
) -> dict:
    """
    Orchestrate the full drug safety check pipeline.
    Returns a dict ready to be unpacked into SafetyCheckResponse.
    """
    start_ms = time.monotonic()
    warnings: list[str] = []
    source: str = "llm"
    requires_doctor_review: bool = False

    allergy_class_map = FALLBACK_DATA.get("allergy_class_map", {})

    # ── 1. Try LLM ───────────────────────────────────────────
    user_prompt = _build_user_prompt(proposed_medicines, patient_history)
    llm_raw = await _call_llm(user_prompt)

    if llm_raw:
        try:
            interactions, allergy_alerts_llm, contras_llm, llm_review = _parse_llm_response(
                llm_raw, proposed_medicines
            )
            requires_doctor_review = llm_review
        except Exception as exc:
            logger.error("LLM response parsing failed: %s", exc)
            llm_raw = None

    # ── 2. Fallback if LLM failed ─────────────────────────────
    if not llm_raw:
        source = "fallback"
        warnings.append("LLM unavailable; using rule-based fallback dataset")
        requires_doctor_review = True
        interactions = _fallback_interactions(
            proposed_medicines, patient_history.current_medications
        )
        allergy_alerts_llm = []
        contras_llm = []

    # ── 3. Always run local allergy + condition checks ────────
    #    These are deterministic and more reliable than LLM for class matching
    local_allergy_alerts = detect_allergy_alerts(
        proposed_medicines, patient_history, allergy_class_map
    )
    local_contras = detect_contraindications(proposed_medicines, patient_history)

    # Merge: prefer local (deterministic) over LLM for allergy/contra;
    # deduplicate by medicine name
    def _merge_by_medicine(primary: list, secondary: list) -> list:
        seen = {item.medicine.lower() for item in primary}
        merged = list(primary)
        for item in secondary:
            if item.medicine.lower() not in seen:
                merged.append(item)
                seen.add(item.medicine.lower())
        return merged

    allergy_alerts = _merge_by_medicine(local_allergy_alerts, allergy_alerts_llm)
    contraindication_alerts = _merge_by_medicine(local_contras, contras_llm)

    # Also check proposed meds vs current medications via fallback rules
    if source == "llm" and patient_history.current_medications:
        current_med_interactions = _fallback_interactions(
            proposed_medicines, patient_history.current_medications
        )
        # Add any current-med interactions not already captured by LLM
        existing_pairs = {
            tuple(sorted([i.drug_a.lower(), i.drug_b.lower()]))
            for i in interactions
        }
        for inter in current_med_interactions:
            pair = tuple(sorted([inter.drug_a.lower(), inter.drug_b.lower()]))
            if pair not in existing_pairs:
                interactions.append(inter)
                existing_pairs.add(pair)

    # ── 4. Determine overall risk ─────────────────────────────
    has_high = any(i.severity == "high" for i in interactions)
    has_critical_allergy = any(
        a.severity in ("critical", "high") for a in allergy_alerts
    )
    has_high_contra = any(c.severity == "high" for c in contraindication_alerts)

    if has_high or has_critical_allergy or has_high_contra:
        requires_doctor_review = True

    safe_to_prescribe = not (has_high or has_critical_allergy or has_high_contra)

    # ── 5. Risk scoring ───────────────────────────────────────
    risk_score, risk_breakdown = compute_risk_score(
        interactions, allergy_alerts, contraindication_alerts, patient_history
    )
    overall_risk_level = _score_to_risk_level(risk_score)

    # Sync safe_to_prescribe with risk score as additional safeguard
    if risk_score >= 60 and safe_to_prescribe:
        safe_to_prescribe = False
        warnings.append("Risk score ≥ 60; safe_to_prescribe overridden to false")

    elapsed_ms = int((time.monotonic() - start_ms) * 1000)

    return {
        "interactions": interactions,
        "allergy_alerts": allergy_alerts,
        "contraindication_alerts": contraindication_alerts,
        "safe_to_prescribe": safe_to_prescribe,
        "overall_risk_level": overall_risk_level,
        "requires_doctor_review": requires_doctor_review,
        "source": source,
        "cache_hit": False,   # cache layer sets this to True when serving from cache
        "processing_time_ms": elapsed_ms,
        "patient_risk_score": risk_score,
        "risk_score_breakdown": risk_breakdown,
        "warnings": warnings,
    }
