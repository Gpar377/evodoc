"""
tests/test_engine.py — Test suite for EvoDoc Clinical Drug Safety Engine.

Run with:
    pytest tests/ -v

Tests cover:
  - Allergy detection (exact + drug-class)
  - Condition contraindication detection
  - Cache key determinism (order independence)
  - Risk scoring boundaries
  - Input validation edge cases
  - Fallback interaction matching
  - FastAPI endpoint integration
"""

import asyncio
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cache import InMemoryCache
from engine import (
    FALLBACK_DATA,
    compute_risk_score,
    detect_allergy_alerts,
    detect_contraindications,
    _fallback_interactions,
    _score_to_risk_level,
)
from main import app
from models import PatientHistory, SafetyCheckRequest


# ─────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────

@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def allergy_class_map():
    return FALLBACK_DATA.get("allergy_class_map", {})


@pytest.fixture
def empty_history():
    return PatientHistory()


@pytest.fixture
def penicillin_allergic_patient():
    return PatientHistory(
        known_allergies=["Penicillin"],
        current_medications=["Metoprolol"],
        conditions=["Hypertension"],
        age=45,
        weight_kg=70.0,
    )


@pytest.fixture
def complex_patient():
    return PatientHistory(
        current_medications=["Warfarin", "Metformin"],
        known_allergies=["Sulfa"],
        conditions=["Chronic kidney disease", "Hypertension"],
        age=68,
        weight_kg=58.0,
    )


# ─────────────────────────────────────────────
# ALLERGY DETECTION TESTS
# ─────────────────────────────────────────────

class TestAllergyDetection:
    def test_exact_allergy_match(self, allergy_class_map, penicillin_allergic_patient):
        """Exact drug name match triggers critical alert."""
        penicillin_allergic_patient.known_allergies = ["Amoxicillin"]
        alerts = detect_allergy_alerts(
            ["Amoxicillin", "Paracetamol"], penicillin_allergic_patient, allergy_class_map
        )
        assert any(a.medicine == "Amoxicillin" for a in alerts)
        assert all(a.severity == "critical" for a in alerts if a.medicine == "Amoxicillin")

    def test_penicillin_class_allergy_flags_amoxicillin(self, allergy_class_map, penicillin_allergic_patient):
        """Penicillin allergy should flag Amoxicillin (same class)."""
        alerts = detect_allergy_alerts(
            ["Amoxicillin", "Ibuprofen"], penicillin_allergic_patient, allergy_class_map
        )
        medicines_flagged = {a.medicine for a in alerts}
        assert "Amoxicillin" in medicines_flagged

    def test_sulfa_class_allergy_flags_cotrimoxazole(self, allergy_class_map):
        """Sulfa allergy should flag Co-trimoxazole."""
        patient = PatientHistory(known_allergies=["Sulfa"])
        alerts = detect_allergy_alerts(
            ["Co-trimoxazole"], patient, allergy_class_map
        )
        assert len(alerts) == 1
        assert alerts[0].severity == "critical"

    def test_no_allergy_no_alerts(self, allergy_class_map, empty_history):
        """No allergies on record means no allergy alerts."""
        alerts = detect_allergy_alerts(["Amoxicillin", "Ibuprofen"], empty_history, allergy_class_map)
        assert alerts == []

    def test_nsaid_class_allergy(self, allergy_class_map):
        """NSAID allergy should flag ibuprofen."""
        patient = PatientHistory(known_allergies=["nsaid"])
        alerts = detect_allergy_alerts(["Ibuprofen", "Paracetamol"], patient, allergy_class_map)
        assert any(a.medicine == "Ibuprofen" for a in alerts)

    def test_safe_drug_not_flagged(self, allergy_class_map, penicillin_allergic_patient):
        """Paracetamol should not be flagged for a Penicillin allergy."""
        alerts = detect_allergy_alerts(
            ["Paracetamol"], penicillin_allergic_patient, allergy_class_map
        )
        assert alerts == []


# ─────────────────────────────────────────────
# CONTRAINDICATION TESTS
# ─────────────────────────────────────────────

class TestContraindications:
    def test_nsaid_flagged_in_ckd(self):
        """Ibuprofen should be flagged for patient with CKD."""
        patient = PatientHistory(conditions=["Chronic kidney disease"])
        alerts = detect_contraindications(["Ibuprofen", "Paracetamol"], patient)
        assert any(a.medicine == "Ibuprofen" for a in alerts)
        flagged = next(a for a in alerts if a.medicine == "Ibuprofen")
        assert flagged.severity == "high"

    def test_metformin_flagged_in_ckd(self):
        """Metformin should be flagged in renal impairment."""
        patient = PatientHistory(conditions=["Renal impairment"])
        alerts = detect_contraindications(["Metformin"], patient)
        assert any(a.medicine == "Metformin" for a in alerts)

    def test_beta_blocker_flagged_in_asthma(self):
        """Metoprolol should be flagged for asthma patient."""
        patient = PatientHistory(conditions=["Asthma"])
        alerts = detect_contraindications(["Metoprolol"], patient)
        assert any(a.medicine == "Metoprolol" for a in alerts)

    def test_benzodiazepine_flagged_for_elderly(self):
        """Diazepam should be flagged for patient aged 70."""
        patient = PatientHistory(age=70)
        alerts = detect_contraindications(["Diazepam"], patient)
        assert any(a.medicine == "Diazepam" for a in alerts)

    def test_no_contraindication_healthy_patient(self):
        """Amoxicillin in a healthy adult should not trigger contraindications."""
        patient = PatientHistory(age=35, conditions=[])
        alerts = detect_contraindications(["Amoxicillin"], patient)
        assert alerts == []

    def test_statin_flagged_in_liver_disease(self):
        """Atorvastatin should be flagged for hepatic impairment."""
        patient = PatientHistory(conditions=["Hepatic impairment"])
        alerts = detect_contraindications(["Atorvastatin"], patient)
        assert any(a.medicine == "Atorvastatin" for a in alerts)


# ─────────────────────────────────────────────
# CACHE TESTS
# ─────────────────────────────────────────────

class TestCache:
    def test_key_is_order_independent_medicines(self):
        """Cache key must be identical regardless of medicine list order."""
        key1 = InMemoryCache.build_key(["Warfarin", "Aspirin"], [])
        key2 = InMemoryCache.build_key(["Aspirin", "Warfarin"], [])
        assert key1 == key2

    def test_key_is_order_independent_medications(self):
        """Cache key must be identical regardless of current_medications order."""
        key1 = InMemoryCache.build_key(["Amoxicillin"], ["Metformin", "Warfarin"])
        key2 = InMemoryCache.build_key(["Amoxicillin"], ["Warfarin", "Metformin"])
        assert key1 == key2

    def test_different_drugs_give_different_keys(self):
        key1 = InMemoryCache.build_key(["Warfarin"], [])
        key2 = InMemoryCache.build_key(["Aspirin"], [])
        assert key1 != key2

    def test_cache_case_insensitive_keys(self):
        """Keys should normalise case: 'warfarin' == 'Warfarin'."""
        key1 = InMemoryCache.build_key(["WARFARIN", "ASPIRIN"], [])
        key2 = InMemoryCache.build_key(["warfarin", "aspirin"], [])
        assert key1 == key2

    def test_cache_set_and_get(self):
        cache = InMemoryCache()

        async def _run():
            await cache.set("testkey", {"foo": "bar"}, ttl=60)
            result = await cache.get("testkey")
            assert result == {"foo": "bar"}

        asyncio.get_event_loop().run_until_complete(_run())

    def test_cache_ttl_expiry(self):
        cache = InMemoryCache()

        async def _run():
            await cache.set("expiring", {"x": 1}, ttl=0)
            import time; time.sleep(0.01)
            result = await cache.get("expiring")
            assert result is None

        asyncio.get_event_loop().run_until_complete(_run())


# ─────────────────────────────────────────────
# RISK SCORING TESTS
# ─────────────────────────────────────────────

class TestRiskScoring:
    def test_no_issues_gives_minimal_risk(self):
        patient = PatientHistory(age=30)
        score, breakdown = compute_risk_score([], [], [], patient)
        assert score == 0.0
        assert _score_to_risk_level(score) == "minimal"

    def test_high_interaction_raises_score(self):
        from models import DrugInteraction
        interactions = [
            DrugInteraction(
                drug_a="Warfarin", drug_b="Aspirin", severity="high",
                mechanism="test", clinical_recommendation="test", source_confidence="high"
            )
        ]
        patient = PatientHistory()
        score, _ = compute_risk_score(interactions, [], [], patient)
        assert score >= 15

    def test_critical_allergy_maxes_allergy_score(self):
        from models import AllergyAlert
        alerts = [
            AllergyAlert(medicine="Amoxicillin", reason="Penicillin class", severity="critical")
        ]
        patient = PatientHistory()
        score, breakdown = compute_risk_score([], alerts, [], patient)
        assert breakdown.allergy_score == 30.0

    def test_score_capped_at_100(self):
        from models import DrugInteraction, AllergyAlert, ContraindicationAlert
        interactions = [
            DrugInteraction(
                drug_a=f"Drug{i}", drug_b=f"Drug{i+1}", severity="high",
                mechanism="x", clinical_recommendation="x", source_confidence="high"
            )
            for i in range(10)
        ]
        alerts = [AllergyAlert(medicine="X", reason="y", severity="critical") for _ in range(5)]
        contras = [ContraindicationAlert(medicine="X", condition="y", reason="z", severity="high") for _ in range(5)]
        patient = PatientHistory(age=80, weight_kg=35)
        score, _ = compute_risk_score(interactions, alerts, contras, patient)
        assert score <= 100.0

    def test_elderly_adds_modifier(self):
        patient_young = PatientHistory(age=30)
        patient_old = PatientHistory(age=80)
        score_young, breakdown_young = compute_risk_score([], [], [], patient_young)
        score_old, breakdown_old = compute_risk_score([], [], [], patient_old)
        assert breakdown_old.age_weight_modifier > breakdown_young.age_weight_modifier


# ─────────────────────────────────────────────
# FALLBACK ENGINE TESTS
# ─────────────────────────────────────────────

class TestFallbackEngine:
    def test_warfarin_aspirin_interaction_detected(self):
        interactions = _fallback_interactions(["Warfarin", "Aspirin"], [])
        assert any(
            {i.drug_a.lower(), i.drug_b.lower()} == {"warfarin", "aspirin"}
            for i in interactions
        )

    def test_current_med_interaction_detected(self):
        """Proposed drug + current medication interaction should be found."""
        interactions = _fallback_interactions(["Aspirin"], ["Warfarin"])
        assert any(
            {i.drug_a.lower(), i.drug_b.lower()} == {"warfarin", "aspirin"}
            for i in interactions
        )

    def test_no_false_positives_for_safe_combo(self):
        interactions = _fallback_interactions(["Paracetamol", "Cetirizine"], [])
        assert interactions == []

    def test_order_independent_detection(self):
        i1 = _fallback_interactions(["Warfarin", "Aspirin"], [])
        i2 = _fallback_interactions(["Aspirin", "Warfarin"], [])
        pairs1 = {tuple(sorted([i.drug_a.lower(), i.drug_b.lower()])) for i in i1}
        pairs2 = {tuple(sorted([i.drug_a.lower(), i.drug_b.lower()])) for i in i2}
        assert pairs1 == pairs2


# ─────────────────────────────────────────────
# INPUT VALIDATION TESTS
# ─────────────────────────────────────────────

class TestInputValidation:
    def test_empty_medicines_rejected(self, client):
        resp = client.post("/check", json={"proposed_medicines": []})
        assert resp.status_code == 422

    def test_negative_age_rejected(self, client):
        resp = client.post("/check", json={
            "proposed_medicines": ["Aspirin"],
            "patient_history": {"age": -5}
        })
        assert resp.status_code == 422

    def test_duplicate_medicines_deduplicated(self, client):
        resp = client.post("/check", json={
            "proposed_medicines": ["Aspirin", "Aspirin", "aspirin"],
        })
        assert resp.status_code == 200

    def test_valid_request_returns_200(self, client):
        resp = client.post("/check", json={
            "proposed_medicines": ["Paracetamol"],
            "patient_history": {"age": 30, "conditions": []}
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "interactions" in data
        assert "allergy_alerts" in data
        assert "cache_hit" in data
        assert "processing_time_ms" in data
        assert isinstance(data["safe_to_prescribe"], bool)

    def test_cache_hit_on_second_request(self, client):
        payload = {
            "proposed_medicines": ["Metformin", "Lisinopril"],
            "patient_history": {"age": 55}
        }
        r1 = client.post("/check", json=payload)
        r2 = client.post("/check", json=payload)
        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r2.json()["cache_hit"] is True

    def test_health_endpoint(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_response_schema_completeness(self, client):
        """Every required field must be present in the response."""
        resp = client.post("/check", json={
            "proposed_medicines": ["Warfarin", "Aspirin"],
            "patient_history": {
                "current_medications": ["Ibuprofen"],
                "known_allergies": ["Penicillin"],
                "conditions": ["Hypertension"],
                "age": 60,
                "weight_kg": 75.0,
            }
        })
        assert resp.status_code == 200
        data = resp.json()
        required_fields = [
            "interactions", "allergy_alerts", "safe_to_prescribe",
            "overall_risk_level", "requires_doctor_review", "source",
            "cache_hit", "processing_time_ms"
        ]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"


# ─────────────────────────────────────────────
# CLINICAL SCENARIO TESTS
# ─────────────────────────────────────────────

class TestClinicalScenarios:
    def test_penicillin_allergy_blocks_amoxicillin(self, client):
        """Core clinical safety: Penicillin allergy must block Amoxicillin."""
        resp = client.post("/check", json={
            "proposed_medicines": ["Amoxicillin"],
            "patient_history": {
                "known_allergies": ["Penicillin"],
                "age": 35,
            }
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["allergy_alerts"]) > 0
        assert data["safe_to_prescribe"] is False

    def test_warfarin_aspirin_flagged_as_high(self, client):
        """Warfarin + Aspirin is a clinically significant interaction."""
        resp = client.post("/check", json={
            "proposed_medicines": ["Warfarin", "Aspirin"],
        })
        assert resp.status_code == 200
        data = resp.json()
        high_interactions = [i for i in data["interactions"] if i["severity"] == "high"]
        assert len(high_interactions) > 0

    def test_ckd_patient_nsaid_contraindicated(self, client):
        """CKD patient should receive contraindication warning for NSAIDs."""
        resp = client.post("/check", json={
            "proposed_medicines": ["Ibuprofen"],
            "patient_history": {
                "conditions": ["Chronic kidney disease"],
                "age": 55,
            }
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data.get("contraindication_alerts", [])) > 0
