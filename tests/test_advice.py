from src.advice import DISCLAIMER, factor_tips, recommendations, risk_tier

HEALTHY = {"trestbps": 110, "chol": 180, "fbs": 0, "exang": 0, "oldpeak": 0, "ca": 0}
RISKY = {"trestbps": 165, "chol": 280, "fbs": 1, "exang": 1, "oldpeak": 2.5, "ca": 2}


def test_risk_tier_boundaries():
    assert risk_tier(0.8) == "high"
    assert risk_tier(0.5) == "high"
    assert risk_tier(0.49) == "moderate"
    assert risk_tier(0.25) == "moderate"
    assert risk_tier(0.24) == "low"


def test_high_risk_guidance_content():
    guidance = recommendations(0.8, RISKY)

    assert guidance["tier"] == "high"
    assert len(guidance["actions"]) >= 5
    joined = " ".join(guidance["actions"])
    assert "exactly as prescribed" in joined
    assert "clinician" in joined
    assert "Seek emergency care" in guidance["emergency_note"]
    assert guidance["disclaimer"] == DISCLAIMER
    assert "not a diagnosis" in guidance["disclaimer"]


def test_factor_tips_flag_elevated_measurements():
    joined = " ".join(factor_tips(RISKY))

    assert "blood pressure" in joined.lower()
    assert "cholesterol" in joined.lower()
    assert "blood sugar" in joined.lower()
    assert "vessels" in joined.lower()


def test_factor_tips_silent_for_healthy_measurements():
    assert factor_tips(HEALTHY) == []


def test_low_risk_guidance_is_light():
    guidance = recommendations(0.05, HEALTHY)

    assert guidance["tier"] == "low"
    assert len(guidance["actions"]) <= 3
    assert guidance["disclaimer"] == DISCLAIMER


def test_moderate_tier_between_thresholds():
    guidance = recommendations(0.35, HEALTHY)

    assert guidance["tier"] == "moderate"
    assert guidance["actions"]
