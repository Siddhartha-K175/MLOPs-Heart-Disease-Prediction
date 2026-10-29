"""General prevention guidance for elevated heart-disease risk predictions.

Educational only: no drug names, no dosages, no diagnosis — a qualified clinician
decides medication. Consumed by the Streamlit app and the single-prediction API route.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

HIGH_THRESHOLD = 0.5
MODERATE_THRESHOLD = 0.25

DISCLAIMER = (
    "General health information only — not a diagnosis or a prescription. "
    "Only a qualified clinician can decide medication for you."
)

EMERGENCY_NOTE = (
    "Seek emergency care now — do not wait — for chest pain or pressure, pain spreading to "
    "the arm or jaw, sudden shortness of breath, fainting, or stroke signs (face drooping, "
    "arm weakness, speech trouble)."
)

MEDICATION_NOTE = (
    "If you already take medicine for blood pressure, cholesterol, or blood clotting, take it "
    "exactly as prescribed, at the same time every day, and never stop or adjust it on your "
    "own — ask the prescriber first."
)

SEEK_CARE = {
    "high": "Book an appointment with a clinician within the next few days to review this "
    "result together with your history and current treatment.",
    "moderate": "Mention this result at your next check-up (within a few months) and ask for "
    "blood pressure and cholesterol to be re-checked.",
    "low": "No urgent action needed — keep your routine check-ups.",
}

_GENERAL_TIPS = {
    "high": [
        "Keep salt under about 5 g a day: limit pickles, packaged and processed foods, and "
        "skip adding salt at the table.",
        "Follow a heart-healthy plate: vegetables, fruit, whole grains, legumes, fish or "
        "lean protein; cut fried food, red meat, sweets, and trans fats.",
        "Aim for about 150 minutes a week of moderate activity (brisk walking suits most "
        "people) once a clinician confirms it is safe for you.",
        "If you smoke or use tobacco, quitting is the single most effective step to slow "
        "this condition — ask about a structured quitting plan.",
        "Keep alcohol low, sleep 7–8 hours, manage stress, and if you have diabetes, keep "
        "blood sugar within the target your doctor set.",
    ],
    "moderate": [
        "Adopt the heart-healthy plate most days: more vegetables and whole grains, less "
        "fried food, red meat, and sweets.",
        "Build up to about 150 minutes a week of moderate activity such as brisk walking.",
        "If you smoke or use tobacco, ask about a quitting plan — it helps at any risk level.",
        "Keep salt and alcohol low, and aim for a healthy weight and 7–8 hours of sleep.",
    ],
    "low": [
        "Your readings sit in the low range — keep the same habits: balanced meals, regular "
        "activity, no tobacco, moderate alcohol, and routine check-ups.",
    ],
}


def risk_tier(probability: float) -> str:
    """Map a risk probability to a guidance tier."""
    if probability >= HIGH_THRESHOLD:
        return "high"
    if probability >= MODERATE_THRESHOLD:
        return "moderate"
    return "low"


def factor_tips(patient: Mapping[str, float]) -> list[str]:
    """Advice tied to the patient's own out-of-range measurements, if any."""
    tips: list[str] = []
    blood_pressure = float(patient.get("trestbps", 0))
    if blood_pressure >= 140:
        tips.append(
            f"Resting blood pressure is high ({blood_pressure:.0f} mm Hg): re-check it at "
            "home on several days and discuss salt reduction and treatment options with "
            "your clinician."
        )
    elif blood_pressure >= 130:
        tips.append(
            f"Resting blood pressure is elevated ({blood_pressure:.0f} mm Hg): measure it "
            "at home a few times this week and share the readings at your next visit."
        )
    cholesterol = float(patient.get("chol", 0))
    if cholesterol >= 240:
        tips.append(
            f"Total cholesterol is high ({cholesterol:.0f}): ask for a lipid test and a "
            "diet review — foods lower in saturated and trans fat help bring it down."
        )
    if float(patient.get("fbs", 0)) >= 1:
        tips.append(
            "Fasting blood sugar is above the usual target: ask for an HbA1c test and "
            "review carbohydrate intake with a clinician or dietitian."
        )
    if float(patient.get("exang", 0)) >= 1 or float(patient.get("oldpeak", 0)) >= 2:
        tips.append(
            "Exercise triggered symptoms or ECG changes: describe exactly when they occur "
            "and get cleared before increasing exercise intensity."
        )
    if float(patient.get("ca", 0)) >= 2:
        tips.append(
            "Several major vessels are affected: a cardiologist should review your "
            "treatment plan and follow-up schedule."
        )
    if float(patient.get("age", 0)) >= 60:
        tips.append(
            "Age is a fixed risk factor: keep every scheduled check-up even when you feel "
            "well, and report any new chest discomfort promptly."
        )
    return tips


def recommendations(probability: float, patient: Mapping[str, float]) -> dict[str, Any]:
    """Return tiered prevention suggestions for a scored patient.

    Keys: tier, headline, actions, emergency_note, disclaimer. `actions` merges the
    tier's general lifestyle tips with factor-specific tips (and, for moderate/high
    tiers, the medication-adherence reminder and seek-care step). General health
    education only — never a diagnosis or a prescription.
    """
    tier = risk_tier(probability)
    headlines = {
        "high": f"Estimated risk is high ({probability:.0%}). Take steps now to protect "
        "your heart and see a clinician soon.",
        "moderate": f"Estimated risk is moderate ({probability:.0%}). A few changes and a "
        "check-up can keep this from progressing.",
        "low": f"Estimated risk is low ({probability:.0%}). Keep up the habits that are "
        "working.",
    }
    actions = [*_GENERAL_TIPS[tier], *factor_tips(patient)]
    if tier != "low":
        actions += [MEDICATION_NOTE, SEEK_CARE[tier]]
    return {
        "tier": tier,
        "headline": headlines[tier],
        "actions": actions,
        "emergency_note": EMERGENCY_NOTE,
        "disclaimer": DISCLAIMER,
    }
