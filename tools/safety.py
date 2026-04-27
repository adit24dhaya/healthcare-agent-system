class SafetyGuard:
    def assess(self, patient, probability, risk):
        alerts = []
        escalation = "routine_followup"

        if patient["bp"] >= 180:
            alerts.append(
                "Severely elevated blood pressure; urgent same-day clinical assessment is recommended."
            )
            escalation = "urgent_clinician_review"
        elif patient["bp"] >= 160:
            alerts.append(
                "Very high blood pressure detected; prompt clinician follow-up is recommended."
            )
            escalation = "prompt_clinician_followup"

        if patient["glucose"] >= 300:
            alerts.append("Very high glucose value; urgent medical evaluation is recommended.")
            escalation = "urgent_clinician_review"
        elif patient["glucose"] >= 200:
            alerts.append(
                "High glucose value detected; prompt confirmatory testing and clinician review are recommended."
            )
            if escalation == "routine_followup":
                escalation = "prompt_clinician_followup"

        if not patient.get("glucose_measured", True):
            alerts.append("Glucose was not measured; risk estimate confidence is reduced.")

        if risk == "High" and probability >= 0.85 and escalation == "routine_followup":
            escalation = "prompt_clinician_followup"
            alerts.append(
                "Model indicates very high predicted risk; expedited clinician review is advised."
            )

        confidence_score, confidence_label = self._confidence(patient)
        disclaimers = [
            "Educational prototype only; not a diagnostic medical device.",
            "Clinical decisions require clinician judgment and measured labs/vitals.",
        ]

        return {
            "alerts": alerts,
            "escalation": escalation,
            "confidence_score": confidence_score,
            "confidence_label": confidence_label,
            "disclaimers": disclaimers,
        }

    def _confidence(self, patient):
        # Simple reliability heuristic based on missingness and plausibility.
        score = 1.0
        if not patient.get("glucose_measured", True):
            score -= 0.25
        if patient["bp"] <= 70 or patient["bp"] >= 220:
            score -= 0.15
        if patient["bmi"] <= 14 or patient["bmi"] >= 50:
            score -= 0.1
        score = max(0.0, min(1.0, score))

        if score >= 0.85:
            label = "High"
        elif score >= 0.65:
            label = "Medium"
        else:
            label = "Low"
        return round(score, 2), label
