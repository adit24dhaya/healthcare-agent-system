import os

from openai import OpenAI


class ExplanationAgent:
    def __init__(self, model_name="gpt-4o-mini"):
        self.model_name = model_name
        self.client = OpenAI() if os.getenv("OPENAI_API_KEY") else None

    def explain(self, patient_data, prob, feature_explanation=None, similar_cases=None):
        if self.client is None:
            return self._fallback_explanation(
                patient_data, prob, feature_explanation, similar_cases
            )

        prompt = f"""
        Patient data: {patient_data}
        Risk probability: {prob}
        Feature explanation: {feature_explanation}
        Similar prior cases: {similar_cases}

        Explain in simple terms why this patient has this risk.
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model_name, messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as exc:
            return f"{self._fallback_explanation(patient_data, prob, feature_explanation, similar_cases)} LLM explanation unavailable: {exc}"

    def _fallback_explanation(
        self, patient_data, prob, feature_explanation=None, similar_cases=None
    ):
        age = patient_data["age"]
        bmi = patient_data["bmi"]
        bp = patient_data["bp"]
        glucose = patient_data["glucose"]
        factors = []

        if age >= 55:
            factors.append("older age")
        if bmi >= 30:
            factors.append("elevated BMI")
        if bp >= 140:
            factors.append("high blood pressure")
        if glucose >= 140:
            factors.append("high glucose")
        if not patient_data.get("glucose_measured", True):
            factors.append("glucose was not measured, so the estimate is less certain")

        if not factors:
            factors.append("the measured values are mostly in lower-risk ranges")

        factor_text = ", ".join(factors)
        top_impacts = self._top_feature_text(feature_explanation)
        memory_text = ""
        if similar_cases:
            memory_text = (
                f" I found {len(similar_cases)} similar prior case(s) in memory for comparison."
            )

        return (
            f"The model estimated a {prob:.1%} risk using a calculated BMI of {bmi:.1f}. "
            f"Key factors: {factor_text}. {top_impacts}{memory_text}"
        )

    def _top_feature_text(self, feature_explanation):
        if not feature_explanation or not feature_explanation.get("features"):
            return ""

        top_features = feature_explanation["features"][:3]
        parts = [f"{item['feature']} {item['direction']}" for item in top_features]
        method = feature_explanation.get("method", "model")
        return f"The {method.upper()} explanation highlights: {', '.join(parts)}. "
