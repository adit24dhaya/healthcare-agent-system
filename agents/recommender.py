import os

from openai import OpenAI


class RecommendationAgent:
    def __init__(self, model_name="gpt-4o-mini"):
        self.model_name = model_name
        self.client = OpenAI() if os.getenv("OPENAI_API_KEY") else None

    def recommend(self, patient_data, risk, retrieved_context=None, similar_cases=None, safety=None):
        if self.client is None:
            return self._fallback_recommendation(patient_data, risk, retrieved_context, similar_cases, safety)

        prompt = f"""
        Patient data: {patient_data}
        Risk level: {risk}
        Retrieved medical context: {retrieved_context}
        Similar prior cases: {similar_cases}
        Safety assessment: {safety}

        Suggest actionable health recommendations.
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as exc:
            return f"{self._fallback_recommendation(patient_data, risk, retrieved_context, similar_cases, safety)} LLM recommendation unavailable: {exc}"

    def _fallback_recommendation(self, patient_data, risk, retrieved_context=None, similar_cases=None, safety=None):
        glucose_note = ""
        if not patient_data.get("glucose_measured", True):
            glucose_note = " Because glucose was not measured, get an A1C or fasting blood glucose test before making clinical decisions."

        if risk == "High":
            recommendation = "Recommend prompt clinician review, repeat vitals/labs, and a care plan focused on blood pressure, glucose, nutrition, activity, and medication adherence."
        elif risk == "Medium":
            recommendation = "Recommend follow-up monitoring, lifestyle coaching, and reviewing blood pressure and glucose trends with a clinician."
        else:
            recommendation = "Recommend routine preventive care, healthy activity, balanced nutrition, and periodic monitoring."

        context_note = self._context_note(retrieved_context)
        memory_note = ""
        if similar_cases:
            memory_note = " Compare against similar historical cases before changing the care plan."
        safety_note = self._safety_note(safety)

        return recommendation + glucose_note + context_note + memory_note + safety_note

    def _context_note(self, retrieved_context):
        if not retrieved_context:
            return ""

        titles = ", ".join(item["title"] for item in retrieved_context[:2])
        return f" Retrieved guidance used: {titles}."

    def _safety_note(self, safety):
        if not safety:
            return ""
        alerts = safety.get("alerts", [])
        escalation = safety.get("escalation", "routine_followup")
        if not alerts and escalation == "routine_followup":
            return ""
        alert_text = " ".join(alerts[:2])
        return f" Safety escalation: {escalation}. {alert_text}"
