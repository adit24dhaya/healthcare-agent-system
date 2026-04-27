from tools.risk_classifier import classify_risk


class Orchestrator:
    def __init__(
        self,
        model,
        explanation_agent,
        recommendation_agent,
        memory_agent,
        feature_explainer=None,
        retrieval_agent=None,
    ):
        self.model = model
        self.explainer = explanation_agent
        self.recommender = recommendation_agent
        self.memory = memory_agent
        self.feature_explainer = feature_explainer
        self.retriever = retrieval_agent

    def run(self, patient_data):
        prob, patient = self.model.assess(patient_data)
        risk = classify_risk(prob)
        feature_explanation = (
            self.feature_explainer.explain(patient)
            if self.feature_explainer is not None
            else {"method": "none", "features": []}
        )
        similar_cases = self.memory.find_similar(patient)
        retrieved_context = (
            self.retriever.retrieve(patient, risk, feature_explanation)
            if self.retriever is not None
            else []
        )

        explanation = self.explainer.explain(
            patient,
            prob,
            feature_explanation=feature_explanation,
            similar_cases=similar_cases,
        )
        recommendation = self.recommender.recommend(
            patient,
            risk,
            retrieved_context=retrieved_context,
            similar_cases=similar_cases,
        )

        self.memory.store(patient, prob, risk)

        return {
            "probability": prob,
            "risk": risk,
            "patient": patient,
            "feature_explanation": feature_explanation,
            "retrieved_context": retrieved_context,
            "similar_cases": similar_cases,
            "explanation": explanation,
            "recommendation": recommendation
        }
