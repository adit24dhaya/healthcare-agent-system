import json
from pathlib import Path

from agents.explainer import ExplanationAgent
from agents.orchestrator import Orchestrator
from agents.recommender import RecommendationAgent
from agents.retriever import RetrievalAgent
from memory.store import MemoryAgent
from models.risk_model import RiskModel
from tools.explainability import FeatureExplainer
from tools.safety import SafetyGuard

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "patients.csv"
KNOWLEDGE_PATH = BASE_DIR / "data" / "medical_knowledge.jsonl"
CHROMA_PATH = BASE_DIR / "data" / "chroma"


def build_agent():
    model = RiskModel()
    model.train_from_csv(DATA_PATH)

    explainer = ExplanationAgent()
    recommender = RecommendationAgent()
    memory = MemoryAgent(CHROMA_PATH)
    feature_explainer = FeatureExplainer(model)
    retriever = RetrievalAgent(KNOWLEDGE_PATH, CHROMA_PATH)
    safety_guard = SafetyGuard()

    return Orchestrator(
        model,
        explainer,
        recommender,
        memory,
        feature_explainer=feature_explainer,
        retrieval_agent=retriever,
        safety_guard=safety_guard,
    )


agent = build_agent()


if __name__ == "__main__":
    patient = {
        "age": 45,
        "height_cm": 170,
        "weight_kg": 82.4,
        "bp": 130,
        "glucose": 180,
    }
    result = agent.run(patient)
    print(json.dumps(result, indent=2))
