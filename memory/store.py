import json
from datetime import datetime, timezone
from uuid import uuid4

from tools.chroma_client import create_persistent_client
from tools.local_embeddings import LocalEmbeddingModel


class MemoryAgent:
    def __init__(self, chroma_path):
        self.embedding_model = LocalEmbeddingModel()
        self.client = create_persistent_client(chroma_path)
        self.collection = self.client.get_or_create_collection("patient_memory")

    def store(self, patient_data, prob, risk):
        record_id = f"patient-{uuid4()}"
        timestamp = datetime.now(timezone.utc).isoformat()
        document = self._patient_summary(patient_data, prob, risk)
        metadata = self._metadata(patient_data, prob, risk, timestamp)

        self.collection.add(
            ids=[record_id],
            documents=[document],
            metadatas=[metadata],
            embeddings=[self.embedding_model.embed_text(document)],
        )

    def get_all(self, limit=25):
        results = self.collection.get(include=["documents", "metadatas"])
        records = []

        for record_id, document, metadata in zip(
            results.get("ids", []),
            results.get("documents", []),
            results.get("metadatas", []),
        ):
            records.append(
                {
                    "id": record_id,
                    "summary": document,
                    "metadata": metadata,
                }
            )

        records.sort(key=lambda item: item["metadata"].get("timestamp", ""), reverse=True)
        return records[:limit]

    def find_similar(self, patient_data, n_results=3):
        count = self.collection.count()
        if count == 0:
            return []

        query = self._patient_summary(patient_data, prob=None, risk=None)
        results = self.collection.query(
            query_embeddings=[self.embedding_model.embed_text(query)],
            n_results=min(n_results, count),
            include=["documents", "metadatas", "distances"],
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        matches = []
        for document, metadata, distance in zip(documents, metadatas, distances):
            matches.append(
                {
                    "summary": document,
                    "metadata": metadata,
                    "distance": float(distance),
                }
            )
        return matches

    def _patient_summary(self, patient_data, prob, risk):
        probability = "unknown" if prob is None else f"{prob:.3f}"
        risk_text = "unknown" if risk is None else risk
        return (
            f"age {patient_data.get('age')}, bmi {patient_data.get('bmi')}, "
            f"blood pressure {patient_data.get('bp')}, glucose {patient_data.get('glucose')}, "
            f"glucose measured {patient_data.get('glucose_measured')}, "
            f"risk {risk_text}, probability {probability}"
        )

    def _metadata(self, patient_data, prob, risk, timestamp):
        metadata = {
            "timestamp": timestamp,
            "risk": risk,
            "probability": float(prob),
            "age": int(patient_data.get("age")),
            "bmi": float(patient_data.get("bmi")),
            "bp": int(patient_data.get("bp")),
            "glucose": float(patient_data.get("glucose")),
            "glucose_measured": bool(patient_data.get("glucose_measured")),
            "patient_json": json.dumps(patient_data),
        }

        if patient_data.get("height_cm") is not None:
            metadata["height_cm"] = float(patient_data["height_cm"])
        if patient_data.get("weight_kg") is not None:
            metadata["weight_kg"] = float(patient_data["weight_kg"])

        return metadata
