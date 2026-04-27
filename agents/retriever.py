import json
from pathlib import Path

from tools.chroma_client import create_persistent_client
from tools.local_embeddings import LocalEmbeddingModel


class RetrievalAgent:
    def __init__(self, knowledge_path, chroma_path):
        self.knowledge_path = Path(knowledge_path)
        self.embedding_model = LocalEmbeddingModel()
        self.client = create_persistent_client(chroma_path)
        self.collection = self.client.get_or_create_collection("medical_knowledge")
        self._seed_knowledge()

    def retrieve(self, patient_data, risk, feature_explanation, n_results=3):
        query = self._query_text(patient_data, risk, feature_explanation)
        if self.collection.count() == 0:
            return []

        query_count = min(max(n_results + 2, 5), self.collection.count())
        results = self.collection.query(
            query_embeddings=[self.embedding_model.embed_text(query)],
            n_results=query_count,
            include=["documents", "metadatas", "distances"],
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        contexts = []
        for document, metadata, distance in zip(documents, metadatas, distances):
            contexts.append(
                {
                    "title": metadata.get("title", "Medical note"),
                    "text": document,
                    "distance": float(distance),
                }
            )
        return self._rerank(contexts, patient_data, risk)[:n_results]

    def _seed_knowledge(self):
        if not self.knowledge_path.exists():
            return

        existing = set(self.collection.get().get("ids", []))
        ids = []
        documents = []
        metadatas = []
        embeddings = []

        with self.knowledge_path.open("r", encoding="utf-8") as file:
            for line in file:
                item = json.loads(line)
                item_id = item["id"]
                if item_id in existing:
                    continue

                text = item["text"]
                ids.append(item_id)
                documents.append(text)
                metadatas.append({"title": item["title"]})
                embeddings.append(self.embedding_model.embed_text(f"{item['title']} {text}"))

        if ids:
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )

    def _query_text(self, patient_data, risk, feature_explanation):
        top_features = ", ".join(
            f"{item['feature']} {item['direction']}"
            for item in feature_explanation.get("features", [])[:3]
        )
        return (
            f"patient risk {risk}. age {patient_data.get('age')}, bmi {patient_data.get('bmi')}, "
            f"blood pressure {patient_data.get('bp')}, glucose {patient_data.get('glucose')}. "
            f"top features: {top_features}"
        )

    def _rerank(self, contexts, patient_data, risk):
        def score(item):
            title = item["title"]
            value = -item["distance"]

            if title == "High risk clinical review":
                value += 2.0 if risk == "High" else -3.0
            if title == "Glucose testing":
                if (
                    not patient_data.get("glucose_measured", True)
                    or patient_data.get("glucose", 0) >= 140
                ):
                    value += 1.5
            if title == "Blood pressure follow-up" and patient_data.get("bp", 0) >= 130:
                value += 1.25
            if title == "BMI context" and patient_data.get("bmi", 0) >= 25:
                value += 1.0
            if title == "Lifestyle foundations":
                value += 0.5

            return value

        return sorted(contexts, key=score, reverse=True)
