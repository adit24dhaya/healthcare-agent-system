import hashlib
import math
import re


class LocalEmbeddingModel:
    def __init__(self, dimensions=48):
        self.dimensions = dimensions

    def embed_text(self, text):
        vector = [0.0] * self.dimensions
        tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:2], "big") % self.dimensions
            sign = 1.0 if digest[2] % 2 == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector

        return [value / norm for value in vector]

    def embed_patient(self, patient):
        text = (
            f"age {patient.get('age')} bmi {patient.get('bmi')} "
            f"blood pressure {patient.get('bp')} glucose {patient.get('glucose')} "
            f"glucose measured {patient.get('glucose_measured')}"
        )
        return self.embed_text(text)
