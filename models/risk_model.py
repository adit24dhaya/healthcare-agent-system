import pandas as pd
from sklearn.linear_model import LogisticRegression


class RiskModel:
    FEATURE_NAMES = ["age", "bmi", "bp", "glucose"]
    INPUT_NAMES = ["age", "height_cm", "weight_kg", "bp", "glucose"]

    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)
        self.is_trained = False
        self.default_glucose = None
        self.training_frame = None

    @staticmethod
    def calculate_bmi(height_cm, weight_kg):
        height_m = height_cm / 100
        if height_m <= 0:
            raise ValueError("Height must be greater than 0.")
        return weight_kg / (height_m**2)

    def train(self, df):
        df = df.copy()
        if "bmi" not in df.columns:
            df["bmi"] = df.apply(
                lambda row: self.calculate_bmi(row["height_cm"], row["weight_kg"]), axis=1
            )

        self.default_glucose = float(df["glucose"].median())
        X = df[self.FEATURE_NAMES]
        y = df["target"]
        self.model.fit(X, y)
        self.training_frame = X.copy()
        self.is_trained = True

    def train_from_csv(self, path):
        df = pd.read_csv(path)
        self.train(df)

    def assess(self, patient_data):
        if not self.is_trained:
            raise RuntimeError("RiskModel must be trained before prediction.")

        patient = self.prepare_patient(patient_data)
        row = self.feature_frame(patient)

        prob = self.model.predict_proba(row)[0][1]
        return float(prob), patient

    def predict(self, patient_data):
        prob, _patient = self.assess(patient_data)
        return prob

    def prepare_patient(self, patient_data):
        if isinstance(patient_data, dict):
            patient = patient_data.copy()
        else:
            patient = dict(zip(self.INPUT_NAMES, patient_data))

        has_height_weight = "height_cm" in patient and "weight_kg" in patient
        has_bmi = "bmi" in patient
        required_fields = ["age", "bp"]
        missing = [field for field in required_fields if field not in patient]
        if not has_height_weight and not has_bmi:
            missing.append("height_cm and weight_kg or bmi")
        if missing:
            raise ValueError(f"Missing required patient fields: {', '.join(missing)}")

        glucose_measured = patient.get("glucose") is not None
        glucose = float(patient["glucose"]) if glucose_measured else self.default_glucose
        bmi = (
            self.calculate_bmi(float(patient["height_cm"]), float(patient["weight_kg"]))
            if has_height_weight
            else float(patient["bmi"])
        )

        return {
            "age": int(patient["age"]),
            "height_cm": float(patient["height_cm"]) if "height_cm" in patient else None,
            "weight_kg": float(patient["weight_kg"]) if "weight_kg" in patient else None,
            "bmi": round(bmi, 1),
            "bp": int(patient["bp"]),
            "glucose": glucose,
            "glucose_measured": glucose_measured,
        }

    def feature_frame(self, patient):
        return pd.DataFrame(
            [
                {
                    "age": patient["age"],
                    "bmi": patient["bmi"],
                    "bp": patient["bp"],
                    "glucose": patient["glucose"],
                }
            ],
            columns=self.FEATURE_NAMES,
        )
