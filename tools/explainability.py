import numpy as np


class FeatureExplainer:
    def __init__(self, risk_model):
        self.risk_model = risk_model
        self.method = "sensitivity"
        self._shap_explainer = None
        self._build_shap_explainer()

    def _build_shap_explainer(self):
        try:
            import shap

            if self.risk_model.training_frame is None:
                return
            if not hasattr(self.risk_model.model, "coef_"):
                return

            self._shap_explainer = shap.LinearExplainer(
                self.risk_model.model,
                self.risk_model.training_frame,
            )
            self.method = "shap"
        except Exception:
            self._shap_explainer = None
            self.method = "sensitivity"

    def explain(self, patient):
        row = self.risk_model.feature_frame(patient)
        values = self._shap_values(row)
        if values is None:
            values = self._sensitivity_values(row)

        impacts = []
        for feature, value, impact in zip(self.risk_model.feature_names, row.iloc[0], values):
            impacts.append(
                {
                    "feature": feature,
                    "value": float(value),
                    "impact": float(impact),
                    "direction": "raises risk" if impact > 0 else "lowers risk",
                    "magnitude": abs(float(impact)),
                }
            )

        impacts.sort(key=lambda item: item["magnitude"], reverse=True)
        return {
            "method": self.method,
            "features": impacts,
        }

    def _shap_values(self, row):
        if self._shap_explainer is None:
            return None

        try:
            raw_values = self._shap_explainer.shap_values(row)
            values = np.asarray(raw_values)

            if values.ndim == 3:
                values = values[:, :, -1]
            if values.ndim == 2:
                values = values[0]

            if values.shape[0] != len(self.risk_model.feature_names):
                return None

            return values
        except Exception:
            return None

    def _sensitivity_values(self, row):
        base_probability = self.risk_model.model.predict_proba(row)[0][1]
        values = []
        defaults = self.risk_model.default_values
        training_medians = self.risk_model.training_frame.median(numeric_only=True).to_dict()

        for feature in self.risk_model.feature_names:
            baseline_row = row.copy()
            baseline_value = defaults.get(
                feature, training_medians.get(feature, row.iloc[0][feature])
            )
            baseline_row.loc[:, feature] = baseline_value
            baseline_probability = self.risk_model.model.predict_proba(baseline_row)[0][1]
            values.append(base_probability - baseline_probability)

        return np.asarray(values)
