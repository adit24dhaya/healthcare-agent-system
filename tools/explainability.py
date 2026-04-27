import numpy as np


class FeatureExplainer:
    def __init__(self, risk_model):
        self.risk_model = risk_model
        self.method = "coefficients"
        self._shap_explainer = None
        self._build_shap_explainer()

    def _build_shap_explainer(self):
        try:
            import shap

            if self.risk_model.training_frame is None:
                return

            self._shap_explainer = shap.LinearExplainer(
                self.risk_model.model,
                self.risk_model.training_frame,
            )
            self.method = "shap"
        except Exception:
            self._shap_explainer = None
            self.method = "coefficients"

    def explain(self, patient):
        row = self.risk_model.feature_frame(patient)
        values = self._shap_values(row)
        if values is None:
            values = self._coefficient_values(row)

        impacts = []
        for feature, value, impact in zip(self.risk_model.FEATURE_NAMES, row.iloc[0], values):
            impacts.append({
                "feature": feature,
                "value": float(value),
                "impact": float(impact),
                "direction": "raises risk" if impact > 0 else "lowers risk",
                "magnitude": abs(float(impact)),
            })

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

            if values.shape[0] != len(self.risk_model.FEATURE_NAMES):
                return None

            return values
        except Exception:
            return None

    def _coefficient_values(self, row):
        coefficients = self.risk_model.model.coef_[0]
        means = self.risk_model.training_frame.mean().to_numpy()
        centered = row.iloc[0].to_numpy() - means
        return coefficients * centered
