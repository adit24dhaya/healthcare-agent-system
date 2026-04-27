import pandas as pd

from models.risk_model import RiskModel


def _training_df():
    return pd.DataFrame(
        [
            {"age": 30, "height_cm": 175, "weight_kg": 70, "bp": 118, "glucose": 90, "target": 0},
            {"age": 62, "height_cm": 165, "weight_kg": 88, "bp": 160, "glucose": 220, "target": 1},
            {"age": 48, "height_cm": 170, "weight_kg": 82, "bp": 138, "glucose": 150, "target": 1},
            {"age": 34, "height_cm": 180, "weight_kg": 72, "bp": 122, "glucose": 95, "target": 0},
        ]
    )


def test_calculate_bmi():
    bmi = RiskModel.calculate_bmi(175, 70)
    assert round(bmi, 1) == 22.9


def test_assess_with_missing_glucose_uses_default():
    model = RiskModel()
    model.train(_training_df())
    prob, patient = model.assess(
        {"age": 45, "height_cm": 172, "weight_kg": 80, "bp": 130, "glucose": None}
    )
    assert 0 <= prob <= 1
    assert patient["glucose_measured"] is False
    assert patient["glucose"] == model.default_glucose
