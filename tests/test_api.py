from fastapi.testclient import TestClient

from api.app import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_v1_health_endpoint():
    response = client.get("/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == "v1"


def test_predict_summary_contains_safety():
    payload = {
        "age": 58,
        "height_cm": 168,
        "weight_kg": 92,
        "bp": 172,
        "glucose": 245,
    }
    response = client.post("/predict/summary", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["risk"] in {"Low", "Medium", "High"}
    assert 0 <= body["probability"] <= 1
    assert body["confidence"] in {"Low", "Medium", "High"}
    assert body["escalation"] in {
        "routine_followup",
        "prompt_clinician_followup",
        "urgent_clinician_review",
    }


def test_predict_summary_v1_contains_request_id():
    payload = {
        "age": 42,
        "height_cm": 172,
        "weight_kg": 79,
        "bp": 130,
        "glucose": 110,
    }
    response = client.post("/v1/predict/summary", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["request_id"], str)
    assert len(body["request_id"]) > 8
