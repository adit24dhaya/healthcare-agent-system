import sys
from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import get_settings
from main import agent
from tools.audit_logger import AuditLogger

app = FastAPI(title="Healthcare Agent API")
settings = get_settings()
audit_logger = AuditLogger(settings.log_dir)


class PatientRequest(BaseModel):
    age: int = Field(..., ge=0, le=120)
    height_cm: Optional[float] = Field(None, ge=50, le=250)
    weight_kg: Optional[float] = Field(None, ge=2, le=300)
    bmi: Optional[float] = Field(None, ge=0, le=80)
    bp: int = Field(..., ge=0, le=260)
    glucose: Optional[float] = Field(None, ge=0, le=500)


class ChatRequest(BaseModel):
    message: str


def _authorize(authorization: Optional[str] = Header(default=None)):
    if not settings.require_api_token:
        return
    if not settings.api_token:
        raise HTTPException(status_code=500, detail="API token is required but not configured.")
    expected = f"Bearer {settings.api_token}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(patient: PatientRequest, _auth=Depends(_authorize)):
    result = agent.run(patient.model_dump())
    audit_logger.log_decision(str(uuid4()), result["patient"], result)
    return result


@app.post("/predict/summary")
def predict_summary(patient: PatientRequest, _auth=Depends(_authorize)):
    result = agent.run(patient.model_dump())
    audit_logger.log_decision(str(uuid4()), result["patient"], result)
    return {
        "risk": result["risk"],
        "probability": result["probability"],
        "confidence": result.get("safety", {}).get("confidence_label"),
        "confidence_score": result.get("safety", {}).get("confidence_score"),
        "escalation": result.get("safety", {}).get("escalation"),
        "alerts": result.get("safety", {}).get("alerts", []),
    }


@app.get("/history")
def history(limit: int = 25, _auth=Depends(_authorize)):
    return {"records": agent.memory.get_all(limit=limit)}


@app.post("/chat")
def chat(request: ChatRequest, _auth=Depends(_authorize)):
    return {
        "response": (
            "This prototype can analyze a patient profile, explain model drivers, "
            "retrieve basic medical context, and compare against stored prior cases. "
            "For clinical decisions, use measured values and clinician review."
        ),
        "message": request.message,
    }


@app.get("/v1/health")
def health_v1():
    return {
        "status": "ok",
        "version": "v1",
        "auth_required": settings.require_api_token,
    }


@app.post("/v1/predict")
def predict_v1(patient: PatientRequest, _auth=Depends(_authorize)):
    result = agent.run(patient.model_dump())
    request_id = str(uuid4())
    audit_logger.log_decision(request_id, result["patient"], result)
    result["request_id"] = request_id
    return result


@app.post("/v1/predict/summary")
def predict_summary_v1(patient: PatientRequest, _auth=Depends(_authorize)):
    result = agent.run(patient.model_dump())
    request_id = str(uuid4())
    audit_logger.log_decision(request_id, result["patient"], result)
    return {
        "request_id": request_id,
        "risk": result["risk"],
        "probability": result["probability"],
        "confidence": result.get("safety", {}).get("confidence_label"),
        "confidence_score": result.get("safety", {}).get("confidence_score"),
        "escalation": result.get("safety", {}).get("escalation"),
        "alerts": result.get("safety", {}).get("alerts", []),
    }
