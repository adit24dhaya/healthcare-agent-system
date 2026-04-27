import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import agent


app = FastAPI(title="Healthcare Agent API")


class PatientRequest(BaseModel):
    age: int = Field(..., ge=0, le=120)
    height_cm: Optional[float] = Field(None, ge=50, le=250)
    weight_kg: Optional[float] = Field(None, ge=2, le=300)
    bmi: Optional[float] = Field(None, ge=0, le=80)
    bp: int = Field(..., ge=0, le=260)
    glucose: Optional[float] = Field(None, ge=0, le=500)


class ChatRequest(BaseModel):
    message: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(patient: PatientRequest):
    return agent.run(patient.model_dump())


@app.get("/history")
def history(limit: int = 25):
    return {"records": agent.memory.get_all(limit=limit)}


@app.post("/chat")
def chat(request: ChatRequest):
    return {
        "response": (
            "This prototype can analyze a patient profile, explain model drivers, "
            "retrieve basic medical context, and compare against stored prior cases. "
            "For clinical decisions, use measured values and clinician review."
        ),
        "message": request.message,
    }
