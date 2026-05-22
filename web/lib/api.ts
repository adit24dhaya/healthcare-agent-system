import type { HistoryResponse, PatientInput, PredictionResult } from "@/types/healthcare";

export async function createPrediction(input: PatientInput): Promise<PredictionResult> {
  const response = await fetch("/api/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(input),
  });

  const payload = await response.json();

  if (!response.ok) {
    const detail = typeof payload?.detail === "string" ? payload.detail : "Prediction failed.";
    throw new Error(detail);
  }

  return payload as PredictionResult;
}

export async function getHistory(limit = 25): Promise<HistoryResponse> {
  const response = await fetch(`/api/history?limit=${limit}`, {
    method: "GET",
  });
  const payload = await response.json();

  if (!response.ok) {
    const detail = typeof payload?.detail === "string" ? payload.detail : "Unable to load history.";
    throw new Error(detail);
  }

  return payload as HistoryResponse;
}
