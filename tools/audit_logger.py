import json
from datetime import datetime, timezone
from pathlib import Path


class AuditLogger:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = self.output_dir / "decisions.jsonl"

    def log_decision(self, request_id, patient, result):
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "patient": {
                "age": patient.get("age"),
                "bmi": patient.get("bmi"),
                "bp": patient.get("bp"),
                "glucose": patient.get("glucose"),
                "glucose_measured": patient.get("glucose_measured"),
            },
            "result": {
                "risk": result.get("risk"),
                "probability": result.get("probability"),
                "confidence": result.get("safety", {}).get("confidence_label"),
                "escalation": result.get("safety", {}).get("escalation"),
                "alerts": result.get("safety", {}).get("alerts", []),
            },
        }
        with self.output_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
