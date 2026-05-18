# Multi-Agent Healthcare Decision System

An autonomous AI system that analyzes patient data, estimates health risk, explains reasoning, recommends actions, retrieves medical context, and learns from prior cases.

## Overview

This project combines:

- A machine learning risk model for baseline probability estimation
- A Kaggle-trained production model path using the CDC Diabetes Health Indicators dataset
- An orchestrator agent that coordinates specialized agents
- LLM-powered explanation and recommendation generation
- Retrieval-augmented context from a local medical knowledge base
- Persistent patient memory with similarity search (ChromaDB)
- SHAP-based feature attribution for explainability
- A Streamlit dashboard and FastAPI backend

The result is an agentic healthcare prototype that demonstrates end-to-end decision flow, transparency, and memory.

## Final Architecture

```text
User Input
   -> Risk Model (Logistic Regression)
      -> Orchestrator Agent
         -> Explanation Agent
         -> Recommendation Agent
         -> Retrieval Agent (medical knowledge)
         -> Memory Agent (past patients)
   -> Final Output (dashboard + API/chat-ready response)
```

Detailed diagram: [`docs/architecture.md`](docs/architecture.md)

## Key Features

- **Risk prediction** from patient vitals, demographics, lifestyle, and access inputs
- **Production training pipeline** for CDC BRFSS diabetes indicators on Kaggle
- **Persisted model artifact loading** with local CSV fallback for development
- **Internal BMI computation** from height and weight
- **Risk classification** into Low / Medium / High
- **SHAP explainability** to show feature impact on risk
- **LLM-generated plain-language explanation** of the prediction
- **LLM-generated actionable recommendations**
- **RAG context retrieval** from local medical guidance documents
- **Persistent memory** with similar historical case retrieval
- **Dual interface**:
  - Streamlit dashboard for interactive use
  - FastAPI endpoints for integration and automation

## Tech Stack

- **Language**: Python 3.10+
- **ML**: scikit-learn, pandas, shap
- **Training**: Kaggle, persisted `joblib` artifacts, model card + metrics JSON
- **LLM**: OpenAI API
- **Memory/Vector Store**: ChromaDB
- **Backend API**: FastAPI, Uvicorn
- **Frontend**: Streamlit
- **Deployment**: Docker, AWS App Runner, ECR, Terraform

## Project Structure

```text
healthcare-agent-system/
├── agents/
│   ├── orchestrator.py
│   ├── explainer.py
│   ├── recommender.py
│   └── retriever.py
├── artifacts/
│   └── risk_model.joblib
├── api/
│   └── app.py
├── data/
│   ├── patients.csv
│   ├── medical_knowledge.jsonl
│   └── chroma/
├── docs/
│   ├── aws_deployment.md
│   └── kaggle_training.md
├── infra/
│   └── aws/apprunner/
├── kaggle/
│   ├── kaggle_train.py
│   └── kernel-metadata.json
├── memory/
│   └── store.py
├── models/
│   └── risk_model.py
├── tools/
│   ├── risk_classifier.py
│   ├── explainability.py
│   ├── chroma_client.py
│   └── local_embeddings.py
├── ui/
│   └── app.py
├── main.py
├── requirements.txt
└── README.md
```

## How the System Works

1. User provides patient attributes (`age`, `height_cm`, `weight_kg`, `bp`, optional `glucose`, and optional BRFSS-style health indicators)
2. System computes BMI internally (`kg / m^2`)
3. Risk model predicts probability of elevated risk
4. Risk classifier maps probability -> `Low` / `Medium` / `High`
5. Feature explainer computes SHAP or local sensitivity contribution scores
6. Retrieval agent fetches relevant medical context
7. Memory agent retrieves similar historical cases and stores new case
8. Explanation and recommendation agents generate human-friendly outputs
9. Final structured response is returned to UI/API

## Setup

### 1) Clone and create environment

```bash
git clone <your-repo-url>
cd healthcare-agent-system
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 2.1) Enable local pre-commit checks (recommended)

```bash
pre-commit install
pre-commit run --all-files
```

### 3) Configure environment variables

Create a `.env` file (or export directly):

```bash
export OPENAI_API_KEY="your_api_key_here"
export MODEL_ARTIFACT_PATH="./artifacts/risk_model.joblib"
```

> The explanation/recommendation agents require a valid OpenAI key.

If `MODEL_ARTIFACT_PATH` points to a trained Kaggle artifact, the app loads it at
startup. If the artifact is missing, the app falls back to `data/patients.csv`.

## Train The Production Model On Kaggle

Training runs on Kaggle against the CDC Diabetes Health Indicators dataset
(`alexteboul/diabetes-health-indicators-dataset`). Your machine only uploads the script
and downloads the trained artifact.

```bash
chmod +x scripts/kaggle_run.sh
./scripts/kaggle_run.sh
```

Kernel: `aditya2402/healthcare-ai-diabetes-risk-training`  
Script: [`kaggle/kaggle_train.py`](kaggle/kaggle_train.py)

See [`docs/kaggle_training.md`](docs/kaggle_training.md) and [`kaggle/README.md`](kaggle/README.md).

## Deploy To AWS

The API deploys as a Dockerized FastAPI service on AWS App Runner with ECR and Terraform.

```bash
cd infra/aws/apprunner
terraform init
terraform apply -target=aws_ecr_repository.api
terraform apply -var "api_token=replace-with-a-strong-token"
```

After the first targeted apply creates ECR, build and push the Docker image from the
repo root, then run the full apply. See [`docs/aws_deployment.md`](docs/aws_deployment.md) and
[`infra/aws/apprunner/README.md`](infra/aws/apprunner/README.md).

## Run the Project

### Run CLI demo

```bash
python main.py
```

### Run Streamlit dashboard

```bash
streamlit run ui/app.py
```

Default local URL: `http://localhost:8501`

### Run FastAPI backend

```bash
uvicorn api.app:app --reload
```

API base URL: `http://127.0.0.1:8000`  
Swagger docs: `http://127.0.0.1:8000/docs`

### Quick API smoke test

After starting FastAPI, verify service health with:

```bash
curl -s http://127.0.0.1:8000/health
```

## API Endpoints (Current)

- `GET /health` -> Health status
- `POST /predict` -> Full multi-agent analysis pipeline
- `POST /predict/summary` -> Condensed triage output
- `GET /v1/health` -> Versioned health endpoint
- `POST /v1/predict` -> Versioned full prediction with `request_id`
- `POST /v1/predict/summary` -> Versioned condensed triage output

Typical `/predict` response includes:

- Risk probability + risk label
- Calculated BMI
- Feature impact scores
- Retrieved medical context
- Similar past cases
- Explanation text
- Recommendation text

### API Authentication

By default, API token auth is optional for local development. To enforce auth:

```bash
export REQUIRE_API_TOKEN=true
export API_TOKEN="replace-with-strong-token"
```

Then call protected endpoints with:

```bash
Authorization: Bearer <API_TOKEN>
```

### Decision Audit Logs

Every prediction request is written to:

- `logs/decisions.jsonl`

Each record includes timestamp, request ID, normalized patient summary, risk result, confidence, escalation, and alerts.

## Example Input

```json
{
  "age": 45,
  "height_cm": 175,
  "weight_kg": 78,
  "bp": 130,
  "glucose": 165
}
```

## Important Medical Disclaimer

This project is a **software/AI prototype for education and engineering demonstration**.  
It is **not a medical device**, **not clinically validated**, and **must not** be used as a sole basis for diagnosis or treatment decisions.

Always consult qualified healthcare professionals for real medical advice.

## Current Limitations

- Kaggle artifact must be trained/downloaded before production deployment
- Recommendations are generated by LLM and may require strict clinical guardrails
- No full production security/compliance pipeline yet (PHI/HIPAA hardening)
- Fairness, calibration, and drift monitoring should be expanded

## Roadmap

### Completed (Phase 1/2)

- Multi-agent orchestration
- SHAP explainability
- ChromaDB persistent memory
- Retrieval agent + local knowledge base
- FastAPI + Streamlit interfaces

### Planned (Phase 3+)

- Fairness reports, calibration plots, and drift monitoring
- Safety guardrails and escalation logic
- Automated tests and CI pipeline
- Better observability and decision tracing
- Feedback loops for continuous learning

### Implemented in Phase 3

- Added a safety guardrail layer with escalation categories:
  - `routine_followup`
  - `prompt_clinician_followup`
  - `urgent_clinician_review`
- Added confidence scoring and safety alerts in orchestrator outputs
- Exposed concise triage endpoint: `POST /predict/summary`
- Added baseline automated tests (`tests/`) for model behavior and API health
- Added evaluation utility: `scripts/evaluate_model.py` for accuracy/F1/ROC-AUC

### Implemented in Phase 4

- Added versioned API routes (`/v1/*`) for forward-compatible contracts
- Added optional bearer-token protection for API endpoints
- Added structured decision audit logging (`logs/decisions.jsonl`)
- Added Dockerfile + Docker Compose for reproducible local deployment
- Added CI workflow (compile + test on push/PR)
- Added lint + coverage quality gates in CI
- Expanded CI to run on Python 3.10 and 3.11 with dependency caching
- Added local pre-commit hooks for code quality
- Added Dependabot automation for pip and GitHub Actions updates

### Implemented in Phase 5

- Added Kaggle training workflow for the CDC Diabetes Health Indicators dataset
- Added persisted `joblib` artifact loading with local fallback
- Expanded API/UI inputs for BRFSS-style health indicators
- Added production training script with model comparison and metrics output
- Added AWS App Runner Terraform deployment stack

## Security Notes

See [`docs/security.md`](docs/security.md) for PII handling guidance, retention policy, and hardening recommendations.

## Contributing

1. Create a feature branch
2. Make focused, testable changes
3. Open a pull request with:
   - clear summary
   - test evidence
   - notes on risk/safety impact

## License

Choose an appropriate license before public/open-source release (for example, MIT or Apache-2.0).
