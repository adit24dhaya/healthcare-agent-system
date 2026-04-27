# Architecture Diagram

This diagram is designed for recruiter/interviewer walkthroughs.

```mermaid
flowchart TD
    A[User Input<br/>UI or API] --> B[RiskModel<br/>Logistic Regression]
    B --> C[Orchestrator Agent]

    C --> D[Explanation Agent<br/>LLM + fallback rules]
    C --> E[Recommendation Agent<br/>LLM + safety context]
    C --> F[Retrieval Agent<br/>Medical knowledge RAG]
    C --> G[Memory Agent<br/>ChromaDB patient memory]
    C --> H[Feature Explainer<br/>SHAP attributions]
    C --> I[Safety Guard<br/>alerts + escalation + confidence]

    F --> J[data/medical_knowledge.jsonl]
    G --> K[data/chroma]
    C --> L[Audit Logger]
    L --> M[logs/decisions.jsonl]

    C --> N[Final Response]
    N --> O[Streamlit Dashboard]
    N --> P[FastAPI /predict + /v1/predict]
```

## Request Lifecycle (High Level)

1. Validate and normalize input (including BMI calculation).
2. Predict probability and classify risk.
3. Compute SHAP feature impacts.
4. Retrieve similar past cases + relevant medical context.
5. Generate explanation and recommendation.
6. Apply safety assessment (alerts, escalation, confidence).
7. Persist memory and write audit trail.
8. Return structured response to UI/API client.
