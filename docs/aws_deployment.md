# AWS Deployment

The deployment target is AWS App Runner running the Dockerized FastAPI service. This
keeps the production path small enough to maintain while still showing a real cloud
stack: Docker, ECR, managed HTTPS, health checks, Terraform, and token-protected APIs.

## Deployment Flow

1. Train the model on Kaggle.
2. Download `risk_model.joblib` into `artifacts/`.
3. Build the API Docker image.
4. Create the ECR repository with Terraform.
5. Push the image to ECR.
6. Deploy App Runner with Terraform from `infra/aws/apprunner`.

## Runtime Contract

The container expects:

- `MODEL_ARTIFACT_PATH=/app/artifacts/risk_model.joblib`
- `REQUIRE_API_TOKEN=true`
- `API_TOKEN=<strong-token>`
- `OPENAI_API_KEY=<optional-key>`

If the model artifact is not present, the service starts with the local demo CSV model.
For the portfolio deployment, include the Kaggle artifact in the image before pushing
to ECR.

## API Checks

```bash
curl -s https://<service-url>/health
curl -s https://<service-url>/v1/predict/summary \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"age":52,"height_cm":170,"weight_kg":88,"bp":142,"glucose":160}'
```

## Portfolio Talking Points

- reproducible Kaggle training
- persisted model artifact with metrics and model card
- FastAPI inference service
- Streamlit decision dashboard
- retrieval, memory, safety checks, and explainability
- Dockerized deployment to AWS App Runner with Terraform
