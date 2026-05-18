# AWS App Runner Deployment

This Terraform stack deploys the trained FastAPI service to AWS App Runner from a
Docker image in Amazon ECR.

## Flow

1. Train on Kaggle and download `artifacts/risk_model.joblib`.
2. Build the Docker image from the repo root.
3. Push the image to ECR.
4. Apply this Terraform stack.

## Commands

Create the ECR repository first:

```bash
cd infra/aws/apprunner
terraform init
terraform apply -target=aws_ecr_repository.api
```

Build and push the image:

```bash
AWS_REGION=us-west-2
ECR_REPO=$(terraform output -raw ecr_repository_url)
cd ../../..
aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "$ECR_REPO"
docker build -t healthcare-ai-risk-api .
docker tag healthcare-ai-risk-api:latest "$ECR_REPO:latest"
docker push "$ECR_REPO:latest"
```

Deploy App Runner:

```bash
cd infra/aws/apprunner
terraform apply \
  -var "api_token=replace-with-a-strong-token" \
  -var "openai_api_key=$OPENAI_API_KEY"
```

App Runner is configured with auto deployments, so pushing `:latest` rolls the API.

## Smoke Test

```bash
curl -s "https://<apprunner_service_url>/health"
curl -s "https://<apprunner_service_url>/v1/predict/summary" \
  -H "Authorization: Bearer <api_token>" \
  -H "Content-Type: application/json" \
  -d '{"age":52,"height_cm":170,"weight_kg":88,"bp":142,"glucose":160}'
```

For a production-grade version, move `API_TOKEN` and `OPENAI_API_KEY` into AWS Secrets
Manager instead of Terraform variables.
