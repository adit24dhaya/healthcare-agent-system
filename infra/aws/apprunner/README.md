# AWS App Runner Deployment

This Terraform stack deploys both services to AWS App Runner from Docker images in
Amazon ECR:

- FastAPI inference API
- Next.js healthcare risk console

## Flow

1. Train on Kaggle and download `artifacts/risk_model.joblib`.
2. Create the API and web ECR repositories.
3. Build and push both Docker images.
4. Apply this Terraform stack to deploy both App Runner services.

## Commands

Create the ECR repository first:

```bash
cd infra/aws/apprunner
terraform init
terraform apply \
  -target=aws_ecr_repository.api \
  -target=aws_ecr_repository.web
```

Build and push the API image:

```bash
AWS_REGION=us-west-2
API_ECR_REPO=$(terraform output -raw ecr_repository_url)
cd ../../..
aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "$API_ECR_REPO"
docker build -t healthcare-ai-risk-api .
docker tag healthcare-ai-risk-api:latest "$API_ECR_REPO:latest"
docker push "$API_ECR_REPO:latest"
```

Build and push the web image:

```bash
cd infra/aws/apprunner
WEB_ECR_REPO=$(terraform output -raw web_ecr_repository_url)
cd ../../..
docker build -t healthcare-ai-risk-web ./web
docker tag healthcare-ai-risk-web:latest "$WEB_ECR_REPO:latest"
docker push "$WEB_ECR_REPO:latest"
```

Deploy App Runner:

```bash
cd infra/aws/apprunner
terraform apply \
  -var "api_token=replace-with-a-strong-token" \
  -var "openai_api_key=$OPENAI_API_KEY"
```

App Runner is configured with auto deployments, so pushing `:latest` rolls the matching
service.

## Smoke Test

```bash
curl -s "https://<api_service_url>/health"
curl -s "https://<api_service_url>/v1/predict/summary" \
  -H "Authorization: Bearer <api_token>" \
  -H "Content-Type: application/json" \
  -d '{"age":52,"height_cm":170,"weight_kg":88,"bp":142,"glucose":160}'
```

Open the Next.js console:

```text
https://<web_service_url>
```

For a production-grade version, move `API_TOKEN` and `OPENAI_API_KEY` into AWS Secrets
Manager instead of Terraform variables.
