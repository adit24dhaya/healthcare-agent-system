output "ecr_repository_url" {
  description = "ECR repository URL for Docker image pushes."
  value       = aws_ecr_repository.api.repository_url
}

output "apprunner_service_url" {
  description = "Public HTTPS URL for the deployed FastAPI service."
  value       = aws_apprunner_service.api.service_url
}

output "image_identifier" {
  description = "Image URI App Runner deploys."
  value       = local.image_identifier
}
