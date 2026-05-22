output "ecr_repository_url" {
  description = "ECR repository URL for Docker image pushes."
  value       = aws_ecr_repository.api.repository_url
}

output "web_ecr_repository_url" {
  description = "ECR repository URL for Next.js web image pushes."
  value       = aws_ecr_repository.web.repository_url
}

output "api_service_url" {
  description = "Public HTTPS URL for the deployed FastAPI service."
  value       = aws_apprunner_service.api.service_url
}

output "web_service_url" {
  description = "Public HTTPS URL for the deployed Next.js console."
  value       = aws_apprunner_service.web.service_url
}

output "api_image_identifier" {
  description = "API image URI App Runner deploys."
  value       = local.api_image_identifier
}

output "web_image_identifier" {
  description = "Web image URI App Runner deploys."
  value       = local.web_image_identifier
}
