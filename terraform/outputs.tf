output "ecr_repository_url" {
  description = "URL of created ECR repository"
  value       = aws_ecr_repository.rag_app.repository_url
}

output "ecs_cluster_name" {
  description = "Name of ECS Cluster"
  value       = aws_ecs_cluster.rag_cluster.name
}

output "task_definition_arn" {
  description = "ARN of ECS task definition"
  value       = aws_ecs_task_definition.rag_task.arn
}
