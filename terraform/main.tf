# Terraform AWS Infrastructure as Code for Enterprise RAG System
terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# AWS ECR Repository for Docker Images
resource "aws_ecr_repository" "rag_app" {
  name                 = "enterprise-rag-system"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

# ECS Cluster
resource "aws_ecs_cluster" "rag_cluster" {
  name = "enterprise-rag-cluster"
}

# IAM Role for ECS Execution
resource "aws_iam_role" "ecs_execution_role" {
  name = "enterprise-rag-ecs-execution-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_execution_policy" {
  role       = aws_iam_role.ecs_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# ECS Fargate Task Definition
resource "aws_ecs_task_definition" "rag_task" {
  family                   = "enterprise-rag-task"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.container_cpu
  memory                   = var.container_memory
  execution_role_arn       = aws_iam_role.ecs_execution_role.arn

  container_definitions = jsonencode([
    {
      name      = "enterprise-rag-api"
      image     = "${aws_ecr_repository.rag_app.repository_url}:latest"
      essential = true
      portMappings = [
        {
          containerPort = 8000
          hostPort      = 8000
        }
      ]
      environment = [
        { name = "QDRANT_IN_MEMORY", value = "true" },
        { name = "ENABLE_METRICS_LOGGING", value = "true" }
      ]
    }
  ])
}
