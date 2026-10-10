provider "aws" {
  region = var.aws_region
}

# The backend configuration handles state locking securely
# terraform {
#   backend "s3" {
#     bucket         = "rag-tfstate-bucket"
#     key            = "environments/dev/terraform.tfstate"
#     region         = "us-east-1"
#     dynamodb_table = "rag-tfstate-lock"
#     encrypt        = true
#   }
# }

module "vpc" {
  source               = "../../modules/vpc"
  environment          = var.environment
  vpc_cidr             = "10.0.0.0/16"
  public_subnet_cidrs  = ["10.0.1.0/24", "10.0.2.0/24"]
  private_subnet_cidrs = ["10.0.10.0/24", "10.0.20.0/24"]
  availability_zones   = ["${var.aws_region}a", "${var.aws_region}b"]
}

module "ecs" {
  source             = "../../modules/ecs"
  environment        = var.environment
  vpc_id             = module.vpc.vpc_id
  public_subnet_ids  = module.vpc.public_subnet_ids
  private_subnet_ids = module.vpc.private_subnet_ids
}

output "load_balancer_dns" {
  value = module.ecs.lb_dns_name
}

module "rds" {
  source             = "../../modules/rds"
  environment        = var.environment
  vpc_id             = module.vpc.vpc_id
  private_subnet_ids = module.vpc.private_subnet_ids
  ecs_sg_id          = module.ecs.ecs_sg_id
  db_password        = var.db_password
}

module "s3" {
  source      = "../../modules/s3"
  environment = var.environment
}
