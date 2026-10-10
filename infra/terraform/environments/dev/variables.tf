variable "aws_region" {
  type    = string
  default = "us-east-1"
}

variable "environment" {
  type    = string
}

variable "db_password" {
  type      = string
  sensitive = true
  default   = "dev_secret_password_123!"
}
