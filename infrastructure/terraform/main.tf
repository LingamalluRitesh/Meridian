terraform {
  required_version = ">= 1.5.0"
}

variable "region" {
  default = "us-east-1"
}

output "cluster_name" {
  value = "modelforge-prod-eks"
}
