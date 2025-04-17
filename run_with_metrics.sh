#!/bin/bash
# Script to run training with metrics exposed to Prometheus

# Clean up any existing containers
docker rm -f training-metrics 2>/dev/null

echo "Starting training with metrics monitoring..."


# 👇 Set the Docker internal URL for MLflow
export MLFLOW_TRACKING_URI=http://mlflow:5000


# Run the container with a fixed name that Prometheus can discover
# docker-compose run --name training-metrics app python src/train.py "$@"
docker-compose run --name training-metrics app python src/train.py --config "$1"

echo "Training completed"

