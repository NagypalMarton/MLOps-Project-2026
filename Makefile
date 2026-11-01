.PHONY: up down logs rebuild dvc-init repro dag trace

# Bring up the local services stack
up:
	docker compose up -d --wait

# Bring down the local services stack
down:
	docker compose down

# View logs of the services
logs:
	docker compose logs -f

# Rebuild and up
rebuild:
	docker compose up -d --wait --build

# Initialize DVC and set up remote
dvc-init:
	dvc init
	dvc remote add -d silo s3://$(DVC_BUCKET_NAME)
	# Enable MLflow autologging for scikit-learn (optional, we'll do manual logging)
	# dvc config core.autostage true

# Run the entire pipeline (only re-runs changed stages)
repro:
	dvc repro

# Show the pipeline DAG
dag:
	dvc dag

# Show the pipeline trace (inputs -> outputs)
trace:
	dvc traces

# Helper target to show environment variables (for debugging)
env:
	@echo "BUCKET_NAME=$(BUCKET_NAME)"
	@echo "MLFLOW_TRACKING_URI=$(MLFLOW_TRACKING_URI)"