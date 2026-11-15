.PHONY: up down logs rebuild validate validation-demo repro dag

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

# Run the raw-data contract gate.
validate:
	dvc repro validate

# Demonstrate rejection and exact restoration of invalid input.
validation-demo:
	uv run python scripts/demo_validation_gate.py

# Run the entire pipeline (only re-runs changed stages)
repro:
	dvc repro

# Show the pipeline DAG
dag:
	dvc dag