.PHONY: up down logs rebuild repro dag

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

# Run the entire pipeline (only re-runs changed stages)
repro:
	dvc repro

# Show the pipeline DAG
dag:
	dvc dag