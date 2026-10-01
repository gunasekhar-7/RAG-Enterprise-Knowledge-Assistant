.PHONY: setup infra-up infra-down run-backend run-frontend test lint format check help

setup:
	@echo "Setting up development environment..."
	@python -m venv .venv
	@.venv/bin/pip install --upgrade pip
	@.venv/bin/pip install -r backend/requirements-dev.txt
	@cd frontend && npm install
	@cp -n .env.example .env || true
	@echo "Setup complete!"

infra-up:
	docker compose up -d

infra-down:
	docker compose down

infra-status:
	docker compose ps

run-backend:
	PYTHONPATH=backend .venv/bin/uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 --app-dir backend

run-frontend:
	cd frontend && npm run dev

test:
	PYTHONPATH=backend .venv/bin/pytest backend/tests -v

lint:
	.venv/bin/ruff check backend
	.venv/bin/mypy --config-file backend/pyproject.toml backend/app

format:
	.venv/bin/ruff format backend
	.venv/bin/ruff check --fix backend

check:
	@echo "Checking environment..."
	@python --version
	@node --version
	@docker --version
