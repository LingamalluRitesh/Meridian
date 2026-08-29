# ModelForge AI Enterprise Makefile
.PHONY: install test lint run clean help docker-build

SHELL := /bin/bash
PYTHON := python3

help:
	@echo "ModelForge AI - Enterprise MLOps & Automated Machine Learning Platform"
	@echo "Available commands:"
	@echo "  make install    - Install Python dependencies, SDK, and frontend packages"
	@echo "  make test       - Execute full automated test suite with pytest"
	@echo "  make lint       - Run code formatting and static analysis checks"
	@echo "  make run        - Launch ModelForge AI unified backend and frontend servers"
	@echo "  make clean      - Remove build artifacts, caches, and temporary files"
	@echo "  make docker-build - Build production multi-stage container image"

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r backend/requirements.txt
	$(PYTHON) -m pip install -e sdk/
	npm install --prefix frontend/

test:
	$(PYTHON) -m pytest backend/tests/ -v --durations=10

lint:
	$(PYTHON) -m flake8 backend/ sdk/ --max-line-length=120 || true
	$(PYTHON) -m mypy backend/ml_engine/ sdk/ || true

run:
	$(PYTHON) run.py --mode all

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf dist build *.egg-info frontend/.next

docker-build:
	docker build -t modelforge-ai:latest .
