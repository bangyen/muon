.PHONY: help install install-dev test lint format clean run-experiments run-single visualize analyze setup-dev build publish

help: ## Show this help message
	@echo "Muon Optimizer: Accelerating Grokking Reproduction"
	@echo "=================================================="
	@echo ""
	@echo "Available commands:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install production dependencies only
	pip install -e .

install-dev: ## Install development dependencies
	pip install -e ".[dev]"

test: ## Run tests
	python -m pytest tests/ -v

lint: ## Run linting checks
	ruff check src/ scripts/
	ruff format --check src/ scripts/
	interrogate src/ --fail-under=90
	mypy src/

format: ## Format code with ruff
	ruff format src/ scripts/
	ruff check --fix src/ scripts/

clean: ## Clean up generated files
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf __pycache__/
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf results/
	rm -rf plots/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete

run-experiments: ## Run comprehensive experiments
	python -m scripts.train_tasks --device cpu

run-single: ## Run quick single task test
	python -m scripts.train_tasks --single_task --device cpu

smoke-test: ## Run CI smoke test (≤2 minutes)
	python -m scripts.smoke_test --timeout 120

LATEST_RESULTS = $(shell ls -t results/experiment_results_*.json | head -1)

analyze: ## Analyze experiment results with statistical tests
	python -m scripts.analyze_results --results_file $(LATEST_RESULTS) --detailed

setup-dev: install-dev ## Setup development environment
	pre-commit install
