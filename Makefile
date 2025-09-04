.PHONY: help install install-prod install-dev test test-quick lint format clean run-experiments visualize setup-dev build publish

help: ## Show this help message
	@echo "Muon Optimizer: Accelerating Grokking Reproduction"
	@echo "=================================================="
	@echo ""
	@echo "Available commands:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install all dependencies (including dev)
	pip install -e ".[dev]"

install-prod: ## Install production dependencies only
	pip install -e .

install-dev: ## Install development dependencies
	pip install -e ".[dev]"

test: ## Run tests
	python -m pytest tests/ -v

test-quick: ## Run quick single task test
	python -m scripts.train_tasks --single_task --device cpu

lint: ## Run linting checks
	ruff check src/ scripts/
	ruff format --check src/ scripts/
	mypy src/
	interrogate src/ --fail-under=90

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
	python -m scripts.train_tasks --device cpu --num_runs 3

visualize: ## Create visualizations from results
	python -m scripts.visualize_results --results_file results/experiment_results_*.json

setup-dev: install-dev ## Setup development environment
	pre-commit install

build: ## Build package
	python -m build

publish: build ## Build and publish to PyPI
	twine upload dist/*
