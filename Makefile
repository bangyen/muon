.PHONY: help install test lint format clean run-experiments run-quick-test visualize

help: ## Show this help message
	@echo "Muon Optimizer: Accelerating Grokking Reproduction"
	@echo "=================================================="
	@echo ""
	@echo "Available commands:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install dependencies
	pip install -e ".[dev]"

install-dev: ## Install development dependencies
	pip install -e ".[dev]"

test: ## Run tests
	python test_implementation.py

test-fast: ## Run quick test experiments
	python scripts/train_tasks.py --quick_test --device cpu

lint: ## Run linting checks
	ruff check src/ scripts/ test_implementation.py
	ruff format --check src/ scripts/ test_implementation.py
	mypy src/
	interrogate src/ --fail-under=90

format: ## Format code with ruff
	ruff format src/ scripts/ test_implementation.py
	ruff check --fix src/ scripts/ test_implementation.py

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
	python scripts/train_tasks.py --device cpu --num_runs 3

run-quick-test: ## Run quick test with limited configurations
	python scripts/train_tasks.py --quick_test --device cpu

visualize: ## Create visualizations from results
	python scripts/visualize_results.py --results_file results/experiment_results_*.json

setup-dev: install-dev ## Setup development environment
	pre-commit install

build: ## Build package
	python -m build

publish: build ## Build and publish to PyPI
	twine upload dist/*
