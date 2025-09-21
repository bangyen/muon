# Muon Optimizer: Accelerating Grokking Reproduction

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains a reproduction of the paper **"Muon Optimizer Accelerates Grokking"** by Tveit et al. (2025). The Muon optimizer, which incorporates spectral norm constraints and second-order information, significantly accelerates the grokking phenomenon—delayed generalization—compared to standard AdamW.

## Paper Summary

**Key Findings:**
- Muon reduces average grokking epoch from 153.09 to 102.89 (33% speedup)
- Statistically significant improvement (t=5.0175, p=6.33e-08) across 7 modular arithmetic tasks
- Combines spectral norm constraints, orthogonalized gradients, and second-order information

**Why This Matters:**
- Grokking is a fascinating phenomenon where models suddenly generalize after extended training
- Optimizer choice plays a crucial role in learning dynamics
- Muon provides a more efficient path from memorization to generalization

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/bangyen/muon.git
cd muon

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install production dependencies
pip install -e .

# Or install with development dependencies (recommended for contributors)
pip install -e ".[dev]"
```

**Note**: This project requires Python 3.9+ and PyTorch 2.0+. The Muon optimizer is installed from the original repository as a dependency.

### Using Makefile (Recommended)

```bash
# See all available commands
make help

# Development setup
make setup-dev          # Install dev dependencies and setup pre-commit hooks
make install            # Install production dependencies only
make install-dev        # Install development dependencies

# Running experiments
make run-single         # Run quick single task test
make run-experiments    # Run comprehensive experiments

# Code quality and testing
make test               # Run all tests
make lint               # Run linting checks (ruff, mypy, interrogate)
make format             # Format code with ruff

# Analysis and cleanup
make analyze            # Analyze latest experiment results
make clean              # Clean up generated files and caches
```

### Manual Commands

```bash
# Run a quick test with limited configurations
python -m scripts.train_tasks --single_task --device cpu

# Run comprehensive experiments (takes several hours)
python -m scripts.train_tasks --device cpu --num_runs 3

# Perform statistical analysis matching the paper
python -m scripts.analyze_results --results_file results/experiment_results_YYYYMMDD_HHMMSS.json --detailed
```


## Project Structure

```
muon/
├── src/
│   ├── __init__.py
│   ├── model.py          # Transformer architecture with identity embeddings & RoPE
│   └── dataset.py        # Modular arithmetic datasets (exact paper configs)
├── scripts/
│   ├── train_tasks.py    # Main training script (faithful to paper methodology)
│   └── analyze_results.py   # Statistical analysis matching paper results
├── tests/
│   ├── __init__.py
│   ├── conftest.py       # Test configuration and fixtures
│   ├── test_model.py     # Model architecture tests
│   ├── test_dataset.py   # Dataset functionality tests
│   └── README.md         # Testing documentation
├── results/              # Experiment results (JSON + CSV) - created during training
├── plots/                # Generated plots and visualizations - created during analysis
├── Makefile              # Build and development commands
├── pyproject.toml        # Project configuration (dependencies, linting, type checking)
└── README.md            # This file
```

**Key Files:**
- `src/model.py`: Complete transformer implementation with RMSNorm, RoPE, and softmax variants
- `src/dataset.py`: Modular arithmetic dataset with exact paper configurations
- `scripts/train_tasks.py`: Training pipeline with Muon and AdamW optimizers
- `scripts/analyze_results.py`: Statistical analysis matching paper methodology
- `pyproject.toml`: Comprehensive project configuration with development tools

## Reproduction Details

### Exact Paper Implementation

This reproduction follows the paper specifications exactly:

1. **Model Architecture**:
   - Identity embeddings (integer value as embedding index)
   - Rotary Positional Embeddings (RoPE) in attention
   - RMSNorm instead of LayerNorm
   - SiLU activation in feed-forward networks

2. **Dataset Configurations** (Figure 2):
   - GCD: mod 97, 50% train split
   - Mod-add: mod 97, 80% train split
   - Mod-div: mod 97, 80% train split
   - Mod-exp: mod 97, 70% train split
   - Mod-mul: mod 97, 50% train split
   - Parity: 10-bit numbers, 50% train split

3. **Softmax Variants** (Figure 3):
   - Standard: `softmax(z)_i = exp(z_i) / sum_j exp(z_j)`
   - Stablemax: `s(z_i) = {z_i + 1 if z_i >= 0, 1/(1-z_i) if z_i < 0}`
   - Sparsemax: `sparsemax(z)_i = max{z_i - tau, 0}`

4. **Grokking Definition**: First epoch where validation accuracy ≥ 95%, occurring after training accuracy stabilizes near 100%

5. **Statistical Analysis**: Two-sample t-test comparing Muon vs AdamW grokking epochs

### Tasks & Architecture

**7 Tasks**: GCD, modular addition/division/exponentiation/multiplication (mod 97), and parity (10-bit)

**Model**: 4-layer transformer, 8 heads, 128 hidden size, RMSNorm, RoPE, SiLU activation

**Softmax Variants**: Standard, Stablemax (numerical stability), Sparsemax (probability simplex)

## Results

### Expected Performance

Based on the paper, you should see:
- **Muon**: Average grokking epoch ~103
- **AdamW**: Average grokking epoch ~153
- **Speedup**: ~1.5x faster grokking with Muon

### Key Metrics

- **Grokking Epoch**: First epoch where validation accuracy ≥ 95%
- **Success Rate**: Percentage of experiments that achieve grokking
- **Statistical Significance**: T-test comparing Muon vs AdamW

## Usage Examples

### Single Experiment

```python
from muon import SingleDeviceMuonWithAuxAdam
from src.model import GrokkingTransformer, ModelConfig
from src.dataset import ModularArithmeticDataset, DatasetConfig

# Create dataset with paper configuration
dataset_config = DatasetConfig(
    task_type='add', 
    modulus=97, 
    train_split=0.8
)
dataset = ModularArithmeticDataset(dataset_config)

# Create model with paper configuration
model_config = ModelConfig(
    vocab_size=dataset.vocab_size,
    hidden_size=128,
    num_layers=4,
    num_heads=8,
    ff_size=512
)
model = GrokkingTransformer(model_config)

# Create Muon optimizer with proper parameter grouping
hidden_weights = [p for p in model.parameters() if p.ndim >= 2]
other_params = [p for p in model.parameters() if p.ndim < 2]

param_groups = [
    dict(params=hidden_weights, use_muon=True, lr=0.02, weight_decay=1e-2),
    dict(params=other_params, use_muon=False, lr=0.002, betas=(0.9, 0.95), weight_decay=1e-2)
]
optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
```

### Custom Configuration

```python
from src.model import ModelConfig
from src.dataset import DatasetConfig

# Custom model configuration
model_config = ModelConfig(
    vocab_size=128,
    hidden_size=256,
    num_layers=6,
    num_heads=16,
    ff_size=1024,
    max_seq_len=15,
    dropout=0.1,
    softmax_variant='stablemax'
)

# Custom dataset configuration
dataset_config = DatasetConfig(
    task_type='mul',
    modulus=97,
    train_split=0.5,
    max_seq_len=7,
    seed=42
)

# Custom Muon optimizer configuration
param_groups = [
    dict(params=hidden_weights, use_muon=True, lr=0.02, weight_decay=5e-3),
    dict(params=other_params, use_muon=False, lr=0.002, betas=(0.9, 0.95), weight_decay=5e-3)
]
```



## Analysis

The analysis script provides statistical analysis matching the paper methodology:

```bash
# Analyze latest results
make analyze
# Or manually: python -m scripts.analyze_results --results_file results/experiment_results_*.json --detailed
```

## Dependencies

**Core**: Python 3.9+, PyTorch 2.0+, NumPy, Pandas, SciPy, Muon Optimizer

**Development**: pytest, ruff, mypy, interrogate, pre-commit

**Optional**: `torch[cuda]` for GPU acceleration

## Experimental Setup

**Hardware**: CPU sufficient, GPU optional (faster), 8GB+ RAM recommended

**Reproducibility**: Fixed random seeds, JSON results, multiple runs for statistics

## References

1. **Original Paper**: Tveit, A., Remseth, B., & Skogvold, A. (2025). Muon Optimizer Accelerates Grokking. arXiv:2504.16041
2. **Grokking Phenomenon**: Power, A., et al. (2022). Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets
3. **Muon Optimizer**: Jordan, K. (2024). Muon Optimizer. GitHub repository

## Development

### Quick Setup for Contributors

```bash
# Install development dependencies and setup pre-commit hooks
make setup-dev

# Run all quality checks
make lint

# Format code and run tests
make format
make test
```

### Code Quality Standards

- **Ruff**: Fast linter and formatter
- **MyPy**: Static type checking (strict mode)
- **Interrogate**: Documentation coverage (100% required)
- **Pre-commit**: Automated quality checks

### Testing

```bash
# Run all tests
make test

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

**Test Coverage:**
- Model architecture (transformer, attention, softmax variants)
- Dataset functionality (generation, tokenization, data loading)

## Contributing

Contributions are welcome! Areas for improvement:

- **Performance**: Optimize for larger models
- **Analysis**: Additional ablation studies
- **Documentation**: More detailed explanations
- **Testing**: Expand test coverage and edge cases

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original authors for the groundbreaking research
- PyTorch team for the excellent framework
- Open source community for inspiration and tools

---

**Note**: This is a reproduction study. For the original research, please refer to the cited papers and repositories.
