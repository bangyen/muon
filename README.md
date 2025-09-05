# Muon Optimizer: Accelerating Grokking Reproduction

This repository contains a reproduction of the paper **"Muon Optimizer Accelerates Grokking"** by Tveit et al. (2025). The Muon optimizer, which incorporates spectral norm constraints and second-order information, significantly accelerates the grokking phenomenon—delayed generalization—compared to standard AdamW.

## 📋 Paper Summary

**Key Findings:**
- Muon reduces average grokking epoch from 153.09 to 102.89 (33% speedup)
- Statistically significant improvement (t=5.0175, p=6.33e-08) across 7 modular arithmetic tasks
- Combines spectral norm constraints, orthogonalized gradients, and second-order information

**Why This Matters:**
- Grokking is a fascinating phenomenon where models suddenly generalize after extended training
- Optimizer choice plays a crucial role in learning dynamics
- Muon provides a more efficient path from memorization to generalization

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd muon

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -e .
```

### Using Makefile (Recommended)

```bash
# Run quick single task test
make run-single

# Run comprehensive experiments
make run-experiments

# Analyze latest results
make analyze

# See all available commands
make help
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


## 📁 Project Structure

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
│   ├── test_model.py     # Model architecture tests
│   └── test_dataset.py   # Dataset functionality tests
├── results/              # Experiment results (JSON + CSV)
├── Makefile              # Build and development commands
├── pyproject.toml         # Project configuration (includes mypy settings)
└── README.md            # This file
```

## 🔬 Reproduction Details

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

### Tasks Implemented

All 7 tasks from the paper:
- **GCD**: Greatest common divisor (mod 97)
- **Modular Addition**: Addition (mod 97)
- **Modular Division**: Division (mod 97)
- **Modular Exponentiation**: Exponentiation (mod 97)
- **Modular Multiplication**: Multiplication (mod 97)
- **Parity**: Parity of 10-bit binary strings

### Model Architecture

- **Transformer**: 4 layers, 8 heads, 128 hidden size
- **RMSNorm**: Replaces LayerNorm for better stability
- **RoPE**: Rotary positional embeddings
- **SiLU**: Sigmoid Linear Unit activation in feed-forward networks

### Softmax Variants

- **Standard**: Traditional exponential normalization
- **Stablemax**: Enhanced numerical stability
- **Sparsemax**: Projects onto probability simplex

## 📊 Results

### Expected Performance

Based on the paper, you should see:
- **Muon**: Average grokking epoch ~103
- **AdamW**: Average grokking epoch ~153
- **Speedup**: ~1.5x faster grokking with Muon

### Key Metrics

- **Grokking Epoch**: First epoch where validation accuracy ≥ 95%
- **Success Rate**: Percentage of experiments that achieve grokking
- **Statistical Significance**: T-test comparing Muon vs AdamW

## 🎯 Usage Examples

### Single Experiment

```python
from muon import SingleDeviceMuonWithAuxAdam
from src.model import GrokkingTransformer
from src.dataset import ModularArithmeticDataset

# Create dataset
dataset = ModularArithmeticDataset('add', modulus=97, train_split=0.8)

# Create model
model = GrokkingTransformer(vocab_size=dataset.vocab_size)

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
# Custom model configuration
model_config = {
    'hidden_size': 256,
    'num_layers': 6,
    'num_heads': 16,
    'ff_size': 1024,
    'max_seq_len': 15,
    'batch_size': 64,
    'dropout': 0.1,
    'max_epochs': 500,
    'grokking_threshold': 0.95
}

# Custom Muon configuration
muon_config = {
    'lr': 0.02,  # Learning rate for Muon parameters
    'weight_decay': 5e-3,
    'betas': (0.9, 0.95),  # For AdamW parameters
}
```

## 🔍 Ablation Studies

The implementation supports various ablation studies:

### Learning Rate
```python
# Test different learning rates for Muon parameters
for lr in [0.01, 0.02, 0.05]:
    param_groups = [
        dict(params=hidden_weights, use_muon=True, lr=lr, weight_decay=1e-2),
        dict(params=other_params, use_muon=False, lr=lr*0.1, betas=(0.9, 0.95), weight_decay=1e-2)
    ]
    optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
```

### Softmax Variants
```python
# Compare different softmax functions
softmax_variants = ['standard', 'stablemax', 'sparsemax']
```


## 📈 Analysis and Results

The analysis script provides:

1. **Statistical Analysis**: T-test comparison of Muon vs AdamW grokking epochs
2. **Summary Statistics**: Mean, standard deviation, and success rates
3. **Task Breakdown**: Results broken down by task and softmax variant
4. **Detailed Reports**: Comprehensive statistical analysis matching paper methodology

```bash
# Perform statistical analysis matching the paper
python -m scripts.analyze_results --results_file results/experiment_results_YYYYMMDD_HHMMSS.json --detailed
```

## 🧪 Experimental Setup

### Hardware Requirements

- **CPU**: Sufficient for small-scale experiments
- **GPU**: Optional, speeds up training significantly
- **Memory**: 8GB+ recommended for full experiments

### Reproducibility

- All experiments use fixed random seeds
- Results are saved in JSON format for analysis
- Multiple runs per configuration for statistical significance

## 📚 References

1. **Original Paper**: Tveit, A., Remseth, B., & Skogvold, A. (2025). Muon Optimizer Accelerates Grokking. arXiv:2504.16041
2. **Grokking Phenomenon**: Power, A., et al. (2022). Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets
3. **Muon Optimizer**: Jordan, K. (2024). Muon Optimizer. GitHub repository

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- **Performance**: Optimize for larger models
- **Analysis**: Additional ablation studies
- **Documentation**: More detailed explanations
- **Testing**: Expand test coverage and edge cases

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Original authors for the groundbreaking research
- PyTorch team for the excellent framework
- Open source community for inspiration and tools

---

**Note**: This is a reproduction study. For the original research, please refer to the cited papers and repositories.
