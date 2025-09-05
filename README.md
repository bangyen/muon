# Muon Optimizer: Accelerating Grokking Reproduction

This repository contains a reproduction of the paper **"Muon Optimizer Accelerates Grokking"** by Tveit et al. (2025). The Muon optimizer, which incorporates spectral norm constraints and second-order information, significantly accelerates the grokking phenomenon—delayed generalization—compared to standard AdamW.

## 📋 Paper Summary

**Key Findings:**
- Muon reduces average grokking epoch from ~153 to ~103 (33% speedup)
- Statistically significant improvement across 7 modular arithmetic tasks
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
pip install -r requirements.txt
```

### Run Quick Test

```bash
# Run a quick test with limited configurations
python -m scripts.train_tasks --quick_test --device cpu
```

### Run Full Experiments

```bash
# Run comprehensive experiments (takes several hours)
python -m scripts.train_tasks --device cpu --num_runs 3
```

### Visualize Results

```bash
# After running experiments, create visualizations
python -m scripts.visualize_results --results_file results/experiment_results_YYYYMMDD_HHMMSS.json
```

## 📁 Project Structure

```
muon/
├── src/
│   ├── __init__.py
│   ├── optimizer.py      # Muon optimizer implementation
│   ├── model.py          # Transformer architecture
│   └── dataset.py        # Modular arithmetic datasets
├── scripts/
│   ├── train_tasks.py    # Main training script
│   └── visualize_results.py  # Visualization and analysis
├── requirements.txt      # Dependencies
└── README.md            # This file
```

## 🔬 Implementation Details

### Muon Optimizer Features

1. **Spectral Norm Constraints**: Prevents runaway weights and "softmax collapse"
2. **Orthogonalized Gradients**: Promotes broader exploration by reducing update redundancy
3. **Second-Order Information**: Approximates Hessian diagonal for better update directions
4. **Layer-wise Scaling**: Matches update size to layer shape for synchronized learning

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

## 📈 Visualization

The visualization script generates:

1. **Grokking Comparison**: Box plots and violin plots comparing Muon vs AdamW
2. **Learning Curves**: Training and validation accuracy over time
3. **Ablation Studies**: Effect of different components
4. **Summary Tables**: Statistical summaries and success rates

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
- **Visualization**: Interactive plots and dashboards

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Original authors for the groundbreaking research
- PyTorch team for the excellent framework
- Open source community for inspiration and tools

---

**Note**: This is a reproduction study. For the original research, please refer to the cited papers and repositories.
