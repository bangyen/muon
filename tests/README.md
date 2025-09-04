# Muon Optimizer Test Suite

This directory contains comprehensive tests for the Muon Optimizer implementation, validating the reproduction of the paper "Muon Optimizer Accelerates Grokking".

## Test Structure

```
tests/
├── __init__.py              # Test package initialization
├── conftest.py             # Test configuration and utilities
├── test_optimizer.py       # Muon optimizer tests
├── test_model.py           # Transformer model tests
├── test_dataset.py         # Dataset and data loading tests
├── test_grokking.py        # Grokking detection and analysis tests
├── test_integration.py     # End-to-end integration tests
└── run_tests.py            # Test runner script
```

## Test Categories

### 1. Optimizer Tests (`test_optimizer.py`)
- **Muon Optimizer Core Functionality**: Basic optimization steps, parameter validation
- **Spectral Norm Constraints**: Testing weight constraint mechanisms
- **Orthogonal Gradient Updates**: Validation of gradient orthogonalization
- **Second-Order Information**: Testing Hessian approximation
- **Comparison with AdamW**: Performance and convergence comparisons
- **Error Handling**: Invalid parameters, sparse gradients, etc.

### 2. Model Tests (`test_model.py`)
- **Transformer Architecture**: Multi-head attention, RMSNorm, RoPE
- **Softmax Variants**: Standard, stablemax, and sparsemax implementations
- **Forward Pass**: Various input shapes and batch sizes
- **Gradient Flow**: Backpropagation validation
- **Model Parameters**: Parameter count and initialization
- **Device Transfer**: CPU/GPU compatibility

### 3. Dataset Tests (`test_dataset.py`)
- **Modular Arithmetic Tasks**: Addition, multiplication, division, exponentiation, GCD
- **Parity Task**: 10-bit binary string parity
- **Data Generation**: Correctness of mathematical operations
- **Train/Validation Split**: Proper data splitting
- **Vocabulary Creation**: Tokenization and vocabulary management
- **Data Loading**: PyTorch DataLoader integration

### 4. Grokking Tests (`test_grokking.py`)
- **Grokking Detection**: Algorithm for detecting grokking epochs
- **Training Dynamics**: Accuracy tracking and loss monitoring
- **Optimizer Comparison**: Muon vs AdamW grokking performance
- **Statistical Analysis**: Mean grokking epochs, standard deviations
- **Grokking Reproduction**: Minimal setups for grokking phenomenon

### 5. Integration Tests (`test_integration.py`)
- **End-to-End Training**: Complete training loops
- **Performance Benchmarks**: Speed and memory comparisons
- **Reproducibility**: Seed-based reproducibility testing
- **Error Handling**: Edge cases and numerical stability
- **Cross-Component Integration**: Model-optimizer-dataset interaction

## Running Tests

### Prerequisites
```bash
# Activate virtual environment
source venv/bin/activate

# Install test dependencies
pip install pytest pytest-cov
```

### Basic Test Execution
```bash
# Run all tests
python tests/run_tests.py

# Run with verbose output
python tests/run_tests.py --verbose

# Run with coverage report
python tests/run_tests.py --coverage
```

### Specific Test Suites
```bash
# Run only optimizer tests
python tests/run_tests.py --suite optimizer

# Run only model tests
python tests/run_tests.py --suite model

# Run only dataset tests
python tests/run_tests.py --suite dataset

# Run only grokking tests
python tests/run_tests.py --suite grokking

# Run only integration tests
python tests/run_tests.py --suite integration
```

### Using pytest directly
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_optimizer.py

# Run specific test class
pytest tests/test_optimizer.py::TestMuonOptimizer

# Run specific test method
pytest tests/test_optimizer.py::TestMuonOptimizer::test_initialization

# Run tests matching pattern
pytest -k "grokking"
```

## Test Configuration

### Test Config (`conftest.py`)
The test configuration provides:
- **TestConfig**: Standard configurations for model, optimizer, dataset, and training
- **Fixtures**: Device, seed, temporary directory, and other test utilities
- **Helper Functions**: Accuracy computation, grokking detection, model creation

### Pytest Configuration (`pytest.ini`)
- **Test Discovery**: Automatic test file and class discovery
- **Output Options**: Verbose output, short tracebacks
- **Markers**: Test categorization (slow, integration, unit, etc.)
- **Coverage**: Code coverage reporting

## Test Coverage

The test suite aims for comprehensive coverage of:

1. **Core Functionality**: All major components and features
2. **Edge Cases**: Error conditions and boundary cases
3. **Integration**: Component interactions and end-to-end workflows
4. **Performance**: Speed and memory usage validation
5. **Reproducibility**: Seed-based deterministic behavior
6. **Paper Reproduction**: Validation against paper claims

## Expected Test Results

### Paper Validation
Based on the paper "Muon Optimizer Accelerates Grokking", tests should validate:

- **Grokking Acceleration**: Muon should achieve grokking earlier than AdamW
- **Statistical Significance**: Mean grokking epoch reduction from 153.09 to 102.89
- **Task Coverage**: All 6 tasks (add, mul, div, exp, gcd, parity)
- **Softmax Variants**: Standard, stablemax, and sparsemax compatibility

### Performance Benchmarks
- **Training Speed**: Muon should complete training in reasonable time
- **Memory Usage**: Efficient memory utilization
- **Numerical Stability**: Robust handling of extreme values
- **Convergence**: Proper loss reduction and accuracy improvement

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure virtual environment is activated
2. **CUDA Errors**: Tests automatically fall back to CPU if CUDA unavailable
3. **Memory Issues**: Tests use small models to avoid memory problems
4. **Timeout Issues**: Long-running tests are marked with `@pytest.mark.slow`

### Debug Mode
```bash
# Run with maximum verbosity
pytest -vvv tests/

# Run single test with debug output
pytest -s tests/test_optimizer.py::TestMuonOptimizer::test_initialization
```

## Contributing

When adding new tests:

1. **Follow Naming Convention**: `test_*.py` files, `Test*` classes, `test_*` methods
2. **Use Fixtures**: Leverage existing fixtures from `conftest.py`
3. **Add Markers**: Use appropriate pytest markers for test categorization
4. **Documentation**: Include docstrings explaining test purpose
5. **Coverage**: Ensure new functionality is adequately tested

## References

- [Muon Optimizer Accelerates Grokking](https://arxiv.org/pdf/2504.16041)
- [Original Grokking Paper](https://arxiv.org/abs/2201.02177)
- [Muon Optimizer Repository](https://github.com/KellerJordan/Muon)
