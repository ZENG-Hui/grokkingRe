# Reference Implementations

This folder contains the original implementations before the unification refactoring.

## Purpose

These scripts preserve the original paper-following implementations for reference:

- **sparse/**: Original sparse training implementation (following circuit sparsity paper)
- **dense/**: Original dense training baseline

## Contents

### Sparse Implementation
- `training_sparse.py` - Sparse training with L0 regularization
- `model_sparse.py` - Sparse model with RMSNorm and AbsTopK
- `config_sparse.py` - Sparse training configuration
- `sparse_utils.py` - Sparsity utility functions

### Dense Implementation  
- `training.py` - Original dense training
- `model.py` - Dense model with LayerNorm
- `cli.py` - Command-line interface for dense training

## Usage

These files are for reference only. For active development, use the unified scripts in the parent directory.

To run the original implementations:

```bash
# Sparse (original)
cd reference_implementations/sparse
python training_sparse.py

# Dense (original)
cd reference_implementations/dense
python cli.py
```

## Backup Date

Created: 2025-11-25
