# Recursive Peptide TRM

A Tiny Recursive Model (TRM) for de novo peptide sequencing from mass spectrometry data.

## Overview

This project implements a novel recursive deep learning architecture for predicting peptide sequences from MS/MS spectra. Unlike standard autoregressive models, it uses iterative refinement to progressively improve predictions, leveraging the physical constraints of mass spectrometry.

### Key Innovations

1. **Recursive Refinement**: Model iterates over T supervision steps, refining predictions each time
2. **Deep Supervision**: Loss computed at every step forces learning of improvement trajectory
3. **Mass-Matching Loss**: Domain-specific auxiliary loss that exploits physical constraints
4. **Curriculum Learning**: Progressive difficulty increase from clean to noisy synthetic data

## Architecture

```
Input: MS/MS Spectrum (peaks + intensities) + Precursor mass/charge
  ↓
Spectrum Encoder (Transformer)
  ↓
Recursive Core:
  For T supervision steps:
    For n latent steps: z = f(x, y, z)  # "Think"
    y = g(y, z)                          # "Act"
    Compute Loss
  ↓
Output: Amino acid sequence
```

**Model Size**: ~945K parameters (MVP config)

## Project Structure

```
recursive_peptide/
├── src/
│   ├── constants.py          # Amino acid masses, vocabulary
│   ├── data/
│   │   ├── synthetic.py      # Forward model (peptide → spectrum)
│   │   ├── encoding.py       # Sinusoidal mass embeddings
│   │   └── dataset.py        # PyTorch datasets
│   ├── model/
│   │   ├── layers.py         # Transformer components
│   │   ├── encoder.py        # Spectrum encoder
│   │   ├── decoder.py        # Recursive decoder (core TRM)
│   │   └── trm.py            # Full model
│   └── training/
│       ├── losses.py         # Deep supervision + spectrum matching
│       ├── metrics.py        # Accuracy metrics
│       ├── curriculum.py     # Curriculum scheduler
│       └── trainer.py        # Training loop with EMA
├── configs/
│   └── default.yaml          # Model & training config
├── scripts/
│   ├── train.py              # Training entry point
│   └── overfit_test.py       # Validation test
└── tests/
    ├── test_physics.py       # Mass calculation tests (18/18 ✓)
    └── test_synthetic.py     # Spectrum generation tests (10/10 ✓)
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

**Requirements**:
- Python 3.9+
- PyTorch 2.0+
- numpy, tqdm, omegaconf, wandb (optional)

## Usage

### Quick Start

Run the overfit test to verify installation:

```bash
python3 scripts/overfit_test.py
```

This trains on a single batch for 500 steps. Expected: loss decreases, accuracy improves.

### Training

Train with default configuration (curriculum learning enabled):

```bash
python3 scripts/train.py
```

The default config uses:
- 3-stage curriculum (clean → moderate → realistic)
- Deep supervision with linear weighting
- EMA for stability
- 50K training steps

### Configuration

Edit `configs/default.yaml` to customize:

**Model**:
```yaml
model:
  hidden_dim: 256
  num_supervision_steps: 8    # T in paper
  num_latent_steps: 4         # n latent reasoning steps
  num_heads: 4
```

**Training**:
```yaml
training:
  batch_size: 64
  learning_rate: 1.0e-4
  use_curriculum: true
  use_ema: true              # Critical for stability!
```

### Monitoring

Enable W&B logging in `scripts/train.py`:
```python
trainer = Trainer(..., use_wandb=True)
```

## Testing

Run all physics tests:
```bash
pytest tests/ -v
```

**Critical**: All physics tests must pass (18/18). These validate mass calculations.

## Current Status

### ✅ Completed (MVP)

- Core TRM architecture with recursive refinement
- Deep supervision loss
- Spectrum matching auxiliary loss
- Curriculum learning (3-stage)
- Synthetic data generation
- EMA training
- Physics validation (all tests passing)

### 🔄 In Progress

- Extended overfit testing (validating convergence)
- Full training run (50K steps)

### 📋 Planned

- Beam search inference
- Uncertainty quantification
- Real data pipeline (Nine-Species benchmark)
- PTM support
- Ablation studies (T=1 vs T=8, +/- spectrum loss)

## Model Details

### Recursive Loop

The core innovation is the recursive reasoning loop:

```python
y, z = initialize()  # Initial guess + latent state

for t in range(T):   # T supervision steps (e.g., 8)
    for _ in range(n):  # n latent steps (e.g., 4)
        z = f(x, y, z)   # Update latent ("think")
    y = g(y, z)          # Update prediction ("act")
    loss_t = CE(y, target) + λ*SpectrumMatch(y, x)
```

**Deep Supervision**: Loss at every step `t` forces model to learn trajectory of improvement.

### Mass-Matching Loss

Novel domain-specific loss exploiting physical constraints:

```python
# Compute expected fragment masses from soft predictions
E[b_i] = Σ P(aa_i) * mass(aa)  # Differentiable!

# Match theoretical to observed peaks
distance = |theoretical - observed|
loss = weighted_distance(intensities)
```

This guides the model toward physically plausible sequences.

### Curriculum Learning

Progressive difficulty increase:

1. **Clean** (0-10K steps): Short peptides, perfect spectra
2. **Moderate** (10K-25K): Medium peptides, 10% dropout, 5 noise peaks
3. **Realistic** (25K-50K): Full length, 20% dropout, 10 noise peaks, mass error

## Performance Targets

**Synthetic Data (Clean)**:
- Token accuracy: >85%
- Sequence accuracy: >60%

**Synthetic Data (Realistic)**:
- Token accuracy: >70%
- Sequence accuracy: >40%

**Recursion Benefit**:
- T=8 should beat T=1 by 10-15%

**Spectrum Loss Benefit**:
- Should add 5-10% improvement

## Citation

Based on "Less is More: Recursive Reasoning with Tiny Networks" (Jolicoeur-Martineau, 2025).

Applied to de novo peptide sequencing with mass spectrometry constraints.

## License

MIT

## Contact

For questions or issues, please open a GitHub issue.
