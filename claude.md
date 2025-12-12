# RecursivePeptide: TRM for De Novo Peptide Sequencing

## Project Overview

**Goal**: Implement a Tiny Recursive Model (TRM) for de novo peptide sequencing from mass spectrometry (MS/MS) data.

**Core Insight**: Unlike standard autoregressive models that generate sequences in a single pass, this model uses a recursive loop to iteratively refine predictions. Mass spectrometry provides hard physical constraints (fragment masses must match observed peaks), making this domain uniquely suited for iterative refinement—the model can learn to "check its work" against mass constraints and self-correct.

**Key Innovation**: We add a mass-matching auxiliary loss that exploits the physical constraints of mass spectrometry—something not present in TRM's original puzzle-solving domains (Sudoku, ARC-AGI, Mazes).

**Reference**: "Less is More: Recursive Reasoning with Tiny Networks" (Jolicoeur-Martineau, 2025)

---

## Architecture Overview

### Core TRM Loop

```
For each supervision step t ∈ {1, ..., T}:
    For n latent reasoning steps:
        z = f_θ(x, y, z)           # Update latent state ("think")
    y = g_θ(y, z)                  # Update sequence prediction ("act")
    Loss_t = CE(y, target) + λ * SpectrumMatch(y, x)

Total Loss = Σ_t w_t * Loss_t     # Deep supervision with weighted sum
```

**Key Components**:
- **x** (Input): MS/MS spectrum (peaks with m/z + intensities) + precursor mass/charge
- **y** (Hypothesis): Predicted amino acid sequence (probability distribution)
- **z** (Latent State): Hidden "scratchpad" for reasoning
- **T**: Number of supervision steps (default: 8)
- **n**: Number of latent steps per supervision (default: 4)

### Model Structure

1. **Spectrum Encoder**: Transformer that encodes peak embeddings (mass + intensity)
2. **Recursive Decoder**: Shared weights used at every step for refinement
3. **Deep Supervision**: Loss computed at ALL steps, not just final output

### Critical Design Decisions

**Deep Supervision** (Core TRM Innovation):
- Compute loss at every supervision step, not just the end
- Forces model to learn a trajectory of improvement
- Uses weighted sum (linear weighting by default: later steps weighted more)

**Soft Embeddings**:
- Convert probability distributions → embeddings via learned projection
- Enables gradient flow through recursive loop
- Different from hard argmax decoding

**EMA (Critical for Stability)**:
- Exponential moving average of parameters (decay=0.999)
- Used for validation and final predictions
- TRM paper reports instability without EMA

**Mass-Matching Auxiliary Loss** (Domain-Specific):
- Computes expected theoretical fragment masses from soft predictions
- Differentiable: E[mass_i] = Σ P(aa) * mass(aa)
- Matches theoretical peaks to observed peaks via soft assignment
- Guides model toward physically plausible sequences

---

## Domain Knowledge: Mass Spectrometry

### Physical Constraints

**Peptide Mass Calculation**:
```
Peptide mass = Σ(residue masses) + H2O
```

**Fragment Ion Types**:
- **b-ions**: N-terminal fragments (mass = sum from N-terminus)
- **y-ions**: C-terminal fragments (mass = sum from C-terminus + H2O)
- Both provide constraints on valid sequences

**Key Physics**:
- All masses are monoisotopic (from UniProt reference)
- Water mass = 18.01056 Da
- Proton mass = 1.00727 Da
- Fragment masses must match observed peaks within tolerance (~0.5 Da or 20 ppm)

### Isobaric Ambiguities

**Unresolvable by mass alone**:
- I/L (Isoleucine/Leucine): Identical mass (113.08406 Da)
- K/Q (Lysine/Glutamine): Very similar mass (Δ = 0.036 Da)

These require additional context or cannot be distinguished.

### Curriculum Learning Rationale

Progressive difficulty helps model learn refinement behavior:

**Stage 1 - Clean** (0-10K steps):
- Short peptides (7-10 AA)
- Perfect spectra (no noise, no dropout)
- Learn basic mass-to-sequence mapping

**Stage 2 - Moderate** (10K-25K steps):
- Medium length (7-15 AA)
- 10% peak dropout, 5 noise peaks
- Learn to handle missing information

**Stage 3 - Realistic** (25K-50K steps):
- Full length (7-20 AA)
- 20% dropout, 10 noise peaks, 20 ppm mass error
- Matches real MS/MS conditions

---

## Development Guidelines

### Running Tests

**Critical: Physics tests must pass 100%**
```bash
pytest tests/test_physics.py -v
```
All mass calculations validated against UniProt reference values.

**All tests**:
```bash
pytest tests/ -v
```

### Training

**Quick validation** (overfit test):
```bash
python3 scripts/overfit_test.py
```
Target: Loss < 0.3, Token accuracy > 80% after 500 steps

**Full training** (with curriculum):
```bash
python3 scripts/train_ms2pip.py
```
100K steps, extended curriculum, automatic difficulty progression

### Configuration

Edit `configs/optimized_extended.yaml` to customize:
- Model size (hidden_dim, num_layers, num_heads)
- Training (batch_size, learning_rate)
- Curriculum stages
- MS2PIP settings (ms2pip_model, top_k_peaks)
- EMA settings (critical: keep enabled!)

### Key Files

**Core Implementation**:
- `src/constants.py`: Amino acid masses, vocabulary (CRITICAL - physics foundation)
- `src/data/ms2pip_dataset.py`: MS2PIP-based data generation
- `src/model/decoder.py`: Recursive core (TRM innovation)
- `src/training/losses.py`: Deep supervision + spectrum matching

**Entry Points**:
- `scripts/train_ms2pip.py`: Main training script
- `scripts/overfit_test.py`: Validation
- `configs/optimized_extended.yaml`: Optimized configuration

---

## Current Status

### ✅ Implemented (MVP Complete)

**Data Pipeline**:
- MS2PIP-based realistic spectrum generator
- Sinusoidal mass embeddings
- Infinite dataset with curriculum support

**Model Architecture**:
- Spectrum encoder (Transformer, 2 layers, 4 heads)
- Recursive decoder with shared weights
- ~945K parameters (MVP config)
- Tested and working

**Training System**:
- Deep supervision loss (linear weighting)
- Spectrum matching auxiliary loss
- 3-stage curriculum learning
- EMA for stability
- Full trainer with logging

**Validation**:
- Physics tests: 18/18 passing (100%)
- Overfit test: 86.2% token accuracy, 62.5% sequence accuracy
- Model successfully learns

### 🔄 In Progress

- Full training run (50K steps with curriculum)
- Performance benchmarking on clean vs. noisy synthetic data

### 📋 Roadmap (Post-MVP)

**Near-term**:
- Ablation studies (T=1 vs T=8, with/without spectrum loss)
- Beam search inference
- Uncertainty quantification (entropy thresholds)

**Medium-term**:
- PTM support (phosphorylation, oxidation, etc.)
- Real data pipeline (pyteomics integration)
- Extended peptide lengths (25-30 AA)

**Long-term**:
- Nine-Species benchmark evaluation
- Comparison with Casanovo baseline
- I/L ambiguity notation
- Attention visualization

---

## Performance Targets

### Synthetic Data (Clean)
- Token accuracy: >85%
- Sequence accuracy: >60%
- Mass error: <10 ppm

### Synthetic Data (Realistic)
- Token accuracy: >70%
- Sequence accuracy: >40%

### Ablation Expectations
- **Recursion benefit**: T=8 should beat T=1 by 10-15%
- **Spectrum loss benefit**: Should add 5-10% improvement
- **Curriculum benefit**: Faster convergence vs. fixed difficulty

---

## Important Constraints & Notes

### Physics Validation
- **All mass calculations must be validated** against known reference values
- Tests in `tests/test_physics.py` are non-negotiable
- Wrong masses = meaningless results

### I/L Ambiguity
- Isoleucine and Leucine have identical mass
- Model cannot distinguish them without additional context
- Should be explicitly noted in uncertainty quantification

### EMA is Critical
- Training unstable without EMA
- Use decay=0.999 (validated in TRM paper)
- Always use EMA model for validation/inference

### Spectrum Loss Gradients
- Must use soft embeddings (not hard argmax)
- Expected mass: E[m] = Σ P(aa) * mass(aa)
- Enables gradient flow through probability distributions

### Curriculum Progression
- Don't skip curriculum stages
- Model needs easy examples first to learn refinement behavior
- Jumping straight to hard examples leads to poor convergence

---

## Code Quality Standards

### When Making Changes

1. **Run physics tests first**: `pytest tests/test_physics.py -v`
2. **Update tests** if changing mass calculations or spectrum generation
3. **Document domain-specific logic** (e.g., why certain loss functions, why EMA)
4. **Maintain type hints** throughout codebase
5. **Test on overfit scenario** before full training

### Architecture Principles

- **Modularity**: Each component (encoder, decoder, losses) is independent
- **Configuration-driven**: No hardcoded hyperparameters
- **Physics-first**: Always validate against ground truth physics
- **Deep supervision**: Never remove loss computation from intermediate steps

---

## Getting Help

### Common Issues

**"Model not learning"**:
1. Run overfit test first - can it memorize 1 batch?
2. Check physics tests pass
3. Verify EMA is enabled
4. Check gradient clipping is active

**"Loss exploding"**:
1. Ensure gradient clipping (max_norm=1.0)
2. Verify EMA is enabled
3. Reduce learning rate
4. Check for NaN in mass calculations

**"Poor accuracy on realistic data"**:
1. Did you use curriculum learning?
2. Is model overfitting clean data?
3. Check spectrum loss weight is appropriate
4. May need more training steps

### Resources

- TRM Paper: Jolicoeur-Martineau et al., 2025
- README.md: Usage instructions and quick start
- IMPLEMENTATION.md: Detailed implementation notes
- Tests: See `tests/` for examples of correct behavior

---

## License

MIT - See LICENSE file
