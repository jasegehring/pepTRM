# PepTRM: Recursive Transformer for De Novo Peptide Sequencing
**Status**: Active Development | **Latest**: Step embeddings + gated residuals implemented
**Last Updated**: December 13, 2024 (Architecture v2.1)

---

## Project Overview

**Goal**: Implement a Tiny Recursive Model (TRM) for de novo peptide sequencing from mass spectrometry (MS/MS) data, capable of handling noisy real-world spectra through iterative refinement.

**Core Insight**: Unlike standard autoregressive models that generate sequences in a single pass, this model uses a recursive loop to iteratively refine predictions. Mass spectrometry provides hard physical constraints (fragment masses must match observed peaks), making this domain uniquely suited for iterative refinement—the model can learn to "check its work" against mass constraints and self-correct.

**Key Innovations**:
1. **Physics-aware losses**: Spectrum matching + precursor mass constraints
2. **Iterative refinement**: 8 supervision steps with 6 latent reasoning steps each
3. **Curriculum learning**: Progressive noise introduction for robustness
4. **Real data integration**: Seamless synthetic → real data transition

**Reference**: "Less is More: Recursive Reasoning with Tiny Networks" (Jolicoeur-Martineau, 2025)

---

## Architecture Overview

### Core TRM Loop (v2.1 - with Step Embeddings & Gated Residuals)

```
Initialize: y₀ (sequence logits), z₀ (latent state)

For each supervision step t ∈ {0, ..., 7}:
    step_emb = StepEmbedding(t)     # NEW: Model knows which iteration it's on

    For n=6 latent reasoning steps:
        z = LayerNorm(z + TransformerBlock(z + step_emb, spectrum_encoding))  # Think

    # NEW: GRU-style gated residual (not just OutputHead)
    candidate = OutputHead(z + step_emb)
    gate = Sigmoid(GateHead(z + step_emb))  # gate ∈ [0,1], initialized conservative
    y = (1 - gate) * y_prev + gate * candidate  # Interpolate prev and new

    Loss_t = CrossEntropy(y, target) + λ_precursor * PrecursorLoss(y, precursor_mass)

Total Loss = Σ_t w_t * Loss_t     # Deep supervision with exponential weighting
```

**Key v2.1 Improvements**:
- **Step embeddings**: Model knows "am I on step 0 (rough draft) or step 7 (polishing)?"
- **Gated residuals**: Model decides per-position how much to update vs keep previous
- **Conservative init**: Gate bias = -2.0, so initial gate ≈ 12% (must "earn" right to change)

**Key Components**:
- **Spectrum Encoding** (x): MS/MS peaks (m/z + intensities) encoded via Transformer
- **Sequence Hypothesis** (y): Predicted amino acid sequence (logits over vocabulary)
- **Latent State** (z): Hidden "scratchpad" for multi-step reasoning
- **Supervision Steps**: 8 (allows 8 attempts to refine prediction)
- **Latent Steps**: 6 per supervision (gives model time to "think")

### Model Architecture

```
Input Spectrum (100 peaks × 2 features)
    ↓
┌─────────────────────────────────┐
│  Spectrum Encoder               │
│  - Peak Embeddings (mass + int) │
│  - 3-layer Transformer          │
│  - 6 attention heads            │
│  Hidden dim: 384                │
└─────────────────────────────────┘
    ↓ (encoded spectrum)
┌─────────────────────────────────┐
│  Recursive Decoder Core         │
│  - Shared weights (reused 8×)   │
│  - 3-layer Transformer          │
│  - Cross-attention to spectrum  │
│  - Soft embeddings from probs   │
│  - Residual connections         │
└─────────────────────────────────┘
    ↓ (8 refinement iterations)
┌─────────────────────────────────┐
│  Deep Supervision               │
│  - CE loss at ALL 8 steps       │
│  - Exponential weighting        │
│  - Precursor mass constraint    │
└─────────────────────────────────┘
    ↓
Output: Refined sequence prediction
```

**Model Size**: 12.5M parameters
**Training Speed**: ~4.4 it/s with torch.compile (RTX 4090)

---

## Current Status (December 2024)

### ✅ **Achievements**

**Infrastructure**:
- ✅ torch.compile working (2.1x speedup with batch_size=80)
- ✅ Mixed precision training (bfloat16)
- ✅ Gradient accumulation for memory efficiency
- ✅ EMA for training stability
- ✅ W&B integration for experiment tracking

**Data Pipeline**:
- ✅ MS2PIP synthetic data (infinite, configurable noise)
- ✅ ProteomeTools dataset loader (21M high-quality synthetic spectra)
- ✅ Nine-Species dataset loader (2.8M real PSMs from 9 species)
- ✅ Identical interface for all datasets (easy switching)
- ✅ Curriculum learning with dynamic noise adjustment

**Model Performance**:
- ✅ **91.3% token accuracy** on clean synthetic data
- ✅ **58% sequence accuracy** on clean data
- ✅ Physics tests: 100% passing
- ✅ Training stable with EMA

### 🔄 **In Progress**

**Architecture v2.1** (Step Embeddings + Gated Residuals) - JUST IMPLEMENTED:
- ✅ Step embeddings: Model knows iteration index (0-7)
- ✅ GRU-style gated residuals in answer_step
- ✅ Gate initialized conservative (bias=-2.0, ~12% initial activation)
- ✅ Local metrics logging (logs/metrics_*.jsonl)
- ✅ Analysis tool (tools/analyze_runs.py)
- **Next**: Train and validate that recursion plateau is fixed
- **Goal**: See progressive improvement across ALL 8 steps

**Previous Run** (Aggressive Noise Curriculum at 57K steps):
- Recursion plateau confirmed (steps 1-6 all similar loss)
- val_hard at 90% while training at 30% (curriculum harder than validation)
- ProteomeTools: 11% token accuracy (better than random but struggling)

### ❌ **Known Challenges**

1. **Recursion Plateau** 🟡 **FIX IMPLEMENTED - NEEDS TESTING**
   - Model refines step 0→1 (loss: 0.98 → 0.83) ✅
   - But plateaus steps 2-7 (all ~0.82 loss) ❌
   - Only using ~2 of 8 available refinement steps
   - **Fix implemented**: Step embeddings + GRU-style gated residuals (v2.1)
   - **Next**: Train and validate the fix works

2. **Spectrum Loss Challenges**
   - Stuck at 94-96% with sigma=0.2 Da (only 4-6% coverage)
   - Root cause: Sigma too narrow for current model accuracy
   - With 9% token errors → 50-200 Da fragment mass errors
   - **Solution**: Disabled for now, testing precursor loss only

3. **Sim-to-Real Gap** (Unknown)
   - Haven't tested on real data yet
   - Expect lower accuracy (60-75%) on Nine-Species benchmark

---

## Key Design Decisions

### **1. Deep Supervision with Exponential Weighting**

**Why**: Force model to optimize ALL refinement steps, not just early ones

**Iteration Weights**:
- Linear (old): [2.8%, 5.6%, 8.3%, ..., 22%] → step 7 only gets 22% of gradient
- **Exponential (new)**: [0.4%, 0.8%, 1.6%, ..., 50%] → step 7 gets 50% of gradient

**Impact**: Should unlock multi-step refinement

### **2. Curriculum Learning (Clean/Noisy Mixing)**

**Philosophy**: Start with mostly clean data, gradually increase noise proportion

**Schedule**:
```
Stage 1 (0-10K):   80% clean, 20% noisy → Learn basics
Stage 2 (10K-20K): 60% clean, 40% noisy → More challenge
Stage 3 (20K-30K): 40% clean, 60% noisy → Majority noisy
Stage 4 (30K-45K): 20% clean, 80% noisy → Force robustness
Stage 5 (45K+):     0% clean, 100% noisy → Pure realistic data
```

**Noise Profile** (Real-world MS/MS):
- 5-30 random noise peaks
- 15-45% peak dropout (missing fragments)
- 5-20 ppm mass error
- 20-70% intensity variation

### **3. Physics-Based Losses**

**Precursor Mass Loss** (Enabled):
```python
# Normalize mass error by average AA mass (110 Da)
# Balances gradient magnitudes with CrossEntropy loss
predicted_mass = Σ P(aa_i) * mass(aa_i) + H2O_mass
error_da = predicted_mass - observed_precursor_mass
loss = SmoothL1(error_da / 110.0)  # Normalized to "AA units"
```

**Spectrum Matching Loss** (Disabled for now):
- Current: Gaussian kernel with sigma=0.2 Da
- Problem: Too narrow for 9% token error rate
- Future: Adaptive sigma curriculum (10 Da → 0.5 Da)
- Alternative: Matched filter loss (recall-only, robust to dropout)

### **4. EMA (Exponential Moving Average)**

**Critical for stability**:
- Decay: 0.999
- Always use EMA model for validation/inference
- TRM paper reports instability without EMA

### **5. Soft Embeddings**

**Enable gradient flow through recursive loop**:
```python
# Convert probability distribution to continuous embedding
probs = softmax(logits)  # (batch, seq_len, vocab_size)
soft_embedding = learned_projection(probs)  # Differentiable
```

**Why**: Allows model to receive gradient signal about "almost correct" predictions

---

## Domain Knowledge: Mass Spectrometry

### Physical Constraints

**Peptide Mass**:
```
Precursor mass = Σ(amino acid masses) + H2O_mass
```

**Fragment Ions**:
- **b-ions**: N-terminal fragments (mass = cumsum from left)
- **y-ions**: C-terminal fragments (mass = cumsum from right + H2O)
- **b++, y++**: Doubly-charged fragments (common in charge 2+ precursors)
  - Formula: (neutral_mass + 2 × proton_mass) / 2
  - **Status**: Not yet implemented (critical missing physics)

**Key Constants** (from UniProt):
- Water mass: 18.01056 Da
- Proton mass: 1.00727 Da
- Average AA mass: ~110 Da

**Tolerances**:
- High-resolution MS: 5-20 ppm (0.01-0.04 Da at m/z 2000)
- Low-resolution MS: 50-100 ppm

### Isobaric Ambiguities

**Unresolvable by mass alone**:
- **I/L** (Isoleucine/Leucine): Identical mass (113.08406 Da)
- **K/Q** (Lysine/Glutamine): Very similar (Δ = 0.036 Da, within instrument tolerance)

**Current handling**: Model must learn from context (future: uncertainty quantification)

---

## Development Guidelines

### Quick Start

**1. Install Dependencies**:
```bash
pip install torch wandb omegaconf ms2pip
```

**2. Run Physics Tests** (Must pass 100%):
```bash
pytest tests/test_physics.py -v
```

**3. Quick Validation** (Overfit test):
```bash
python scripts/overfit_test.py
```
Target: >80% token accuracy after 500 steps on 1 batch

**4. Full Training**:
```bash
# Current aggressive noise run
python scripts/train_aggressive_noise.py

# Or standard optimized training
python scripts/train_optimized.py
```

### Configuration

**Main config**: `configs/aggressive_noise_test.yaml`

```yaml
model:
  hidden_dim: 384
  num_encoder_layers: 3
  num_decoder_layers: 3
  num_heads: 6
  num_supervision_steps: 8
  num_latent_steps: 6

training:
  batch_size: 80
  learning_rate: 1.5e-4
  use_amp: true
  use_compile: true
  iteration_weights: 'exponential'  # Force late-step optimization
  spectrum_weight: 0.0  # Disabled (sigma too narrow)

  # Curriculum
  use_curriculum: true
```

### Key Files

**Core Implementation**:
```
src/
├── constants.py              # Amino acid masses (physics foundation)
├── data/
│   ├── ms2pip_dataset.py    # MS2PIP synthetic data
│   ├── proteometools_dataset.py  # High-quality synthetic
│   └── nine_species_dataset.py   # Real biological data
├── model/
│   ├── encoder.py           # Spectrum encoder
│   ├── decoder.py           # Recursive core (TRM innovation)
│   └── trm.py              # Full model
└── training/
    ├── losses.py            # Deep supervision + physics losses
    ├── trainer_optimized.py # Optimized trainer (compile, AMP, etc.)
    └── curriculum_aggressive_noise.py  # Current curriculum
```

**Entry Points**:
```
scripts/
├── train_aggressive_noise.py  # Current experiment
├── train_optimized.py         # Standard optimized training
└── overfit_test.py           # Validation test
```

**Analysis Tools**:
```
tools/
└── analyze_runs.py           # Parse wandb & local logs
                              # --live, --run <ID>, --local, --compare
logs/
└── metrics_*.jsonl           # Local metrics (every 1000 steps)
```

**Documentation**:
```
docs/
├── MASTER_ROADMAP_2024-12-12.md  # Complete development plan
├── README_DOCUMENTATION.md        # Documentation index
├── training_with_real_data.md    # Real data guide
├── proteometools_dataset.md      # ProteomeTools setup
└── nine_species_dataset.md       # Nine-Species benchmark
```

---

## Roadmap & Future Work

**See `docs/MASTER_ROADMAP_2024-12-12.md` for complete details.**

### **Phase 1: Unlock Multi-Step Recursion** 🔴 **CURRENT**

**Goal**: Prove model can use all 8 refinement steps + achieve length generalization

**Current Experiment** (Dec 13):
- ⏳ Two-tier noise curriculum (`curriculum_complex_noise.py`)
  - Flat length 7-30 from step 0 (no progressive length)
  - "Grass" noise (low-intensity background) + "Spikes" (high-intensity decoys)
  - Signal suppression (crush real peaks to break intensity bias)
  - Pure foundation phase (5K steps, 100% clean, all lengths)
- ⏳ DataLoader iterator bug FIXED (workers now see curriculum updates)
- ⏳ Monitor: length-bucketed accuracy, val_hard, real data transfer

**Previous Run Learnings** (36K steps):
- ✅ Bug fix working (better val_hard performance)
- ⚠️ Short peptides plateau early (positional overfitting)
- ⚠️ Long peptides struggle (introduced too late)
- 📈 ProteomeTools reached 12% token acc (better than before!)

**Architecture v2.1 (Just Implemented)**:
- ✅ Step embeddings (model knows iteration 0-7)
- ✅ GRU-style gated residuals (conservative gate init, ~12% activation)
- 🔲 Train and validate recursion plateau is fixed
- 🔲 Integrate refinement tracker (monitor sequence changes per step)

### **Phase 1.5: Architecture Experiments** (After curriculum validated)

**Experiment: Perception-Heavy Architecture**
```yaml
# Hypothesis: Bigger encoder for noisy spectra, smaller decoder for fast iteration
hidden_dim: 384        # Keep
num_heads: 6           # Keep
encoder_layers: 6      # INCREASE (was 3) - Better noise parsing
decoder_layers: 2      # DECREASE (was 3) - Faster recursive loop
supervision_steps: 8   # Keep
latent_steps: 4        # DECREASE (was 6) - Peptide logic may need less
```
- Rationale: Spectrum parsing is hard (150+ noise peaks), sequence logic is constrained by mass
- Tradeoff: ~15M params, 32 thinking steps (vs 12.5M, 48 steps)
- Test AFTER proving curriculum works

**Experiment: Tiny TRM Challenge**
- Can we match Casanovo (47M) with <5M params?
- Slim config: hidden=256, enc=2, dec=2 → 3.7M params
- Test AFTER architecture experiments

### **Phase 2: Advanced Architecture Features**

**Mass Gap Token** (for unknown modifications):
- Add `[GAP]` token to vocabulary
- Predict mass shift per position: delta_mass_head(z) → ±500 Da
- Compute fragment masses: base_mass + predicted_delta
- **Use case**: PTMs, non-standard AAs, open search

**Residual Spectrum Embedding** (physics feedback):
- Render observed vs predicted spectra
- Compute residual: what did we miss?
- Feed through CNN to detect "shift patterns"
- Inject into latent state for refinement
- **Use case**: Modified peptides, mass shift detection

### **Phase 3: Spectrum Loss Improvements** (Week 5)

**Matched Filter Loss** (recall-only, robust to dropout):
- Trust observed peaks (they're real)
- Don't trust absence (peaks may have dropped out)
- Let precursor loss handle precision (prevent hallucinations)

**Adaptive Sigma Curriculum**:
- Start wide: 10 Da (forgiving, learn rough patterns)
- Narrow over time: 10 → 5 → 2 → 0.5 Da
- Match sigma to model accuracy

### **Phase 4: Real Data Transition** (Weeks 6-8) ⭐

**Mixed Synthetic/Real Curriculum** (Recommended):
```
100% synthetic → 80/20 → 50/50 → 20/80 → 100% real
Over 60-80K steps
```

**Benchmarks**:
- 🔲 ProteomeTools pre-training (21M spectra, synthetic)
- 🔲 Nine-Species 9-fold CV (2.8M PSMs, real data)
- 🔲 Compare vs Casanovo baseline

**Expected Performance**:
- ProteomeTools: 85-90% token accuracy
- Nine-Species: 60-75% token accuracy (real data is harder)

### **Phase 5: Missing Physics & Advanced Features** (Weeks 9-12)

**Critical Missing Physics**:
- 🔲 Doubly-charged ions (b++, y++) - common in real data
- 🔲 Common PTMs (oxidation, phosphorylation, acetylation)

**Production Features**:
- 🔲 Uncertainty quantification (entropy-based confidence)
- 🔲 Beam search inference
- 🔲 I/L ambiguity notation

---

## Performance Targets

### **Synthetic Data**

| Dataset | Token Acc | Sequence Acc | Notes |
|---------|-----------|--------------|-------|
| MS2PIP (clean) | **>90%** | **>60%** | Current: 91.3% / 58% ✅ |
| MS2PIP (noisy) | >75% | >30% | 45% dropout, 30 noise peaks |
| ProteomeTools | >85% | >50% | High-quality synthetic |

### **Real Data**

| Dataset | Token Acc | Sequence Acc | Notes |
|---------|-----------|--------------|-------|
| Nine-Species | >70% | >30% | Averaged across 9 folds |
| Nine-Species (easy) | >80% | >50% | Clean subset |

### **Ablation Studies** (Planned)

| Configuration | Expected Impact |
|---------------|-----------------|
| T=8 vs T=1 | Recursion adds +10-15% accuracy |
| With vs without spectrum loss | +5-10% (when sigma is right) |
| With vs without curriculum | Faster convergence, better final accuracy |
| Exponential vs linear weighting | Unlocks multi-step refinement |

---

## Important Constraints & Notes

### **Physics Validation**
- **All mass calculations must be validated** against UniProt reference values
- Tests in `tests/test_physics.py` are non-negotiable
- Wrong masses = meaningless results

### **EMA is Critical**
- Training unstable without EMA (from TRM paper)
- Use decay=0.999
- Always use EMA model for validation/inference

### **Recursion Debugging**
When recursion isn't working:
1. Check iteration weighting (exponential > linear)
2. Verify noise is challenging enough (easy data → no need to refine)
3. Monitor per-step losses (should see gradient: step_0 > step_7)
4. Check refinement tracker (edit_rate per step)

### **Curriculum is Essential**
- Don't skip curriculum stages
- Model needs easy examples first
- Catastrophic forgetting happens with distribution shifts (0 noise → 5 noise peaks)
- Use gradual transitions (0 → 1 → 2 → 5 noise peaks)

### **I/L Ambiguity**
- Isoleucine and Leucine are **indistinguishable** by mass alone
- Model should learn to output one consistently (or indicate uncertainty)
- Future: Explicit uncertainty quantification

---

## Monitoring & Debugging

### **Key W&B Metrics**

**Per-step losses** (recursion diagnostics):
```python
ce_step_0: 0.98  # Initial rough guess
ce_step_1: 0.83  # First refinement (working! ✅)
ce_step_2: 0.82  # Plateau starts (problem ❌)
ce_step_7: 0.82  # Should be << ce_step_0 if recursion works
```

**Refinement tracker** (when integrated):
```python
edit_rate_step_{t}:      # % of tokens changed at step t
accuracy_step_{t}:       # % correct at step t
improvement_step_{t}:    # Net improvement from edits
```

**Loss breakdown**:
```python
train/ce_loss:          # Cross-entropy
train/precursor_loss:   # Precursor mass constraint
train/spectrum_loss:    # Spectrum matching (disabled for now)
```

### **Common Issues**

**"Model not refining past step 1"**:
- ✅ Switch to exponential weighting
- ✅ Increase noise difficulty (force model to need recursion)
- ✅ Add step embeddings (model knows iteration 0-7)
- ✅ GRU-style gated residuals (conservative init, ~12% gate activation)

**"Spectrum loss stuck at 95%"**:
- ✅ Disable for now (sigma too narrow)
- 🔲 Implement adaptive sigma curriculum
- 🔲 Try matched filter loss (recall-only)

**"Poor accuracy on real data"**:
- 🔲 Pre-train on ProteomeTools first
- 🔲 Use mixed synthetic/real curriculum
- 🔲 Add doubly-charged ions (critical missing physics)
- 🔲 Fine-tune with lower learning rate

---

## Code Quality Standards

### **When Making Changes**

1. **Run physics tests first**: `pytest tests/test_physics.py -v`
2. **Test on overfit scenario**: Can it memorize 1 batch?
3. **Update tests** if changing mass calculations or data generation
4. **Document domain-specific logic** (why EMA, why certain losses)
5. **Maintain type hints** throughout codebase

### **Architecture Principles**

- **Modularity**: Each component (encoder, decoder, losses) is independent
- **Configuration-driven**: No hardcoded hyperparameters
- **Physics-first**: Always validate against ground truth
- **Deep supervision**: Loss at ALL intermediate steps
- **Curriculum learning**: Progressive difficulty essential

---

## Resources

### **Documentation**
- **MASTER_ROADMAP_2024-12-12.md**: Complete development plan with priorities
- **README_DOCUMENTATION.md**: Documentation index
- **training_with_real_data.md**: Real data pipeline guide
- **memory_management.md**: GPU optimization strategies

### **Papers**
- TRM: Jolicoeur-Martineau et al., 2025
- ProteomeTools: Gessulat et al., Nature Methods, 2019
- Nine-Species: Eloff et al., Scientific Data, 2024

### **Datasets**
- MS2PIP: On-the-fly synthetic (unlimited)
- ProteomeTools: 21M synthetic spectra (2.3 GB download)
- Nine-Species: 2.8M real PSMs (15-50 GB download)

---

## License

MIT - See LICENSE file

---

**Quick Links**:
- 📊 [Master Roadmap](docs/MASTER_ROADMAP_2024-12-12.md)
- 📚 [Documentation Index](docs/README_DOCUMENTATION.md)
- 🔬 [Training Guide](docs/training_with_real_data.md)
- ⚙️ [Memory Optimization](docs/memory_management.md)

**Status**: Active development, architecture v2.1 ready for training
**Next Milestone**: Validate step embeddings + gated residuals fix recursion plateau
