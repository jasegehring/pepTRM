# Proposed Improvements & Next Steps

This file tracks ideas and enhancements to implement after the baseline training run completes.

---

## High Priority 🔴

### 1. Add Protonation to Fragment Ions (Critical Physics Bug!)
**Status**: Not implemented
**Priority**: **CRITICAL**
**Effort**: 5 minutes

**Problem**: MS detectors measure **ionized** fragments, not neutral masses. Our synthetic data generates neutral masses, creating a systematic ~1 Da offset from real data.

**Physics**:
- b-ions: Detector sees `[M+H]+` = sum(residues) + **PROTON_MASS** (~1.007 Da)
- y-ions: Detector sees `[M+H2O+H]+` = sum(residues) + WATER_MASS + **PROTON_MASS**

**Current Code** (`src/data/synthetic.py`):
```python
# Line 115 (b-ions) - WRONG
mass = cumulative  # Neutral mass

# Line 145 (y-ions) - WRONG
mass = cumulative  # Neutral mass (already has H2O, but missing H+)
```

**Proposed Fix**:
```python
# Line 115 (b-ions)
mass = cumulative + PROTON_MASS  # [M+H]+

# Line 145 (y-ions)
mass = cumulative + PROTON_MASS  # [M+H2O+H]+
```

**Impact**:
- **Internally consistent now**: Both synthetic data AND loss function use neutral masses, so training works
- **Real data will fail**: Real .mzML files have +1 Da offset
- **Must fix before real data integration**

**Also update** `src/training/losses.py:compute_theoretical_peaks()`:
```python
# Line 156 (b-ions)
b_ions = torch.cumsum(residue_masses, dim=1)[:, :-1] + PROTON_MASS

# Line 159-162 (y-ions)
y_ions = torch.flip(...) + WATER_MASS + PROTON_MASS
```

**Why it matters**: Without this, model trained on synthetic won't transfer to real data (systematic 1 Da error).

---

### 2. Add Precursor Noise to Synthetic Data
**Status**: Not implemented
**Priority**: High
**Effort**: 5 minutes

**Problem**: Currently `precursor_mass` is computed perfectly from `sum(residue_masses) + H2O`. Real mass spectrometers have measurement error (±20 ppm typical). If the model learns to trust precursor mass as a perfect constraint during training, it will fail on real data.

**Location**: `src/data/synthetic.py:106`

**Current Code**:
```python
# Line 106
precursor_mass = sum(residue_masses) + WATER_MASS
precursor_mz = (precursor_mass + charge * PROTON_MASS) / charge
```

**Proposed Fix**:
```python
# Compute true mass
precursor_mass_true = sum(residue_masses) + WATER_MASS

# Apply mass error (like we do for fragment peaks)
if mass_error_ppm > 0:
    error = precursor_mass_true * random.gauss(0, mass_error_ppm * 1e-6)
    precursor_mass = precursor_mass_true + error
else:
    precursor_mass = precursor_mass_true

precursor_mz = (precursor_mass + charge * PROTON_MASS) / charge
```

**Why it matters**: Real-world robustness. Model must learn that precursor mass is a soft constraint, not hard truth.

**Validation**: After implementing, check that curriculum Stage 3 (realistic) applies noise to both fragments AND precursor.

---

## Medium Priority 🟡

### 2. Add Precursor Mass Loss (Global Constraint)
**Status**: Partially implemented (fragment matching exists, but no direct precursor loss)
**Priority**: Medium
**Effort**: 15 minutes

**Problem**: Current `SpectrumMatchingLoss` matches individual fragment peaks. This is computationally expensive and doesn't enforce the most basic constraint: **the sum of predicted amino acids must equal the observed precursor mass**.

**Location**: `src/training/losses.py:SpectrumMatchingLoss`

**Current Implementation**: Only fragment peak matching (lines 148-221)

**Proposed Addition**:
```python
# In SpectrumMatchingLoss class

def compute_precursor_mass_loss(
    self,
    sequence_probs: Tensor,      # (batch, seq_len, vocab)
    observed_precursor: Tensor,  # (batch,)
) -> Tensor:
    """
    Compute MSE between predicted total mass and observed precursor.

    This is cheap, interpretable, and enforces the global mass constraint.
    """
    # Expected mass at each position
    expected_masses = torch.einsum('bsv,v->bs', sequence_probs, self.aa_masses)

    # Sum residue masses (skip SOS/EOS tokens at positions 0 and -1)
    predicted_mass = expected_masses[:, 1:-1].sum(dim=1) + WATER_MASS

    # MSE with observed precursor
    return F.mse_loss(predicted_mass, observed_precursor)


def forward(
    self,
    sequence_probs: Tensor,
    observed_masses: Tensor,
    observed_intensities: Tensor,
    peak_mask: Tensor,
    precursor_mass: Tensor,  # NEW parameter
) -> Tensor:
    # Existing fragment matching
    fragment_loss = self._compute_fragment_loss(...)

    # NEW: Direct precursor constraint
    precursor_loss = self.compute_precursor_mass_loss(
        sequence_probs, precursor_mass
    )

    # Combine (could make weight configurable)
    return fragment_loss + 0.5 * precursor_loss
```

**Why it matters**:
- Cheap to compute (one sum vs many peak matches)
- Interpretable metric: "predicted mass error in Da"
- Global constraint (entire sequence must sum correctly)
- May help with convergence speed

**Ablation Study**: Train 3 models:
1. Baseline (CE only)
2. + Fragment matching
3. + Fragment matching + Precursor loss

Compare token accuracy at 10K steps.

---

## Low Priority / Ideas 🟢

### 3. Log Per-Step Improvement Metrics
**Status**: Not implemented
**Priority**: Low
**Effort**: 10 minutes

**Idea**: Currently we log `ce_step_0` through `ce_step_7` separately. Add a metric that shows **improvement from step to step**:

```python
improvement_metrics = {
    f'delta_step_{t}': metrics[f'ce_step_{t-1}'] - metrics[f'ce_step_{t}']
    for t in range(1, num_steps)
}
```

**Why**: Directly measure whether recursion is helping. If `delta_step_t` is positive, step t+1 is better than step t.

**Expected pattern**: Early steps show large improvement, later steps show diminishing returns.

---

### 4. Visualization: Attention Maps
**Status**: Not implemented
**Priority**: Low (research/debugging tool)
**Effort**: 1 hour

**Idea**: Visualize which peaks the decoder attends to when predicting each amino acid.

**Location**: Add hook to `src/model/decoder.py:latent_step()` cross-attention

**Use case**: Debugging. If model predicts "A" at position 5, which peaks is it looking at? Should correlate with b5 or y_n-5 ions.

---

### 5. Beam Search Inference
**Status**: Partially stubbed in `src/inference/predict.py`
**Priority**: Medium (after baseline works)
**Effort**: 2 hours

**Current**: Only greedy decoding (argmax at each position)
**Proposed**: Beam search to explore top-k hypotheses

**Why**: Peptide sequencing often has ambiguities (I/L, K/Q). Beam search can return multiple candidates with confidence scores.

**Implementation**: See `src/inference/predict.py:_beam_search()` (currently just a stub)

---

### 6. I/L Ambiguity Handling (Physics Constraint)
**Status**: Detected but not handled
**Priority**: Medium (affects evaluation metrics)

**Problem**: Isoleucine (I) and Leucine (L) have **identical mass** (113.084 Da). Mass spectrometry **cannot distinguish them**. Current training penalizes the model for "guessing wrong" on an impossible task.

**Physics**: `I` and `L` fragments produce identical m/z peaks. Only database search or additional experiments (MS3, retention time) can distinguish them.

**Current behavior**:
- Synthetic data generates random I/L
- Model must choose I or L (50/50 guess)
- Loss penalizes "wrong" choice even though it's physically impossible
- Training loss can never reach zero even on perfect predictions

**Three Options**:

#### **Option A: Collapse to Single Token** (Recommended for MVP)
```python
# In constants.py
AMINO_ACID_MASSES = {
    ...
    '[I/L]': 113.08406,  # Single isobaric token
    # Remove separate 'I' and 'L'
}

# In synthetic.py:generate_random_peptide()
def generate_random_peptide(...):
    # Always use [I/L] token
    available_aa = [aa for aa in AMINO_ACID_MASSES.keys() ...]
```

**Pros**:
- Model never penalized for impossible choice
- Vocabulary: 23 tokens (cleaner)
- Evaluation is straightforward
- Honest about MS limitations

**Cons**:
- Can't leverage database info if available
- Less granular than real proteins

**Effort**: 30 minutes (update constants, dataset, metrics)

---

#### **Option B: Multi-label Training with Soft Targets**
```python
# When ground truth is 'I', allow both I and L
if target_aa in ['I', 'L']:
    soft_targets[pos, AA_TO_IDX['I']] = 0.5
    soft_targets[pos, AA_TO_IDX['L']] = 0.5
else:
    soft_targets[pos, AA_TO_IDX[target_aa]] = 1.0

# Use KL divergence loss instead of cross-entropy
loss = F.kl_div(log_predictions, soft_targets)
```

**Pros**:
- Most principled approach
- Model learns "these are equivalent"
- Can still express preference if other evidence exists
- Generalizes to other isobaric pairs (K/Q: 128.095 vs 128.059)

**Cons**:
- Requires changing loss function (CE → KL divergence)
- More complex implementation
- Slower convergence (model explores both options)

**Effort**: 2 hours (update losses.py, synthetic.py, metrics.py)

---

#### **Option C: Fix Evaluation Metrics Only** (Quick fix)
```python
# In metrics.py:token_accuracy()
def is_isobaric_match(pred_idx, target_idx):
    """Check if prediction is isobaric equivalent of target."""
    I_IDX = AA_TO_IDX['I']
    L_IDX = AA_TO_IDX['L']

    return ((pred_idx == I_IDX) & (target_idx == L_IDX)) | \
           ((pred_idx == L_IDX) & (target_idx == I_IDX))

def token_accuracy(logits, targets, mask):
    predictions = logits.argmax(dim=-1)
    exact_match = (predictions == targets)
    isobaric_match = is_isobaric_match(predictions, targets)

    correct = (exact_match | isobaric_match) & mask
    return correct.sum() / mask.sum()
```

**Pros**:
- No change to training pipeline
- 10 minutes to implement
- Evaluation is fair
- Can apply retroactively to current run

**Cons**:
- Training loss still penalizes I/L confusion
- Model wastes capacity learning impossible distinction
- Not principled (training/eval mismatch)

**Effort**: 10 minutes (only update metrics.py)

---

**Recommendation**:
1. **Immediate (current baseline)**: Use **Option C** - fix eval metrics now
2. **V2 (after baseline works)**: Implement **Option A** - collapse to `[I/L]` token
3. **Research paper**: Implement **Option B** - multi-label with soft targets

**Also handle K/Q**: Similar issue (128.095 vs 128.059 Da, Δ=0.036). With 20ppm mass error, these become ambiguous too.

**Test**: After implementing, verify that sequences differing only by I↔L swaps have 100% accuracy

---

## Model Architecture Ideas 💡

### 7. Sinusoidal Positional Encoding for Decoder
**Status**: ✅ Already implemented
**Note**: This was discussed but confirmed working in `src/model/decoder.py:41`

No action needed.

---

### 8. Learnable Temperature for Spectrum Matching
**Status**: Not implemented
**Priority**: Low

**Current**: Fixed `temperature = 0.1` in soft assignment (line 191)
**Proposed**: Make it learnable: `self.temperature = nn.Parameter(torch.tensor(0.1))`

**Why**: Let model learn optimal sharpness for peak matching. Too sharp → brittle to noise. Too soft → doesn't match well.

---

## Experiments to Run 🧪

### A. Ablation: T=1 vs T=8 (Core TRM Validation)
**Goal**: Prove recursion helps
**Setup**:
- Train model with `num_supervision_steps: 1` (standard autoregressive)
- Train model with `num_supervision_steps: 8` (recursive)
- Compare final token accuracy on clean synthetic data

**Expected**: T=8 outperforms T=1 by 10-15%

**Implementation**: Create `configs/ablation_t1.yaml` with single step

---

### B. Ablation: Spectrum Loss On/Off
**Goal**: Measure value of domain-specific loss
**Setup**:
- Train with `spectrum_weight: 0.0` (pure CE)
- Train with `spectrum_weight: 0.1` (with physics constraint)
- Compare accuracy

**Expected**: Spectrum loss adds 5-10% improvement

---

### C. Scaling: Hidden Dim Sweep
**Goal**: Find optimal model size
**Current**: 256 hidden dim = 3.7M params
**Proposed**: Try [128, 256, 512]

**Hypothesis**: 256 may be overkill for clean synthetic. Could train faster with 128.

---

## Future: Real Data Pipeline 🔮

### 9. Real Data Integration
**Priority**: After synthetic baseline works
**Effort**: 1 week

**Tasks**:
1. Install pyteomics for reading .mzML files
2. Download Nine-Species benchmark (MSV000081382)
3. Create `RealPeptideDataset` class
4. Preprocess: peak filtering, normalization
5. Finetuning strategy: start from synthetic checkpoint

**Challenges**:
- Missing peaks (20-50% dropout typical)
- Noise peaks
- PTMs (need extended vocabulary)
- Charge state > 4

---

## Notes & Decisions 📝

### Why Stage Improvements Instead of Implementing Now?

**Reasoning**:
1. **Baseline first**: Need to see if basic approach works before optimizing
2. **Clean ablations**: Implementing all changes at once makes it impossible to attribute improvements
3. **Learning**: Current training run will reveal which issues are real vs theoretical

### When to Implement?

**After baseline 10K run completes** (~2.5 hours), we should:

1. ✅ Check if model converged (loss decreased, accuracy increased)
2. ✅ Review W&B curves (are we learning?)
3. 🔴 Implement High Priority items (precursor noise)
4. 🟡 Run ablations (T=1, spectrum loss on/off)
5. 🟢 Decide on low priority items based on results

---

## Questions to Answer 🤔

1. **Is the model learning at all?**
   → Wait for step 1000 (first eval checkpoint)

2. **Does loss decrease consistently?**
   → Check W&B dashboard after ~500 steps

3. **Does recursion help?**
   → Need ablation study (T=1 vs T=8)

4. **Is spectrum loss helping or hurting?**
   → Currently disabled (weight=0) in Stage 1. Will be enabled at 10K steps if we continue.

5. **Is 3.7M parameters too large?**
   → Consider 128 hidden_dim (250K params) for faster iteration

---

**Last Updated**: December 7, 2025
**Current Training Run**: https://wandb.ai/jase-g/peptide-trm/runs/in7xsbjd
**Baseline Status**: In progress (step 274 / 10,000)
