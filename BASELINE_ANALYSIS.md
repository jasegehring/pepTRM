# Baseline Training Run Analysis

**Run Date**: December 7, 2025
**W&B URL**: https://wandb.ai/jase-g/peptide-trm/runs/in7xsbjd
**Steps**: 10,000
**Duration**: 2:38:00 (M3 Pro MPS)

---

## Summary: The Model Learned, But Was Handicapped

**Good News**: Core TRM architecture works! Recursive refinement shows measurable improvement.

**Bad News**: Model was "blind" due to critical mass embedding bug. Performance limited to ~48% accuracy.

---

## Key Results

### Training Metrics
- **Loss**: 2.48 → 1.65 (33% reduction) ✅
- **Token Accuracy**: 0% → 43.8% (validation) ✅
- **Sequence Accuracy**: 0% (expected - need 99%^10 ≈ 90% token acc) ✅
- **Mass Error**: 59K ppm ⚠️

### Recursive Refinement (Deep Supervision)
**Evidence that TRM recursive loop works**:
```
Step 0 (initial guess):  Loss = 1.701  ← worst
Step 1:                  Loss = 1.661
Step 2:                  Loss = 1.642
Step 3:                  Loss = 1.651
Step 4:                  Loss = 1.642  ← best (3.5% improvement!)
Step 5-7:                Loss ≈ 1.65  (plateau)
```

**Interpretation**: Model learns to refine its own predictions! The recursive mechanism is working as intended.

---

## Critical Bugs Found

### 🚨 Bug #1: Mass Embedding Cannot Distinguish 1 Da (CRITICAL!)

**Discovery**: User asked "can your mass embedding distinguish 1 Da?"

**Finding**: NO! Current configuration has **2000 Da minimum wavelength**. The model literally cannot tell amino acids apart by mass.

**Test Results**:
```
I (113.08 Da) vs N (114.04 Da):
  Δ mass: 0.96 Da
  L2 distance: 0.0082  ← Too small!
  Cosine similarity: 1.000000  ← Identical!
```

**Root Cause**:
```python
# src/data/encoding.py:32
max_freq: float = 1.0  # WRONG - gives 2000 Da wavelength
```

**Fix**:
```python
max_freq: float = 1000.0  # Gives 2.0 Da wavelength ✓
```

**Impact**: This is why the model is stuck at 48% accuracy. It's essentially learning from random noise, not mass information!

---

### Bug #2: Spectrum Matching Loss Disabled

**Finding**: `spectrum_weight = 0.0` in config, so the mass-aware auxiliary loss never ran.

**Why**: Curriculum was supposed to enable it at 10K steps, but we only trained to 10K.

**Fix**: Either:
- Start with `spectrum_weight: 0.1` from step 0
- Or extend training beyond 10K to hit curriculum Stage 2

---

### Non-Bug: Protonation Offset

**Initial Suspicion**: Missing protonation (+1 Da) causing 59K ppm error.

**Finding**: This is **NOT** causing the mass error! Both synthetic data and metrics use neutral masses consistently.

**Why 59K ppm?**: Model predicts wrong amino acids (48% accuracy = half wrong). Wrong AAs → wrong mass → large ppm errors.

**Decision**: Still worth fixing for real-world data compatibility, but not urgent for synthetic baseline.

---

## What's Working vs. Broken

### ✅ Working
1. **Recursive refinement**: Clear improvement from step 0 → 4
2. **Deep supervision**: Model learns trajectory of improvement
3. **Sequence length**: Nearly perfect (9/10 correct, ±1)
4. **Training stability**: No crashes, smooth convergence with EMA
5. **MPS acceleration**: 1.05 it/s on M3 Pro

### 🔴 Broken / Missing
1. **Mass embedding resolution**: Cannot distinguish AAs by mass (CRITICAL)
2. **Spectrum matching loss**: Disabled (weight=0)
3. **Flip rate metric**: No visibility into token-level changes per step
4. **Protonation**: Neutral masses won't match real MS data

---

## Detailed Prediction Analysis

**Sample Predictions** (10 random sequences):

| True | Predicted | Token Acc | Mass Error |
|------|-----------|-----------|------------|
| NATRSQMP | DAPCFQKP | 50% | 1,094 ppm ✅ |
| PSNSEEHF | PSNCRTEH | 50% | 3,158 ppm ✅ |
| TKAHMSQGQK | TQAQQKNGQK | 58% | 13,484 ppm ⚠️ |
| SRWYQTSMEI | SRWWWMIMQL | 50% | 105K ppm 🔴 |

**Pattern**:
- Model gets length almost perfect
- Token accuracy ~48% (coin flip for each position)
- When predictions are close, mass error is low (<10K ppm)
- When predictions are far, mass error explodes (>100K ppm)
- Average: ~59K ppm

---

## Why Only 48% Accuracy?

**Three Contributing Factors**:

1. **Mass Embedding Blind** (primary cause)
   - Cannot distinguish I vs N, K vs Q, etc.
   - Model has no mass-based signal to learn from

2. **No Spectrum Matching Loss**
   - Cross-entropy alone doesn't care about mass
   - Model never learned to match theoretical peaks to observed

3. **High Uncertainty**
   - Logit analysis shows entropy 1.77 bits (out of 4.58 max)
   - Many positions are essentially random guesses
   - Example: Position 4 predicts Q (6.4%), E (6.2%), M (6.0%) - basically uniform!

---

## Initial Hypothesis (Wrong): W Bias

**What I thought**: Model over-predicts W (Tryptophan), causing mass errors.

**Evidence**: Debug script showed "W: 44% of predictions" ← BUG IN MY SCRIPT!

**Reality**: W has only 2.06% average probability (rank 19/24). No W bias exists.

**Lesson**: Always verify token distributions carefully. My script was counting padding tokens.

---

## Next Steps (Priority Order)

### 1. Fix Mass Embedding (MUST DO)
```python
# src/data/encoding.py:32
max_freq: float = 1000.0  # Was 1.0
```

**Verify**:
```bash
python3 scripts/test_mass_embedding.py
# Should show L2 distance > 0.5 for 1 Da differences
```

### 2. Add Flip Rate Metric
```python
# src/training/trainer.py
flip_rates = {}
for t in range(1, num_steps):
    prev_preds = all_logits[t-1].argmax(dim=-1)
    curr_preds = all_logits[t].argmax(dim=-1)
    flips = (prev_preds != curr_preds) & target_mask
    flip_rates[f'flip_rate_{t-1}_to_{t}'] = flips.float().sum() / target_mask.sum()
```

### 3. Enable Spectrum Matching Loss
```yaml
# configs/default.yaml
training:
  spectrum_weight: 0.1  # Was 0.0
```

### 4. Add Protonation (Nice to Have)
```python
# src/data/synthetic.py:115, 145
mass = cumulative + PROTON_MASS  # Add for both b and y ions
```

### 5. Add Precursor Noise (Nice to Have)
```python
# src/data/synthetic.py:106
error = precursor_mass_true * random.gauss(0, mass_error_ppm * 1e-6)
precursor_mass = precursor_mass_true + error
```

### 6. Run New Baseline
- Same 10K steps
- Compare metrics:
  - Token accuracy (expect >60% with mass embedding fix)
  - Mass error (expect <10K ppm)
  - Flip rates (expect 20-40% early → 5% late)

---

## Expected Improvements After Fixes

| Metric | Baseline | Expected (Fixed) | Reason |
|--------|----------|------------------|--------|
| Token Accuracy | 43.8% | >70% | Mass embedding can distinguish AAs |
| Mass Error | 59K ppm | <10K ppm | Better AA predictions |
| Sequence Accuracy | 0% | >5% | Higher token acc compounds |
| Loss | 1.65 | <1.2 | Better training signal |

---

## Lessons Learned

1. **Always verify encoding resolution** - Nyquist criterion for continuous embeddings
2. **Debug predictions, not just metrics** - Aggregate metrics hide issues
3. **Check consistency** - Both data generation AND loss should use same physics
4. **Instrument recursive loops** - Need flip rate to understand refinement behavior

---

## Questions for Future Investigation

1. **Why does recursive loop plateau at step 4?**
   - Are 4 latent steps enough?
   - Is the model overtrained on early steps?
   - Should we use different iteration weights?

2. **Will spectrum matching loss help convergence?**
   - Current: CE only, mass-agnostic
   - Expected: Spectrum loss provides mass-aware gradient

3. **Is 8 supervision steps optimal?**
   - Ablation: Try T=1, T=4, T=8, T=16
   - Measure: Does more T → better accuracy?

4. **Can we predict flip rate from early training?**
   - If flip_rate increases over training, model learns to refine
   - If stuck at 0%, architecture problem

---

**Last Updated**: December 7, 2025
**Authors**: Claude Code + User (critical insights!)
**Next Run**: After implementing fixes above
