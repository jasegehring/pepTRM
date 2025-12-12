# Revised Curriculum: Phased Auxiliary Loss Introduction

## Problem with Current Curriculum

**Current**:
- Spectrum loss @ step 10k (40% accuracy)
  - Gradient strength: 0.16 (very weak)
  - Risk: Noisy signal, potential interference

- Precursor loss @ step 20k (60% accuracy)
  - Gradient strength: 0.47 (moderate)
  - Could be introduced later for stronger signal

**Issue**: Introducing auxiliary losses too early when gradients are weak and noisy.

---

## Proposed Revised Curriculum

### Phase 1: Foundation (0-15k steps)
**Pure Cross-Entropy Learning**

```python
CurriculumStage(
    name="stage_1_foundation",
    steps=15000,  # Extended from 10k
    min_length=7,
    max_length=10,
    noise_peaks=0,
    spectrum_loss_weight=0.0,
    precursor_loss_weight=0.0,
)
```

**Goal**: Let model learn basic token distributions and sequence patterns
**Expected accuracy**: 40% → 55%

### Phase 2: Fragment Physics (15k-30k steps)
**Introduce Spectrum Loss**

```python
CurriculumStage(
    name="stage_2a_fragments_intro",
    steps=7500,  # 15k-22.5k
    min_length=7,
    max_length=11,
    noise_peaks=1,
    spectrum_loss_weight=0.08,  # Start gentle
    precursor_loss_weight=0.0,
),
CurriculumStage(
    name="stage_2b_fragments_ramp",
    steps=7500,  # 22.5k-30k
    min_length=7,
    max_length=12,
    noise_peaks=2,
    spectrum_loss_weight=0.12,  # Increase
    precursor_loss_weight=0.0,
),
```

**Goal**: Learn fragment mass constraints while model is still plastic
**Expected accuracy**: 55% → 70%
**Gradient strength**: 0.25 → 0.49 (moderate to strong)

### Phase 3: Total Mass Constraint (30k-45k steps)
**Introduce Precursor Loss**

```python
CurriculumStage(
    name="stage_3a_precursor_intro",
    steps=7500,  # 30k-37.5k
    min_length=7,
    max_length=13,
    noise_peaks=3,
    spectrum_loss_weight=0.15,
    precursor_loss_weight=0.01,  # Start here!
),
CurriculumStage(
    name="stage_3b_precursor_ramp",
    steps=7500,  # 37.5k-45k
    min_length=7,
    max_length=14,
    noise_peaks=5,
    spectrum_loss_weight=0.18,
    precursor_loss_weight=0.02,
),
```

**Goal**: Add total mass constraint when soft predictions are meaningful
**Expected accuracy**: 70% → 80%
**Gradient strength**: 0.49 → 0.64 (strong)

### Phase 4: Full Multi-Task Learning (45k-100k steps)
**All Losses Active**

```python
# Stages 4-10: Continue with increasing noise and loss weights
# spectrum_loss_weight: 0.18 → 0.25
# precursor_loss_weight: 0.02 → 0.08
```

**Goal**: Full physics-aware learning with realistic noise
**Expected accuracy**: 80% → 95%

---

## Comparison: Current vs Revised

| Milestone | Current Curriculum | Revised Curriculum | Change |
|-----------|-------------------|-------------------|--------|
| **Spectrum loss introduced** | Step 10k (40% acc) | Step 15k (55% acc) | **+5k steps** |
| **Precursor loss introduced** | Step 20k (60% acc) | Step 30k (70% acc) | **+10k steps** |
| **Gradient strength @ intro** | 0.16 (weak) | 0.25 (moderate) | **+56% stronger** |
| **Full physics** | Step 30k | Step 45k | +15k steps |

---

## Rationale

### 1. **Gradient Information Content**
At 40% accuracy, spectrum loss gradient is only 0.16 (weak signal).
Waiting until 55% gives gradient of 0.25 (moderate, actionable signal).

### 2. **Fragment vs Total Mass**
- Fragments are **local**: Can learn even with some errors
- Total mass is **global**: Needs high accuracy to be meaningful
- This justifies different introduction times

### 3. **Avoiding Interference**
Early auxiliary losses can **interfere** with CE learning:
- Conflicting gradients when model is unstable
- "Tug of war" between different objectives
- Risk of settling into compromise that's bad at both

### 4. **Multitask Learning Theory**
Introduce auxiliary tasks when:
- Main task is reasonably stable (55% >> 40%)
- Auxiliary task gradients are actionable (0.25 >> 0.16)
- Tasks are complementary, not conflicting

---

## Alternative: Even More Conservative

If you want maximum stability, consider:

```
Spectrum @ step 20k (60% acc, gradient=0.36)
Precursor @ step 35k (75% acc, gradient=0.65)
```

**Trade-off**:
- More stable training, stronger gradients
- But may miss opportunity to guide early learning
- Model may have settled into local minimum

---

## Testing Both Curricula

To empirically determine which is better, run A/B test:

**Run A: Current curriculum** (early introduction)
```bash
python scripts/train_optimized.py  # Uses current curriculum
```

**Run B: Revised curriculum** (delayed introduction)
```bash
python scripts/train_revised_curriculum.py  # Uses phased intro
```

**Compare at step 50k**:
- Validation accuracy (primary metric)
- Training stability (NaN, loss spikes)
- Final loss values (CE, spectrum, precursor)
- Convergence speed (steps to reach 90% acc)

---

## My Prediction

**Revised curriculum will be better** because:

1. **Stronger gradients when introduced** (0.25 vs 0.16)
2. **Less interference** with CE learning
3. **More stable training** (lower risk of conflicting signals)
4. **Still early enough** to guide learning (not in local minimum yet)

**But**: Current curriculum might reach high accuracy faster by providing early guidance.

**Solution**: Test both and compare!

---

## Implementation

I can create the revised curriculum file if you want to test this. It would be a new version of `curriculum_extended.py` with the phased introduction schedule.

Would you like me to:
1. Create the revised curriculum file?
2. Create a training script for A/B testing?
3. Both?
