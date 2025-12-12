# A/B Testing Guide: Early vs Phased Auxiliary Loss Introduction

## Overview

This guide walks you through comparing two curriculum strategies:

**A) Early Introduction (Current)**
- Spectrum loss @ step 10k (40% accuracy, gradient=0.16)
- Precursor loss @ step 20k (60% accuracy, gradient=0.47)

**B) Phased Introduction (New)**
- Spectrum loss @ step 15k (55% accuracy, gradient=0.25)
- Precursor loss @ step 30k (70% accuracy, gradient=0.59)

**Hypothesis**: Delaying auxiliary losses until gradients are stronger leads to better training.

---

## Quick Start (Recommended)

### Option 1: Quick Test (10k steps, ~15 minutes)

Fast sanity check to see if phased curriculum works:

```bash
# Run both curricula with 10k steps each
python scripts/compare_curricula.py --quick_test
```

**What to expect**:
- Early: Spectrum @ 10k (you'll see loss potentially jump)
- Phased: Pure CE for all 10k steps (smooth curve)
- Quick comparison to verify everything works

### Option 2: Full Comparison (50k steps, ~4-6 hours)

Comprehensive test to step 50k (covers both spectrum and precursor introduction):

```bash
# Run both curricula to 50k steps
python scripts/compare_curricula.py --max_steps 50000
```

**What to expect**:
- Early: Spectrum @ 10k, Precursor @ 20k
- Phased: Spectrum @ 15k, Precursor @ 30k
- Both reach full multi-task learning by step 50k

### Option 3: Complete Training (100k steps, ~8-12 hours)

Full training to completion:

```bash
# Run both curricula to 100k steps (full training)
python scripts/compare_curricula.py --max_steps 100000
```

---

## Detailed Testing Workflow

### Step 1: Run Individual Curricula

If you want more control, run each curriculum separately:

**A) Early Introduction (Current)**
```bash
python scripts/train_optimized.py --max_steps 50000
# Saves to: checkpoints_optimized/
```

**B) Phased Introduction (New)**
```bash
python scripts/train_phased_curriculum.py --max_steps 50000
# Saves to: checkpoints_phased/
```

### Step 2: Analyze Results

After both runs complete:

```bash
# Create comparison plots
python scripts/plot_curriculum_comparison.py
```

This will:
- Parse training logs from both runs
- Generate `curriculum_comparison.png` with 4 plots
- Print numerical comparison table

### Step 3: Compare Metrics

Check the following at key milestones:

| Milestone | Step | What to Check |
|-----------|------|---------------|
| **Spectrum intro (Early)** | 10k | Loss spike? Accuracy drop? |
| **Spectrum intro (Phased)** | 15k | Smooth transition? |
| **Precursor intro (Early)** | 20k | Loss spike? NaN? |
| **Precursor intro (Phased)** | 30k | Smooth transition? |
| **Mid-training** | 40k | Which has higher accuracy? |
| **Final** | 50k+ | Final validation accuracy? |

---

## What to Look For

### 1. Training Stability

**Good**: Smooth loss curve, no sudden jumps
```
Step 15000 | Loss: 2.1 → 2.2  ← Small increase when spectrum added (OK)
Step 15100 | Loss: 2.2
Step 15200 | Loss: 2.1
```

**Bad**: Large jumps or NaN
```
Step 10000 | Loss: 2.1 → 7.5  ← Huge jump (BAD)
Step 10100 | Loss: 8.2
Step 11600 | Loss: NaN  ← Explosion (VERY BAD)
```

### 2. Validation Accuracy

Compare at step 50k:
- **Phased > Early**: Phased curriculum is better ✓
- **Early > Phased**: Early introduction is better
- **Similar**: Check stability as tiebreaker

### 3. Convergence Speed

Which reaches 90% validation accuracy first?
- If phased reaches 90% at step 45k but early at step 55k → Phased wins
- Even if final accuracy is similar, faster convergence is valuable

### 4. Loss Magnitude

At introduction points, check raw loss values:

**Early (step 10k)**:
```
Spectrum loss: ~0.15 (raw value)
With weight 0.05: contributes 0.0075 to total
```

**Phased (step 15k)**:
```
Spectrum loss: ~0.10 (lower raw value, clearer signal)
With weight 0.08: contributes 0.008 to total
```

Lower raw loss at introduction = model is more ready for that constraint.

---

## Decision Criteria

After comparing, choose the winner based on:

### Primary Criterion: **Final Validation Accuracy**
- Higher accuracy on validation (hard) set wins
- Difference > 2% is significant
- Difference < 1% → consider a tie

### Secondary Criteria (if similar accuracy):

1. **Training Stability**
   - No NaN → ✓
   - No large loss spikes → ✓
   - Smoother curves → ✓

2. **Convergence Speed**
   - Reaches 90% accuracy faster → ✓
   - Lower final loss → ✓

3. **Theoretical Soundness**
   - Stronger gradients when introduced → ✓
   - Less risk of interference → ✓

### Example Decision

```
Results at step 50k:

Early curriculum:
  Val (easy): 92.5%
  Val (hard): 38.2%
  Training: Stable, small spike at 10k

Phased curriculum:
  Val (easy): 93.1%  ← 0.6% better
  Val (hard): 39.8%  ← 1.6% better
  Training: Very stable, smooth throughout

Decision: PHASED WINS
- Better accuracy on both sets
- More stable training
- Stronger theoretical foundation
```

---

## Running with Gaussian Losses (Optional)

Want to test the new Gaussian spectral rendering losses too?

### Phased + Gaussian
```bash
# Edit train_gaussian_losses.py to use PHASED_CURRICULUM instead of EXTENDED_CURRICULUM
python scripts/train_gaussian_losses.py --max_steps 50000
```

This gives you a third option to compare:
- A) Early + Old losses
- B) Phased + Old losses
- C) Phased + Gaussian losses

---

## Troubleshooting

### Issue: "CUDA out of memory"

**Solution**: Reduce batch size
```bash
# Edit configs/optimized_extended.yaml
training:
  batch_size: 64  # Was 96
```

### Issue: "Training very slow"

**Solution**: Enable compilation (if not already)
```bash
# Edit configs/optimized_extended.yaml
training:
  use_compile: true
  compile_mode: 'max-autotune'
```

### Issue: "Can't find log files"

Check these locations:
```bash
ls wandb/latest-run/files/output.log  # Early curriculum
ls checkpoints_phased/training.log     # Phased curriculum
```

If missing, training may have crashed. Check error logs.

### Issue: "Comparison plot fails"

Ensure matplotlib is installed:
```bash
pip install matplotlib
```

---

## Advanced: Resume from Checkpoint

If training crashes or you want to extend:

```bash
# Resume early curriculum
python scripts/train_optimized.py --resume_from checkpoints_optimized/checkpoint_step_30000.pt

# Resume phased curriculum
python scripts/train_phased_curriculum.py --resume_from checkpoints_phased/checkpoint_step_30000.pt
```

---

## Expected Timeline

| Test Type | Steps | Time (RTX 4090) | Purpose |
|-----------|-------|-----------------|---------|
| Quick test | 10k | ~15 min | Sanity check |
| Comparison test | 50k | 4-6 hours | Full comparison |
| Complete training | 100k | 8-12 hours | Production model |

**Note**: Times assume batch_size=96, mixed precision enabled, compilation enabled.

---

## Files Created

```
src/training/curriculum_phased.py          # Phased curriculum definition
scripts/train_phased_curriculum.py         # Training script for phased
scripts/compare_curricula.py               # A/B test runner
scripts/plot_curriculum_comparison.py      # Analysis and plotting
scripts/analyze_gradient_information.py    # Gradient strength analysis
AB_TEST_GUIDE.md                          # This guide
REVISED_CURRICULUM_PROPOSAL.md            # Detailed rationale
```

---

## Summary

**Recommended workflow**:

1. **Quick test** (15 min): `python scripts/compare_curricula.py --quick_test`
2. **Full comparison** (4-6 hours): `python scripts/compare_curricula.py --max_steps 50000`
3. **Analyze**: `python scripts/plot_curriculum_comparison.py`
4. **Decide**: Based on validation accuracy + stability
5. **Deploy winner** for full 100k training

**My prediction**: Phased curriculum will be more stable and achieve similar or better accuracy due to stronger gradient signals when auxiliary losses are introduced.

**Good luck with testing! Let me know the results!**
