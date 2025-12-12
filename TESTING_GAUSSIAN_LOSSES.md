# Testing Gaussian Losses - Quick Start Guide

## What We're Testing

Your new `gaussian_spectral_rendering_losses.py` file implements two key improvements:

1. **Gaussian Spectral Rendering** (spectrum loss)
   - Smooth, differentiable gradients (no hard min/max operations)
   - Bounded loss values [0, 2]
   - Better gradient flow

2. **Scaled L1** (precursor loss)
   - Constant gradients (fixes vanishing gradient problem)
   - Simple, interpretable: 1 Da error = 0.004 loss
   - Tuned scale_factor=0.004 for stability

## Why This Should Fix Gradient Issues

### Problem 1: Gradient Vanishing (log1p)
**Old**: At 200 Da error, gradient = 0.003 (model barely learns)
**New**: At 200 Da error, gradient = 0.004 (constant, strong signal)

### Problem 2: Gradient Explosion (hard min operations)
**Old**: Hard argmin in Huber loss creates gradient discontinuities
**New**: Smooth Gaussian kernels everywhere

## Quick Test (5 minutes)

Run a sanity check:

```bash
# Create a quick test config
cat > configs/gaussian_test.yaml << EOF
# Inherits from optimized_extended.yaml but with shorter run
training:
  max_steps: 5000  # Just 5k steps for quick test
  use_compile: false  # Disable compilation for first test
EOF

# Run test
python scripts/train_gaussian_losses.py
```

**Watch for**:
- ✓ No NaN losses at any point
- ✓ Loss values stay in reasonable range (0-5)
- ✓ Gradients flow smoothly (no sudden spikes)

**Success criteria** (by step 1000):
```
Step 1000 | Loss: ~2.5 | Token Acc: ~0.15 | Spectrum Loss: ~1.2 | Precursor: ~0.6
```

If you see NaN or loss > 10, stop and we'll debug.

## Full Comparison Test

### Setup

Create two training runs:

**Run 1: Old losses (baseline)**
```bash
# Uses the fixed log1p losses from PRECURSOR_LOSS_FIX.md
python scripts/train_optimized.py --max_steps 30000
```

**Run 2: New Gaussian losses**
```bash
# Uses Gaussian rendering + L1
python scripts/train_gaussian_losses.py --max_steps 30000
```

### Compare at Step 30k

| Metric | Old Losses | Gaussian Losses | Winner |
|--------|------------|-----------------|--------|
| Validation Accuracy (easy) | ? | ? | ? |
| Validation Accuracy (hard) | ? | ? | ? |
| Training Stability | Check for NaN | Check for NaN | ? |
| Loss Smoothness | Plot curve | Plot curve | ? |
| Training Speed (it/s) | Measure | Measure | ? |
| Memory Usage (GB) | ~6 GB | ~7 GB (Gaussian) | Old (lower) |

## What to Monitor

### 1. Loss Values (Every 100 Steps)

**Healthy training** looks like:
```
Step 0    | Total: 3.3 | CE: 3.3 | Spectrum: 0.0 | Precursor: 0.0
Step 1000 | Total: 2.8 | CE: 2.8 | Spectrum: 0.0 | Precursor: 0.0
Step 10000| Total: 2.0 | CE: 2.0 | Spectrum: 0.0 | Precursor: 0.0
Step 20000| Total: 2.0 | CE: 1.8 | Spectrum: 0.3 | Precursor: 0.4  ← Precursor introduced
Step 30000| Total: 1.7 | CE: 1.5 | Spectrum: 0.25| Precursor: 0.3
```

**Problematic training** looks like:
```
Step 20000| Total: 8.5 | CE: 1.8 | Spectrum: 0.3 | Precursor: 6.4  ← Too large!
or
Step 25000| Total: NaN | CE: NaN | Spectrum: NaN | Precursor: NaN  ← Explosion!
```

### 2. Gradient Norms

Add this to trainer to monitor:
```python
total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
print(f"Gradient norm: {total_norm:.4f}")
```

**Healthy**: 0.1 - 1.0
**Warning**: > 5.0 (approaching clip threshold)
**Problem**: NaN

### 3. Memory Usage

```bash
# Monitor GPU memory while training
watch -n 1 nvidia-smi
```

**Expected**:
- Old losses: ~6 GB
- Gaussian losses: ~7 GB (extra 1 GB for spectral rendering)

If you see OOM (out of memory), reduce batch_size from 96 to 64.

## Tuning Knobs (If Needed)

If Gaussian losses are **too strong** (dominating CE):

**Option 1: Reduce weights in curriculum**
```python
# In curriculum_extended.py, stage 3:
spectrum_loss_weight=0.05,    # Was 0.08
precursor_loss_weight=0.005,  # Was 0.01
```

**Option 2: Reduce scale factors**
```python
# In gaussian_spectral_rendering_losses.py:
PrecursorMassLoss(scale_factor=0.002)  # Was 0.004
```

**Option 3: Increase sigma (smoother Gaussians)**
```python
# In SpectrumMatchingLoss:
sigma=0.1,  # Was 0.05, makes peaks broader/smoother
```

If Gaussian losses are **too weak** (not learning):

**Option 1: Increase weights**
```python
spectrum_loss_weight=0.12,   # Was 0.08
precursor_loss_weight=0.02,  # Was 0.01
```

**Option 2: Increase scale factor**
```python
PrecursorMassLoss(scale_factor=0.006)  # Was 0.004
```

## Expected Results

### Best Case Scenario
- ✓ Stable training, no NaN
- ✓ Auxiliary losses converge smoothly
- ✓ **Higher final accuracy** (cleaner gradients → better optimization)
- ✓ Faster convergence (fewer steps to reach same accuracy)

### Realistic Scenario
- ✓ Stable training, no NaN
- ✓ Similar final accuracy to old losses
- ✓ Smoother loss curves
- ≈ Slightly slower iteration speed (Gaussian rendering overhead)

### Worst Case Scenario
- Gaussian rendering too slow (>20% slower)
- Memory issues on smaller GPUs
- → Solution: Keep spectrum loss, revert to log1p precursor loss

## Decision Tree

After testing, choose the best configuration:

```
Run 5k step sanity check
├─ NaN losses?
│  ├─ Yes → Reduce curriculum weights by 50%, retest
│  └─ No → Continue to 30k test
│
Run 30k step comparison
├─ Gaussian losses better accuracy?
│  ├─ Yes → **Adopt Gaussian losses** ✓
│  └─ No → Check loss curves
│
├─ Gaussian losses smoother/more stable?
│  ├─ Yes → **Adopt Gaussian losses** ✓ (stability > raw accuracy)
│  └─ No → Check speed
│
└─ Gaussian losses much slower (>20%)?
   ├─ Yes → Hybrid: Keep old spectrum, use L1 precursor
   └─ No → **Adopt Gaussian losses** ✓
```

## Summary

**Your Gaussian loss file is well-designed and theoretically sound.** The fixes I made:
- Tuned `scale_factor=0.004` (was 0.01)
- Set explicit hyperparameters for spectral rendering

**Next steps**:
1. Run 5k quick test (5 min)
2. If stable, run 30k comparison (1-2 hours)
3. Compare results and decide

**My prediction**: Gaussian losses will be **more stable** with **similar or better accuracy**. The smooth gradients are a fundamental improvement.

Let me know when you want to start the test run!
