# Precursor Loss NaN Fix - Summary

## Problem Diagnosis

### Issue 1: Loss Explosion at Step 10k
At step 10,000, when precursor loss was introduced, the total loss jumped from **2.2 → 7.1**.

**Root cause**: The precursor loss formula had incorrect scaling:
```python
# OLD (WRONG):
loss_scale = 100.0
loss = torch.log1p(ppm_error / loss_scale) * loss_scale
```

With PPM errors of ~170,000 early in training:
```
loss = log1p(170000/100) * 100
     = log(1701) * 100
     ≈ 7.44 * 100 = 744
```

With curriculum weight of 0.01:
```
contribution = 0.01 * 744 = 7.44  ← This dominated the CE loss of ~2!
```

### Issue 2: NaN at Step 11,600
The large loss values (8-9) caused:
1. Large gradients flowing backward
2. Gradient explosion over 1,500 steps
3. Complete model collapse (NaN)

## Diagnostic Results

Running `scripts/diagnose_loss_magnitudes.py` revealed:

```
Stage: Before precursor loss (step 10k)
Current (scale=100):
  Raw loss value: 700.66
  PPM error: 170932.9 ppm  ← Model hasn't learned mass constraints yet!
  Mass error (Da): 167.31
  With weight=0.01: contributes 7.007 to total loss  ← WAY TOO MUCH
```

## Fixes Applied

### Fix 1: Corrected Precursor Loss Scale

**File**: `src/training/losses.py`

Changed the loss formula from:
```python
# OLD: Multiplies by scale → huge values
loss = torch.log1p(ppm_error / 100.0) * 100.0
```

To:
```python
# NEW: Just the log, no multiplication
loss_scale = 100000.0  # 100k ppm = 10% error
loss = torch.log1p(ppm_error / loss_scale)
```

**Effect**: With 100k ppm scale:
- 170k ppm error → loss = log1p(1.7) ≈ **1.0** ✓
- 10k ppm error → loss = log1p(0.1) ≈ **0.095** ✓
- 1k ppm error → loss = log1p(0.01) ≈ **0.01** ✓

### Fix 2: Reduced Curriculum Weights

**File**: `src/training/curriculum_extended.py`

Reduced precursor loss weights by ~20-40%:

| Stage | Old Weight | New Weight | Expected Loss | Contribution |
|-------|------------|------------|---------------|--------------|
| 2 (10k) | 0.01 | 0.01 | ~1.0 | ~0.01 ✓ |
| 3 (20k) | 0.02 | 0.015 | ~0.7 | ~0.01 ✓ |
| 4 (30k) | 0.03 | 0.02 | ~0.5 | ~0.01 ✓ |
| 5 (40k) | 0.04 | 0.025 | ~0.3 | ~0.008 ✓ |
| 6 (50k) | 0.05 | 0.03 | ~0.2 | ~0.006 ✓ |
| 7 (60k) | 0.06 | 0.04 | ~0.1 | ~0.004 ✓ |
| 8 (70k) | 0.07 | 0.05 | ~0.05 | ~0.0025 ✓ |
| 9 (80k) | 0.08 | 0.06 | ~0.02 | ~0.0012 ✓ |
| 10 (90k) | 0.10 | 0.08 | ~0.01 | ~0.0008 ✓ |

Target: Precursor loss contributes **0.005-0.01** to total loss (vs CE loss of ~2-3).

### Fix 3: Gradient Clipping (Already Present)

**File**: `src/training/trainer_optimized.py:298`

Verified gradient clipping is enabled:
```python
torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
```

This prevents gradient explosion and NaN.

## Expected Behavior After Fix

### At Step 10k (Stage 2 start):
- CE loss: ~2.0
- Precursor raw loss: ~1.0 (was ~700)
- Precursor contribution: 0.01 × 1.0 = **0.01**
- **Total loss: ~2.01** (was 7.1) ✓

### Training progression:
- Steps 0-10k: Loss decreases 3.3 → 2.0 (pure CE)
- **Step 10k: Loss stays ~2.0** (gentle precursor introduction)
- Steps 10k-20k: Loss continues decreasing as model learns mass constraints
- No NaN, stable gradients throughout

## Files Modified

1. `src/training/losses.py`
   - Changed `PrecursorMassLoss.__init__`: `loss_scale = 100.0 → 100000.0`
   - Changed loss formula: removed `* scale` multiplication
   - Updated `CombinedLoss.__init__`: `loss_scale = 100.0 → 100000.0`

2. `src/training/curriculum_extended.py`
   - Reduced precursor_loss_weight for stages 3-10
   - Updated comments with expected PPM error targets

3. `scripts/diagnose_loss_magnitudes.py` (NEW)
   - Diagnostic script to measure loss magnitudes at different training stages

## How to Test

Run a short training test:
```bash
python scripts/train_optimized.py
```

Monitor at step 10,000:
- ✓ Total loss should stay ~2-3 (not jump to 7+)
- ✓ Precursor loss should be ~1.0 (not ~700)
- ✓ No NaN after step 11k

Or run the diagnostic:
```bash
python scripts/diagnose_loss_magnitudes.py
```

Expected output:
```
Current (scale=100000):
  Raw loss value: ~1.0    ← Good!
  PPM error: ~170000
  With weight=0.01: contributes 0.01 to total loss  ← Perfect!
```

## Theory: Why This Works

The log1p formulation provides:

1. **Bounded growth** for large errors:
   - Prevents explosion even when PPM errors are 100k+
   - Loss grows as O(log(error)), not O(error)

2. **Strong gradients** for small errors:
   - log(1 + x) ≈ x for small x
   - Acts like MSE when model is close to correct mass

3. **Never zero gradient**:
   - Even at 1M ppm error, gradient is non-zero
   - Model can always improve

4. **Appropriate scale**:
   - loss_scale = 100k matches typical early errors
   - Normalizes loss to ~0-2 range
   - Compatible with CE loss magnitude

## References

- Training logs: `wandb/run-20251211_122248-9m26te9s/files/output.log`
- Diagnostic output: `scripts/diagnose_loss_magnitudes.py`
- Loss implementation: `src/training/losses.py:245-332`
- Curriculum settings: `src/training/curriculum_extended.py:42-175`
