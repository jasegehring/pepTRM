# Comparing Old vs New Training Runs

## Setup for Scientific Comparison

You're starting from checkpoint step 15K with a **NEW W&B run** to compare:

### Old Run (nbs1e6hk / "gallant-water-62")
- **Steps:** 0-31K
- **Spectrum loss:** BUGGY (PAD tokens included)
- **Curriculum:** Introduced spectrum at step 15K
- **What happened:** Accuracy at 48% at step 15K, spectrum loss introduced, accuracy dropped

### New Run (will be created)
- **Steps:** 15K-100K
- **Spectrum loss:** FIXED (PAD tokens masked)
- **Curriculum:** DELAYED (no spectrum until step 30K)
- **Expected:** Accuracy improves 48% → 65%+ by step 30K

---

## What to Compare in W&B

### 1. Token Accuracy Trajectory

**Old run (nbs1e6hk):**
```
Step 0-15K:   ? → 48% (pure CE)
Step 15K-31K: 48% → ? (buggy spectrum introduced, likely dropped)
```

**New run (this one):**
```
Step 15K-30K: 48% → 65%+ (pure CE, delayed curriculum)
Step 30K+:    65%+ → ? (fixed spectrum introduced with good signal)
```

**What to look for:**
- ✓ New run should show **steady improvement** 15K → 30K
- ✓ Old run likely shows **drop or plateau** at 15K

---

### 2. Spectrum Loss Behavior

**Old run (nbs1e6hk):**
```
Step 15K-31K: Spectrum loss stuck at ~0.96-0.98 (2-4% coverage)
              Predicted peaks up to 3150 Da (WRONG - includes PAD)
```

**New run (this one):**
```
Step 15K-30K: Spectrum loss not computed (weight = 0.0)
Step 30K+:    Spectrum loss starts ~0.85 and should DECREASE
              Predicted peaks max ~1700 Da (CORRECT - PAD masked)
```

**What to look for:**
- ✓ New run at step 30K+ should show **decreasing spectrum loss**
- ✓ Old run was **stuck/flat** at high values

---

### 3. CE Loss (Cross-Entropy)

**Both runs:**
Should continue decreasing, but:

**Old run:** CE loss may have gotten worse after step 15K (interference from buggy spectrum)
**New run:** CE loss should keep improving smoothly

---

## How to Set Up Comparison in W&B

### Method 1: Compare View (Recommended)

1. Go to your W&B project
2. Click "Runs" tab
3. Select both runs:
   - Old: nbs1e6hk
   - New: (the new run that gets created)
4. Click "Compare" button
5. Add these charts:
   - `train/token_accuracy` vs step
   - `train/spectrum_loss` vs step
   - `train/ce_final` vs step
   - `train/total_loss` vs step

### Method 2: Custom Dashboard

Create a dashboard with:
- Line charts showing both runs overlaid
- Filter to show steps 15K-45K (the overlap period)
- Color code: Old run = red, New run = green

---

## Expected Outcomes

### Success (Fix Works):

**By step 20K:**
- New run: Token accuracy 55-60% ✓
- Old run: Was stuck ~48-50%

**By step 30K:**
- New run: Token accuracy 65%+ ✓
- New run: Ready for spectrum loss introduction
- Old run: Never recovered

**By step 35K (after spectrum introduction):**
- New run: Spectrum loss decreasing (0.85 → 0.80) ✓
- New run: Coverage improving (15% → 20%) ✓
- Old run: Was stuck at 0.96-0.98

### Failure (Fix Doesn't Help):

If the new run also struggles:
- Token accuracy stuck below 60% at step 30K
- Spectrum loss still high (>0.90) when introduced
- Need to investigate further

---

## Key Metrics to Screenshot

For documentation/paper:

1. **Side-by-side accuracy plot** (steps 15K-45K)
   - Shows old run plateau/drop vs new run improvement

2. **Spectrum loss comparison** (steps 30K-45K)
   - Shows old run stuck vs new run decreasing

3. **Coverage statistics**
   - Old run: 2-4% (too weak)
   - New run at 30K: 15-20% (useful signal)

---

## Tags to Find Runs

The new run will have these tags:
- `spectrum-loss-FIXED`
- `delayed-curriculum`
- `comparison-run`
- `from-step-15k`

Search in W&B: `tags:"spectrum-loss-FIXED"`

---

## Run Name

The new run will be named:
```
fixed-spectrum-delayed-{run_id}
```

For example: `fixed-spectrum-delayed-abc123de`

---

## Notes in Config

Check the new run's config in W&B - it includes:
```yaml
spectrum_loss_bug_fixed: true
bug_fix: "PAD tokens now masked in theoretical peak computation"
comparison_to_run: "nbs1e6hk"
curriculum_modified: true
```

This documents exactly what changed for future reference.

---

## Questions?

- **Q: Why start from step 15K instead of 0?**
  A: The bug only affects spectrum loss (introduced at 15K). Steps 0-15K are identical.

- **Q: Can I compare steps 0-15K?**
  A: No, new run starts at 15K. But those steps are identical anyway (pure CE).

- **Q: What if I want to re-run from step 0 with the fix?**
  A: You can, but it won't change anything for steps 0-15K (no spectrum loss yet).

- **Q: Should I stop the old run?**
  A: It already stopped at 31K. Just leave it for comparison.
