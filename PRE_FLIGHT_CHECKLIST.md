# Pre-Flight Checklist - Training Run

✅ **All systems verified and ready for production training!**

## System Components Status

### ✅ Curriculum System
- **File:** `src/training/curriculum.py`
- **Status:** Production-ready
- **Stages:** 10 stages, 100K steps total
- **Key Features:**
  - Spectrum loss @ 15K (not 10K!)
  - Precursor loss @ 30K (not 20K!)
  - Clean data through 30K
  - Smooth mixing 80%→20%→0% clean (30K-70K)
  - Dropout delayed until 70K

### ✅ Loss Functions
- **File:** `src/training/losses.py`
- **Status:** Production-ready
- **Improvements:**
  - Gaussian spectrum rendering (bounded [0,2], low variance)
  - Log-scaled precursor loss (robust gradients)
  - Smooth, differentiable gradients
  - Lower variance expected

### ✅ Dataset
- **File:** `src/data/dataset.py`
- **Status:** Updated with clean/noisy mixing
- **Feature:** `clean_data_ratio` parameter for probability-based mixing
- **Integration:** Controlled by curriculum scheduler

### ✅ Training Loop
- **File:** `src/training/trainer_optimized.py`
- **Status:** Properly integrated
- **Verified:**
  - Imports new curriculum
  - Imports new losses
  - Updates loss weights from curriculum
  - Curriculum scheduler properly hooked in

### ✅ Training Script
- **File:** `scripts/train_optimized.py`
- **Status:** Updated to use new curriculum
- **Fixed:** Now imports `DEFAULT_CURRICULUM` (not EXTENDED_CURRICULUM)

### ✅ Configuration
- **File:** `configs/optimized_extended.yaml`
- **Status:** Ready
- **Settings:**
  - Batch size: 96
  - Learning rate: 1.5e-4
  - AMP: enabled (bfloat16)
  - Model: 384 hidden_dim, 3 layers, 6 heads
  - MS2PIP model: HCDch2 (b, y, b++, y++ ions)

## Integration Tests

✅ All imports working
✅ Curriculum transitions verified
✅ Loss function creation verified
✅ Gaussian rendering confirmed
✅ Clean/noisy mixing confirmed
✅ Trainer integration verified

## Expected Training Behavior

### Phase 1: Foundation (0-15K)
- **Loss:** Smooth CE decrease
- **Accuracy:** Climb to ~55%
- **Spectrum loss:** N/A (not active yet)
- **Precursor loss:** N/A (not active yet)

### Phase 2: Fragment Physics (15K-30K)
- **Loss:** Spectrum loss appears, starts ~1.0
- **Accuracy:** Continue to ~70%
- **Spectrum loss:** Decrease to ~0.5-0.8
- **Precursor loss:** N/A (not active yet)
- **Data:** 100% clean - learn physics clearly!

### Phase 3: Noise Introduction (30K-70K)
- **Loss:** Precursor appears, smooth transitions
- **Accuracy:** Stable 65-70% (slight dip, then recover)
- **Spectrum loss:** ~0.4-0.6 (stable, low variance)
- **Precursor ppm:** Decrease from 50k → 5k
- **Data:** Gradual mixing 80%→60%→40%→20% clean

### Phase 4: Final Realism (70K-100K)
- **Loss:** All losses stable
- **Accuracy:** Final performance 60-65%
- **Spectrum loss:** ~0.3-0.5
- **Precursor ppm:** <1000 ppm
- **Data:** Dropout added, fully noisy

## Red Flags to Watch For

❌ **Curriculum shock** - Sudden loss jumps at transitions (should NOT happen)
❌ **Spiky spectrum loss** - High variance (should be smooth with Gaussian)
❌ **Accuracy collapse** - Sudden drop >10% (should be gradual)
❌ **Precursor staying high** - ppm error not decreasing (check gradients)

## Launch Command

```bash
python scripts/train_optimized.py
```

## Optional: Resume Training

If you need to resume from a checkpoint:

```bash
python scripts/train_optimized.py --resume_from checkpoints_optimized/checkpoint_step_XXXXX.pt
```

## Monitoring

Key metrics to watch:
1. **Total loss** - Should decrease smoothly
2. **Spectrum loss** - Should be smooth, low variance
3. **Precursor ppm error** - Should decrease from 50k → <1k
4. **Token accuracy** - Should climb then stabilize 60-70%
5. **Sequence accuracy** - Should reach 20-25% by end

## Archive Status

Old files properly archived:
- ✅ `curriculum_old_*.py` (3 files)
- ✅ `losses_old_*.py` (2 files)
- ✅ Documentation updated

## Git Status

```
717cb12 - feat: Final authoritative curriculum
2b13b60 - feat: Major training improvements - Curriculum V2 + Low-Variance Losses
41b102f - refactor: Clean up main branch
```

Main branch is clean and production-ready!

---

## 🚀 READY FOR LAUNCH!

All systems verified. No blocking issues found.

Good luck with the training run! 🎉
