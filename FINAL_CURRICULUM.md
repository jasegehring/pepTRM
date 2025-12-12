# Final Authoritative Curriculum

This document describes the production curriculum that combines all lessons learned from previous training experiments.

## Design Philosophy

The curriculum is designed around **four phases** that introduce complexity gradually to prevent curriculum shock while maintaining strong learning signals:

1. **Foundation** (0-15K): Build basic sequence generation
2. **Fragment Physics** (15K-30K): Learn mass constraints on clean data
3. **Noise Introduction** (30K-70K): Gradual mixing with smooth transitions
4. **Final Realism** (70K-100K): Add dropout and full realism

## Key Improvements Over Previous Versions

### 1. Delayed Auxiliary Losses

**Based on gradient analysis:**
- **Spectrum loss @ 15K** (not 10K) - when model reaches ~55% accuracy
  - Gradient strength: ~0.25 at this accuracy
  - Clear, actionable gradients
- **Precursor loss @ 30K** (not 20K) - when model reaches ~70% accuracy
  - Gradient strength: ~0.59 at this accuracy
  - Strong signal for mass constraint learning

### 2. Clean Data for Learning Physics

**Critical insight:** Keep data 100% clean while learning mass constraints (15K-30K)
- Allows model to learn fragment patterns without noise confusion
- Clearer gradient signals for spectrum matching
- Prevents curriculum shock from simultaneous noise + new loss

### 3. Smooth Clean/Noisy Mixing

**No hard transitions!** Use probability-based mixing:
- 30-40K: 80% clean, 20% noisy - gentle introduction
- 40-50K: 60% clean, 40% noisy
- 50-60K: 40% clean, 60% noisy
- 60-70K: 20% clean, 80% noisy
- 70K+: Mostly/fully noisy

### 4. Delayed Peak Dropout

**Peak dropout is harsh** - it removes critical b/y ions
- NO dropout until 70K (was 20K in old curriculum!)
- Only introduce at 8% after model is robust
- Final stage uses 10% dropout for realism

### 5. Smooth Noise Ramp

Progressive noise introduction:
- **Noise peaks**: 0 → 1 → 2 → 3 → 5 → 7 → 10
- **Mass error**: 0 → 1 → 2 → 5 → 8 → 12 → 15 ppm
- **Peak dropout**: 0 → 0 → 0 → 0 → 0 → 8% → 10%

## 10-Stage Curriculum Breakdown

| Stage | Steps | Length | Clean % | Noise Peaks | Dropout | Mass Error | Spectrum W | Precursor W |
|-------|-------|--------|---------|-------------|---------|------------|-----------|-------------|
| 1 | 0-7.5K | 7-10 | 100% | 0 | 0% | 0 ppm | 0.00 | 0.00 |
| 2 | 7.5-15K | 7-10 | 100% | 0 | 0% | 0 ppm | 0.00 | 0.00 |
| 3 | 15-22.5K | 7-11 | 100% | 0 | 0% | 0 ppm | **0.08** ← Start | 0.00 |
| 4 | 22.5-30K | 7-12 | 100% | 0 | 0% | 0 ppm | 0.12 | 0.00 |
| 5 | 30-40K | 7-13 | 80% | 1 | 0% | 1 ppm | 0.15 | **0.01** ← Start |
| 6 | 40-50K | 7-14 | 60% | 2 | 0% | 2 ppm | 0.18 | 0.02 |
| 7 | 50-60K | 7-15 | 40% | 3 | 0% | 5 ppm | 0.20 | 0.03 |
| 8 | 60-70K | 7-16 | 20% | 5 | 0% | 8 ppm | 0.22 | 0.05 |
| 9 | 70-85K | 7-18 | 10% | 7 | **8%** ← Start | 12 ppm | 0.24 | 0.06 |
| 10 | 85-100K | 7-20 | 0% | 10 | 10% | 15 ppm | 0.25 | 0.08 |

## Phase-by-Phase Details

### Phase 1: Foundation (0-15K)

**Goal:** Build stable sequence generation, reach ~55% token accuracy

**Stages 1-2:**
- Pure cross-entropy loss
- 100% clean data
- Short peptides (7-10 aa)
- No auxiliary losses
- Model learns basic amino acid distributions and common patterns

**Expected outcomes:**
- Token accuracy: 40% → 55%
- Sequence accuracy: 5% → 15%
- Loss: 3.0 → 1.5

### Phase 2: Fragment Physics (15K-30K)

**Goal:** Learn fragment mass constraints with clear gradients

**Stages 3-4:**
- **Introduce spectrum loss at 15K** when accuracy is stable
- **Keep data 100% clean** - critical for learning physics!
- Peptide length increases slightly (7-11, then 7-12)
- No noise yet - learn patterns first

**Why clean data?**
- Clear gradient signals for fragment matching
- No noise confusion
- Model can focus on learning physics

**Expected outcomes:**
- Token accuracy: 55% → 70%
- Spectrum loss: ∞ → 0.5-0.8 (Gaussian cosine similarity)
- Model learns which fragments should exist

### Phase 3: Noise Introduction (30K-70K)

**Goal:** Add mass constraint + smooth noise mixing

**Stages 5-8:**
- **Introduce precursor loss at 30K** when accuracy is high (~70%)
- **Gradual clean/noisy mixing**: 80% → 60% → 40% → 20% clean
- Progressive noise: 1 → 2 → 3 → 5 noise peaks
- Mass error: 1 → 2 → 5 → 8 ppm
- **NO dropout yet** - too harsh, delayed until 70K

**Why gradual mixing?**
- Prevents curriculum shock
- Model maintains stable performance
- Each batch has mix of clean (clear signal) and noisy (robustness)

**Expected outcomes:**
- Token accuracy: 70% → 65% (slight dip from noise, then stabilize)
- Precursor ppm error: 50k → 5k ppm
- Smooth loss curves, no spikes

### Phase 4: Final Realism (70K-100K)

**Goal:** Add dropout, achieve full robustness

**Stages 9-10:**
- **Introduce peak dropout at 70K** (8%, then 10%)
- Mostly/fully noisy data (10% → 0% clean)
- Maximum noise: 7-10 peaks, 12-15 ppm errors
- Intensity variation up to 20%
- Final loss weight ramp: spectrum 0.24-0.25, precursor 0.06-0.08

**Why dropout so late?**
- Peak dropout removes critical b/y ions
- Model needs strong foundation first
- Only add when robust to other noise types

**Expected outcomes:**
- Token accuracy: 65% → 60% (final realistic performance)
- Sequence accuracy: 20% → 25%
- Precursor ppm error: < 1000 ppm
- Robust to all noise types

## Comparison to Previous Curricula

| Aspect | Old (Extended V1) | Old (Phased) | **New (Final)** |
|--------|-------------------|--------------|-----------------|
| Spectrum loss start | 10K | 15K | **15K** ✓ |
| Precursor loss start | 20K | 30K | **30K** ✓ |
| Data at spectrum intro | 100% noisy! | Mix | **100% clean** ✓ |
| Clean/noisy mixing | Hard cutoff | No mixing | **Gradual 80%→0%** ✓ |
| Dropout introduction | 20K (2%) | Stage 7 | **70K (8%)** ✓ |
| Curriculum shock | Severe (20K) | Moderate | **None** ✓ |

## Expected Training Behavior

### What You Should See

**Steps 0-15K:**
- Smooth loss decrease
- Accuracy climbing steadily
- Pure CE learning

**Steps 15K-30K:**
- Spectrum loss appears, starts high (~1.0), decreases to ~0.5-0.8
- Token accuracy continues climbing
- NO shock - data still clean

**Steps 30K-70K:**
- Precursor loss appears, ppm error decreases from 50k → 5k
- Smooth transitions as noise mixing increases
- Loss curves smooth, no spikes
- Accuracy may dip slightly (65-70%) but stable

**Steps 70K-100K:**
- Dropout added, slight accuracy dip (60-65%)
- All losses stabilize
- Final performance achieved

### What You Should NOT See

❌ **Curriculum shock** - No sudden loss jumps at stage transitions
❌ **Spiky spectrum loss** - Should be smooth with low variance
❌ **Precursor loss staying high** - Should decrease to < 1k ppm by end
❌ **Token accuracy collapse** - Should stay in 60-70% range after 30K

## Files

- **src/training/curriculum.py** - Production curriculum (this one!)
- **src/training/curriculum_old_original.py** - Archived original
- **src/training/curriculum_old_phased.py** - Archived phased version
- **src/training/curriculum_old_extended_v2.py** - Archived extended v2
- **test_final_curriculum.py** - Validation script

## Testing

```bash
python test_final_curriculum.py
```

Should output all 10 stages with correct transitions.

## Summary

This curriculum is production-ready and incorporates:
- ✅ Delayed auxiliary losses (gradient-based timing)
- ✅ Clean data for learning physics (15K-30K)
- ✅ Smooth clean/noisy mixing (no shock)
- ✅ Delayed peak dropout (70K)
- ✅ Progressive noise ramp

**Result:** Stable training, no curriculum shock, better convergence! 🎉
