# Curriculum Comparison: Original vs Delayed Spectrum

## Quick Summary

**Change:** Delay spectrum loss introduction from **step 15K → 30K**
**Reason:** Model at 48% accuracy needs time to reach 65% before spectrum loss provides useful gradient
**W&B Run:** Resumes run ID `nbs1e6hk` from step 15K

---

## Detailed Comparison

### Original Curriculum

| Steps | Stage | Spectrum Weight | Peptide Length | Purpose |
|-------|-------|-----------------|----------------|---------|
| 0-7.5K | Foundation | 0.0 | 7-10 | Learn basics |
| 7.5K-15K | Stabilization | 0.0 | 7-10 | Build to 55% acc |
| **15K-22.5K** | **Spectrum Intro** | **0.08** | 7-11 | Add spectrum loss ← **WE ARE HERE** |
| 22.5K-30K | Spectrum Ramp | 0.12 | 7-12 | Increase weight |
| 30K+ | ... | ... | ... | Continue |

**Problem at step 15K:**
- Expected: 55% token accuracy
- Actual: 48% token accuracy ❌
- Spectrum coverage at 48%: ~2.1% (too weak!)

---

### New Delayed Curriculum

| Steps | Stage | Spectrum Weight | Peptide Length | Purpose |
|-------|-------|-----------------|----------------|---------|
| 0-7.5K | Foundation | 0.0 | 7-10 | Learn basics |
| 7.5K-15K | Foundation | 0.0 | 7-10 | Build foundation |
| **15K-22.5K** | **Extended Stabilization** | **0.0** | 7-11 | **Give time to mature** ← **START HERE** |
| **22.5K-30K** | **Pre-Spectrum Ramp** | **0.0** | 7-12 | **Reach 65% accuracy** |
| **30K-37.5K** | **Spectrum Intro** | **0.06** | 7-12 | **Add spectrum with strong signal** |
| 37.5K-45K | Spectrum Ramp | 0.10 | 7-13 | Increase weight |
| 45K+ | ... | ... | ... | Continue |

**Expected at step 30K:**
- Token accuracy: 65-70% ✓
- Spectrum coverage: 10-20% (useful signal!)

---

## Why This Works Better

### Coverage vs Token Accuracy

Based on our testing:

```
Token Accuracy → Spectrum Coverage → Gradient Quality
    25%        →      0.7%         →  Extremely weak/noisy
    48%        →      2.1%         →  Very weak/noisy ❌ CURRENT
    60%        →      8-12%        →  Weak but usable
    65%        →     15-20%        →  Good signal ✓ TARGET
    70%        →     25-30%        →  Strong signal
   100%        →     60-64%        →  Maximum (ground truth)
```

### The Problem with Early Introduction

At 48% accuracy with 2.1% coverage:
- **CE Loss gradient:** Strong, clear signal pointing toward correct tokens
- **Spectrum Loss gradient:** Weak, noisy signal (98% miss rate)
- **Result:** Noisy gradient interferes with CE learning

At 65% accuracy with 15-20% coverage:
- **CE Loss gradient:** Still strong and clear
- **Spectrum Loss gradient:** Meaningful signal (80% miss rate but consistent)
- **Result:** Spectrum reinforces CE learning by favoring sequences with correct fragment patterns

---

## What to Expect

### Steps 15K-30K (Next 15,000 steps)
**Goal:** Build token accuracy from 48% → 65%+

**Training:**
- Pure CE loss (spectrum weight = 0.0)
- Clean data (100% clean, no noise)
- Gradually longer peptides (7-10 → 7-12)

**Monitoring:**
- `train/token_accuracy`: Should steadily improve 48% → 60%+ by step 25K
- `train/ce_final`: Should continue decreasing
- `val_easy/token_accuracy`: Should stay high or improve

**Success criteria at step 30K:**
- ✓ Token accuracy ≥ 65%
- ✓ CE loss stable and decreasing
- ✓ Ready for spectrum loss introduction

---

### Steps 30K+ (After reaching 65%)
**Goal:** Add spectrum loss with meaningful gradient signal

**Training:**
- Spectrum weight: 0.06 → 0.10 (gentle ramp)
- Still clean data initially
- Expected coverage: 15-20%

**Monitoring:**
- `train/spectrum_loss`: Should start ~0.85 and decrease
- `train/token_accuracy`: May dip 2-3% then recover
- `train/total_loss`: Should decrease overall

**Success criteria:**
- ✓ Spectrum loss drops from 0.85 → 0.75 (coverage 15% → 25%)
- ✓ Token accuracy recovers and continues improving
- ✓ Model learns to jointly optimize CE + spectrum

---

## How to Run

```bash
./RESUME_WITH_DELAYED_SPECTRUM.sh
```

Or directly:
```bash
python3 scripts/resume_with_delayed_spectrum.py
```

This will:
1. Load checkpoint from step 15000
2. Resume W&B run `nbs1e6hk`
3. Use delayed curriculum (no spectrum until step 30K)
4. Train to step 100K (or until stopped)

---

## Files Changed

### New Files Created:
- `src/training/curriculum_delayed_spectrum.py` - Modified curriculum
- `scripts/resume_with_delayed_spectrum.py` - Resume script
- `RESUME_WITH_DELAYED_SPECTRUM.sh` - Convenience wrapper

### Original Files (Unchanged):
- `src/training/curriculum.py` - Original curriculum (preserved)
- Other training code remains the same

### Spectrum Loss Fix Applied:
- `src/data/ion_types.py` - PAD token masking ✓
- `src/training/losses.py` - Sequence mask passed through ✓

---

## Questions?

- **Q: Will this take longer to train?**
  A: Yes, 15K more steps before spectrum loss. But you'll get better results!

- **Q: Can I switch back to original curriculum?**
  A: Yes, just use `curriculum.py` instead of `curriculum_delayed_spectrum.py`

- **Q: What if accuracy doesn't reach 65% by step 30K?**
  A: You can modify the curriculum to delay even further (e.g., to step 35K)

- **Q: Will this resume my exact wandb run?**
  A: Yes, using `WANDB_RUN_ID=nbs1e6hk` to continue logging to the same run
