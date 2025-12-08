# Overnight Training Run Plan (50K Steps)

## Executive Summary

**Yes, you're ready for a longer run!** The model has proven it can learn (68% token accuracy), and now we need to test whether your spectrum matching loss improvements help with mass constraints.

**Expected runtime:** ~13 hours (at 1.08 it/s)

## Why the High Mass Error (50k ppm) is Expected

With 68% token accuracy but no mass constraints:
- For a 10-residue peptide: ~3.2 wrong amino acids
- Single G→W substitution: 129 Da error = 12,900 ppm
- Multiple random substitutions easily accumulate to 50k ppm

**The model hasn't learned that precursor mass is a hard physical constraint.**

This is exactly what your spectrum matching loss fixes!

## New 4-Stage Curriculum

### Stage 1: Warmup (0-10K steps)
```
Purpose: Learn basic sequence patterns
Data: Clean, short peptides (7-10 residues)
Loss: Pure cross-entropy (spectrum_weight=0.0)
Expected:
  - Token accuracy → 68% (like your previous runs)
  - Mass error stays high (~50k ppm)
```

### Stage 2: Introduce Spectrum Loss (10K-20K steps) ⭐ CRITICAL STAGE
```
Purpose: Test your spectrum matching improvements!
Data: SAME as Stage 1 (clean, short) - isolate the loss effect
Loss: Cross-entropy + spectrum matching (spectrum_weight=0.1)
Expected:
  - Token accuracy: stable or slight improvement (68-72%)
  - Mass error: SHOULD DROP significantly (50k → 5-10k ppm)
  - If mass error doesn't drop, spectrum loss isn't helping
Key insight: Don't add data difficulty AND new loss at same time
```

### Stage 3: Moderate Difficulty (20K-35K steps)
```
Purpose: Test if mass constraints hold with harder data
Data: Longer peptides (up to 15), some noise (5 peaks), dropout (10%)
Loss: spectrum_weight=0.1
Expected:
  - Token accuracy: may dip slightly (65-70%)
  - Mass error: should stay improved (<15k ppm)
  - Model learning to generalize
```

### Stage 4: Realistic Conditions (35K-50K steps)
```
Purpose: Real-world simulation
Data: Full length (up to 20), noisy (10 peaks), dropout (20%), mass error (20 ppm)
Loss: spectrum_weight=0.15 (slightly higher for noisy data)
Expected:
  - Token accuracy: 60-70% (harder data)
  - Mass error: <20k ppm (if learned constraints)
  - Sequence accuracy: 5-15% (hopeful!)
```

## Key Metrics to Watch

### 1. **Stage Transitions (Check Logs)**
```bash
grep "Curriculum: Advanced" <output_log>
```
Should show 4 transitions at steps 10K, 20K, 35K

### 2. **Mass Error Trajectory (Most Important!)**
- **0-10K**: ~50k ppm (expected, no spectrum loss)
- **10K mark**: Should see DROP when spectrum loss activates
- **10K-20K**: Target <10k ppm (proves your fixes work!)
- **20K-35K**: Stay <15k ppm (generalization)
- **35K-50K**: <20k ppm (realistic conditions)

### 3. **Token Accuracy**
- Should stay stable or improve slightly (68-72%)
- May dip in Stage 3/4 due to harder data

### 4. **Sequence Accuracy**
- Currently ~1% (13/1000 sequences)
- If mass constraints help: could reach 5-15%

### 5. **Spectrum Loss Value**
- Should appear in metrics starting at step 10K
- Typical values: 0.1-1.0 (depends on normalization)
- Should generally decrease over time

## Success Criteria

### ✅ **Success**: Your spectrum loss improvements work
- Mass error drops significantly at 10K step boundary
- Mass error stays <15k ppm in Stages 2-4
- Sequence accuracy improves

### ❌ **Failure**: Spectrum loss doesn't help
- Mass error stays ~50k ppm throughout
- No improvement at 10K boundary
- Need to debug the spectrum matching loss

### 🤔 **Partial Success**: Helps but not enough
- Small drop at 10K (50k → 30k ppm)
- Some improvement but not dramatic
- May need to tune `spectrum_weight` hyperparameter

## What Could Go Wrong

### 1. **Spectrum Loss Causes Instability**
**Symptoms:** Loss explodes, NaNs in metrics
**Solution:** Reduce `spectrum_weight` from 0.1 to 0.05

### 2. **Spectrum Loss Has No Effect**
**Symptoms:** Mass error unchanged at 10K
**Possible causes:**
- Bug in compute_theoretical_peaks (check a-ions are included)
- Mass tolerance window too tight
- Loss weight too small

### 3. **Model Forgets Earlier Learning**
**Symptoms:** Token accuracy drops when spectrum loss added
**Solution:** This suggests loss weights are unbalanced

## How to Monitor Progress

### During Training
```bash
# Watch for stage transitions
tail -f <output_log> | grep -E "(Curriculum|Step [0-9]+K)"

# Check WandB dashboard
# Key plots: mass_error_ppm, token_accuracy, spectrum_loss
```

### After Training
```bash
# Check final metrics
python3 scripts/evaluate.py --checkpoint checkpoints/final_model.pt

# Plot learning curves
python3 scripts/plot_training.py
```

## Compute Efficiency Analysis

**Is this worth ~13 hours of compute?**

YES, because:
1. **Critical test** of your spectrum matching improvements
2. **Clear success criteria** - you'll know if it works
3. **Ablation study** built in (Stage 1 vs Stage 2 isolates effect)
4. **Realistic end-to-end** test of full curriculum
5. **One run tests everything** - efficient use of time

If this works, you have validation that:
- Your a-ion fix was important
- Your mass tolerance window works
- The TRM approach + mass constraints is viable

If it doesn't work, you have clear diagnostic information:
- Exactly which stage failed
- Whether it's the loss function or hyperparameters
- Next debugging steps are obvious

## Next Steps After Overnight Run

### If Successful (Mass Error Drops)
1. Document results in IMPROVEMENTS.md
2. Commit with message: "Validate spectrum loss improvements - 50K curriculum run"
3. Consider even longer run (100K) to see limits
4. Start thinking about real data integration

### If Unsuccessful (Mass Error Stays High)
1. Inspect model predictions manually - what's it getting wrong?
2. Check spectrum loss values - is it too small to matter?
3. Debug compute_theoretical_peaks - print intermediate values
4. Consider increasing spectrum_weight

### If Partially Successful
1. Tune spectrum_weight (try 0.05, 0.15, 0.2)
2. Analyze which peptides have low vs high mass error
3. May need to adjust curriculum pacing

## Launch Command

```bash
python3 scripts/train.py
```

Configuration will auto-load from `configs/default.yaml` (now set to 50K steps).

Good luck! 🚀
