# Validation Accuracy Failure Analysis

## Summary

The model achieves near-perfect accuracy on short sequences (length 7-12) but fails catastrophically on longer sequences (length 14+), dropping from 99% to ~40%.

## Key Findings

### 1. The Cliff is at Length 13

| Sequence Length | Token Accuracy |
|-----------------|----------------|
| 10 | 99.8% |
| 12 | 99.6% |
| 14 | 41.5% |
| 16 | 34.4% |
| 18 | 34.8% |

### 2. Failure Pattern is Position-Dependent

For length 14 sequences, accuracy by position:
- Position 1-2: 94-95% (good)
- Position 3-4: 88-93% (slight drop)
- Position 5-6: 58-77% (significant drop)
- Position 7+: 5-25% (essentially random)

The model gets the first few positions right, then "gives up" around position 6-7.

### 3. NOT Caused by Precursor Mass

Experiment: Feed length-14 spectrum with fake precursor mass (as if length 10):
- Result: Accuracy stays at ~42% regardless of precursor mass provided
- Conclusion: Model is NOT using precursor mass as a "length hint"

Experiment: Feed length-10 spectrum with fake precursor mass (as if length 14):
- Result: Accuracy drops from 89% to 61%
- Conclusion: Model does use precursor mass somewhat, but it's not the cause of length-14 failure

### 4. Spectrum Encoder is the Bottleneck

The model cannot interpret the peak patterns from longer sequences. A length-14 peptide has 52 fragment ions (vs 36 for length-10), and the cross-attention mechanism struggles to extract the right information.

### 5. Training Never Learned Length 14+

Checkpoint analysis across training:
```
Step      L=10    L=12    L=14    L=18
5K        57.0%   48.8%   25.5%   21.9%   (Stage 1: len 7-12)
10K       95.2%   86.5%   39.6%   30.9%   (Stage 2: len 10-15)
15K       97.6%   95.9%   42.2%   33.7%   (Stage 3: len 12-18)
73K       99.8%   99.6%   42.2%   34.0%   (Best model)
```

The model improved on L=14 from 25% to 40% in early training, then **completely plateaued** despite continued exposure.

### 6. Curriculum Issues

Clean data exposure by length:
- Length 7-9: 1,333 clean samples each (80% clean)
- Length 10-11: 2,333 clean samples each (70% clean)
- Length 12: 3,238 clean samples (30% clean)
- Length 13-15: 1,905 clean samples each (21% clean)
- Length 16-18: 905 clean samples each (12% clean)
- Length 21+: 0 clean samples (0% clean!)

Lengths 13-15 are introduced at step 10K with 60% clean / 40% noisy, when the model has already learned strong priors for lengths 7-12.

## Root Cause Analysis

The model has learned **length-specific representations** that don't generalize:

1. **Early training (Stage 1)**: Model learns optimal patterns for length 7-12 spectra with 80% clean data
2. **Stage 2 onwards**: Longer sequences introduced, but model has already "locked in" to short-sequence patterns
3. **Gradient conflict**: Easy short sequences (where model is perfect) don't generate gradients; hard long sequences generate conflicting gradients that don't lead to improvement
4. **Plateau**: Model converges to a local minimum where it essentially "guesses" for longer sequences

## Why Val_Hard Caps at ~40%

Val_hard uses lengths 8-25. Accuracy breakdown:
- Lengths 8-12 (~25% of samples): ~95% accuracy
- Lengths 13-25 (~75% of samples): ~35% accuracy

Weighted average: 0.25 * 95% + 0.75 * 35% = **50%**

Actual observed ~40% suggests even worse performance on very long sequences.

## Recommendations

### Option 1: Extended Clean Phase
Add a curriculum stage that trains on longer sequences (13-20) with HIGH clean ratio (70-80%) BEFORE introducing noise. The model needs more clean exposure to longer sequences while it's still plastic.

### Option 2: Length Balancing
Modify the curriculum to give equal TOTAL exposure to each length, not just equal probability within each stage. Lengths 13-15 need ~2x more training time.

### Option 3: Progressive Length Extension
Instead of jumping from max_length=12 to max_length=15, extend one length at a time:
- Stage 1: 7-12 (80% clean)
- Stage 2: 7-13 (80% clean)
- Stage 3: 7-14 (80% clean)
- Stage 4: 7-15 (70% clean)
- ... etc

### Option 4: Separate Length Heads
Architectural change: Use different output heads for different length ranges, then gradually merge them. This prevents interference between length-specific patterns.

## Files Created for Diagnosis

1. `scripts/diagnose_length_failure.py` - Position-level accuracy analysis
2. `scripts/test_precursor_mass_impact.py` - Tests if precursor mass causes the failure
3. `scripts/analyze_curriculum_exposure.py` - Analyzes training exposure per length
4. `scripts/test_length_generalization.py` - Original length sweep test
5. `scripts/debug_validation_accuracy.py` - Validation debugging
