# Precursor Mass Constraint Enhancement

## Problem Identified

Diagnostic testing revealed that the model was **ignoring the precursor mass constraint**:
- Model maintained 96% accuracy even with ±100 Da precursor mass errors
- This means it was pattern-matching on spectra alone, not using physics
- Missing a critical constraint: `sum(amino_acid_masses) + H2O = precursor_mass`

## Solution Implemented

### 1. New PrecursorMassLoss Component

Added direct precursor mass constraint loss in `src/training/losses.py`:

```python
class PrecursorMassLoss:
    """
    Directly penalizes predictions where sum(aa_masses) ≠ precursor_mass

    Computes expected mass from probability distribution:
        predicted_mass = sum_i sum_aa P(aa_i) * mass(aa) + H2O

    Loss = |predicted_mass - true_precursor_mass| in ppm
    """
```

**Features:**
- Uses ppm (parts per million) for interpretability
- Differentiable through soft probability distribution
- Clamped to 100 ppm to prevent gradient explosion
- Works on final prediction probabilities

### 2. Enhanced Curriculum

Updated `DEFAULT_CURRICULUM` with much stronger mass constraints:

| Stage | Steps | Spectrum Weight | Precursor Weight | Notes |
|-------|-------|-----------------|------------------|-------|
| 1: Warmup | 0-8K | 0.0 | 0.0 | Pure CE, learn patterns |
| 2: Mass Constraints | 8K-16K | 0.2 → | 0.2 → | **Start enforcing mass** |
| 2.5: Strong Mass | 16K-22K | 0.3 → | 0.5 → | **Strong before noise** |
| 2.75: Gradual Noise | 22K-28K | 0.4 → | 0.7 → | Maintain strong constraint |
| 3: Moderate | 28K-38K | 0.5 → | 0.9 → | Near-full constraint |
| 4: Realistic | 38K-50K | **0.6** | **1.0** | **Full constraint** |

**Key Changes:**
- `spectrum_loss_weight`: Increased from max 0.15 → **0.6** (4x higher!)
- `precursor_loss_weight`: New ramp from 0.0 → **1.0**
- By stage 4, the model **must** use precursor mass to succeed

### 3. Trainer Integration

Updated `src/training/trainer_optimized.py`:
- Dynamically updates both `spectrum_weight` and `precursor_weight` during training
- Passes `precursor_mass` to loss function forward call
- Logs precursor loss in metrics

### 4. CurriculumScheduler Updates

Added to `src/training/curriculum.py`:
- `precursor_loss_weight` field in `CurriculumStage`
- `get_precursor_loss_weight()` method
- Logs precursor weight during stage transitions

## Expected Behavior After Retraining

### Before (Current Model):
```
Correct precursor:  96.18% accuracy
+100 Da offset:     96.06% accuracy  ← BAD! Model ignores precursor
```

### After (With New Loss):
```
Correct precursor:  ~95% accuracy (similar)
+100 Da offset:     ~20-30% accuracy  ← GOOD! Model requires correct precursor
```

The model should **fail dramatically** when given wrong precursor mass, proving it's using the constraint.

## Loss Weight Interpretation

At stage 4 (final), the total loss is:
```
total_loss = 1.0 * CE_loss + 0.6 * spectrum_loss + 1.0 * precursor_loss
```

Relative importance:
- **Cross-entropy: 38%** (pattern learning)
- **Spectrum matching: 23%** (fragment mass constraints)
- **Precursor mass: 38%** (total mass constraint)

Much more balanced than before (was 85% CE, 15% spectrum, 0% precursor)!

## Files Modified

1. `src/training/losses.py`
   - Added `PrecursorMassLoss` class
   - Updated `CombinedLoss` to include precursor component
   - Forward pass now accepts `precursor_mass` parameter

2. `src/training/curriculum.py`
   - Added `precursor_loss_weight` to `CurriculumStage`
   - Increased `spectrum_loss_weight` across all stages
   - Updated `CurriculumScheduler` with `get_precursor_loss_weight()`

3. `src/training/trainer_optimized.py`
   - Updates `precursor_weight` during curriculum transitions
   - Passes `precursor_mass` to loss function

4. `scripts/test_precursor_loss.py` (new)
   - Comprehensive tests for new loss component
   - Validates correct computation on perfect/wrong/soft predictions

## How to Use

### Option 1: Retrain from Scratch (Recommended)
```bash
python scripts/train_optimized.py
```
This will use the new enhanced curriculum with precursor mass constraints.

### Option 2: Train with MS2PIP
```bash
python scripts/train_ms2pip.py
```
Combines realistic MS2PIP spectra with strong mass constraints.

### Option 3: Resume Training
If you want to add mass constraints to an existing checkpoint, you can resume training but the model may need to "unlearn" its current pattern-matching strategy.

## Validation

After training, run the precursor mass diagnostic again:
```bash
python scripts/diagnostic_precursor.py
```

You should see **large accuracy drops** with wrong precursor masses, confirming the model is using the constraint.

## Technical Details

### Why Clamp to 100 ppm?

The loss is clamped to prevent gradient explosion:
- Typical MS errors: 5-20 ppm
- Random guesses: 1000-100,000 ppm
- Clamping at 100 ppm provides strong signal while maintaining stability
- Gradients for errors >100 ppm are still non-zero (linear penalty)

### Why Curriculum Progression?

Starting with full precursor constraint (weight=1.0) from step 0 would be too hard:
1. Stage 1: Learn basic amino acid patterns first
2. Stage 2: Introduce mass constraint on clean data
3. Stages 2.5-3: Strengthen constraint before adding significant noise
4. Stage 4: Full constraint with realistic noise

This gradual ramp prevents catastrophic forgetting.

### Relationship to Spectrum Loss

- **Spectrum loss**: Checks if fragment ions match observed peaks
- **Precursor loss**: Checks if total mass equals precursor mass

Both are complementary:
- Spectrum loss: "Do the pieces match what we see?"
- Precursor loss: "Does the whole add up to the right total?"

Together they provide strong physics-based constraints.

## A-ions Note

The `SpectrumMatchingLoss` includes a-ions (b-ions minus CO, ~28 Da lighter). These are computed in `losses.py:164-165`:
```python
a_ions = b_ions - CO_MASS
```

However, MS2PIP may not predict a-ions depending on the model. The simple synthetic generator includes them but MS2PIP's HCD2021 model might not (HCD fragmentation typically produces fewer a-ions than CID). This is fine - the spectrum loss just won't have a-ions to match in that case.

## Next Steps

1. **Retrain the model** with new curriculum
2. **Run diagnostics** to verify precursor mass sensitivity
3. **Compare performance** between:
   - Old curriculum (weak constraints)
   - New curriculum (strong constraints)
   - MS2PIP + strong constraints

The model should show:
- Similar or better accuracy on correct data
- **Much worse** accuracy on wrong precursor mass (this is good!)
- Better generalization to real experimental data
