# Precursor Mass Loss Bug Fix

## Problem

During diagnostic training runs, the precursor mass error was exploding to the clamp value (100 ppm) as soon as precursor loss was introduced in the curriculum. The predicted precursor mass was consistently way off from the target.

## Root Cause

**Vocabulary Indexing Mismatch** between the MS2PIP dataset and the loss functions.

### The Bug

1. **MS2PIP Dataset** (`src/data/ms2pip_dataset.py`):
   - Created custom token indexing: `{aa: i+1 for i, aa in enumerate(STANDARD_VOCAB)}`
   - Resulted in: `0=PAD, 1=A, 2=R, 3=N, 4=D, ...`
   - Only used PAD as special token

2. **Loss Functions** (`src/training/losses.py`):
   - Used main vocabulary from `constants.py`: `VOCAB = ['<PAD>', '<SOS>', '<EOS>', '<UNK>'] + STANDARD_VOCAB`
   - Resulted in: `0=PAD, 1=SOS, 2=EOS, 3=UNK, 4=A, 5=R, 6=N, 7=D, ...`
   - Has 4 special tokens before amino acids

### The Impact

When the dataset encoded a sequence like "ACDEF":
- Dataset used indices: `[1, 5, 7, 9, 11]` (meaning A, C, D, E, F)

When the loss function decoded these indices:
- Index 1 → Looked up `aa_masses[1]` → Got mass for `<SOS>` → **0.0 Da** (should be A = 71.0 Da)
- Index 5 → Looked up `aa_masses[5]` → Got mass for `R` → **156.1 Da** (should be C = 103.0 Da)
- And so on...

The loss function was looking up masses with indices shifted by 3 positions! Most amino acids were mapped to wrong masses, and the first few were getting 0.0 from special tokens.

Result: Predicted precursor mass ≈ 0-200 Da instead of the correct 800-2000 Da → massive ppm error → hits 100 ppm clamp immediately.

## The Fix

**Changes to `src/data/ms2pip_dataset.py`:**

1. Import the standard vocabulary mapping:
   ```python
   from ..constants import AA_TO_IDX, PAD_IDX
   ```

2. Use the standard mapping instead of creating a custom one:
   ```python
   # Before:
   self.token_to_idx = {aa: i+1 for i, aa in enumerate(self.amino_acids)}
   self.token_to_idx['<PAD>'] = 0

   # After:
   self.token_to_idx = AA_TO_IDX  # Use main vocabulary indices
   ```

## Verification

Created `scripts/verify_vocabulary_fix.py` to validate:

1. ✓ Dataset uses correct token indices matching `AA_TO_IDX`
2. ✓ Special tokens (`<PAD>`, `<SOS>`, `<EOS>`, `<UNK>`) have 0 mass
3. ✓ All amino acids map to correct masses
4. ✓ Loss function calculates precursor mass with < 0.001 Da error (< 0.01 ppm)

Test output:
```
Sample sequence: CQDQVID
Expected precursor mass: 819.3433 Da
Loss function predicted mass: 819.3433 Da
Error: 0.000001 Da (0.00 ppm)
✓ Loss function calculates correct mass!
```

## Impact

With this fix:
- Precursor mass loss will now correctly penalize sequences with wrong total mass
- The model can learn to respect mass constraints during training
- The curriculum can successfully introduce precursor loss starting at stage 2 (step 10K)
- Expected ppm error should be < 5 ppm during training (well below the 100 ppm clamp)

## Files Modified

1. `src/data/ms2pip_dataset.py` - Fixed vocabulary indexing
2. `scripts/verify_vocabulary_fix.py` - Added verification script (new)

## Next Steps

1. ✓ Run verification script to confirm fix
2. Resume diagnostic training from last checkpoint
3. Monitor that precursor mass error stays reasonable (< 10 ppm) when loss is introduced
4. Verify that final model respects precursor mass constraints during inference
