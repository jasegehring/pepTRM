# Curriculum V2: Smooth Transitions with Clean/Noisy Mixing

## Problem Identified

The previous curriculum (V1) caused **severe curriculum shock** at step 20K:
- Loss jumped from 1.5-1.7 → 2.4-2.8
- Token accuracy dropped from 65-75% → 26-50%
- Sequence accuracy dropped to nearly 0%

**Root cause:** Hard transition from 100% clean to 100% noisy data at step 20K, introducing:
- Noise peaks
- Peak dropout (very harsh)
- Mass errors
- Precursor loss (new)
- All simultaneously

## Solution: Curriculum V2 with Probability Mixing

### Key Improvements

1. **Introduce precursor loss EARLY on clean data (step 10K)**
   - Allows model to learn mass constraints without noise confusion
   - Clean data provides clear gradient signals

2. **Keep data 100% clean through step 40K**
   - Stages 1-4 focus purely on learning fragmentation patterns
   - Build strong foundation before introducing noise

3. **Gradual clean/noisy mixing (steps 40K-100K)**
   - Stage 5 (40-50K): 80% clean, 20% noisy
   - Stage 6 (50-60K): 60% clean, 40% noisy
   - Stage 7 (60-70K): 40% clean, 60% noisy
   - Stage 8 (70-80K): 20% clean, 80% noisy
   - Stage 9 (80-90K): 10% clean, 90% noisy
   - Stage 10 (90-100K): 100% noisy
   - **No more hard transitions!**

4. **Delay peak_dropout until step 70K**
   - Peak dropout removes critical b/y ions, destroying gradient signal
   - Only introduce at 5% after model is robust (stage 8)

5. **Smooth noise ramp**
   - Mass error: 0 → 1 → 2 → 5 → 8 → 12 → 15 ppm
   - Noise peaks: 0 → 1 → 2 → 3 → 5 → 7 → 10
   - Peak dropout: 0 → 0 → 0 → 0 → 5% → 8% → 10%

## 10-Stage Curriculum Breakdown

| Stage | Steps | Length | Clean % | Noise Peaks | Dropout | Mass Error | Spectrum Loss | Precursor Loss |
|-------|-------|--------|---------|-------------|---------|------------|---------------|----------------|
| 1 | 0-10K | 7-10 | 100% | 0 | 0% | 0 ppm | 0.00 | 0.00 |
| 2 | 10-20K | 7-10 | 100% | 0 | 0% | 0 ppm | 0.05 | **0.01** ← Start here! |
| 3 | 20-30K | 7-12 | 100% | 0 | 0% | 0 ppm | 0.08 | 0.02 |
| 4 | 30-40K | 7-14 | 100% | 0 | 0% | 0 ppm | 0.10 | 0.03 |
| 5 | 40-50K | 7-15 | 80% | 1 | 0% | 1 ppm | 0.12 | 0.04 |
| 6 | 50-60K | 7-16 | 60% | 2 | 0% | 2 ppm | 0.13 | 0.05 |
| 7 | 60-70K | 7-17 | 40% | 3 | 0% | 5 ppm | 0.14 | 0.06 |
| 8 | 70-80K | 7-18 | 20% | 5 | 5% | 8 ppm | 0.14 | 0.07 |
| 9 | 80-90K | 7-20 | 10% | 7 | 8% | 12 ppm | 0.15 | 0.08 |
| 10 | 90-100K | 7-20 | 0% | 10 | 10% | 15 ppm | 0.15 | 0.08 |

## Implementation Details

### Modified Files

1. **src/training/curriculum_extended.py**
   - Added `clean_data_ratio` field to `CurriculumStage`
   - Redesigned `EXTENDED_CURRICULUM` with smooth transitions
   - Updated logging to show clean/noisy ratio

2. **src/data/dataset.py**
   - Added `clean_data_ratio` parameter to `SyntheticPeptideDataset`
   - Modified `_generate_sample()` to randomly choose clean vs noisy
   - Updated `set_difficulty()` to accept `clean_data_ratio`

3. **Probability Mixing Logic**
   ```python
   # In dataset._generate_sample():
   use_clean = random.random() < self.clean_data_ratio

   # Generate spectrum with conditional noise
   spectrum = generate_theoretical_spectrum(
       noise_peaks=0 if use_clean else self.noise_peaks,
       peak_dropout=0.0 if use_clean else self.peak_dropout,
       mass_error_ppm=0.0 if use_clean else self.mass_error_ppm,
       # ...
   )
   ```

## Expected Benefits

1. **No curriculum shock** - Smooth transitions prevent performance collapse
2. **Better convergence** - Learning mass constraints on clean data first
3. **Robust to noise** - Gradual exposure builds resilience
4. **Higher final accuracy** - Strong foundation + smooth difficulty ramp
5. **Stable training** - No more spiky loss curves

## Next Steps

1. Update `configs/optimized_extended.yaml` if needed
2. Start a fresh training run with the new curriculum
3. Monitor for smooth transitions at steps 10K, 20K, 30K, 40K
4. Compare loss curves to previous run
5. Expect to see:
   - Precursor loss decreasing from step 10K
   - Stable accuracy through step 40K
   - Smooth (not spiky) transitions at 40K, 50K, 60K, etc.

## Test Results

✓ Curriculum test passed (see `test_curriculum_v2.py`)
✓ All 10 stages configured correctly
✓ Clean/noisy mixing verified
✓ Dataset updates working properly
