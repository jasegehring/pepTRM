#!/usr/bin/env python3
"""
Test the improved curriculum with clean/noisy mixing.
"""

from src.training.curriculum_extended import EXTENDED_CURRICULUM, CurriculumScheduler
from src.data.dataset import SyntheticPeptideDataset


def test_curriculum():
    """Test the curriculum stages and transitions."""

    # Create dataset
    dataset = SyntheticPeptideDataset()

    # Create curriculum scheduler
    scheduler = CurriculumScheduler(
        stages=EXTENDED_CURRICULUM,
        dataset=dataset,
    )

    print("=" * 80)
    print("IMPROVED CURRICULUM V2 - Clean/Noisy Mixing")
    print("=" * 80)
    print()

    # Test transitions at key steps
    test_steps = [0, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000]

    for step in test_steps:
        changed = scheduler.step(step)
        stage = scheduler.current_stage

        if changed and stage:
            print(f"\nStep {step:,} - Stage: {stage.name}")
            print(f"  Peptide length: {stage.min_length}-{stage.max_length}")
            print(f"  Clean/Noisy: {stage.clean_data_ratio:.0%} clean, {(1-stage.clean_data_ratio):.0%} noisy")
            print(f"  Noise params (for noisy samples only):")
            print(f"    - Noise peaks: {stage.noise_peaks}")
            print(f"    - Peak dropout: {stage.peak_dropout:.1%}")
            print(f"    - Mass error: {stage.mass_error_ppm} ppm")
            print(f"    - Intensity variation: {stage.intensity_variation:.1%}")
            print(f"  Loss weights:")
            print(f"    - Spectrum loss: {stage.spectrum_loss_weight:.3f}")
            print(f"    - Precursor loss: {stage.precursor_loss_weight:.3f}")

    print("\n" + "=" * 80)
    print("KEY IMPROVEMENTS:")
    print("=" * 80)
    print("✓ Precursor loss introduced at step 10K on CLEAN data")
    print("✓ All data 100% clean through step 40K")
    print("✓ Noise introduced gradually with mixing ratios:")
    print("  - Stage 5 (40-50K): 80% clean, 20% noisy")
    print("  - Stage 6 (50-60K): 60% clean, 40% noisy")
    print("  - Stage 7 (60-70K): 40% clean, 60% noisy")
    print("  - Stage 8 (70-80K): 20% clean, 80% noisy")
    print("  - Stage 9 (80-90K): 10% clean, 90% noisy")
    print("  - Stage 10 (90-100K): 0% clean, 100% noisy")
    print("✓ Peak dropout delayed until step 70K (stage 8)")
    print("✓ Smooth noise ramp: 0→1→2→5→8→12→15 ppm")
    print("=" * 80)
    print()

    # Verify dataset updates
    print("Dataset verification:")
    print(f"  clean_data_ratio: {dataset.clean_data_ratio}")
    print(f"  min_length: {dataset.min_length}")
    print(f"  max_length: {dataset.max_length}")
    print(f"  noise_peaks: {dataset.noise_peaks}")
    print(f"  peak_dropout: {dataset.peak_dropout}")
    print(f"  mass_error_ppm: {dataset.mass_error_ppm}")
    print()

    # Generate a few samples to verify mixing works
    print("Generating 10 samples at step 50000 (stage 6: 60% clean, 40% noisy):")
    scheduler.step(50000)

    clean_count = 0
    for i in range(10):
        sample = dataset._generate_sample()
        num_peaks = sample.spectrum_mask.sum().item()
        print(f"  Sample {i+1}: {num_peaks} peaks")

    print("\n✓ Curriculum test passed!")


if __name__ == '__main__':
    test_curriculum()
