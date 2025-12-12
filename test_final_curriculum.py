#!/usr/bin/env python3
"""
Test the final authoritative curriculum.
"""

from src.training.curriculum import DEFAULT_CURRICULUM, CurriculumScheduler
from src.data.dataset import SyntheticPeptideDataset


def test_curriculum():
    """Test the final curriculum stages and transitions."""

    dataset = SyntheticPeptideDataset()
    scheduler = CurriculumScheduler(stages=DEFAULT_CURRICULUM, dataset=dataset)

    print("=" * 80)
    print("FINAL AUTHORITATIVE CURRICULUM")
    print("=" * 80)
    print()

    # Test all stage transitions
    test_steps = [0, 7500, 15000, 22500, 30000, 40000, 50000, 60000, 70000, 85000]

    for step in test_steps:
        changed = scheduler.step(step)
        stage = scheduler.current_stage

        if changed and stage:
            print(f"\n{'='*70}")
            print(f"Step {step:,} - {stage.name}")
            print(f"{'='*70}")
            print(f"  Length: {stage.min_length}-{stage.max_length}")
            print(f"  Data: {stage.clean_data_ratio:.0%} clean, {(1-stage.clean_data_ratio):.0%} noisy")

            if stage.clean_data_ratio < 1.0:
                print(f"  Noise (for noisy samples):")
                print(f"    - Peaks: {stage.noise_peaks}")
                print(f"    - Dropout: {stage.peak_dropout:.1%}")
                print(f"    - Mass error: {stage.mass_error_ppm} ppm")
                print(f"    - Intensity var: {stage.intensity_variation:.1%}")

            print(f"  Loss weights:")
            print(f"    - Spectrum: {stage.spectrum_loss_weight:.2f}")
            print(f"    - Precursor: {stage.precursor_loss_weight:.2f}")

    print("\n" + "=" * 80)
    print("KEY DESIGN PRINCIPLES:")
    print("=" * 80)
    print("Phase 1 (0-15K): Pure CE, 100% clean")
    print("  → Build foundation, reach ~55% accuracy")
    print()
    print("Phase 2 (15K-30K): Add spectrum loss, 100% clean")
    print("  → Learn fragment physics with clear gradients")
    print("  → Spectrum loss starts at 15K (not 10K!)")
    print()
    print("Phase 3 (30K-70K): Add precursor + gradual mixing 80%→20% clean")
    print("  → Smooth noise introduction, no curriculum shock")
    print("  → NO peak dropout yet - too harsh")
    print()
    print("Phase 4 (70K-100K): Add dropout + full realism 10%→0% clean")
    print("  → Final robustness training")
    print("  → Peak dropout only introduced at 70K")
    print("=" * 80)
    print()

    # Verify total steps
    total_steps = sum(s.steps for s in DEFAULT_CURRICULUM)
    print(f"Total curriculum steps: {total_steps:,}")
    print(f"Scheduler total steps: {scheduler.total_steps:,}")
    assert total_steps == 100000, f"Expected 100K steps, got {total_steps}"
    print("✓ Curriculum test passed!")


if __name__ == '__main__':
    test_curriculum()
