"""
Progressive Length Curriculum - designed to fix length generalization failure.

Key insight: The model locks into length-specific patterns during early training.
When longer sequences are introduced with noise, the model can't adapt.

Solution:
1. Extend length range ONE length at a time
2. Keep high clean ratio (80%) when introducing each new length
3. Give adequate exposure (5K steps) per new length before moving on
4. Only introduce noise AFTER all lengths are learned clean
5. Then gradually increase noise while maintaining all lengths

This ensures the model learns clean patterns for ALL lengths before
having to deal with noise.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class CurriculumStage:
    """Single stage in curriculum."""
    name: str
    steps: int
    min_length: int
    max_length: int
    # Noise parameters
    noise_peaks: int = 0
    peak_dropout: float = 0.0
    mass_error_ppm: float = 0.0
    intensity_variation: float = 0.0
    clean_data_ratio: float = 1.0  # 1.0=all clean, 0.0=all noisy
    # Loss weights
    ce_weight: float = 1.0
    spectrum_loss_weight: float = 0.0
    precursor_loss_weight: float = 0.0


# Progressive Length Curriculum
# Phase 1: Learn all lengths with clean data (0-50K steps)
# Phase 2: Introduce noise gradually (50K-100K steps)

PROGRESSIVE_LENGTH_CURRICULUM = [
    # ===== PHASE 1: CLEAN LENGTH EXTENSION (0-50K) =====
    # Goal: Learn clean patterns for ALL lengths before introducing noise

    # Stage 1 (0-5K): Foundation - short peptides
    CurriculumStage(
        name="Foundation (7-10)",
        steps=5000,
        min_length=7,
        max_length=10,
        clean_data_ratio=1.0,  # 100% clean
        noise_peaks=0,
        peak_dropout=0.0,
        mass_error_ppm=0.0,
        intensity_variation=0.0,
        precursor_loss_weight=0.0,
    ),

    # Stage 2 (5K-10K): Extend to 12
    CurriculumStage(
        name="Extend to 12",
        steps=5000,
        min_length=7,
        max_length=12,
        clean_data_ratio=1.0,  # 100% clean
        noise_peaks=0,
        peak_dropout=0.0,
        mass_error_ppm=0.0,
        intensity_variation=0.0,
        precursor_loss_weight=0.0,
    ),

    # Stage 3 (10K-15K): Extend to 14
    CurriculumStage(
        name="Extend to 14",
        steps=5000,
        min_length=7,
        max_length=14,
        clean_data_ratio=1.0,  # 100% clean
        noise_peaks=0,
        peak_dropout=0.0,
        mass_error_ppm=0.0,
        intensity_variation=0.0,
        precursor_loss_weight=0.0,
    ),

    # Stage 4 (15K-20K): Extend to 16
    CurriculumStage(
        name="Extend to 16",
        steps=5000,
        min_length=7,
        max_length=16,
        clean_data_ratio=1.0,  # 100% clean
        noise_peaks=0,
        peak_dropout=0.0,
        mass_error_ppm=0.0,
        intensity_variation=0.0,
        precursor_loss_weight=0.0,
    ),

    # Stage 5 (20K-25K): Extend to 18
    CurriculumStage(
        name="Extend to 18",
        steps=5000,
        min_length=7,
        max_length=18,
        clean_data_ratio=1.0,  # 100% clean
        noise_peaks=0,
        peak_dropout=0.0,
        mass_error_ppm=0.0,
        intensity_variation=0.0,
        precursor_loss_weight=0.0,
    ),

    # Stage 6 (25K-30K): Extend to 20
    CurriculumStage(
        name="Extend to 20",
        steps=5000,
        min_length=7,
        max_length=20,
        clean_data_ratio=1.0,  # 100% clean
        noise_peaks=0,
        peak_dropout=0.0,
        mass_error_ppm=0.0,
        intensity_variation=0.0,
        precursor_loss_weight=0.0,
    ),

    # Stage 7 (30K-35K): Extend to 22
    CurriculumStage(
        name="Extend to 22",
        steps=5000,
        min_length=7,
        max_length=22,
        clean_data_ratio=1.0,  # 100% clean
        noise_peaks=0,
        peak_dropout=0.0,
        mass_error_ppm=0.0,
        intensity_variation=0.0,
        precursor_loss_weight=0.0,
    ),

    # Stage 8 (35K-40K): Extend to 25 (full range)
    CurriculumStage(
        name="Extend to 25",
        steps=5000,
        min_length=7,
        max_length=25,
        clean_data_ratio=1.0,  # 100% clean
        noise_peaks=0,
        peak_dropout=0.0,
        mass_error_ppm=0.0,
        intensity_variation=0.0,
        precursor_loss_weight=0.0,
    ),

    # Stage 9 (40K-50K): Consolidate full range clean
    CurriculumStage(
        name="Consolidate Clean",
        steps=10000,
        min_length=7,
        max_length=25,
        clean_data_ratio=1.0,  # Still 100% clean
        noise_peaks=0,
        peak_dropout=0.0,
        mass_error_ppm=0.0,
        intensity_variation=0.0,
        precursor_loss_weight=0.05,  # Start using precursor
    ),

    # ===== PHASE 2: NOISE INTRODUCTION (50K-100K) =====
    # Goal: Add noise gradually while maintaining all lengths

    # Stage 10 (50K-60K): Light noise
    CurriculumStage(
        name="Light Noise",
        steps=10000,
        min_length=7,
        max_length=25,
        clean_data_ratio=0.7,  # 70% clean, 30% noisy
        noise_peaks=5,
        peak_dropout=0.10,
        mass_error_ppm=5.0,
        intensity_variation=0.1,
        precursor_loss_weight=0.1,
    ),

    # Stage 11 (60K-70K): Moderate noise
    CurriculumStage(
        name="Moderate Noise",
        steps=10000,
        min_length=7,
        max_length=25,
        clean_data_ratio=0.5,  # 50% clean, 50% noisy
        noise_peaks=10,
        peak_dropout=0.20,
        mass_error_ppm=10.0,
        intensity_variation=0.2,
        precursor_loss_weight=0.15,
    ),

    # Stage 12 (70K-80K): Realistic noise
    CurriculumStage(
        name="Realistic Noise",
        steps=10000,
        min_length=7,
        max_length=25,
        clean_data_ratio=0.3,  # 30% clean, 70% noisy
        noise_peaks=15,
        peak_dropout=0.25,
        mass_error_ppm=15.0,
        intensity_variation=0.3,
        precursor_loss_weight=0.2,
    ),

    # Stage 13 (80K-90K): Heavy noise
    CurriculumStage(
        name="Heavy Noise",
        steps=10000,
        min_length=7,
        max_length=25,
        clean_data_ratio=0.1,  # 10% clean, 90% noisy
        noise_peaks=20,
        peak_dropout=0.35,
        mass_error_ppm=15.0,
        intensity_variation=0.4,
        precursor_loss_weight=0.25,
    ),

    # Stage 14 (90K-100K): Extreme noise (all noisy)
    CurriculumStage(
        name="Extreme Noise",
        steps=10000,
        min_length=7,
        max_length=25,
        clean_data_ratio=0.0,  # 0% clean, 100% noisy
        noise_peaks=25,
        peak_dropout=0.40,
        mass_error_ppm=20.0,
        intensity_variation=0.5,
        precursor_loss_weight=0.3,
    ),
]


class CurriculumScheduler:
    """Scheduler that updates dataset difficulty over training."""

    def __init__(self, stages: list[CurriculumStage], dataset):
        self.stages = stages
        self.dataset = dataset
        self.current_stage_idx = 0
        self.total_steps = sum(s.steps for s in stages)

        # Set initial difficulty
        self._update_stage(0)

    def _update_stage(self, stage_idx: int):
        """Apply a curriculum stage's settings to the dataset."""
        if stage_idx >= len(self.stages):
            return  # Stay at final stage

        stage = self.stages[stage_idx]
        self.dataset.set_difficulty(
            min_length=stage.min_length,
            max_length=stage.max_length,
            noise_peaks=stage.noise_peaks,
            peak_dropout=stage.peak_dropout,
            mass_error_ppm=stage.mass_error_ppm,
            intensity_variation=stage.intensity_variation,
            clean_data_ratio=stage.clean_data_ratio,
        )
        self.current_stage_idx = stage_idx

        print(f"\n{'='*60}")
        print(f"CURRICULUM STAGE {stage_idx + 1}/{len(self.stages)}: {stage.name}")
        print(f"{'='*60}")
        print(f"  Steps: {stage.steps:,}")
        print(f"  Sequence length: {stage.min_length}-{stage.max_length}")
        print(f"  Clean data: {stage.clean_data_ratio:.0%} | Noisy: {(1-stage.clean_data_ratio):.0%}")
        if stage.clean_data_ratio < 1.0:
            print(f"  Noise peaks: {stage.noise_peaks}")
            print(f"  Peak dropout: {stage.peak_dropout:.0%}")
            print(f"  Mass error: {stage.mass_error_ppm} ppm")
            print(f"  Intensity variation: {stage.intensity_variation:.0%}")
        print(f"  Precursor loss weight: {stage.precursor_loss_weight}")

    def step(self, global_step: int) -> bool:
        """
        Update difficulty based on global step.

        Returns:
            True if stage changed, False otherwise
        """
        cumulative_steps = 0
        for idx, stage in enumerate(self.stages):
            if global_step < cumulative_steps + stage.steps:
                # Check if we need to transition to this stage
                if idx != self.current_stage_idx:
                    self._update_stage(idx)
                    return True
                return False
            cumulative_steps += stage.steps

        # Past end of curriculum - stay at final stage
        return False

    def get_current_stage(self) -> CurriculumStage:
        """Get current curriculum stage."""
        return self.stages[self.current_stage_idx]

    @property
    def current_stage(self) -> CurriculumStage:
        """Property for compatibility with trainer."""
        return self.stages[self.current_stage_idx]

    def get_spectrum_loss_weight(self) -> float:
        """Get current spectrum loss weight."""
        return self.stages[self.current_stage_idx].spectrum_loss_weight

    def get_precursor_loss_weight(self) -> float:
        """Get current precursor mass loss weight."""
        return self.stages[self.current_stage_idx].precursor_loss_weight


def print_curriculum_summary():
    """Print a summary of the progressive length curriculum."""
    print("\n" + "=" * 70)
    print("PROGRESSIVE LENGTH CURRICULUM SUMMARY")
    print("=" * 70)

    total_steps = sum(s.steps for s in PROGRESSIVE_LENGTH_CURRICULUM)

    print(f"\nTotal training steps: {total_steps:,}")
    print(f"Total stages: {len(PROGRESSIVE_LENGTH_CURRICULUM)}")

    print("\n--- PHASE 1: CLEAN LENGTH EXTENSION (0-50K) ---")
    print(f"{'Stage':<25} {'Steps':<10} {'Lengths':<12} {'Clean':<10}")
    print("-" * 60)

    cumulative = 0
    for i, stage in enumerate(PROGRESSIVE_LENGTH_CURRICULUM):
        if cumulative >= 50000:
            if cumulative == 50000:
                print("\n--- PHASE 2: NOISE INTRODUCTION (50K-100K) ---")
                print(f"{'Stage':<25} {'Steps':<10} {'Lengths':<12} {'Clean':<10} {'Noise':<10}")
                print("-" * 70)
            noise_info = f"{stage.noise_peaks}pk/{stage.peak_dropout:.0%}do"
            print(f"{i+1}. {stage.name:<21} {stage.steps:<10,} {stage.min_length}-{stage.max_length:<9} {stage.clean_data_ratio:.0%}       {noise_info}")
        else:
            print(f"{i+1}. {stage.name:<21} {stage.steps:<10,} {stage.min_length}-{stage.max_length:<9} {stage.clean_data_ratio:.0%}")
        cumulative += stage.steps

    # Length exposure analysis
    print("\n--- LENGTH EXPOSURE ANALYSIS ---")

    length_exposure = {}
    length_clean_exposure = {}

    for stage in PROGRESSIVE_LENGTH_CURRICULUM:
        lengths_in_stage = range(stage.min_length, stage.max_length + 1)
        num_lengths = len(lengths_in_stage)

        for length in lengths_in_stage:
            if length not in length_exposure:
                length_exposure[length] = 0
                length_clean_exposure[length] = 0
            length_exposure[length] += stage.steps / num_lengths
            length_clean_exposure[length] += stage.steps * stage.clean_data_ratio / num_lengths

    print(f"\n{'Length':<10} {'Total Exposure':<18} {'Clean Exposure':<18} {'% Clean':<10}")
    print("-" * 60)

    for length in sorted(length_exposure.keys()):
        total = length_exposure[length]
        clean = length_clean_exposure[length]
        pct = clean / total * 100 if total > 0 else 0
        print(f"{length:<10} {total:>12,.0f}       {clean:>12,.0f}       {pct:>5.1f}%")

    # Compare to old curriculum
    print("\n--- COMPARISON TO AGGRESSIVE NOISE CURRICULUM ---")
    print("Old curriculum clean exposure for length 14: ~1,905 steps (21% of total)")
    print(f"New curriculum clean exposure for length 14: {length_clean_exposure.get(14, 0):,.0f} steps ({length_clean_exposure.get(14, 0)/length_exposure.get(14, 1)*100:.0f}% of total)")


if __name__ == '__main__':
    print_curriculum_summary()
