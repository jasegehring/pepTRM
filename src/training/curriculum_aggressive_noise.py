"""
Aggressive noise curriculum designed to force multi-step refinement.

Philosophy:
- Start with HARD data (not easy) to force model to learn refinement
- Recursion should be necessary, not optional
- Noise profile matches real-world MS/MS complexity
"""
from dataclasses import dataclass
from typing import Optional
from src.training.curriculum import CurriculumStage


@dataclass
class CurriculumStage:
    """Single stage in curriculum."""
    name: str
    steps: int
    min_length: int
    max_length: int
    # Noise parameters (realistic MS/MS)
    noise_peaks: int = 0
    peak_dropout: float = 0.0
    mass_error_ppm: float = 0.0
    intensity_variation: float = 0.0
    clean_data_ratio: float = 0.0  # 1.0=all clean, 0.0=all noisy
    # Loss weights
    ce_weight: float = 1.0
    spectrum_loss_weight: float = 0.0
    precursor_loss_weight: float = 0.0


# Aggressive curriculum: Force recursion from the start
# Strategy: Start with mostly clean data, gradually increase noise proportion
AGGRESSIVE_NOISE_CURRICULUM = [
    # Stage 1 (0-10K): Learn on clean data, gentle noise introduction
    CurriculumStage(
        name="Warmup",
        steps=10000,
        min_length=7,
        max_length=12,
        clean_data_ratio=0.8,    # 80% clean, 20% noisy
        noise_peaks=5,
        peak_dropout=0.15,
        mass_error_ppm=5.0,
        intensity_variation=0.2,
        precursor_loss_weight=0.05,
    ),

    # Stage 2 (10K-20K): Increase noise proportion
    CurriculumStage(
        name="Longer Sequences",
        steps=10000,
        min_length=10,
        max_length=15,
        clean_data_ratio=0.6,    # 60% clean, 40% noisy
        noise_peaks=10,
        peak_dropout=0.20,
        mass_error_ppm=8.0,
        intensity_variation=0.3,
        precursor_loss_weight=0.1,
    ),

    # Stage 3 (20K-30K): Majority noisy now
    CurriculumStage(
        name="Realistic Noise",
        steps=10000,
        min_length=12,
        max_length=18,
        clean_data_ratio=0.4,    # 40% clean, 60% noisy
        noise_peaks=15,
        peak_dropout=0.30,       # Real-world levels!
        mass_error_ppm=10.0,
        intensity_variation=0.4,
        precursor_loss_weight=0.15,
    ),

    # Stage 4 (30K-45K): Mostly noisy data
    CurriculumStage(
        name="Challenging Data",
        steps=15000,
        min_length=12,
        max_length=20,
        clean_data_ratio=0.2,    # 20% clean, 80% noisy
        noise_peaks=20,
        peak_dropout=0.35,
        mass_error_ppm=12.0,
        intensity_variation=0.5,
        precursor_loss_weight=0.2,
    ),

    # Stage 5 (45K-60K): Pure noisy - low-quality spectra
    CurriculumStage(
        name="Low Quality Spectra",
        steps=15000,
        min_length=12,
        max_length=22,
        clean_data_ratio=0.0,    # 100% noisy - no more crutches!
        noise_peaks=25,
        peak_dropout=0.40,
        mass_error_ppm=15.0,
        intensity_variation=0.6,
        precursor_loss_weight=0.25,
    ),

    # Stage 6 (60K-80K): Extreme difficulty
    CurriculumStage(
        name="Extreme Difficulty",
        steps=20000,
        min_length=12,
        max_length=25,
        clean_data_ratio=0.0,    # 100% noisy
        noise_peaks=30,
        peak_dropout=0.45,
        mass_error_ppm=20.0,
        intensity_variation=0.7,
        precursor_loss_weight=0.3,
    ),

    # Stage 7 (80K-100K): Maintain extreme difficulty
    CurriculumStage(
        name="Final Challenge",
        steps=20000,
        min_length=12,
        max_length=25,
        clean_data_ratio=0.0,    # 100% noisy
        noise_peaks=30,
        peak_dropout=0.45,
        mass_error_ppm=20.0,
        intensity_variation=0.7,
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

        print(f"\n📚 Curriculum Stage {stage_idx + 1}/{len(self.stages)}: {stage.name}")
        print(f"  Steps: {stage.steps:,}")
        print(f"  Sequence length: {stage.min_length}-{stage.max_length}")
        print(f"  Clean data: {stage.clean_data_ratio:.0%} | Noisy: {(1-stage.clean_data_ratio):.0%}")
        print(f"  Noise peaks: {stage.noise_peaks}")
        print(f"  Peak dropout: {stage.peak_dropout:.1%}")
        print(f"  Mass error: {stage.mass_error_ppm} ppm")
        print(f"  Intensity variation: {stage.intensity_variation:.1%}")
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
