"""
Extended, smoother curriculum learning scheduler.

Improvements over default curriculum:
1. Longer stages (100K steps total vs 50K)
2. Smoother transitions (more intermediate stages)
3. Gradual noise introduction (0→1→2→3→5→8 peaks)
4. Better sim-to-real transfer
"""

from dataclasses import dataclass
from typing import Optional
import logging

from ..data.dataset import SyntheticPeptideDataset

log = logging.getLogger(__name__)


@dataclass
class CurriculumStage:
    """Configuration for a curriculum stage."""
    name: str
    steps: int

    # Peptide parameters
    min_length: int = 7
    max_length: int = 20

    # Noise parameters
    noise_peaks: int = 0
    peak_dropout: float = 0.0
    mass_error_ppm: float = 0.0
    intensity_variation: float = 0.0

    # Loss weights
    spectrum_loss_weight: float = 0.0


# Extended 10-stage curriculum for smoother sim-to-real transfer
EXTENDED_CURRICULUM = [
    # Stage 1 (0-10K): Foundation - Pure CE on clean data
    CurriculumStage(
        name="stage_1_foundation",
        steps=10000,
        min_length=7,
        max_length=10,
        noise_peaks=0,
        peak_dropout=0.0,
        mass_error_ppm=0.0,
        spectrum_loss_weight=0.0,  # Pure CE first
    ),

    # Stage 2 (10K-20K): Physics constraints - Add spectrum loss, keep data clean
    CurriculumStage(
        name="stage_2_physics",
        steps=10000,  # 10K-20K
        min_length=7,
        max_length=10,
        noise_peaks=0,
        peak_dropout=0.0,
        mass_error_ppm=0.0,
        spectrum_loss_weight=0.1,  # Learn mass constraints
    ),

    # Stage 3 (20K-30K): Minimal noise - Very gentle introduction
    CurriculumStage(
        name="stage_3_minimal_noise",
        steps=10000,  # 20K-30K
        min_length=7,
        max_length=11,
        noise_peaks=1,  # Just 1 noise peak
        peak_dropout=0.02,
        mass_error_ppm=2.0,
        spectrum_loss_weight=0.1,
    ),

    # Stage 4 (30K-40K): Light noise - Gradual increase
    CurriculumStage(
        name="stage_4_light_noise",
        steps=10000,  # 30K-40K
        min_length=7,
        max_length=12,
        noise_peaks=2,  # Increase to 2
        peak_dropout=0.05,
        mass_error_ppm=5.0,
        spectrum_loss_weight=0.12,
    ),

    # Stage 5 (40K-50K): Mild noise - Continue gradual ramp
    CurriculumStage(
        name="stage_5_mild_noise",
        steps=10000,  # 40K-50K
        min_length=7,
        max_length=13,
        noise_peaks=3,  # NEW intermediate stage
        peak_dropout=0.08,
        mass_error_ppm=7.5,
        spectrum_loss_weight=0.13,
    ),

    # Stage 6 (50K-60K): Moderate noise - Bridge to realistic
    CurriculumStage(
        name="stage_6_moderate_noise",
        steps=10000,  # 50K-60K
        min_length=7,
        max_length=14,
        noise_peaks=5,  # Was causing issues before
        peak_dropout=0.10,
        mass_error_ppm=10.0,
        spectrum_loss_weight=0.14,
    ),

    # Stage 7 (60K-70K): Moderate-high noise
    CurriculumStage(
        name="stage_7_moderate_high",
        steps=10000,  # 60K-70K
        min_length=7,
        max_length=15,
        noise_peaks=6,  # NEW intermediate stage
        peak_dropout=0.12,
        mass_error_ppm=12.0,
        spectrum_loss_weight=0.14,
    ),

    # Stage 8 (70K-80K): High noise
    CurriculumStage(
        name="stage_8_high_noise",
        steps=10000,  # 70K-80K
        min_length=7,
        max_length=16,
        noise_peaks=7,  # NEW intermediate stage
        peak_dropout=0.13,
        mass_error_ppm=13.0,
        intensity_variation=0.1,
        spectrum_loss_weight=0.15,
    ),

    # Stage 9 (80K-90K): Near-realistic
    CurriculumStage(
        name="stage_9_near_realistic",
        steps=10000,  # 80K-90K
        min_length=7,
        max_length=17,
        noise_peaks=8,
        peak_dropout=0.14,
        mass_error_ppm=14.0,
        intensity_variation=0.15,
        spectrum_loss_weight=0.15,
    ),

    # Stage 10 (90K-100K): Fully realistic
    CurriculumStage(
        name="stage_10_realistic",
        steps=10000,  # 90K-100K
        min_length=7,
        max_length=18,
        noise_peaks=10,  # Even more realistic
        peak_dropout=0.15,
        mass_error_ppm=15.0,
        intensity_variation=0.2,
        spectrum_loss_weight=0.15,
    ),
]


class CurriculumScheduler:
    """
    Manages curriculum progression during training.

    Automatically adjusts dataset difficulty and loss weights
    based on training progress.
    """

    def __init__(
        self,
        stages: list[CurriculumStage] = None,
        dataset: SyntheticPeptideDataset = None,
    ):
        self.stages = stages or EXTENDED_CURRICULUM
        self.dataset = dataset
        self.current_stage_idx = -1
        self._cumulative_steps = self._compute_cumulative_steps()

    def _compute_cumulative_steps(self) -> list[int]:
        """Compute cumulative step counts for stage boundaries."""
        cumulative = []
        total = 0
        for stage in self.stages:
            total += stage.steps
            cumulative.append(total)
        return cumulative

    @property
    def current_stage(self) -> Optional[CurriculumStage]:
        """Get current curriculum stage."""
        if 0 <= self.current_stage_idx < len(self.stages):
            return self.stages[self.current_stage_idx]
        return None

    @property
    def total_steps(self) -> int:
        """Total steps across all stages."""
        return self._cumulative_steps[-1] if self._cumulative_steps else 0

    def step(self, global_step: int) -> bool:
        """
        Update curriculum based on current step.

        Returns:
            True if stage changed, False otherwise
        """
        # Find which stage we should be in
        new_stage_idx = 0
        for i, boundary in enumerate(self._cumulative_steps):
            if global_step < boundary:
                new_stage_idx = i
                break
        else:
            new_stage_idx = len(self.stages) - 1

        # Check if stage changed
        if new_stage_idx != self.current_stage_idx:
            self.current_stage_idx = new_stage_idx
            stage = self.stages[new_stage_idx]

            # Update dataset if provided
            if self.dataset is not None:
                self.dataset.set_difficulty(
                    min_length=stage.min_length,
                    max_length=stage.max_length,
                    noise_peaks=stage.noise_peaks,
                    peak_dropout=stage.peak_dropout,
                    mass_error_ppm=stage.mass_error_ppm,
                    intensity_variation=stage.intensity_variation,
                )

            log.info(
                f"\n{'='*60}\n"
                f"Curriculum: Advanced to '{stage.name}' at step {global_step}\n"
                f"{'='*60}\n"
                f"  Peptide length: {stage.min_length}-{stage.max_length}\n"
                f"  Noise peaks: {stage.noise_peaks}\n"
                f"  Peak dropout: {stage.peak_dropout:.1%}\n"
                f"  Mass error: {stage.mass_error_ppm} ppm\n"
                f"  Intensity variation: {stage.intensity_variation:.1%}\n"
                f"  Spectrum loss weight: {stage.spectrum_loss_weight}\n"
                f"{'='*60}"
            )

            return True

        return False

    def get_spectrum_loss_weight(self) -> float:
        """Get current spectrum loss weight."""
        if self.current_stage is not None:
            return self.current_stage.spectrum_loss_weight
        return 0.0
