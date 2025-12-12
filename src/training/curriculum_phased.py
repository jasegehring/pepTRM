"""
Phased curriculum with delayed auxiliary loss introduction.

Based on gradient information analysis, this curriculum introduces:
- Spectrum loss @ step 15k (55% accuracy, gradient strength 0.25)
- Precursor loss @ step 30k (70% accuracy, gradient strength 0.59)

This provides clearer, more actionable gradients compared to early introduction.
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
    precursor_loss_weight: float = 0.0


# Phased 10-stage curriculum with delayed auxiliary loss introduction
PHASED_CURRICULUM = [
    # ========================================================================
    # PHASE 1: FOUNDATION (0-15k) - Pure CE, Build Stable Representations
    # ========================================================================

    # Stage 1 (0-7.5k): Foundation - Learn basic token distributions
    CurriculumStage(
        name="stage_1_foundation",
        steps=7500,
        min_length=7,
        max_length=10,
        noise_peaks=0,
        peak_dropout=0.0,
        mass_error_ppm=0.0,
        spectrum_loss_weight=0.0,  # Pure CE
        precursor_loss_weight=0.0,
    ),

    # Stage 2 (7.5k-15k): Stabilization - Let model reach ~55% accuracy
    CurriculumStage(
        name="stage_2_stabilization",
        steps=7500,  # 7.5k-15k
        min_length=7,
        max_length=10,
        noise_peaks=0,
        peak_dropout=0.0,
        mass_error_ppm=0.0,
        spectrum_loss_weight=0.0,  # Still pure CE
        precursor_loss_weight=0.0,
    ),

    # ========================================================================
    # PHASE 2: FRAGMENT PHYSICS (15k-30k) - Introduce Spectrum Loss
    # ========================================================================

    # Stage 3 (15k-22.5k): Spectrum introduction - Gentle start
    CurriculumStage(
        name="stage_3_spectrum_intro",
        steps=7500,  # 15k-22.5k
        min_length=7,
        max_length=11,
        noise_peaks=1,  # Minimal noise
        peak_dropout=0.02,
        mass_error_ppm=2.0,
        spectrum_loss_weight=0.08,  # START: Gradient strength ~0.25 at 55% acc
        precursor_loss_weight=0.0,  # Not yet
    ),

    # Stage 4 (22.5k-30k): Spectrum ramp-up
    CurriculumStage(
        name="stage_4_spectrum_ramp",
        steps=7500,  # 22.5k-30k
        min_length=7,
        max_length=12,
        noise_peaks=2,
        peak_dropout=0.05,
        mass_error_ppm=5.0,
        spectrum_loss_weight=0.12,  # Increase: Model at ~65% acc
        precursor_loss_weight=0.0,
    ),

    # ========================================================================
    # PHASE 3: TOTAL MASS CONSTRAINT (30k-45k) - Introduce Precursor Loss
    # ========================================================================

    # Stage 5 (30k-37.5k): Precursor introduction - Model at ~70% accuracy
    CurriculumStage(
        name="stage_5_precursor_intro",
        steps=7500,  # 30k-37.5k
        min_length=7,
        max_length=13,
        noise_peaks=3,
        peak_dropout=0.08,
        mass_error_ppm=7.5,
        spectrum_loss_weight=0.15,
        precursor_loss_weight=0.01,  # START: Gradient strength ~0.59 at 70% acc
    ),

    # Stage 6 (37.5k-45k): Precursor ramp-up
    CurriculumStage(
        name="stage_6_precursor_ramp",
        steps=7500,  # 37.5k-45k
        min_length=7,
        max_length=14,
        noise_peaks=5,
        peak_dropout=0.10,
        mass_error_ppm=10.0,
        spectrum_loss_weight=0.18,
        precursor_loss_weight=0.02,  # Model at ~75% acc
    ),

    # ========================================================================
    # PHASE 4: FULL MULTI-TASK LEARNING (45k-100k) - All Losses Active
    # ========================================================================

    # Stage 7 (45k-55k): Moderate-high noise
    CurriculumStage(
        name="stage_7_moderate_high",
        steps=10000,  # 45k-55k
        min_length=7,
        max_length=15,
        noise_peaks=6,
        peak_dropout=0.12,
        mass_error_ppm=12.0,
        spectrum_loss_weight=0.20,
        precursor_loss_weight=0.03,
    ),

    # Stage 8 (55k-67.5k): High noise
    CurriculumStage(
        name="stage_8_high_noise",
        steps=12500,  # 55k-67.5k
        min_length=7,
        max_length=16,
        noise_peaks=7,
        peak_dropout=0.13,
        mass_error_ppm=13.0,
        intensity_variation=0.1,
        spectrum_loss_weight=0.22,
        precursor_loss_weight=0.05,
    ),

    # Stage 9 (67.5k-82.5k): Near-realistic
    CurriculumStage(
        name="stage_9_near_realistic",
        steps=15000,  # 67.5k-82.5k
        min_length=7,
        max_length=17,
        noise_peaks=8,
        peak_dropout=0.14,
        mass_error_ppm=14.0,
        intensity_variation=0.15,
        spectrum_loss_weight=0.24,
        precursor_loss_weight=0.06,
    ),

    # Stage 10 (82.5k-100k): Fully realistic
    CurriculumStage(
        name="stage_10_realistic",
        steps=17500,  # 82.5k-100k
        min_length=7,
        max_length=18,
        noise_peaks=10,
        peak_dropout=0.15,
        mass_error_ppm=15.0,
        intensity_variation=0.2,
        spectrum_loss_weight=0.25,  # Final weight
        precursor_loss_weight=0.08,  # Final weight
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
        self.stages = stages or PHASED_CURRICULUM
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
                f"  Precursor loss weight: {stage.precursor_loss_weight}\n"
                f"{'='*60}"
            )

            return True

        return False

    def get_spectrum_loss_weight(self) -> float:
        """Get current spectrum loss weight."""
        if self.current_stage is not None:
            return self.current_stage.spectrum_loss_weight
        return 0.0

    def get_precursor_loss_weight(self) -> float:
        """Get current precursor mass loss weight."""
        if self.current_stage is not None:
            return self.current_stage.precursor_loss_weight
        return 0.0
