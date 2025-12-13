"""
MODIFIED CURRICULUM - Delayed Spectrum Loss Introduction

Changes from original curriculum:
1. Extended pure CE training from 15K → 25K steps (give model time to reach 65% accuracy)
2. Delayed spectrum loss introduction from step 15K → step 25K
3. Gentler progression with more time at each stage

Rationale:
- Model at 48% token accuracy at step 15K (below expected 55%)
- Spectrum loss needs ~60-65% accuracy for meaningful gradient signal (>10% coverage)
- Better to build strong CE foundation before adding spectrum supervision
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

    # Noise parameters (for noisy examples only when clean_data_ratio < 1.0)
    noise_peaks: int = 0
    peak_dropout: float = 0.0
    mass_error_ppm: float = 0.0
    intensity_variation: float = 0.0

    # Data mixing - probability of clean vs noisy data
    clean_data_ratio: float = 1.0  # 1.0 = 100% clean, 0.0 = 100% noisy

    # Loss weights
    spectrum_loss_weight: float = 0.0
    precursor_loss_weight: float = 0.0


# Modified 10-stage curriculum: 100K steps total
DELAYED_SPECTRUM_CURRICULUM = [
    # ========================================================================
    # PHASE 1: EXTENDED FOUNDATION (0-25K) - Pure CE, Build to 65% Accuracy
    # ========================================================================

    # Stage 1 (0-7.5K): Initial foundation
    CurriculumStage(
        name="stage_1_foundation",
        steps=7500,
        min_length=7,
        max_length=10,
        noise_peaks=0,
        peak_dropout=0.0,
        mass_error_ppm=0.0,
        clean_data_ratio=1.0,
        spectrum_loss_weight=0.0,  # Pure CE
        precursor_loss_weight=0.0,
    ),

    # Stage 2 (7.5K-15K): Continued foundation
    CurriculumStage(
        name="stage_2_foundation",
        steps=7500,
        min_length=7,
        max_length=10,
        noise_peaks=0,
        peak_dropout=0.0,
        mass_error_ppm=0.0,
        clean_data_ratio=1.0,
        spectrum_loss_weight=0.0,  # Still pure CE
        precursor_loss_weight=0.0,
    ),

    # Stage 3 (15K-22.5K): Extended stabilization - CHANGED FROM ORIGINAL
    # ORIGINAL: Introduced spectrum loss here (0.08)
    # MODIFIED: Continue pure CE to reach 65% accuracy
    CurriculumStage(
        name="stage_3_extended_stabilization",
        steps=7500,
        min_length=7,
        max_length=11,  # Slightly longer peptides
        noise_peaks=0,
        peak_dropout=0.0,
        mass_error_ppm=0.0,
        clean_data_ratio=1.0,
        spectrum_loss_weight=0.0,  # STILL PURE CE - Let model mature
        precursor_loss_weight=0.0,
    ),

    # Stage 4 (22.5K-30K): Pre-spectrum ramp - NEW STAGE
    # Give model time to reach 65%+ accuracy before spectrum introduction
    CurriculumStage(
        name="stage_4_pre_spectrum_ramp",
        steps=7500,
        min_length=7,
        max_length=12,
        noise_peaks=0,
        peak_dropout=0.0,
        mass_error_ppm=0.0,
        clean_data_ratio=1.0,
        spectrum_loss_weight=0.0,  # LAST STAGE OF PURE CE
        precursor_loss_weight=0.0,
    ),

    # ========================================================================
    # PHASE 2: FRAGMENT PHYSICS (30K-45K) - Introduce Spectrum Loss
    # ========================================================================

    # Stage 5 (30K-37.5K): Spectrum introduction - DELAYED FROM 15K
    # Now model should be at ~65-70% accuracy with 10-15% spectrum coverage
    CurriculumStage(
        name="stage_5_spectrum_intro",
        steps=7500,
        min_length=7,
        max_length=12,
        noise_peaks=0,
        peak_dropout=0.0,
        mass_error_ppm=0.0,
        clean_data_ratio=1.0,
        spectrum_loss_weight=0.06,  # GENTLER START (was 0.08)
        precursor_loss_weight=0.0,
    ),

    # Stage 6 (37.5K-45K): Spectrum ramp-up
    CurriculumStage(
        name="stage_6_spectrum_ramp",
        steps=7500,
        min_length=7,
        max_length=13,
        noise_peaks=0,
        peak_dropout=0.0,
        mass_error_ppm=0.0,
        clean_data_ratio=1.0,
        spectrum_loss_weight=0.10,
        precursor_loss_weight=0.0,
    ),

    # ========================================================================
    # PHASE 3: MASS CONSTRAINT + GRADUAL MIXING (45K-75K)
    # ========================================================================

    # Stage 7 (45K-55K): Precursor intro + Light mixing
    CurriculumStage(
        name="stage_7_precursor_light_mix",
        steps=10000,
        min_length=7,
        max_length=14,
        noise_peaks=1,
        peak_dropout=0.0,
        mass_error_ppm=1.0,
        clean_data_ratio=0.8,
        spectrum_loss_weight=0.15,
        precursor_loss_weight=0.01,
    ),

    # Stage 8 (55K-65K): Moderate mixing
    CurriculumStage(
        name="stage_8_moderate_mix",
        steps=10000,
        min_length=7,
        max_length=15,
        noise_peaks=3,
        peak_dropout=0.0,
        mass_error_ppm=5.0,
        clean_data_ratio=0.5,
        spectrum_loss_weight=0.18,
        precursor_loss_weight=0.03,
    ),

    # Stage 9 (65K-75K): Heavy mixing
    CurriculumStage(
        name="stage_9_heavy_mix",
        steps=10000,
        min_length=7,
        max_length=16,
        noise_peaks=5,
        peak_dropout=0.0,
        mass_error_ppm=8.0,
        clean_data_ratio=0.3,
        spectrum_loss_weight=0.20,
        precursor_loss_weight=0.05,
    ),

    # ========================================================================
    # PHASE 4: FULL REALISM (75K-100K)
    # ========================================================================

    # Stage 10 (75K-87.5K): Near-realistic + dropout
    CurriculumStage(
        name="stage_10_near_realistic",
        steps=12500,
        min_length=7,
        max_length=18,
        noise_peaks=7,
        peak_dropout=0.08,
        mass_error_ppm=12.0,
        intensity_variation=0.1,
        clean_data_ratio=0.1,
        spectrum_loss_weight=0.22,
        precursor_loss_weight=0.06,
    ),

    # Stage 11 (87.5K-100K): Fully realistic
    CurriculumStage(
        name="stage_11_realistic",
        steps=12500,
        min_length=7,
        max_length=20,
        noise_peaks=10,
        peak_dropout=0.10,
        mass_error_ppm=15.0,
        intensity_variation=0.2,
        clean_data_ratio=0.0,
        spectrum_loss_weight=0.25,
        precursor_loss_weight=0.08,
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
        self.stages = stages or DELAYED_SPECTRUM_CURRICULUM
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
                    clean_data_ratio=stage.clean_data_ratio,
                )

            log.info(
                f"\n{'='*60}\n"
                f"Curriculum: Advanced to '{stage.name}' at step {global_step}\n"
                f"{'='*60}\n"
                f"  Peptide length: {stage.min_length}-{stage.max_length}\n"
                f"  Clean data ratio: {stage.clean_data_ratio:.0%} clean, {(1-stage.clean_data_ratio):.0%} noisy\n"
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
