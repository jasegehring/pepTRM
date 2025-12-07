"""
Curriculum learning scheduler for progressive difficulty increase.
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


# Simplified 3-stage curriculum for MVP
DEFAULT_CURRICULUM = [
    # Stage 1: Easy - short peptides, clean spectra
    CurriculumStage(
        name="clean",
        steps=10000,
        min_length=7,
        max_length=10,
        noise_peaks=0,
        peak_dropout=0.0,
        mass_error_ppm=0.0,
        spectrum_loss_weight=0.0,  # Pure CE first
    ),

    # Stage 2: Moderate difficulty
    CurriculumStage(
        name="moderate",
        steps=15000,
        min_length=7,
        max_length=15,
        noise_peaks=5,
        peak_dropout=0.1,
        mass_error_ppm=0.0,
        spectrum_loss_weight=0.1,
    ),

    # Stage 3: Realistic synthetic
    CurriculumStage(
        name="realistic",
        steps=25000,
        min_length=7,
        max_length=20,
        noise_peaks=10,
        peak_dropout=0.2,
        mass_error_ppm=20.0,
        intensity_variation=0.2,
        spectrum_loss_weight=0.1,
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
        self.stages = stages or DEFAULT_CURRICULUM
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
                f"Curriculum: Advanced to stage '{stage.name}' at step {global_step}\n"
                f"  - Peptide length: {stage.min_length}-{stage.max_length}\n"
                f"  - Peak dropout: {stage.peak_dropout:.1%}\n"
                f"  - Noise peaks: {stage.noise_peaks}\n"
                f"  - Mass error: {stage.mass_error_ppm} ppm\n"
                f"  - Spectrum loss weight: {stage.spectrum_loss_weight}"
            )

            return True

        return False

    def get_spectrum_loss_weight(self) -> float:
        """Get current spectrum loss weight."""
        if self.current_stage is not None:
            return self.current_stage.spectrum_loss_weight
        return 0.0
