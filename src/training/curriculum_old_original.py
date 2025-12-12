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
    precursor_loss_weight: float = 0.0


# Enhanced 7-stage curriculum with stronger mass constraint enforcement
# Extended to support peptides up to length 30
# Key changes from original:
# 1. Increased spectrum_loss_weight from max 0.15 → 0.6
# 2. Added precursor_loss_weight ramping from 0 → 1.0
# 3. Extended max_length from 18 → 30 across stages 5-7
DEFAULT_CURRICULUM = [
    # Stage 1 (0-8K): Pure CE on clean data - learn basic sequence patterns
    CurriculumStage(
        name="stage_1_warmup",
        steps=8000,
        min_length=7,
        max_length=10,
        noise_peaks=0,
        peak_dropout=0.0,
        mass_error_ppm=0.0,
        spectrum_loss_weight=0.0,  # Pure CE first
        precursor_loss_weight=0.0,  # No mass constraint yet
    ),

    # Stage 2 (8K-16K): Add spectrum loss + precursor mass constraint
    # Start enforcing that sum(aa_masses) = precursor_mass
    CurriculumStage(
        name="stage_2_mass_constraints",
        steps=8000,  # 8K-16K
        min_length=7,
        max_length=10,  # Same as Stage 1
        noise_peaks=0,   # Keep clean!
        peak_dropout=0.0,
        mass_error_ppm=0.0,
        spectrum_loss_weight=0.2,  # Increased from 0.1
        precursor_loss_weight=0.2,  # NEW: Start learning precursor mass
    ),

    # Stage 2.5 (16K-22K): Increase mass constraints before adding noise
    # Critical: Make sure mass constraint is learned before corrupting data
    CurriculumStage(
        name="stage_2.5_strong_mass",
        steps=6000,  # 16K-22K
        min_length=7,
        max_length=12,  # Slightly longer peptides
        noise_peaks=1,   # Start with 1 noise peak only
        peak_dropout=0.02,  # Minimal dropout
        mass_error_ppm=2.0,  # Small mass error
        spectrum_loss_weight=0.3,  # Increased from 0.1
        precursor_loss_weight=0.5,  # Strong precursor constraint
    ),

    # Stage 2.75 (22K-28K): Maintain strong mass constraints with gradual noise
    CurriculumStage(
        name="stage_2.75_gradual_noise",
        steps=6000,  # 22K-28K
        min_length=7,
        max_length=15,  # INCREASED from 12 to 15
        noise_peaks=2,   # Increase to 2 noise peaks
        peak_dropout=0.05,
        mass_error_ppm=5.0,
        spectrum_loss_weight=0.4,  # Increased from 0.12
        precursor_loss_weight=0.7,  # Strong precursor constraint
    ),

    # Stage 3 (28K-38K): Moderate difficulty with near-full mass constraint
    CurriculumStage(
        name="stage_3_moderate",
        steps=10000,  # 28K-38K
        min_length=7,
        max_length=18,  # INCREASED from 15 to 18
        noise_peaks=5,   # Moderate noise
        peak_dropout=0.10,
        mass_error_ppm=10.0,
        spectrum_loss_weight=0.5,  # Increased from 0.15
        precursor_loss_weight=0.9,  # Nearly full precursor constraint
    ),

    # Stage 4 (38K-50K): Longer peptides with realistic conditions
    CurriculumStage(
        name="stage_4_longer",
        steps=12000,  # 38K-50K
        min_length=7,
        max_length=22,  # INCREASED from 18 to 22
        noise_peaks=8,   # More realistic noise level
        peak_dropout=0.15,
        mass_error_ppm=15.0,
        intensity_variation=0.2,
        spectrum_loss_weight=0.6,  # Increased from 0.15
        precursor_loss_weight=1.0,  # FULL precursor constraint
    ),

    # Stage 5 (50K-62K): Extended length peptides
    # NEW: Bridge to very long peptides
    CurriculumStage(
        name="stage_5_extended",
        steps=12000,  # 50K-62K
        min_length=10,  # Increase min_length to focus on longer seqs
        max_length=26,
        noise_peaks=8,
        peak_dropout=0.15,
        mass_error_ppm=15.0,
        intensity_variation=0.2,
        spectrum_loss_weight=0.6,
        precursor_loss_weight=1.0,
    ),

    # Stage 6 (62K-76K): Very long peptides
    # NEW: Approach the model's max_seq_len limit
    CurriculumStage(
        name="stage_6_very_long",
        steps=14000,  # 62K-76K
        min_length=12,
        max_length=30,  # Target max_seq_len
        noise_peaks=8,
        peak_dropout=0.15,
        mass_error_ppm=15.0,
        intensity_variation=0.2,
        spectrum_loss_weight=0.6,
        precursor_loss_weight=1.0,
    ),

    # Stage 7 (76K-100K): Final polish on full distribution
    # NEW: Train on full length range with realistic noise
    CurriculumStage(
        name="stage_7_final",
        steps=24000,  # 76K-100K
        min_length=7,   # Full range
        max_length=30,  # Full range
        noise_peaks=8,
        peak_dropout=0.15,
        mass_error_ppm=15.0,
        intensity_variation=0.2,
        spectrum_loss_weight=0.6,
        precursor_loss_weight=1.0,
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
                f"  - Spectrum loss weight: {stage.spectrum_loss_weight}\n"
                f"  - Precursor loss weight: {stage.precursor_loss_weight}"
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
