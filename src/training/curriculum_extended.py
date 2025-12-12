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

    # Noise parameters (for noisy examples only)
    noise_peaks: int = 0
    peak_dropout: float = 0.0
    mass_error_ppm: float = 0.0
    intensity_variation: float = 0.0

    # Data mixing - probability of clean vs noisy data
    clean_data_ratio: float = 1.0  # 1.0 = 100% clean, 0.0 = 100% noisy

    # Loss weights
    spectrum_loss_weight: float = 0.0
    precursor_loss_weight: float = 0.0


# IMPROVED 10-stage curriculum with smooth transitions and clean/noisy mixing
# Key improvements:
# 1. Introduce precursor loss early on CLEAN data (stage 2)
# 2. Keep data 100% clean through stage 4 (40K steps)
# 3. Mix clean/noisy data gradually (stages 5-9)
# 4. Delay peak_dropout until stage 8 (it's very harsh)
# 5. Smooth noise ramp: mass_error 1→2→5→8→12→15 ppm
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
        clean_data_ratio=1.0,  # 100% clean
        spectrum_loss_weight=0.0,  # Pure CE first
        precursor_loss_weight=0.0,  # No mass constraint yet
    ),

    # Stage 2 (10K-20K): Add spectrum + precursor on CLEAN data
    CurriculumStage(
        name="stage_2_clean_constraints",
        steps=10000,
        min_length=7,
        max_length=10,
        noise_peaks=0,
        peak_dropout=0.0,
        mass_error_ppm=0.0,
        clean_data_ratio=1.0,  # 100% clean - CRITICAL: Learn mass constraints first!
        spectrum_loss_weight=0.05,   # Start learning fragment masses
        precursor_loss_weight=0.01,  # EARLY introduction on clean data
    ),

    # Stage 3 (20K-30K): Stabilize losses on clean data
    CurriculumStage(
        name="stage_3_clean_stabilization",
        steps=10000,
        min_length=7,
        max_length=12,
        noise_peaks=0,
        peak_dropout=0.0,
        mass_error_ppm=0.0,
        clean_data_ratio=1.0,  # 100% clean - let losses stabilize
        spectrum_loss_weight=0.08,
        precursor_loss_weight=0.02,
    ),

    # Stage 4 (30K-40K): Pre-noise refinement
    CurriculumStage(
        name="stage_4_clean_refinement",
        steps=10000,
        min_length=7,
        max_length=14,
        noise_peaks=0,
        peak_dropout=0.0,
        mass_error_ppm=0.0,
        clean_data_ratio=1.0,  # 100% clean - get accuracy high before noise
        spectrum_loss_weight=0.10,
        precursor_loss_weight=0.03,
    ),

    # Stage 5 (40K-50K): Minimal noise introduction (mostly clean)
    CurriculumStage(
        name="stage_5_minimal_noise",
        steps=10000,
        min_length=7,
        max_length=15,
        noise_peaks=1,       # Minimal noise when used
        peak_dropout=0.0,    # NO dropout yet
        mass_error_ppm=1.0,  # Very small mass errors
        clean_data_ratio=0.8,  # 80% clean, 20% noisy - smooth introduction
        spectrum_loss_weight=0.12,
        precursor_loss_weight=0.04,
    ),

    # Stage 6 (50K-60K): Light noise (balanced mix)
    CurriculumStage(
        name="stage_6_light_noise",
        steps=10000,
        min_length=7,
        max_length=16,
        noise_peaks=2,       # Light noise
        peak_dropout=0.0,    # STILL no dropout
        mass_error_ppm=2.0,
        clean_data_ratio=0.6,  # 60% clean, 40% noisy
        spectrum_loss_weight=0.13,
        precursor_loss_weight=0.05,
    ),

    # Stage 7 (60K-70K): Moderate noise (mostly noisy)
    CurriculumStage(
        name="stage_7_moderate_noise",
        steps=10000,
        min_length=7,
        max_length=17,
        noise_peaks=3,       # Moderate noise
        peak_dropout=0.0,    # STILL no dropout
        mass_error_ppm=5.0,
        clean_data_ratio=0.4,  # 40% clean, 60% noisy
        spectrum_loss_weight=0.14,
        precursor_loss_weight=0.06,
    ),

    # Stage 8 (70K-80K): Heavy noise + introduce dropout
    CurriculumStage(
        name="stage_8_heavy_noise",
        steps=10000,
        min_length=7,
        max_length=18,
        noise_peaks=5,       # Heavy noise
        peak_dropout=0.05,   # FIRST TIME: Minimal dropout
        mass_error_ppm=8.0,
        clean_data_ratio=0.2,  # 20% clean, 80% noisy
        spectrum_loss_weight=0.14,
        precursor_loss_weight=0.07,
    ),

    # Stage 9 (80K-90K): Near-realistic
    CurriculumStage(
        name="stage_9_near_realistic",
        steps=10000,
        min_length=7,
        max_length=20,
        noise_peaks=7,       # High noise
        peak_dropout=0.08,   # Moderate dropout
        mass_error_ppm=12.0,
        intensity_variation=0.1,
        clean_data_ratio=0.1,  # 10% clean, 90% noisy
        spectrum_loss_weight=0.15,
        precursor_loss_weight=0.08,
    ),

    # Stage 10 (90K-100K): Fully realistic
    CurriculumStage(
        name="stage_10_realistic",
        steps=10000,
        min_length=7,
        max_length=20,
        noise_peaks=10,      # Maximum noise
        peak_dropout=0.10,   # Realistic dropout
        mass_error_ppm=15.0,
        intensity_variation=0.2,
        clean_data_ratio=0.0,  # 100% noisy - full realism
        spectrum_loss_weight=0.15,
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
