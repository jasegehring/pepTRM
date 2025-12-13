"""
Advanced Curriculum Learning for TRM Peptide Sequencing.

Integrates "Sim-to-Real" strategies:
1. Flattened Length Curriculum: Prevents positional overfitting.
2. Two-Tier Noise: Adds both low-intensity "grass" and high-intensity "contaminants".
3. Signal Suppression: Decorrelates intensity from validity.
4. Physics Ramp: Increases mass constraint weight as perception becomes harder.
"""

from dataclasses import dataclass
from typing import List

@dataclass
class CurriculumStage:
    """
    Configuration for a single stage of training difficulty.
    """
    name: str
    steps: int                  # Duration of this stage
    
    # --- Sequence Parameters ---
    min_length: int = 7
    max_length: int = 30        # Flat curriculum: Expose full range early
    
    # --- Noise Parameters (The "Sim-to-Real" Bridge) ---
    clean_data_ratio: float = 1.0  # Fraction of batch that remains untouched
    
    # 1. The "Grass" (Chemical Background)
    # Low intensity peaks (0.0 - 0.1 intensity)
    noise_peaks_low: int = 0    
    
    # 2. The "Spikes" (Contaminants/Chimeras)
    # High intensity peaks (0.2 - 1.0 intensity) - Kills "Big = True" bias
    noise_peaks_high: int = 0   
    
    # 3. Signal Corruption
    peak_dropout: float = 0.0   # Probability of deleting a real peak completely
    signal_suppression: float = 0.0 # Prob. of crushing a real peak to <0.05 intensity
    
    # 4. Precision
    mass_error_ppm: float = 0.0 # Jitter in m/z values
    intensity_variation: float = 0.0 # Jitter in peak heights
    
    # --- Loss Weights ---
    ce_weight: float = 1.0
    spectrum_loss_weight: float = 0.0 # Usually 0 unless fine-tuning
    precursor_loss_weight: float = 0.05 # Starts low, ramps up with noise


# ==============================================================================
# THE CURRICULUM
# Strategy: "Classroom" -> "Playground" -> "Real World" -> "Stress Test"
# ==============================================================================

AGGRESSIVE_NOISE_CURRICULUM = [

    # Stage 0: Pure Foundation (No Noise Whatsoever)
    # Goal: Let positional embeddings learn ALL lengths (7-30) on perfectly clean data.
    # This prevents "position 20+ is rare" bias before noise makes learning harder.
    CurriculumStage(
        name="Pure Foundation",
        steps=5000,
        min_length=7,
        max_length=30,           # Full range immediately
        clean_data_ratio=1.0,    # 100% CLEAN - no noise at all

        noise_peaks_low=0,
        noise_peaks_high=0,
        peak_dropout=0.0,
        signal_suppression=0.0,

        mass_error_ppm=0.0,
        intensity_variation=0.0,
        precursor_loss_weight=0.02,  # Light physics from the start
    ),

    # Stage 1: The Classroom (Syntax & Grammar)
    # Goal: Learn b/y ion rules and positional embeddings for ALL lengths.
    CurriculumStage(
        name="Syntax & Grammar",
        steps=10000,
        min_length=7,
        max_length=30,           # Full range (No length curriculum)
        clean_data_ratio=0.8,    # Mostly clean to establish baseline accuracy

        noise_peaks_low=10,      # Light background
        noise_peaks_high=0,      # No confuse-bots yet
        peak_dropout=0.10,       # Gentle dropout
        signal_suppression=0.0,

        mass_error_ppm=2.0,
        intensity_variation=0.1,
        precursor_loss_weight=0.05,
    ),

    # Stage 2: The Playground (Basic Robustness)
    # Goal: Stop trusting intensity blindly. Introduce "Spikes".
    CurriculumStage(
        name="Robustness & Intensity Decorrelation",
        steps=15000,
        min_length=7,
        max_length=30,
        clean_data_ratio=0.5,    # 50/50 Real vs Sim
        
        noise_peaks_low=25,      # More grass
        noise_peaks_high=2,      # INTRODUCE SPIKES: 2 random loud peaks
        peak_dropout=0.20,
        signal_suppression=0.10, # 10% of real peaks get crushed to noise level
        
        mass_error_ppm=5.0,
        intensity_variation=0.2,
        precursor_loss_weight=0.10, # Ramp up physics as vision gets harder
    ),

    # Stage 3: The Real World (Standard Proteomics)
    # Goal: Simulate a standard Orbitrap RAW file (HeLa/ProteomeTools).
    CurriculumStage(
        name="Realistic Orbitrap Simulation",
        steps=20000,
        min_length=7,
        max_length=30,
        clean_data_ratio=0.2,    # Mostly noisy now
        
        noise_peaks_low=50,      # Heavy background (common in real data)
        noise_peaks_high=5,      # Significant contamination
        peak_dropout=0.30,       # 30% of ions missing (standard)
        signal_suppression=0.20, # 20% of ions are barely visible
        
        mass_error_ppm=10.0,
        intensity_variation=0.4,
        precursor_loss_weight=0.20, # Physics is now critical
    ),

    # Stage 4: The Stress Test (Haystack Mode)
    # Goal: Force the model to use Recursive Refinement and Precursor Constraints.
    # If the model can solve this, Real Data will feel easy.
    CurriculumStage(
        name="Extreme Stress Test",
        steps=55000,             # Long tail training
        min_length=7,
        max_length=30,
        clean_data_ratio=0.0,    # Zero crutches. Sink or swim.
        
        noise_peaks_low=150,     # Massive haystack (realistic for complex samples)
        noise_peaks_high=10,     # Very messy spectrum
        peak_dropout=0.50,       # Missing HALF the sequence evidence
        signal_suppression=0.30, # Heavy suppression
        
        mass_error_ppm=15.0,
        intensity_variation=0.5,
        precursor_loss_weight=0.30, # MAX PHYSICS: "Trust the mass, not the eyes"
    ),
]


class CurriculumScheduler:
    """
    Manages the transition between curriculum stages and updates the dataset.
    """

    def __init__(self, stages: List[CurriculumStage], dataset):
        self.stages = stages
        self.dataset = dataset
        self.current_stage_idx = 0
        self.total_steps = sum(s.steps for s in stages)

        # Initialize
        self._update_stage(0)

    def _update_stage(self, stage_idx: int):
        """Apply stage settings to the dataset and update internal state."""
        if stage_idx >= len(self.stages):
            return

        stage = self.stages[stage_idx]
        
        # Update dataset parameters (Assuming your Dataset class supports these)
        # Note: You may need to update your Dataset class to accept the new noise args
        if hasattr(self.dataset, 'set_difficulty'):
            self.dataset.set_difficulty(
                min_length=stage.min_length,
                max_length=stage.max_length,
                
                # New Noise Params
                noise_peaks_low=stage.noise_peaks_low,
                noise_peaks_high=stage.noise_peaks_high,
                signal_suppression=stage.signal_suppression,
                
                # Standard Params
                peak_dropout=stage.peak_dropout,
                mass_error_ppm=stage.mass_error_ppm,
                intensity_variation=stage.intensity_variation,
                clean_data_ratio=stage.clean_data_ratio,
            )

        self.current_stage_idx = stage_idx
        self._log_stage_transition(stage, stage_idx)

    def step(self, global_step: int) -> bool:
        """
        Check if we need to advance to the next stage based on global_step.
        Returns True if a transition occurred.
        """
        cumulative_steps = 0
        for idx, stage in enumerate(self.stages):
            if global_step < cumulative_steps + stage.steps:
                if idx != self.current_stage_idx:
                    self._update_stage(idx)
                    return True
                return False
            cumulative_steps += stage.steps
        
        # If we exceed total steps, stay in the last stage
        return False

    def get_current_stage(self) -> CurriculumStage:
        return self.stages[self.current_stage_idx]

    @property
    def current_stage(self) -> CurriculumStage:
        """Property for compatibility with trainer."""
        return self.stages[self.current_stage_idx]

    def get_precursor_loss_weight(self) -> float:
        return self.stages[self.current_stage_idx].precursor_loss_weight

    def get_spectrum_loss_weight(self) -> float:
        return self.stages[self.current_stage_idx].spectrum_loss_weight
        
    def _log_stage_transition(self, stage: CurriculumStage, idx: int):
        print(f"\n🚀 CURRICULUM UPDATE: Stage {idx + 1}/{len(self.stages)} - {stage.name}")
        print(f"   Steps: {stage.steps:,}")
        print(f"   Lengths: {stage.min_length}-{stage.max_length} (Flat)")
        print(f"   Clean Data: {stage.clean_data_ratio:.0%}")
        print(f"   Noise: {stage.noise_peaks_low} (Low) | {stage.noise_peaks_high} (High/Spikes)")
        print(f"   Signal Damage: Dropout {stage.peak_dropout:.0%} | Suppress {stage.signal_suppression:.0%}")
        print(f"   Physics Weight: {stage.precursor_loss_weight:.2f}")