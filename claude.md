# RecursivePeptide: Complete Project Specification

## Project Overview

**Goal**: Implement a Tiny Recursive Model (TRM) for De Novo Peptide Sequencing from Mass Spectrometry data.

**Core Insight**: Unlike standard autoregressive models that generate sequences left-to-right in a single pass, this model uses a recursive loop to iteratively refine the sequence hypothesis. Mass spectrometry provides hard physical constraints (fragment masses must match observed peaks), making this domain uniquely suited for iterative refinement—the model can learn to "check its work" against mass constraints and self-correct.

**Core Paper**: "Less is More: Recursive Reasoning with Tiny Networks" (Jolicoeur-Martineau, 2025)

**Key Differentiator from Original TRM**: We add a mass-matching auxiliary loss that exploits the physical constraints of mass spectrometry—something not present in TRM's original puzzle-solving domains (Sudoku, ARC-AGI, Mazes).

---

## Theoretical Architecture

The model follows the TRM framework with domain-specific adaptations:

### Core Variables
- **x** (The Problem): The Mass Spectrum (list of m/z peaks with intensities) + precursor mass
- **y** (The Hypothesis): The predicted Peptide Sequence (probability distribution over amino acids)
- **z** (The Latent State): A hidden memory vector ("scratchpad") tracking the reasoning state

### The Recursive Loop

For T supervision steps (e.g., T=8 or T=16):

```
For each supervision step t:
    For n latent reasoning steps:
        z = f_θ(x, y, z)           # Update latent state
    y = g_θ(y, z)                  # Update sequence prediction
    Loss_t = CE(y, target) + λ * SpectrumMatch(y, x)
    
Total Loss = Σ_t w_t * Loss_t     # Deep supervision with weighted sum
```

**Deep Supervision**: We calculate loss at every supervision step, not just the end. This forces the model to learn a trajectory of improvement and is the primary driver of TRM's performance gains.

---

## Directory Structure

```
recursive_peptide/
├── src/
│   ├── __init__.py
│   ├── constants.py              # Amino acid masses, PTM masses, vocabulary
│   ├── data/
│   │   ├── __init__.py
│   │   ├── synthetic.py          # Theoretical spectrum generator (forward model)
│   │   ├── encoding.py           # Sinusoidal embeddings for mass values
│   │   ├── dataset.py            # PyTorch Dataset classes
│   │   └── preprocessing.py      # Real data loading and preprocessing
│   ├── model/
│   │   ├── __init__.py
│   │   ├── layers.py             # Shared layers (sinusoidal embedding, etc.)
│   │   ├── encoder.py            # Spectrum encoder (Transformer)
│   │   ├── decoder.py            # Recursive decoder core
│   │   └── trm.py                # Full TRM model combining components
│   ├── training/
│   │   ├── __init__.py
│   │   ├── losses.py             # CE loss, spectrum matching loss, combined loss
│   │   ├── metrics.py            # Accuracy metrics, mass error
│   │   ├── trainer.py            # Main training loop
│   │   └── curriculum.py         # Curriculum learning scheduler
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── predict.py            # Inference with uncertainty quantification
│   │   └── beam_search.py        # Beam search for top-k hypotheses
│   └── utils/
│       ├── __init__.py
│       ├── visualization.py      # Spectrum plotting, attention visualization
│       └── evaluation.py         # Benchmark evaluation utilities
├── configs/
│   ├── default.yaml              # Base configuration
│   ├── curriculum.yaml           # Curriculum learning stages
│   ├── finetune_real.yaml        # Real data finetuning config
│   └── model/
│       ├── tiny.yaml             # 7M parameter model
│       └── small.yaml            # 20M parameter model
├── scripts/
│   ├── train.py                  # Training entry point
│   ├── evaluate.py               # Evaluation entry point
│   ├── predict.py                # Inference entry point
│   └── download_data.py          # Download benchmark datasets
├── tests/
│   ├── test_physics.py           # CRITICAL: Verify mass calculations
│   ├── test_synthetic.py         # Test spectrum generation
│   ├── test_model.py             # Test model forward pass
│   ├── test_losses.py            # Test loss calculations
│   └── test_curriculum.py        # Test curriculum progression
├── notebooks/
│   ├── exploration.ipynb         # Data exploration
│   ├── visualize_recursion.ipynb # Visualize iterative refinement
│   └── benchmark_analysis.ipynb  # Analyze benchmark results
├── data/                         # Data directory (gitignored)
│   ├── synthetic/
│   └── real/
├── checkpoints/                  # Model checkpoints (gitignored)
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Phase 1: Constants and Data Generation

### 1.1 Constants (`src/constants.py`)

```python
"""
Physical constants for peptide mass spectrometry.
All masses are monoisotopic masses in Daltons (Da).
"""

# Standard amino acid masses (monoisotopic)
AMINO_ACID_MASSES: dict[str, float] = {
    'A': 71.03711,    # Alanine
    'R': 156.10111,   # Arginine
    'N': 114.04293,   # Asparagine
    'D': 115.02694,   # Aspartic acid
    'C': 103.00919,   # Cysteine
    'E': 129.04259,   # Glutamic acid
    'Q': 128.05858,   # Glutamine
    'G': 57.02146,    # Glycine
    'H': 137.05891,   # Histidine
    'I': 113.08406,   # Isoleucine  (SAME MASS AS LEUCINE)
    'L': 113.08406,   # Leucine     (SAME MASS AS ISOLEUCINE)
    'K': 128.09496,   # Lysine
    'M': 131.04049,   # Methionine
    'F': 147.06841,   # Phenylalanine
    'P': 97.05276,    # Proline
    'S': 87.03203,    # Serine
    'T': 101.04768,   # Threonine
    'W': 186.07931,   # Tryptophan
    'Y': 163.06333,   # Tyrosine
    'V': 99.06841,    # Valine
}

# Common post-translational modifications (mass deltas)
PTM_MASS_DELTAS: dict[str, float] = {
    'Phospho': 79.96633,      # Phosphorylation (S, T, Y)
    'Oxidation': 15.99491,    # Oxidation (M)
    'Acetyl': 42.01056,       # Acetylation (K, N-term)
    'Deamidation': 0.98402,   # Deamidation (N, Q)
    'Carbamidomethyl': 57.02146,  # Carbamidomethylation (C) - common artifact
}

# Extended vocabulary including modified amino acids
EXTENDED_AMINO_ACID_MASSES: dict[str, float] = {
    **AMINO_ACID_MASSES,
    # Phosphorylation
    'S[Phospho]': AMINO_ACID_MASSES['S'] + PTM_MASS_DELTAS['Phospho'],
    'T[Phospho]': AMINO_ACID_MASSES['T'] + PTM_MASS_DELTAS['Phospho'],
    'Y[Phospho]': AMINO_ACID_MASSES['Y'] + PTM_MASS_DELTAS['Phospho'],
    # Oxidation
    'M[Oxidation]': AMINO_ACID_MASSES['M'] + PTM_MASS_DELTAS['Oxidation'],
    # Acetylation
    'K[Acetyl]': AMINO_ACID_MASSES['K'] + PTM_MASS_DELTAS['Acetyl'],
    # Deamidation
    'N[Deamidation]': AMINO_ACID_MASSES['N'] + PTM_MASS_DELTAS['Deamidation'],
    'Q[Deamidation]': AMINO_ACID_MASSES['Q'] + PTM_MASS_DELTAS['Deamidation'],
    # Carbamidomethylation (fixed modification for cysteine)
    'C[Carbamidomethyl]': AMINO_ACID_MASSES['C'] + PTM_MASS_DELTAS['Carbamidomethyl'],
}

# Physical constants
WATER_MASS = 18.01056       # H2O
PROTON_MASS = 1.00727       # H+
AMMONIA_MASS = 17.02655     # NH3 (for neutral loss)
CO_MASS = 27.99491          # CO (for a-ion calculation)

# Vocabulary for model
STANDARD_VOCAB = list(AMINO_ACID_MASSES.keys())  # 20 amino acids
SPECIAL_TOKENS = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
VOCAB = SPECIAL_TOKENS + STANDARD_VOCAB

# Token indices
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3
AA_START_IDX = 4

VOCAB_SIZE = len(VOCAB)

# For PTM-aware model, use extended vocabulary
EXTENDED_VOCAB = SPECIAL_TOKENS + list(EXTENDED_AMINO_ACID_MASSES.keys())
EXTENDED_VOCAB_SIZE = len(EXTENDED_VOCAB)

# Isobaric groups (amino acids with same/similar masses)
ISOBARIC_GROUPS = {
    'IL': {'I', 'L'},           # Exactly same mass (113.08406)
    'KQ': {'K', 'Q'},           # Very similar (128.095 vs 128.059, Δ=0.036)
}

# Create lookup dictionaries
AA_TO_IDX = {aa: i for i, aa in enumerate(VOCAB)}
IDX_TO_AA = {i: aa for i, aa in enumerate(VOCAB)}
AA_MASS_TENSOR_ORDER = [AMINO_ACID_MASSES.get(aa, 0.0) for aa in VOCAB]
```

### 1.2 Theoretical Spectrum Generator (`src/data/synthetic.py`)

```python
"""
Forward model: Generate theoretical MS/MS spectrum from peptide sequence.

This is the "physics simulator" that creates training data with perfect labels.
"""

from dataclasses import dataclass
import random
import torch
from torch import Tensor

from ..constants import (
    AMINO_ACID_MASSES, WATER_MASS, PROTON_MASS, 
    AMMONIA_MASS, CO_MASS, VOCAB, AA_TO_IDX
)


@dataclass
class TheoreticalSpectrum:
    """Output of the forward model."""
    peaks: list[tuple[float, float]]  # List of (mass, intensity) tuples
    precursor_mass: float             # Total peptide mass (M+H)+
    precursor_mz: float               # m/z of precursor
    charge: int                       # Precursor charge state
    peptide: str                      # Original peptide sequence
    ion_annotations: dict[float, str] # Mass -> ion type (e.g., "b3", "y5")


def compute_peptide_mass(peptide: str) -> float:
    """
    Compute monoisotopic mass of peptide.
    
    Peptide mass = sum of residue masses + H2O (for N and C termini)
    """
    residue_mass = sum(AMINO_ACID_MASSES[aa] for aa in peptide)
    return residue_mass + WATER_MASS


def generate_theoretical_spectrum(
    peptide: str,
    charge: int = 2,
    ion_types: list[str] = ['b', 'y'],
    include_neutral_losses: bool = False,
    noise_peaks: int = 0,
    peak_dropout: float = 0.0,
    mass_error_ppm: float = 0.0,
    intensity_variation: float = 0.0,
) -> TheoreticalSpectrum:
    """
    Generate theoretical MS/MS spectrum from peptide sequence.
    
    Physics:
    - b-ions: N-terminal fragments. Mass = sum of residues from N-terminus
    - y-ions: C-terminal fragments. Mass = sum of residues from C-terminus + H2O
    - a-ions: b-ions minus CO (28 Da) - optional
    - Neutral losses: Loss of H2O (-18) or NH3 (-17) from b/y ions - optional
    
    Args:
        peptide: Amino acid sequence (e.g., "PEPTIDE")
        charge: Precursor charge state (affects m/z calculation)
        ion_types: Which ion series to generate ('b', 'y', 'a')
        include_neutral_losses: Include -H2O and -NH3 losses
        noise_peaks: Number of random noise peaks to add (for realistic simulation)
        peak_dropout: Fraction of theoretical peaks to randomly remove
        mass_error_ppm: Add Gaussian mass error (parts per million)
        intensity_variation: Add variation to intensities
    
    Returns:
        TheoreticalSpectrum with peaks, precursor info, and annotations
    """
    n = len(peptide)
    peaks = []
    annotations = {}
    
    # Calculate residue masses
    residue_masses = [AMINO_ACID_MASSES[aa] for aa in peptide]
    
    # Precursor mass (full peptide + H2O + charge*proton for m/z)
    precursor_mass = sum(residue_masses) + WATER_MASS
    precursor_mz = (precursor_mass + charge * PROTON_MASS) / charge
    
    # Generate b-ions (N-terminal fragments)
    # b_i = mass of first i residues
    if 'b' in ion_types:
        cumulative = 0.0
        for i in range(1, n):  # b1 to b_{n-1}
            cumulative += residue_masses[i - 1]
            mass = cumulative
            intensity = _theoretical_intensity(i, n, 'b')
            peaks.append((mass, intensity))
            annotations[mass] = f'b{i}'
            
            # Neutral losses from b-ions
            if include_neutral_losses:
                if peptide[i-1] in 'STED':  # Residues prone to H2O loss
                    peaks.append((mass - WATER_MASS, intensity * 0.3))
                    annotations[mass - WATER_MASS] = f'b{i}-H2O'
                if peptide[i-1] in 'RKNQ':  # Residues prone to NH3 loss
                    peaks.append((mass - AMMONIA_MASS, intensity * 0.3))
                    annotations[mass - AMMONIA_MASS] = f'b{i}-NH3'
    
    # Generate a-ions (b-ions minus CO)
    if 'a' in ion_types:
        cumulative = 0.0
        for i in range(1, n):
            cumulative += residue_masses[i - 1]
            mass = cumulative - CO_MASS
            intensity = _theoretical_intensity(i, n, 'a') * 0.5  # a-ions typically weaker
            peaks.append((mass, intensity))
            annotations[mass] = f'a{i}'
    
    # Generate y-ions (C-terminal fragments)
    # y_i = mass of last i residues + H2O
    if 'y' in ion_types:
        cumulative = WATER_MASS  # y-ions include the C-terminal OH
        for i in range(1, n):  # y1 to y_{n-1}
            cumulative += residue_masses[n - i]
            mass = cumulative
            intensity = _theoretical_intensity(i, n, 'y')
            peaks.append((mass, intensity))
            annotations[mass] = f'y{i}'
            
            # Neutral losses from y-ions
            if include_neutral_losses:
                if peptide[n-i] in 'STED':
                    peaks.append((mass - WATER_MASS, intensity * 0.3))
                    annotations[mass - WATER_MASS] = f'y{i}-H2O'
                if peptide[n-i] in 'RKNQ':
                    peaks.append((mass - AMMONIA_MASS, intensity * 0.3))
                    annotations[mass - AMMONIA_MASS] = f'y{i}-NH3'
    
    # Apply peak dropout (simulate missing peaks)
    if peak_dropout > 0:
        peaks = [p for p in peaks if random.random() > peak_dropout]
    
    # Apply mass error
    if mass_error_ppm > 0:
        peaks = [
            (m + m * random.gauss(0, mass_error_ppm * 1e-6), i)
            for m, i in peaks
        ]
    
    # Apply intensity variation
    if intensity_variation > 0:
        peaks = [
            (m, max(0.01, i * random.gauss(1.0, intensity_variation)))
            for m, i in peaks
        ]
    
    # Add noise peaks
    if noise_peaks > 0:
        max_mass = precursor_mass
        for _ in range(noise_peaks):
            noise_mass = random.uniform(50, max_mass)
            noise_intensity = random.uniform(0.01, 0.2)
            peaks.append((noise_mass, noise_intensity))
    
    # Sort by mass
    peaks.sort(key=lambda x: x[0])
    
    return TheoreticalSpectrum(
        peaks=peaks,
        precursor_mass=precursor_mass,
        precursor_mz=precursor_mz,
        charge=charge,
        peptide=peptide,
        ion_annotations=annotations,
    )


def _theoretical_intensity(ion_index: int, peptide_length: int, ion_type: str) -> float:
    """
    Generate theoretical intensity for a fragment ion.
    
    In real spectra, intensity depends on many factors. For synthetic data,
    we use a simple model where middle fragments are more intense.
    """
    # Normalized position (0 to 1)
    if ion_type in ['b', 'a']:
        pos = ion_index / peptide_length
    else:  # y-ions
        pos = ion_index / peptide_length
    
    # Bell curve - middle fragments slightly more intense
    base_intensity = 0.5 + 0.5 * (1 - abs(pos - 0.5) * 2)
    
    # y-ions typically more intense than b-ions
    if ion_type == 'y':
        base_intensity *= 1.2
    elif ion_type == 'a':
        base_intensity *= 0.5
    
    return min(1.0, base_intensity)


def generate_random_peptide(
    min_length: int = 7,
    max_length: int = 20,
    exclude_aa: set[str] = None,
) -> str:
    """Generate a random peptide sequence."""
    exclude_aa = exclude_aa or set()
    available_aa = [aa for aa in AMINO_ACID_MASSES.keys() if aa not in exclude_aa]
    length = random.randint(min_length, max_length)
    return ''.join(random.choices(available_aa, k=length))
```

### 1.3 Mass Encoding (`src/data/encoding.py`)

```python
"""
Sinusoidal embeddings for continuous mass values.

Similar to positional encoding in Transformers, but the input is mass (float)
instead of position (integer). This allows the model to learn mass relationships.
"""

import math
import torch
import torch.nn as nn
from torch import Tensor


class SinusoidalMassEmbedding(nn.Module):
    """
    Project scalar mass values to high-dimensional space using sinusoidal functions.
    
    Uses multiple frequency bands to capture both coarse mass differences
    (hundreds of Da) and fine differences (fractions of Da).
    
    For a mass value m, the embedding is:
        [sin(m * f_0), cos(m * f_0), sin(m * f_1), cos(m * f_1), ...]
    
    where frequencies f_i are logarithmically spaced.
    """
    
    def __init__(
        self,
        dim: int = 256,
        max_mass: float = 2000.0,
        min_freq: float = 1e-4,
        max_freq: float = 1.0,
        learnable_scale: bool = True,
    ):
        """
        Args:
            dim: Output embedding dimension (must be even)
            max_mass: Maximum expected mass value (for normalization)
            min_freq: Minimum frequency for sinusoidal functions
            max_freq: Maximum frequency for sinusoidal functions
            learnable_scale: Whether to include learnable scaling
        """
        super().__init__()
        assert dim % 2 == 0, "Dimension must be even for sin/cos pairs"
        
        self.dim = dim
        self.max_mass = max_mass
        
        # Create logarithmically spaced frequencies
        num_freqs = dim // 2
        freqs = torch.exp(
            torch.linspace(
                math.log(min_freq),
                math.log(max_freq),
                num_freqs
            )
        )
        self.register_buffer('freqs', freqs)
        
        # Optional learnable scaling
        if learnable_scale:
            self.scale = nn.Parameter(torch.ones(dim))
        else:
            self.register_buffer('scale', torch.ones(dim))
    
    def forward(self, mass: Tensor) -> Tensor:
        """
        Args:
            mass: Tensor of any shape containing mass values
            
        Returns:
            Tensor of shape (*mass.shape, dim) with embeddings
        """
        # Normalize mass to reasonable range
        mass_normalized = mass / self.max_mass
        
        # Expand for broadcasting: (..., 1) @ (num_freqs,) -> (..., num_freqs)
        mass_expanded = mass_normalized.unsqueeze(-1)
        angles = mass_expanded * self.freqs * 2 * math.pi
        
        # Concatenate sin and cos
        embeddings = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        
        return embeddings * self.scale


class PeakEncoder(nn.Module):
    """
    Encode a single spectrum peak (mass, intensity) pair.
    
    Combines sinusoidal mass embedding with intensity information.
    """
    
    def __init__(
        self,
        dim: int = 256,
        max_mass: float = 2000.0,
        intensity_dim: int = 32,
    ):
        super().__init__()
        
        self.mass_embedding = SinusoidalMassEmbedding(
            dim=dim - intensity_dim,
            max_mass=max_mass,
        )
        
        # Simple MLP for intensity (log-scaled)
        self.intensity_encoder = nn.Sequential(
            nn.Linear(1, intensity_dim),
            nn.ReLU(),
            nn.Linear(intensity_dim, intensity_dim),
        )
        
        self.output_dim = dim
    
    def forward(self, mass: Tensor, intensity: Tensor) -> Tensor:
        """
        Args:
            mass: (batch, num_peaks) mass values
            intensity: (batch, num_peaks) intensity values
            
        Returns:
            (batch, num_peaks, dim) peak embeddings
        """
        # Log-scale intensity (add small epsilon to avoid log(0))
        log_intensity = torch.log(intensity + 1e-8).unsqueeze(-1)
        
        mass_emb = self.mass_embedding(mass)  # (batch, num_peaks, dim - intensity_dim)
        intensity_emb = self.intensity_encoder(log_intensity)  # (batch, num_peaks, intensity_dim)
        
        return torch.cat([mass_emb, intensity_emb], dim=-1)


class PrecursorEncoder(nn.Module):
    """
    Encode precursor information (mass and charge).
    """
    
    def __init__(self, dim: int = 256, max_mass: float = 3000.0, max_charge: int = 6):
        super().__init__()
        
        self.mass_embedding = SinusoidalMassEmbedding(dim=dim - 16, max_mass=max_mass)
        self.charge_embedding = nn.Embedding(max_charge + 1, 16)
        
    def forward(self, precursor_mass: Tensor, charge: Tensor) -> Tensor:
        """
        Args:
            precursor_mass: (batch,) precursor masses
            charge: (batch,) charge states (integers)
            
        Returns:
            (batch, dim) precursor embeddings
        """
        mass_emb = self.mass_embedding(precursor_mass)
        charge_emb = self.charge_embedding(charge)
        return torch.cat([mass_emb, charge_emb], dim=-1)
```

### 1.4 Dataset Class (`src/data/dataset.py`)

```python
"""
PyTorch Dataset classes for peptide sequencing.
"""

import random
import torch
from torch import Tensor
from torch.utils.data import IterableDataset, Dataset
from dataclasses import dataclass
from typing import Optional

from ..constants import AMINO_ACID_MASSES, VOCAB, AA_TO_IDX, PAD_IDX, SOS_IDX, EOS_IDX
from .synthetic import generate_theoretical_spectrum, generate_random_peptide


@dataclass
class PeptideSpectrumSample:
    """A single training sample."""
    spectrum_masses: Tensor      # (max_peaks,) - mass values
    spectrum_intensities: Tensor # (max_peaks,) - intensity values
    spectrum_mask: Tensor        # (max_peaks,) - True for real peaks, False for padding
    precursor_mass: Tensor       # (1,) - precursor mass
    precursor_charge: Tensor     # (1,) - charge state (integer)
    sequence: Tensor             # (max_seq_len,) - token indices
    sequence_mask: Tensor        # (max_seq_len,) - True for real tokens


class SyntheticPeptideDataset(IterableDataset):
    """
    Infinite synthetic peptide dataset for training.
    
    Generates random peptides and their theoretical spectra on the fly.
    Supports curriculum learning by adjusting difficulty parameters.
    """
    
    def __init__(
        self,
        # Peptide parameters
        min_length: int = 7,
        max_length: int = 20,
        
        # Spectrum parameters
        max_peaks: int = 100,
        max_seq_len: int = 25,
        ion_types: list[str] = ['b', 'y'],
        include_neutral_losses: bool = False,
        
        # Difficulty parameters (for curriculum learning)
        noise_peaks: int = 0,
        peak_dropout: float = 0.0,
        mass_error_ppm: float = 0.0,
        intensity_variation: float = 0.0,
        
        # Charge state distribution
        charge_distribution: dict[int, float] = None,
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.max_peaks = max_peaks
        self.max_seq_len = max_seq_len
        self.ion_types = ion_types
        self.include_neutral_losses = include_neutral_losses
        
        # Difficulty parameters
        self.noise_peaks = noise_peaks
        self.peak_dropout = peak_dropout
        self.mass_error_ppm = mass_error_ppm
        self.intensity_variation = intensity_variation
        
        # Default charge distribution (charge 2 most common)
        self.charge_distribution = charge_distribution or {2: 0.7, 3: 0.25, 4: 0.05}
    
    def set_difficulty(
        self,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        noise_peaks: Optional[int] = None,
        peak_dropout: Optional[float] = None,
        mass_error_ppm: Optional[float] = None,
        intensity_variation: Optional[float] = None,
    ):
        """Update difficulty parameters for curriculum learning."""
        if min_length is not None:
            self.min_length = min_length
        if max_length is not None:
            self.max_length = max_length
        if noise_peaks is not None:
            self.noise_peaks = noise_peaks
        if peak_dropout is not None:
            self.peak_dropout = peak_dropout
        if mass_error_ppm is not None:
            self.mass_error_ppm = mass_error_ppm
        if intensity_variation is not None:
            self.intensity_variation = intensity_variation
    
    def _sample_charge(self) -> int:
        """Sample charge state from distribution."""
        charges = list(self.charge_distribution.keys())
        probs = list(self.charge_distribution.values())
        return random.choices(charges, weights=probs, k=1)[0]
    
    def _generate_sample(self) -> PeptideSpectrumSample:
        """Generate a single training sample."""
        # Generate random peptide
        peptide = generate_random_peptide(self.min_length, self.max_length)
        charge = self._sample_charge()
        
        # Generate theoretical spectrum
        spectrum = generate_theoretical_spectrum(
            peptide=peptide,
            charge=charge,
            ion_types=self.ion_types,
            include_neutral_losses=self.include_neutral_losses,
            noise_peaks=self.noise_peaks,
            peak_dropout=self.peak_dropout,
            mass_error_ppm=self.mass_error_ppm,
            intensity_variation=self.intensity_variation,
        )
        
        # Convert peaks to tensors
        num_peaks = min(len(spectrum.peaks), self.max_peaks)
        
        masses = torch.zeros(self.max_peaks)
        intensities = torch.zeros(self.max_peaks)
        peak_mask = torch.zeros(self.max_peaks, dtype=torch.bool)
        
        for i, (mass, intensity) in enumerate(spectrum.peaks[:num_peaks]):
            masses[i] = mass
            intensities[i] = intensity
            peak_mask[i] = True
        
        # Convert sequence to token indices with SOS and EOS
        seq_tokens = [SOS_IDX] + [AA_TO_IDX[aa] for aa in peptide] + [EOS_IDX]
        seq_len = len(seq_tokens)
        
        sequence = torch.full((self.max_seq_len,), PAD_IDX, dtype=torch.long)
        seq_mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
        
        for i, token in enumerate(seq_tokens[:self.max_seq_len]):
            sequence[i] = token
            seq_mask[i] = True
        
        return PeptideSpectrumSample(
            spectrum_masses=masses,
            spectrum_intensities=intensities,
            spectrum_mask=peak_mask,
            precursor_mass=torch.tensor([spectrum.precursor_mass]),
            precursor_charge=torch.tensor([charge], dtype=torch.long),
            sequence=sequence,
            sequence_mask=seq_mask,
        )
    
    def __iter__(self):
        """Infinite iterator."""
        while True:
            yield self._generate_sample()


def collate_peptide_samples(samples: list[PeptideSpectrumSample]) -> dict[str, Tensor]:
    """Collate function for DataLoader."""
    return {
        'spectrum_masses': torch.stack([s.spectrum_masses for s in samples]),
        'spectrum_intensities': torch.stack([s.spectrum_intensities for s in samples]),
        'spectrum_mask': torch.stack([s.spectrum_mask for s in samples]),
        'precursor_mass': torch.stack([s.precursor_mass for s in samples]).squeeze(-1),
        'precursor_charge': torch.stack([s.precursor_charge for s in samples]).squeeze(-1),
        'sequence': torch.stack([s.sequence for s in samples]),
        'sequence_mask': torch.stack([s.sequence_mask for s in samples]),
    }
```

---

## Phase 2: Model Architecture

### 2.1 Shared Layers (`src/model/layers.py`)

```python
"""
Shared model components.
"""

import math
import torch
import torch.nn as nn
from torch import Tensor


class LayerNorm(nn.Module):
    """Layer normalization with optional bias."""
    
    def __init__(self, dim: int, bias: bool = True, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None
        self.eps = eps
    
    def forward(self, x: Tensor) -> Tensor:
        return nn.functional.layer_norm(
            x, self.weight.shape, self.weight, self.bias, self.eps
        )


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation."""
    
    def __init__(self, dim: int, hidden_dim: int = None, dropout: float = 0.1):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional masking."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        causal: bool = False,
    ):
        super().__init__()
        assert dim % num_heads == 0
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.causal = causal
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        q: Tensor,
        k: Tensor = None,
        v: Tensor = None,
        mask: Tensor = None,
    ) -> Tensor:
        """
        Args:
            q: Query tensor (batch, seq_len, dim)
            k: Key tensor (batch, seq_len, dim) - defaults to q (self-attention)
            v: Value tensor (batch, seq_len, dim) - defaults to k
            mask: Attention mask (batch, seq_len) or (batch, q_len, k_len)
        """
        if k is None:
            k = q
        if v is None:
            v = k
        
        batch, q_len, _ = q.shape
        _, k_len, _ = k.shape
        
        # Project and reshape to (batch, num_heads, seq_len, head_dim)
        q = self.q_proj(q).view(batch, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).view(batch, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).view(batch, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply mask
        if mask is not None:
            if mask.dim() == 2:
                # (batch, k_len) -> (batch, 1, 1, k_len)
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                # (batch, q_len, k_len) -> (batch, 1, q_len, k_len)
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(~mask, float('-inf'))
        
        # Causal mask
        if self.causal:
            causal_mask = torch.triu(
                torch.ones(q_len, k_len, dtype=torch.bool, device=q.device),
                diagonal=1
            )
            scores = scores.masked_fill(causal_mask, float('-inf'))
        
        # Softmax and apply to values
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, q_len, self.dim)
        
        return self.out_proj(out)


class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer."""
    
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(dim, num_heads, dropout)
        self.ff = FeedForward(dim, dropout=dropout)
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        # Self-attention with residual
        x = x + self.dropout(self.self_attn(self.norm1(x), mask=mask))
        # Feed-forward with residual
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class TransformerDecoderLayer(nn.Module):
    """Single transformer decoder layer with cross-attention."""
    
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(dim, num_heads, dropout, causal=False)
        self.cross_attn = MultiHeadAttention(dim, num_heads, dropout)
        self.ff = FeedForward(dim, dropout=dropout)
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)
        self.norm3 = LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: Tensor,
        context: Tensor,
        self_mask: Tensor = None,
        context_mask: Tensor = None,
    ) -> Tensor:
        # Self-attention
        x = x + self.dropout(self.self_attn(self.norm1(x), mask=self_mask))
        # Cross-attention to context (encoded spectrum)
        x = x + self.dropout(self.cross_attn(self.norm2(x), context, context, mask=context_mask))
        # Feed-forward
        x = x + self.dropout(self.ff(self.norm3(x)))
        return x
```

### 2.2 Spectrum Encoder (`src/model/encoder.py`)

```python
"""
Spectrum encoder: Transforms MS/MS spectrum into contextual representations.
"""

import torch
import torch.nn as nn
from torch import Tensor

from .layers import TransformerEncoderLayer, LayerNorm
from ..data.encoding import PeakEncoder, PrecursorEncoder


class SpectrumEncoder(nn.Module):
    """
    Encode MS/MS spectrum (list of peaks) into contextual representations.
    
    Architecture:
    1. Encode each peak (mass, intensity) independently
    2. Add precursor information
    3. Apply Transformer encoder layers
    4. Output: contextualized peak embeddings
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        max_peaks: int = 100,
        max_mass: float = 2000.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Peak encoding (mass + intensity -> hidden_dim)
        self.peak_encoder = PeakEncoder(
            dim=hidden_dim,
            max_mass=max_mass,
            intensity_dim=32,
        )
        
        # Precursor encoding
        self.precursor_encoder = PrecursorEncoder(
            dim=hidden_dim,
            max_mass=max_mass * 1.5,  # Precursor can be larger
        )
        
        # Learnable [CLS]-like token for precursor info
        self.precursor_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        
        # Transformer encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = LayerNorm(hidden_dim)
        
    def forward(
        self,
        masses: Tensor,           # (batch, max_peaks)
        intensities: Tensor,      # (batch, max_peaks)
        peak_mask: Tensor,        # (batch, max_peaks)
        precursor_mass: Tensor,   # (batch,)
        precursor_charge: Tensor, # (batch,)
    ) -> tuple[Tensor, Tensor]:
        """
        Encode spectrum.
        
        Returns:
            encoded: (batch, max_peaks + 1, hidden_dim) - peak representations + precursor
            mask: (batch, max_peaks + 1) - attention mask
        """
        batch_size = masses.shape[0]
        
        # Encode peaks
        peak_embeddings = self.peak_encoder(masses, intensities)  # (batch, max_peaks, hidden)
        
        # Encode precursor and add as first token
        precursor_emb = self.precursor_encoder(precursor_mass, precursor_charge)  # (batch, hidden)
        precursor_token = self.precursor_token.expand(batch_size, -1, -1)  # (batch, 1, hidden)
        precursor_token = precursor_token + precursor_emb.unsqueeze(1)
        
        # Concatenate: [precursor, peaks...]
        x = torch.cat([precursor_token, peak_embeddings], dim=1)  # (batch, 1 + max_peaks, hidden)
        
        # Update mask to include precursor token
        full_mask = torch.cat([
            torch.ones(batch_size, 1, dtype=torch.bool, device=peak_mask.device),
            peak_mask
        ], dim=1)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, mask=full_mask)
        
        x = self.norm(x)
        
        return x, full_mask
```

### 2.3 Recursive Decoder (`src/model/decoder.py`)

```python
"""
Recursive decoder: The core TRM component that iteratively refines predictions.
"""

import torch
import torch.nn as nn
from torch import Tensor

from .layers import TransformerDecoderLayer, LayerNorm
from ..constants import VOCAB_SIZE, AA_MASS_TENSOR_ORDER


class RecursiveDecoder(nn.Module):
    """
    Recursive decoder that iteratively refines sequence predictions.
    
    Following TRM, this uses a single shared network that:
    1. Updates latent state z based on (x, y, z) - "think"
    2. Updates prediction y based on (y, z) - "act"
    
    The same network is used for both operations (distinguished by input).
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        max_seq_len: int = 25,
        vocab_size: int = VOCAB_SIZE,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        
        # Token embedding for sequence positions
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)
        
        # Project soft sequence representation (probabilities -> embeddings)
        self.soft_embed_proj = nn.Linear(vocab_size, hidden_dim)
        
        # Shared transformer decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = LayerNorm(hidden_dim)
        
        # Output head: hidden -> vocab logits
        self.output_head = nn.Linear(hidden_dim, vocab_size)
        
        # Learnable initial latent state
        self.initial_z = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim) * 0.02)
        
        # Learnable initial sequence guess (uniform-ish)
        self.initial_logits = nn.Parameter(torch.zeros(1, max_seq_len, vocab_size))
        
    def get_initial_state(self, batch_size: int, device: torch.device) -> tuple[Tensor, Tensor]:
        """
        Get initial prediction (y) and latent state (z).
        
        Returns:
            y: (batch, max_seq_len, vocab_size) - initial logits
            z: (batch, max_seq_len, hidden_dim) - initial latent state
        """
        y = self.initial_logits.expand(batch_size, -1, -1).clone()
        z = self.initial_z.expand(batch_size, -1, -1).clone()
        return y, z
    
    def soft_embed(self, probs: Tensor) -> Tensor:
        """
        Convert probability distribution to soft embedding.
        
        Args:
            probs: (batch, seq_len, vocab_size) - probability distribution
            
        Returns:
            (batch, seq_len, hidden_dim) - soft embeddings
        """
        return self.soft_embed_proj(probs)
    
    def latent_step(
        self,
        encoded_spectrum: Tensor,  # (batch, num_peaks, hidden)
        spectrum_mask: Tensor,     # (batch, num_peaks)
        current_probs: Tensor,     # (batch, seq_len, vocab_size)
        latent_z: Tensor,          # (batch, seq_len, hidden)
    ) -> Tensor:
        """
        Single latent reasoning step: z <- f(x, y, z)
        
        Updates the latent state based on spectrum, current prediction, and previous state.
        """
        batch_size, seq_len = latent_z.shape[:2]
        
        # Create input by combining current prediction embedding and latent state
        y_embed = self.soft_embed(current_probs)
        
        # Add positional information
        positions = torch.arange(seq_len, device=latent_z.device)
        pos_embed = self.position_embedding(positions)
        
        # Combine: y_embed + z + position
        x = y_embed + latent_z + pos_embed
        
        # Apply decoder layers with cross-attention to spectrum
        for layer in self.layers:
            x = layer(
                x,
                context=encoded_spectrum,
                context_mask=spectrum_mask,
            )
        
        # Return updated latent state (residual connection)
        return self.norm(x) + latent_z
    
    def answer_step(
        self,
        current_probs: Tensor,  # (batch, seq_len, vocab_size)
        latent_z: Tensor,       # (batch, seq_len, hidden)
    ) -> Tensor:
        """
        Answer update step: y <- g(y, z)
        
        Updates the prediction based on current prediction and latent state.
        
        Returns:
            (batch, seq_len, vocab_size) - updated logits
        """
        batch_size, seq_len = latent_z.shape[:2]
        
        # Combine current prediction and latent state
        y_embed = self.soft_embed(current_probs)
        
        positions = torch.arange(seq_len, device=latent_z.device)
        pos_embed = self.position_embedding(positions)
        
        x = y_embed + latent_z + pos_embed
        
        # Self-attention only (no cross-attention for answer step)
        for layer in self.layers:
            # Use self-attention only by passing x as both input and context
            x = layer(x, context=x)
        
        x = self.norm(x)
        
        # Project to logits
        return self.output_head(x)


class RecursiveCore(nn.Module):
    """
    Complete recursive reasoning core.
    
    Implements the full TRM recursive loop:
    For each supervision step:
        For n latent steps: z <- f(x, y, z)
        y <- g(y, z)
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        max_seq_len: int = 25,
        vocab_size: int = VOCAB_SIZE,
        num_latent_steps: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.num_latent_steps = num_latent_steps
        
        self.decoder = RecursiveDecoder(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            vocab_size=vocab_size,
            dropout=dropout,
        )
    
    def forward(
        self,
        encoded_spectrum: Tensor,
        spectrum_mask: Tensor,
        num_supervision_steps: int = 8,
        return_all_steps: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """
        Full recursive forward pass.
        
        Args:
            encoded_spectrum: (batch, num_peaks, hidden) - encoded spectrum
            spectrum_mask: (batch, num_peaks) - attention mask
            num_supervision_steps: Number of supervision steps (T in paper)
            return_all_steps: If True, return logits from all steps
            
        Returns:
            all_logits: (T, batch, seq_len, vocab) if return_all_steps else (batch, seq_len, vocab)
            final_z: (batch, seq_len, hidden) - final latent state
        """
        batch_size = encoded_spectrum.shape[0]
        device = encoded_spectrum.device
        
        # Initialize
        y_logits, z = self.decoder.get_initial_state(batch_size, device)
        
        all_logits = []
        
        for t in range(num_supervision_steps):
            # Convert logits to probabilities for soft embedding
            y_probs = torch.softmax(y_logits, dim=-1)
            
            # Latent reasoning: n steps of z <- f(x, y, z)
            for _ in range(self.num_latent_steps):
                z = self.decoder.latent_step(
                    encoded_spectrum, spectrum_mask, y_probs, z
                )
            
            # Answer update: y <- g(y, z)
            y_logits = self.decoder.answer_step(y_probs, z)
            
            if return_all_steps:
                all_logits.append(y_logits)
        
        if return_all_steps:
            return torch.stack(all_logits, dim=0), z  # (T, batch, seq, vocab)
        else:
            return y_logits, z
```

### 2.4 Full Model (`src/model/trm.py`)

```python
"""
Complete Tiny Recursive Model for peptide sequencing.
"""

import torch
import torch.nn as nn
from torch import Tensor
from dataclasses import dataclass
from typing import Optional

from .encoder import SpectrumEncoder
from .decoder import RecursiveCore
from ..constants import VOCAB_SIZE


@dataclass
class TRMConfig:
    """Model configuration."""
    hidden_dim: int = 256
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    num_heads: int = 4
    max_peaks: int = 100
    max_seq_len: int = 25
    max_mass: float = 2000.0
    vocab_size: int = VOCAB_SIZE
    num_supervision_steps: int = 8
    num_latent_steps: int = 6
    dropout: float = 0.1


class RecursivePeptideModel(nn.Module):
    """
    Tiny Recursive Model for de novo peptide sequencing.
    
    Architecture:
        1. Spectrum Encoder: Transformer encoder on peak embeddings
        2. Recursive Core: Shared weights, runs T supervision steps
        3. Output: Sequence logits at each step (for deep supervision)
    
    Key features:
        - Uses both mass AND intensity from spectrum peaks
        - Includes precursor mass as explicit input
        - Soft embeddings from probability distributions
        - Deep supervision at every step
    """
    
    def __init__(self, config: TRMConfig):
        super().__init__()
        self.config = config
        
        # Spectrum encoder
        self.encoder = SpectrumEncoder(
            hidden_dim=config.hidden_dim,
            num_layers=config.num_encoder_layers,
            num_heads=config.num_heads,
            max_peaks=config.max_peaks,
            max_mass=config.max_mass,
            dropout=config.dropout,
        )
        
        # Recursive decoder core
        self.recursive_core = RecursiveCore(
            hidden_dim=config.hidden_dim,
            num_layers=config.num_decoder_layers,
            num_heads=config.num_heads,
            max_seq_len=config.max_seq_len,
            vocab_size=config.vocab_size,
            num_latent_steps=config.num_latent_steps,
            dropout=config.dropout,
        )
    
    def forward(
        self,
        spectrum_masses: Tensor,
        spectrum_intensities: Tensor,
        spectrum_mask: Tensor,
        precursor_mass: Tensor,
        precursor_charge: Tensor,
        num_supervision_steps: Optional[int] = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Forward pass.
        
        Args:
            spectrum_masses: (batch, max_peaks) - peak m/z values
            spectrum_intensities: (batch, max_peaks) - peak intensities
            spectrum_mask: (batch, max_peaks) - True for real peaks
            precursor_mass: (batch,) - precursor mass
            precursor_charge: (batch,) - precursor charge state
            num_supervision_steps: Override default number of steps
            
        Returns:
            all_logits: (T, batch, max_seq_len, vocab_size) - logits at each step
            final_z: (batch, max_seq_len, hidden_dim) - final latent state
        """
        num_steps = num_supervision_steps or self.config.num_supervision_steps
        
        # Encode spectrum
        encoded_spectrum, full_mask = self.encoder(
            spectrum_masses,
            spectrum_intensities,
            spectrum_mask,
            precursor_mass,
            precursor_charge,
        )
        
        # Recursive refinement
        all_logits, final_z = self.recursive_core(
            encoded_spectrum,
            full_mask,
            num_supervision_steps=num_steps,
            return_all_steps=True,
        )
        
        return all_logits, final_z
    
    def predict(
        self,
        spectrum_masses: Tensor,
        spectrum_intensities: Tensor,
        spectrum_mask: Tensor,
        precursor_mass: Tensor,
        precursor_charge: Tensor,
    ) -> Tensor:
        """
        Inference: return only final predictions.
        
        Returns:
            logits: (batch, max_seq_len, vocab_size)
        """
        all_logits, _ = self.forward(
            spectrum_masses,
            spectrum_intensities,
            spectrum_mask,
            precursor_mass,
            precursor_charge,
        )
        return all_logits[-1]  # Return final step


def create_model(config: TRMConfig = None) -> RecursivePeptideModel:
    """Factory function to create model."""
    config = config or TRMConfig()
    return RecursivePeptideModel(config)
```

---

## Phase 3: Loss Functions and Metrics

### 3.1 Loss Functions (`src/training/losses.py`)

```python
"""
Loss functions including deep supervision and mass-matching auxiliary loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..constants import (
    AMINO_ACID_MASSES, WATER_MASS, VOCAB, AA_START_IDX, PAD_IDX
)


class DeepSupervisionLoss(nn.Module):
    """
    Cross-entropy loss summed over all supervision steps.
    
    This is the core TRM training signal - computing loss at every
    intermediate step forces the model to learn a trajectory of improvement.
    """
    
    def __init__(
        self,
        iteration_weights: str = 'linear',
        label_smoothing: float = 0.0,
    ):
        """
        Args:
            iteration_weights: How to weight different steps
                - 'uniform': All steps weighted equally
                - 'linear': Later steps weighted more (1, 2, 3, ...)
                - 'exponential': Exponential weighting toward later steps
            label_smoothing: Label smoothing for cross-entropy
        """
        super().__init__()
        self.iteration_weights = iteration_weights
        self.label_smoothing = label_smoothing
    
    def _get_weights(self, num_steps: int, device: torch.device) -> Tensor:
        """Get normalized weights for each step."""
        if self.iteration_weights == 'uniform':
            weights = torch.ones(num_steps, device=device)
        elif self.iteration_weights == 'linear':
            weights = torch.arange(1, num_steps + 1, dtype=torch.float, device=device)
        elif self.iteration_weights == 'exponential':
            weights = torch.exp(torch.arange(num_steps, dtype=torch.float, device=device) * 0.5)
        else:
            raise ValueError(f"Unknown weighting: {self.iteration_weights}")
        
        return weights / weights.sum()
    
    def forward(
        self,
        all_logits: Tensor,      # (T, batch, seq_len, vocab)
        targets: Tensor,         # (batch, seq_len)
        target_mask: Tensor,     # (batch, seq_len)
    ) -> tuple[Tensor, dict]:
        """
        Compute deep supervision loss.
        
        Returns:
            loss: Scalar loss value
            metrics: Dict with per-step losses for logging
        """
        num_steps = all_logits.shape[0]
        batch_size = all_logits.shape[1]
        device = all_logits.device
        
        weights = self._get_weights(num_steps, device)
        
        total_loss = 0.0
        step_losses = []
        
        for t in range(num_steps):
            logits = all_logits[t]  # (batch, seq_len, vocab)
            
            # Reshape for cross entropy
            logits_flat = logits.view(-1, logits.shape[-1])
            targets_flat = targets.view(-1)
            
            # Compute cross entropy with mask
            ce_loss = F.cross_entropy(
                logits_flat,
                targets_flat,
                ignore_index=PAD_IDX,
                label_smoothing=self.label_smoothing,
                reduction='none',
            )
            
            # Apply mask and average
            mask_flat = target_mask.view(-1).float()
            masked_loss = (ce_loss * mask_flat).sum() / mask_flat.sum().clamp(min=1)
            
            step_losses.append(masked_loss.item())
            total_loss = total_loss + weights[t] * masked_loss
        
        metrics = {
            f'ce_step_{t}': step_losses[t] for t in range(num_steps)
        }
        metrics['ce_final'] = step_losses[-1]
        
        return total_loss, metrics


class SpectrumMatchingLoss(nn.Module):
    """
    Auxiliary loss that compares predicted peptide's theoretical spectrum
    to the observed spectrum.
    
    This exploits the unique structure of peptide sequencing: unlike language
    modeling, we have an implicit verification oracle (mass constraints).
    
    The loss encourages predictions whose theoretical fragment masses
    match observed peaks.
    """
    
    def __init__(
        self,
        mass_tolerance: float = 0.5,   # Da, for matching window
        temperature: float = 0.1,       # For soft assignment
    ):
        super().__init__()
        self.mass_tolerance = mass_tolerance
        self.temperature = temperature
        
        # Register amino acid masses as buffer
        aa_masses = torch.tensor([
            AMINO_ACID_MASSES.get(aa, 0.0) for aa in VOCAB
        ])
        self.register_buffer('aa_masses', aa_masses)
    
    def compute_theoretical_peaks(
        self,
        sequence_probs: Tensor,  # (batch, seq_len, vocab)
    ) -> Tensor:
        """
        Compute expected theoretical peak masses from probability distribution.
        
        This is differentiable because we use expected mass:
        E[mass_i] = sum_aa P(aa_i) * mass(aa)
        
        Returns:
            (batch, num_theoretical_peaks) expected masses for b and y ions
        """
        batch_size, seq_len, vocab_size = sequence_probs.shape
        
        # Expected mass at each position
        # (batch, seq_len, vocab) @ (vocab,) -> (batch, seq_len)
        expected_masses = torch.einsum('bsv,v->bs', sequence_probs, self.aa_masses)
        
        # Compute b-ions: cumulative mass from N-terminus
        # b_i = sum of first i residues
        # Skip position 0 (SOS token) and last position (EOS token)
        residue_masses = expected_masses[:, 1:-1]  # (batch, seq_len - 2)
        
        # Cumulative sum for b-ions
        b_ions = torch.cumsum(residue_masses, dim=1)[:, :-1]  # b1 to b_{n-1}
        
        # Compute y-ions: cumulative mass from C-terminus + H2O
        y_ions = torch.flip(
            torch.cumsum(torch.flip(residue_masses, [1]), dim=1),
            [1]
        )[:, :-1] + WATER_MASS
        
        # Concatenate all theoretical peaks
        theoretical_peaks = torch.cat([b_ions, y_ions], dim=1)
        
        return theoretical_peaks
    
    def forward(
        self,
        sequence_probs: Tensor,       # (batch, seq_len, vocab)
        observed_masses: Tensor,      # (batch, max_peaks)
        observed_intensities: Tensor, # (batch, max_peaks)
        peak_mask: Tensor,            # (batch, max_peaks)
    ) -> Tensor:
        """
        Compute spectrum matching loss.
        
        For each theoretical peak, find the closest observed peak and penalize
        the distance. Weight by observed peak intensity (more intense = more reliable).
        """
        # Compute theoretical peaks from soft predictions
        theoretical = self.compute_theoretical_peaks(sequence_probs)  # (batch, num_theo)
        
        # Compute pairwise distances between theoretical and observed peaks
        # theoretical: (batch, num_theo, 1)
        # observed: (batch, 1, max_peaks)
        theo_expanded = theoretical.unsqueeze(-1)  # (batch, num_theo, 1)
        obs_expanded = observed_masses.unsqueeze(1)  # (batch, 1, max_peaks)
        
        # Absolute distance: (batch, num_theo, max_peaks)
        distances = torch.abs(theo_expanded - obs_expanded)
        
        # Soft assignment: which observed peak matches each theoretical peak?
        # Use softmin over distances
        scores = -distances / self.temperature
        
        # Mask out padding peaks
        mask_expanded = peak_mask.unsqueeze(1).float()  # (batch, 1, max_peaks)
        scores = scores.masked_fill(~peak_mask.unsqueeze(1), float('-inf'))
        
        # Soft assignment weights
        soft_assignment = F.softmax(scores, dim=-1)  # (batch, num_theo, max_peaks)
        
        # Expected distance for each theoretical peak (weighted by soft assignment)
        matched_distances = (soft_assignment * distances).sum(dim=-1)  # (batch, num_theo)
        
        # Weight by intensity of matched peaks
        intensity_weights = (soft_assignment * observed_intensities.unsqueeze(1)).sum(dim=-1)
        
        # Average over theoretical peaks
        loss = (matched_distances * intensity_weights).mean()
        
        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss with deep supervision and spectrum matching.
    """
    
    def __init__(
        self,
        ce_weight: float = 1.0,
        spectrum_weight: float = 0.1,
        iteration_weights: str = 'linear',
        label_smoothing: float = 0.0,
        mass_tolerance: float = 0.5,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.spectrum_weight = spectrum_weight
        
        self.ce_loss = DeepSupervisionLoss(
            iteration_weights=iteration_weights,
            label_smoothing=label_smoothing,
        )
        
        self.spectrum_loss = SpectrumMatchingLoss(
            mass_tolerance=mass_tolerance,
        )
    
    def forward(
        self,
        all_logits: Tensor,
        targets: Tensor,
        target_mask: Tensor,
        observed_masses: Tensor,
        observed_intensities: Tensor,
        peak_mask: Tensor,
    ) -> tuple[Tensor, dict]:
        """
        Compute combined loss.
        
        Returns:
            total_loss: Scalar loss
            metrics: Dict with component losses
        """
        # Cross-entropy with deep supervision
        ce_loss, ce_metrics = self.ce_loss(all_logits, targets, target_mask)
        
        # Spectrum matching on final prediction
        final_probs = F.softmax(all_logits[-1], dim=-1)
        spec_loss = self.spectrum_loss(
            final_probs,
            observed_masses,
            observed_intensities,
            peak_mask,
        )
        
        # Combine
        total_loss = self.ce_weight * ce_loss + self.spectrum_weight * spec_loss
        
        metrics = {
            **ce_metrics,
            'spectrum_loss': spec_loss.item(),
            'total_loss': total_loss.item(),
        }
        
        return total_loss, metrics
```

### 3.2 Metrics (`src/training/metrics.py`)

```python
"""
Evaluation metrics for peptide sequencing.
"""

import torch
from torch import Tensor
from typing import Optional

from ..constants import PAD_IDX, EOS_IDX, IDX_TO_AA, AMINO_ACID_MASSES, WATER_MASS


def token_accuracy(
    logits: Tensor,
    targets: Tensor,
    mask: Tensor,
) -> float:
    """
    Per-amino-acid accuracy (ignoring padding).
    
    Args:
        logits: (batch, seq_len, vocab)
        targets: (batch, seq_len)
        mask: (batch, seq_len)
    """
    predictions = logits.argmax(dim=-1)
    correct = (predictions == targets) & mask
    return correct.sum().item() / mask.sum().item()


def sequence_accuracy(
    logits: Tensor,
    targets: Tensor,
    mask: Tensor,
) -> float:
    """
    Fraction of perfectly reconstructed sequences.
    """
    predictions = logits.argmax(dim=-1)
    
    # For each sequence, check if all masked positions match
    correct = (predictions == targets) | ~mask
    all_correct = correct.all(dim=-1)
    
    return all_correct.float().mean().item()


def decode_sequence(indices: Tensor, mask: Optional[Tensor] = None) -> str:
    """Convert token indices to amino acid string."""
    seq = []
    for i, idx in enumerate(indices.tolist()):
        if mask is not None and not mask[i]:
            break
        if idx == EOS_IDX:
            break
        if idx >= len(IDX_TO_AA):
            continue
        aa = IDX_TO_AA[idx]
        if aa not in ['<PAD>', '<SOS>', '<EOS>', '<UNK>']:
            seq.append(aa)
    return ''.join(seq)


def compute_mass_error(
    predicted_sequence: str,
    target_mass: float,
) -> float:
    """
    Compute mass error between predicted sequence and target precursor mass.
    
    Returns error in Daltons.
    """
    if not predicted_sequence:
        return float('inf')
    
    predicted_mass = sum(
        AMINO_ACID_MASSES.get(aa, 0) for aa in predicted_sequence
    ) + WATER_MASS
    
    return abs(predicted_mass - target_mass)


def compute_mass_error_ppm(
    predicted_sequence: str,
    target_mass: float,
) -> float:
    """Compute mass error in parts per million."""
    error_da = compute_mass_error(predicted_sequence, target_mass)
    return (error_da / target_mass) * 1e6


def compute_metrics(
    logits: Tensor,
    targets: Tensor,
    mask: Tensor,
    precursor_masses: Optional[Tensor] = None,
) -> dict:
    """
    Compute all metrics for a batch.
    
    Returns:
        dict with token_acc, seq_acc, and optionally mass_error
    """
    metrics = {
        'token_accuracy': token_accuracy(logits, targets, mask),
        'sequence_accuracy': sequence_accuracy(logits, targets, mask),
    }
    
    if precursor_masses is not None:
        mass_errors = []
        predictions = logits.argmax(dim=-1)
        
        for i in range(len(predictions)):
            pred_seq = decode_sequence(predictions[i], mask[i])
            target_mass = precursor_masses[i].item()
            error = compute_mass_error_ppm(pred_seq, target_mass)
            if error < float('inf'):
                mass_errors.append(error)
        
        if mass_errors:
            metrics['mean_mass_error_ppm'] = sum(mass_errors) / len(mass_errors)
    
    return metrics
```

---

## Phase 4: Curriculum Learning

### 4.1 Curriculum Scheduler (`src/training/curriculum.py`)

```python
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
    epochs: int
    
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


DEFAULT_CURRICULUM = [
    # Stage 1: Easy - short peptides, clean spectra
    CurriculumStage(
        name="warmup",
        epochs=10000,
        min_length=7,
        max_length=10,
        noise_peaks=0,
        peak_dropout=0.0,
        mass_error_ppm=0,
        spectrum_loss_weight=0.0,  # Pure CE first
    ),
    
    # Stage 2: Introduce spectrum matching loss
    CurriculumStage(
        name="add_spectrum_loss",
        epochs=20000,
        min_length=7,
        max_length=12,
        noise_peaks=0,
        peak_dropout=0.0,
        mass_error_ppm=0,
        spectrum_loss_weight=0.1,
    ),
    
    # Stage 3: Longer peptides
    CurriculumStage(
        name="longer_peptides",
        epochs=30000,
        min_length=8,
        max_length=16,
        noise_peaks=0,
        peak_dropout=0.0,
        mass_error_ppm=0,
        spectrum_loss_weight=0.1,
    ),
    
    # Stage 4: Add missing peaks (simulate real conditions)
    CurriculumStage(
        name="missing_peaks",
        epochs=40000,
        min_length=8,
        max_length=18,
        noise_peaks=0,
        peak_dropout=0.2,  # 20% peaks randomly dropped
        mass_error_ppm=0,
        spectrum_loss_weight=0.15,
    ),
    
    # Stage 5: Add noise peaks
    CurriculumStage(
        name="noisy",
        epochs=50000,
        min_length=7,
        max_length=20,
        noise_peaks=10,
        peak_dropout=0.2,
        mass_error_ppm=0,
        spectrum_loss_weight=0.15,
    ),
    
    # Stage 6: Add mass error (most realistic)
    CurriculumStage(
        name="realistic_synthetic",
        epochs=50000,
        min_length=7,
        max_length=20,
        noise_peaks=15,
        peak_dropout=0.3,
        mass_error_ppm=20,  # Typical high-res MS error
        intensity_variation=0.3,
        spectrum_loss_weight=0.2,
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
        self._cumulative_epochs = self._compute_cumulative_epochs()
    
    def _compute_cumulative_epochs(self) -> list[int]:
        """Compute cumulative epoch counts for stage boundaries."""
        cumulative = []
        total = 0
        for stage in self.stages:
            total += stage.epochs
            cumulative.append(total)
        return cumulative
    
    @property
    def current_stage(self) -> Optional[CurriculumStage]:
        """Get current curriculum stage."""
        if 0 <= self.current_stage_idx < len(self.stages):
            return self.stages[self.current_stage_idx]
        return None
    
    @property
    def total_epochs(self) -> int:
        """Total epochs across all stages."""
        return self._cumulative_epochs[-1] if self._cumulative_epochs else 0
    
    def step(self, epoch: int) -> bool:
        """
        Update curriculum based on current epoch.
        
        Returns:
            True if stage changed, False otherwise
        """
        # Find which stage we should be in
        new_stage_idx = 0
        for i, boundary in enumerate(self._cumulative_epochs):
            if epoch < boundary:
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
                f"Curriculum: Advanced to stage '{stage.name}' at epoch {epoch}\n"
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
```

---

## Phase 5: Training Loop

### 5.1 Trainer (`src/training/trainer.py`)

```python
"""
Main training loop with curriculum learning and logging.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
import logging
from typing import Optional
from dataclasses import dataclass
from copy import deepcopy

from ..model.trm import RecursivePeptideModel, TRMConfig
from ..data.dataset import SyntheticPeptideDataset, collate_peptide_samples
from .losses import CombinedLoss
from .metrics import compute_metrics
from .curriculum import CurriculumScheduler, DEFAULT_CURRICULUM

log = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 64
    max_epochs: int = 200000
    warmup_epochs: int = 1000
    
    # Loss weights
    ce_weight: float = 1.0
    spectrum_weight: float = 0.1  # Will be overridden by curriculum
    iteration_weights: str = 'linear'
    label_smoothing: float = 0.0
    
    # EMA (important for TRM stability)
    use_ema: bool = True
    ema_decay: float = 0.999
    
    # Logging
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 10000
    
    # Paths
    checkpoint_dir: str = 'checkpoints'
    
    # Curriculum
    use_curriculum: bool = True


class EMA:
    """Exponential Moving Average of model parameters."""
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad = False
    
    @torch.no_grad()
    def update(self, model: nn.Module):
        for ema_p, model_p in zip(self.shadow.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1 - self.decay)
    
    def state_dict(self):
        return self.shadow.state_dict()
    
    def load_state_dict(self, state_dict):
        self.shadow.load_state_dict(state_dict)


class Trainer:
    """
    Main trainer class.
    """
    
    def __init__(
        self,
        model: RecursivePeptideModel,
        train_dataset: SyntheticPeptideDataset,
        config: TrainingConfig,
        val_dataset: Optional[SyntheticPeptideDataset] = None,
    ):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Datasets and loaders
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            collate_fn=collate_peptide_samples,
            num_workers=4,
            pin_memory=True,
        )
        
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                collate_fn=collate_peptide_samples,
            )
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.max_epochs,
            eta_min=config.learning_rate * 0.01,
        )
        
        # Loss
        self.loss_fn = CombinedLoss(
            ce_weight=config.ce_weight,
            spectrum_weight=config.spectrum_weight,
            iteration_weights=config.iteration_weights,
            label_smoothing=config.label_smoothing,
        )
        
        # Curriculum
        self.curriculum = None
        if config.use_curriculum:
            self.curriculum = CurriculumScheduler(
                stages=DEFAULT_CURRICULUM,
                dataset=train_dataset,
            )
        
        # EMA
        self.ema = None
        if config.use_ema:
            self.ema = EMA(model, decay=config.ema_decay)
        
        # Tracking
        self.epoch = 0
        self.global_step = 0
        self.best_val_acc = 0.0
        
        # Checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self):
        """Train for one epoch (actually one batch in infinite dataset)."""
        self.model.train()
        
        batch = next(iter(self.train_loader))
        
        # Move to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Forward pass
        all_logits, _ = self.model(
            spectrum_masses=batch['spectrum_masses'],
            spectrum_intensities=batch['spectrum_intensities'],
            spectrum_mask=batch['spectrum_mask'],
            precursor_mass=batch['precursor_mass'],
            precursor_charge=batch['precursor_charge'],
        )
        
        # Update spectrum loss weight from curriculum
        if self.curriculum:
            self.loss_fn.spectrum_weight = self.curriculum.get_spectrum_loss_weight()
        
        # Compute loss
        loss, metrics = self.loss_fn(
            all_logits=all_logits,
            targets=batch['sequence'],
            target_mask=batch['sequence_mask'],
            observed_masses=batch['spectrum_masses'],
            observed_intensities=batch['spectrum_intensities'],
            peak_mask=batch['spectrum_mask'],
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        self.scheduler.step()
        
        # Update EMA
        if self.ema:
            self.ema.update(self.model)
        
        # Compute accuracy metrics
        with torch.no_grad():
            acc_metrics = compute_metrics(
                all_logits[-1],
                batch['sequence'],
                batch['sequence_mask'],
                batch['precursor_mass'],
            )
        
        metrics.update(acc_metrics)
        metrics['lr'] = self.scheduler.get_last_lr()[0]
        
        return metrics
    
    @torch.no_grad()
    def evaluate(self):
        """Evaluate on validation set."""
        if self.val_dataset is None:
            return {}
        
        model = self.ema.shadow if self.ema else self.model
        model.eval()
        
        total_correct = 0
        total_tokens = 0
        total_sequences = 0
        correct_sequences = 0
        
        for batch in self.val_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            logits = model.predict(
                spectrum_masses=batch['spectrum_masses'],
                spectrum_intensities=batch['spectrum_intensities'],
                spectrum_mask=batch['spectrum_mask'],
                precursor_mass=batch['precursor_mass'],
                precursor_charge=batch['precursor_charge'],
            )
            
            predictions = logits.argmax(dim=-1)
            mask = batch['sequence_mask']
            targets = batch['sequence']
            
            # Token accuracy
            correct = (predictions == targets) & mask
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()
            
            # Sequence accuracy
            all_correct = ((predictions == targets) | ~mask).all(dim=-1)
            correct_sequences += all_correct.sum().item()
            total_sequences += len(predictions)
        
        return {
            'val_token_accuracy': total_correct / max(total_tokens, 1),
            'val_sequence_accuracy': correct_sequences / max(total_sequences, 1),
        }
    
    def save_checkpoint(self, name: str = None):
        """Save model checkpoint."""
        name = name or f'checkpoint_epoch_{self.epoch}.pt'
        path = self.checkpoint_dir / name
        
        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
        }
        
        if self.ema:
            state['ema_state_dict'] = self.ema.state_dict()
        
        torch.save(state, path)
        log.info(f"Saved checkpoint: {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        state = torch.load(path, map_location=self.device)
        
        self.epoch = state['epoch']
        self.global_step = state['global_step']
        self.model.load_state_dict(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        self.scheduler.load_state_dict(state['scheduler_state_dict'])
        self.best_val_acc = state.get('best_val_acc', 0.0)
        
        if self.ema and 'ema_state_dict' in state:
            self.ema.load_state_dict(state['ema_state_dict'])
        
        log.info(f"Loaded checkpoint from epoch {self.epoch}")
    
    def train(self):
        """Main training loop."""
        log.info(f"Starting training on {self.device}")
        log.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        while self.epoch < self.config.max_epochs:
            # Update curriculum
            if self.curriculum:
                self.curriculum.step(self.epoch)
            
            # Train step
            metrics = self.train_epoch()
            self.global_step += 1
            self.epoch += 1
            
            # Logging
            if self.epoch % self.config.log_interval == 0:
                log.info(
                    f"Epoch {self.epoch} | "
                    f"Loss: {metrics['total_loss']:.4f} | "
                    f"Token Acc: {metrics['token_accuracy']:.3f} | "
                    f"Seq Acc: {metrics['sequence_accuracy']:.3f} | "
                    f"LR: {metrics['lr']:.2e}"
                )
            
            # Evaluation
            if self.epoch % self.config.eval_interval == 0:
                val_metrics = self.evaluate()
                if val_metrics:
                    log.info(
                        f"Validation | "
                        f"Token Acc: {val_metrics['val_token_accuracy']:.3f} | "
                        f"Seq Acc: {val_metrics['val_sequence_accuracy']:.3f}"
                    )
                    
                    # Save best model
                    if val_metrics['val_sequence_accuracy'] > self.best_val_acc:
                        self.best_val_acc = val_metrics['val_sequence_accuracy']
                        self.save_checkpoint('best_model.pt')
            
            # Periodic checkpoint
            if self.epoch % self.config.save_interval == 0:
                self.save_checkpoint()
        
        # Final save
        self.save_checkpoint('final_model.pt')
        log.info("Training complete!")
```

---

## Phase 6: Inference with Uncertainty

### 6.1 Prediction with Uncertainty (`src/inference/predict.py`)

```python
"""
Inference with uncertainty quantification and ambiguity detection.
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from dataclasses import dataclass
from typing import Optional

from ..model.trm import RecursivePeptideModel
from ..constants import (
    VOCAB, IDX_TO_AA, AMINO_ACID_MASSES, WATER_MASS,
    ISOBARIC_GROUPS, EOS_IDX, PAD_IDX
)
from ..data.synthetic import generate_theoretical_spectrum


@dataclass
class PeptideHypothesis:
    """Single peptide prediction with confidence."""
    sequence: str                      # Amino acid sequence
    confidence: float                  # Overall confidence (mean of position probs)
    position_confidences: list[float]  # Per-position confidence
    mass_error_da: float               # Difference from precursor mass (Da)
    mass_error_ppm: float              # Mass error in ppm
    spectrum_match_score: float        # How well theoretical matches observed


@dataclass
class SequencingResult:
    """Full output from the model."""
    top_k_hypotheses: list[PeptideHypothesis]
    
    # Per-position information
    position_distributions: Tensor      # (seq_len, vocab_size)
    uncertain_positions: list[int]      # Positions with high entropy
    isobaric_positions: list[int]       # I/L or K/Q ambiguities
    potential_ptm_positions: list[int]  # Mass gaps suggesting modification
    
    # Mass analysis
    unexplained_peaks: list[float]      # Observed peaks not matched
    unexplained_mass: float             # Precursor minus prediction
    
    def to_string(self) -> str:
        """Human-readable representation."""
        if not self.top_k_hypotheses:
            return "No predictions"
        
        top = self.top_k_hypotheses[0]
        seq_chars = list(top.sequence)
        
        # Mark ambiguous positions
        for pos in self.isobaric_positions:
            if pos < len(seq_chars):
                aa = seq_chars[pos]
                if aa in 'IL':
                    seq_chars[pos] = '[I/L]'
                elif aa in 'KQ':
                    seq_chars[pos] = '[K/Q]'
        
        result = ''.join(seq_chars)
        result += f"\nConfidence: {top.confidence:.2%}"
        result += f"\nMass error: {top.mass_error_ppm:.1f} ppm"
        
        if self.potential_ptm_positions:
            result += f"\nPossible PTMs at positions: {self.potential_ptm_positions}"
        
        if abs(self.unexplained_mass) > 10:
            result += f"\nUnexplained mass: {self.unexplained_mass:.1f} Da"
        
        return result


def compute_entropy(probs: Tensor, dim: int = -1) -> Tensor:
    """Compute entropy of probability distribution."""
    log_probs = torch.log(probs + 1e-10)
    return -(probs * log_probs).sum(dim=dim)


def detect_isobaric_ambiguity(
    position_probs: Tensor,
    threshold: float = 0.2,
) -> list[str]:
    """Detect if position has ambiguity between isobaric amino acids."""
    ambiguities = []
    
    for group_name, group_aas in ISOBARIC_GROUPS.items():
        group_probs = []
        for aa in group_aas:
            if aa in VOCAB:
                idx = VOCAB.index(aa)
                group_probs.append(position_probs[idx].item())
        
        if len(group_probs) >= 2 and all(p > threshold for p in group_probs):
            ambiguities.append(group_name)
    
    return ambiguities


def compute_spectrum_match_score(
    predicted_sequence: str,
    observed_masses: list[float],
    observed_intensities: list[float],
    tolerance_da: float = 0.5,
) -> float:
    """
    Compute how well predicted sequence's theoretical spectrum matches observed.
    
    Returns fraction of observed peaks that are explained by prediction.
    """
    if not predicted_sequence:
        return 0.0
    
    # Generate theoretical spectrum
    theoretical = generate_theoretical_spectrum(predicted_sequence)
    theoretical_masses = set(m for m, _ in theoretical.peaks)
    
    # Count explained peaks (weighted by intensity)
    explained_intensity = 0.0
    total_intensity = sum(observed_intensities)
    
    for mass, intensity in zip(observed_masses, observed_intensities):
        for theo_mass in theoretical_masses:
            if abs(mass - theo_mass) < tolerance_da:
                explained_intensity += intensity
                break
    
    return explained_intensity / max(total_intensity, 1e-10)


class PeptidePredictor:
    """
    Inference wrapper with uncertainty quantification.
    """
    
    def __init__(
        self,
        model: RecursivePeptideModel,
        device: torch.device = None,
        entropy_threshold: float = 1.5,  # For flagging uncertain positions
        isobaric_threshold: float = 0.2, # For detecting I/L, K/Q ambiguity
    ):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.entropy_threshold = entropy_threshold
        self.isobaric_threshold = isobaric_threshold
    
    @torch.no_grad()
    def predict(
        self,
        spectrum_masses: Tensor,
        spectrum_intensities: Tensor,
        spectrum_mask: Tensor,
        precursor_mass: Tensor,
        precursor_charge: Tensor,
        top_k: int = 5,
    ) -> SequencingResult:
        """
        Full prediction with uncertainty quantification.
        """
        # Ensure batch dimension
        if spectrum_masses.dim() == 1:
            spectrum_masses = spectrum_masses.unsqueeze(0)
            spectrum_intensities = spectrum_intensities.unsqueeze(0)
            spectrum_mask = spectrum_mask.unsqueeze(0)
            precursor_mass = precursor_mass.unsqueeze(0)
            precursor_charge = precursor_charge.unsqueeze(0)
        
        # Move to device
        spectrum_masses = spectrum_masses.to(self.device)
        spectrum_intensities = spectrum_intensities.to(self.device)
        spectrum_mask = spectrum_mask.to(self.device)
        precursor_mass = precursor_mass.to(self.device)
        precursor_charge = precursor_charge.to(self.device)
        
        # Get predictions
        logits = self.model.predict(
            spectrum_masses,
            spectrum_intensities,
            spectrum_mask,
            precursor_mass,
            precursor_charge,
        )
        
        probs = F.softmax(logits, dim=-1)[0]  # (seq_len, vocab)
        
        # Decode top-k hypotheses using beam search
        hypotheses = self._beam_search(probs, precursor_mass[0].item(), top_k)
        
        # Analyze uncertainty
        entropy = compute_entropy(probs, dim=-1)  # (seq_len,)
        uncertain_positions = (entropy > self.entropy_threshold).nonzero().squeeze(-1).tolist()
        if isinstance(uncertain_positions, int):
            uncertain_positions = [uncertain_positions]
        
        # Detect isobaric ambiguities
        isobaric_positions = []
        for pos in range(probs.shape[0]):
            ambiguities = detect_isobaric_ambiguity(probs[pos], self.isobaric_threshold)
            if ambiguities:
                isobaric_positions.append(pos)
        
        # Compute spectrum match scores
        observed_masses_list = spectrum_masses[0][spectrum_mask[0]].cpu().tolist()
        observed_intensities_list = spectrum_intensities[0][spectrum_mask[0]].cpu().tolist()
        
        for hyp in hypotheses:
            hyp.spectrum_match_score = compute_spectrum_match_score(
                hyp.sequence,
                observed_masses_list,
                observed_intensities_list,
            )
        
        # Analyze unexplained mass
        if hypotheses:
            predicted_mass = sum(
                AMINO_ACID_MASSES.get(aa, 0) for aa in hypotheses[0].sequence
            ) + WATER_MASS
            unexplained_mass = precursor_mass[0].item() - predicted_mass
        else:
            unexplained_mass = 0.0
        
        # TODO: Detect potential PTM positions based on mass gaps
        potential_ptm_positions = []
        
        return SequencingResult(
            top_k_hypotheses=hypotheses,
            position_distributions=probs.cpu(),
            uncertain_positions=uncertain_positions,
            isobaric_positions=isobaric_positions,
            potential_ptm_positions=potential_ptm_positions,
            unexplained_peaks=[],  # TODO: implement
            unexplained_mass=unexplained_mass,
        )
    
    def _beam_search(
        self,
        probs: Tensor,
        precursor_mass: float,
        beam_width: int,
    ) -> list[PeptideHypothesis]:
        """Simple beam search for top-k hypotheses."""
        # For now, just use greedy + variants
        # TODO: Implement proper beam search
        
        hypotheses = []
        
        # Greedy decoding
        greedy_indices = probs.argmax(dim=-1)
        greedy_seq = self._decode_indices(greedy_indices)
        greedy_probs = probs.max(dim=-1).values
        
        if greedy_seq:
            predicted_mass = sum(AMINO_ACID_MASSES.get(aa, 0) for aa in greedy_seq) + WATER_MASS
            mass_error_da = abs(predicted_mass - precursor_mass)
            mass_error_ppm = (mass_error_da / precursor_mass) * 1e6
            
            hypotheses.append(PeptideHypothesis(
                sequence=greedy_seq,
                confidence=greedy_probs.mean().item(),
                position_confidences=greedy_probs.tolist(),
                mass_error_da=mass_error_da,
                mass_error_ppm=mass_error_ppm,
                spectrum_match_score=0.0,  # Filled in later
            ))
        
        return hypotheses[:beam_width]
    
    def _decode_indices(self, indices: Tensor) -> str:
        """Convert token indices to sequence string."""
        seq = []
        for idx in indices.tolist():
            if idx == EOS_IDX:
                break
            if idx == PAD_IDX:
                continue
            if idx < len(VOCAB):
                aa = VOCAB[idx]
                if aa not in ['<PAD>', '<SOS>', '<EOS>', '<UNK>']:
                    seq.append(aa)
        return ''.join(seq)
```

---

## Phase 7: Configuration Files

### 7.1 Default Configuration (`configs/default.yaml`)

```yaml
# RecursivePeptide Default Configuration

model:
  hidden_dim: 256
  num_encoder_layers: 2
  num_decoder_layers: 2
  num_heads: 4
  max_peaks: 100
  max_seq_len: 25
  max_mass: 2000.0
  vocab_size: 24  # 20 AA + 4 special tokens
  num_supervision_steps: 8
  num_latent_steps: 6
  dropout: 0.1

data:
  # Peptide generation
  min_length: 7
  max_length: 20
  
  # Spectrum generation
  ion_types: ['b', 'y']
  include_neutral_losses: false
  
  # Initial difficulty (overridden by curriculum)
  noise_peaks: 0
  peak_dropout: 0.0
  mass_error_ppm: 0.0
  intensity_variation: 0.0
  
  # Charge distribution
  charge_distribution:
    2: 0.7
    3: 0.25
    4: 0.05

training:
  # Optimization
  learning_rate: 1.0e-4
  weight_decay: 0.01
  batch_size: 64
  max_epochs: 200000
  warmup_epochs: 1000
  
  # Loss weights
  ce_weight: 1.0
  spectrum_weight: 0.1
  iteration_weights: 'linear'
  label_smoothing: 0.0
  
  # EMA
  use_ema: true
  ema_decay: 0.999
  
  # Logging
  log_interval: 100
  eval_interval: 1000
  save_interval: 10000
  
  # Curriculum
  use_curriculum: true
  
  # Paths
  checkpoint_dir: 'checkpoints'

inference:
  entropy_threshold: 1.5
  isobaric_threshold: 0.2
  beam_width: 5
  mass_tolerance_da: 0.5
```

### 7.2 Finetuning Configuration (`configs/finetune_real.yaml`)

```yaml
# Configuration for finetuning on real data

# Inherit from default
defaults:
  - default

# Override for finetuning
training:
  # Lower learning rate for finetuning
  learning_rate: 1.0e-5
  
  # Shorter training
  max_epochs: 50000
  
  # Disable curriculum (real data has fixed difficulty)
  use_curriculum: false
  
  # Lower spectrum loss weight (real spectra are noisier)
  spectrum_weight: 0.05
  
  checkpoint_dir: 'checkpoints/finetune'

# Finetuning-specific settings
finetune:
  # Start from synthetic-trained checkpoint
  pretrained_checkpoint: 'checkpoints/synthetic_final.pt'
  
  # Freeze encoder initially
  freeze_encoder_epochs: 5
  
  # Mix real and synthetic data to prevent forgetting
  data_mix:
    real: 0.8
    synthetic: 0.2
  
  # Only use high-confidence labels
  min_search_score: 0.95

# Real data sources
real_data:
  nine_species:
    path: 'data/real/nine_species'
    identifier: 'MSV000081382'
    species: ['Human', 'Mouse', 'Yeast', 'M.mazei', 'Honeybee', 
              'Tomato', 'RiceBean', 'Bacillus', 'Clam']
```

---

## Testing Requirements

### Critical Tests (`tests/test_physics.py`)

```python
"""
CRITICAL: Physics tests - run these FIRST before any model development.

These tests verify that mass calculations are correct. If these fail,
all downstream results will be meaningless.
"""

import pytest
import torch
from src.constants import AMINO_ACID_MASSES, WATER_MASS
from src.data.synthetic import (
    compute_peptide_mass,
    generate_theoretical_spectrum,
    generate_random_peptide,
)


class TestMassCalculations:
    """Test basic mass calculations."""
    
    def test_single_amino_acid_masses(self):
        """Verify individual amino acid masses against known values."""
        # Reference values from UniProt
        known_masses = {
            'G': 57.02146,
            'A': 71.03711,
            'S': 87.03203,
            'P': 97.05276,
            'V': 99.06841,
        }
        
        for aa, expected in known_masses.items():
            assert abs(AMINO_ACID_MASSES[aa] - expected) < 0.001, \
                f"Mass of {aa} is wrong: {AMINO_ACID_MASSES[aa]} vs {expected}"
    
    def test_peptide_mass(self):
        """Test peptide mass calculation against known values."""
        # PEPTIDE: Calculate manually
        # P(97.05) + E(129.04) + P(97.05) + T(101.05) + I(113.08) + D(115.03) + E(129.04)
        # = 781.34 + H2O(18.01) = 799.35 Da
        expected = 799.35
        calculated = compute_peptide_mass("PEPTIDE")
        assert abs(calculated - expected) < 0.1, \
            f"PEPTIDE mass wrong: {calculated} vs {expected}"
    
    def test_glycine_polymer(self):
        """Test with simple glycine polymer."""
        # 5 glycines + water
        expected = 5 * 57.02146 + WATER_MASS
        calculated = compute_peptide_mass("GGGGG")
        assert abs(calculated - expected) < 0.01


class TestTheoreticalSpectrum:
    """Test theoretical spectrum generation."""
    
    def test_b_ion_series(self):
        """Verify b-ion masses."""
        spectrum = generate_theoretical_spectrum("AG", ion_types=['b'])
        
        # b1 = A = 71.04
        b_ions = [(m, a) for m, a in zip(
            [p[0] for p in spectrum.peaks],
            [spectrum.ion_annotations.get(p[0], '') for p in spectrum.peaks]
        ) if 'b' in a]
        
        assert len(b_ions) >= 1
        # b1 should be mass of A
        assert abs(b_ions[0][0] - AMINO_ACID_MASSES['A']) < 0.01
    
    def test_y_ion_series(self):
        """Verify y-ion masses (include water)."""
        spectrum = generate_theoretical_spectrum("AG", ion_types=['y'])
        
        y_ions = [(m, a) for m, a in zip(
            [p[0] for p in spectrum.peaks],
            [spectrum.ion_annotations.get(p[0], '') for p in spectrum.peaks]
        ) if 'y' in a]
        
        assert len(y_ions) >= 1
        # y1 = G + H2O
        expected_y1 = AMINO_ACID_MASSES['G'] + WATER_MASS
        assert abs(y_ions[0][0] - expected_y1) < 0.01
    
    def test_precursor_mass(self):
        """Verify precursor mass calculation."""
        spectrum = generate_theoretical_spectrum("ACDEF")
        calculated_precursor = compute_peptide_mass("ACDEF")
        
        assert abs(spectrum.precursor_mass - calculated_precursor) < 0.01
    
    def test_spectrum_has_peaks(self):
        """Ensure spectrum generation produces peaks."""
        spectrum = generate_theoretical_spectrum("PEPTIDE")
        
        assert len(spectrum.peaks) > 0
        # For a 7-residue peptide with b and y ions, expect ~12 peaks
        assert len(spectrum.peaks) >= 10
    
    def test_peak_dropout(self):
        """Test that peak dropout reduces number of peaks."""
        full_spectrum = generate_theoretical_spectrum("PEPTIDE", peak_dropout=0.0)
        dropped_spectrum = generate_theoretical_spectrum("PEPTIDE", peak_dropout=0.5)
        
        # With 50% dropout, expect roughly half the peaks (with some variance)
        assert len(dropped_spectrum.peaks) < len(full_spectrum.peaks)
    
    def test_noise_peaks(self):
        """Test that noise peaks are added."""
        clean_spectrum = generate_theoretical_spectrum("PEP", noise_peaks=0)
        noisy_spectrum = generate_theoretical_spectrum("PEP", noise_peaks=10)
        
        assert len(noisy_spectrum.peaks) > len(clean_spectrum.peaks)


class TestRandomPeptide:
    """Test random peptide generation."""
    
    def test_length_constraints(self):
        """Test length constraints."""
        for _ in range(100):
            peptide = generate_random_peptide(min_length=7, max_length=10)
            assert 7 <= len(peptide) <= 10
    
    def test_valid_amino_acids(self):
        """Test that only valid amino acids are generated."""
        peptide = generate_random_peptide()
        for aa in peptide:
            assert aa in AMINO_ACID_MASSES, f"Invalid amino acid: {aa}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

---

## Development Timeline

| Week | Phase | Deliverables |
|------|-------|--------------|
| 1-2 | Data & Physics | `constants.py`, `synthetic.py`, `encoding.py`, `test_physics.py` (all passing) |
| 3-4 | Core Model | `layers.py`, `encoder.py`, `decoder.py`, `trm.py`, basic forward pass working |
| 5-6 | Training | `losses.py`, `metrics.py`, `trainer.py`, train on clean synthetic |
| 7 | Curriculum | `curriculum.py`, progressive difficulty training |
| 8 | Mass-Matching | Spectrum matching loss, ablation studies |
| 9 | Baselines | Non-recursive baseline (T=1), comparison experiments |
| 10-11 | Real Data | Download benchmarks, finetuning experiments |
| 12 | Uncertainty | `predict.py`, beam search, ambiguity detection |
| 13-14 | Evaluation | Full benchmark comparison, visualization, documentation |

---

## Technical Requirements

### Dependencies (`requirements.txt`)

```
torch>=2.0.0
einops>=0.6.0
hydra-core>=1.3.0
omegaconf>=2.3.0
wandb>=0.15.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
pytest>=7.3.0
pyteomics>=4.5.0  # For reading real MS data
```

### Hardware Requirements

- Minimum: 1 GPU with 8GB VRAM (for small model, batch_size=32)
- Recommended: 1 GPU with 24GB VRAM (for full experiments)
- Training time estimate: ~24-48 hours for full curriculum on single GPU

---

## Key Design Decisions Summary

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| Spectrum input | Mass + Intensity | Intensity provides signal reliability information |
| Precursor | Explicit input | Critical constraint for valid predictions |
| Sequence representation | Soft embeddings | Allows gradient flow through probability distributions |
| Latent initialization | Learned parameter | More flexible than zeros |
| Deep supervision | All steps | Primary driver of TRM performance |
| Mass-matching loss | Auxiliary signal | Exploits domain-specific physical constraints |
| Curriculum | Progressive | Helps learn refinement behavior |
| I/L ambiguity | Explicit notation | Physically unresolvable, should be acknowledged |
| PTM handling | Extended vocabulary + mass gap detection | Practical for real-world use |
| Uncertainty | Entropy + top-k | Multiple valid outputs common in MS/MS |

---

## Expected Outcomes

### Synthetic Data (Clean)
- Token accuracy: >95%
- Sequence accuracy: >85%
- This validates the core approach works

### Synthetic Data (Realistic noise)
- Token accuracy: >85%
- Sequence accuracy: >60%
- Gap shows the challenge of noise handling

### Real Data (Nine-Species benchmark)
- Target: Competitive with Casanovo baseline
- This is the true test of generalization

### Key Research Questions
1. Does the recursive mechanism provide measurable benefit over T=1?
2. Does the mass-matching auxiliary loss help self-correction?
3. How does performance scale with number of supervision steps?
4. Can the model learn to recognize its own uncertainty?
