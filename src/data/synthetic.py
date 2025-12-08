"""
Forward model: Generate theoretical MS/MS spectrum from peptide sequence.

This is the "physics simulator" that creates training data with perfect labels.
"""

from dataclasses import dataclass
import random
from typing import Optional

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


def _theoretical_intensity(ion_index: int, peptide_length: int, ion_type: str) -> float:
    """
    Generate theoretical intensity for a fragment ion.

    In real spectra, intensity is highly variable. We use a stochastic
    model based on a beta distribution to mimic this and remove any
    simple deterministic patterns from the data.
    """
    # Heuristic: y-ions are often more prominent and have a wider intensity range.
    # b- and a-ions are often less intense.
    if ion_type == 'y':
        # Beta distribution centered around ~0.5, creating a wide range of intensities
        intensity = random.betavariate(2, 2)
    else:  # b- and a-ions
        # Beta distribution skewed towards lower values
        intensity = random.betavariate(2, 5)

    return max(0.01, intensity)  # Ensure a minimum intensity


def generate_theoretical_spectrum(
    peptide: str,
    charge: int = 2,
    ion_types: list[str] = None,
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
    if ion_types is None:
        ion_types = ['b', 'y']

    n = len(peptide)
    peaks = []
    annotations = {}

    # Calculate residue masses
    residue_masses = [AMINO_ACID_MASSES[aa] for aa in peptide]

    # Precursor mass (full peptide + H2O)
    precursor_mass_true = sum(residue_masses) + WATER_MASS

    # Apply mass error to precursor (like we do for fragment peaks)
    if mass_error_ppm > 0:
        error = precursor_mass_true * random.gauss(0, mass_error_ppm * 1e-6)
        precursor_mass = precursor_mass_true + error
    else:
        precursor_mass = precursor_mass_true

    precursor_mz = (precursor_mass + charge * PROTON_MASS) / charge

    # Generate b-ions (N-terminal fragments)
    # b_i = mass of first i residues + H+ (ionized form: [M+H]+)
    if 'b' in ion_types:
        cumulative = 0.0
        for i in range(1, n):  # b1 to b_{n-1}
            cumulative += residue_masses[i - 1]
            mass = cumulative + PROTON_MASS  # Add proton for ionization
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
            mass = cumulative - CO_MASS + PROTON_MASS  # Add proton for ionization
            intensity = _theoretical_intensity(i, n, 'a')
            peaks.append((mass, intensity))
            annotations[mass] = f'a{i}'

    # Generate y-ions (C-terminal fragments)
    # y_i = mass of last i residues + H2O + H+ (ionized form: [M+H2O+H]+)
    if 'y' in ion_types:
        cumulative = WATER_MASS  # y-ions include the C-terminal OH
        for i in range(1, n):  # y1 to y_{n-1}
            cumulative += residue_masses[n - i]
            mass = cumulative + PROTON_MASS  # Add proton for ionization
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


def generate_random_peptide(
    min_length: int = 7,
    max_length: int = 20,
    exclude_aa: Optional[set[str]] = None,
) -> str:
    """Generate a random peptide sequence."""
    exclude_aa = exclude_aa or set()
    available_aa = [aa for aa in AMINO_ACID_MASSES.keys() if aa not in exclude_aa]
    length = random.randint(min_length, max_length)
    return ''.join(random.choices(available_aa, k=length))
