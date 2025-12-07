"""
PyTorch Dataset classes for peptide sequencing.
"""

import random
import torch
from torch import Tensor
from torch.utils.data import IterableDataset
from dataclasses import dataclass
from typing import Optional

from ..constants import AA_TO_IDX, PAD_IDX, SOS_IDX, EOS_IDX
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
        ion_types: list[str] = None,
        include_neutral_losses: bool = False,

        # Difficulty parameters (for curriculum learning)
        noise_peaks: int = 0,
        peak_dropout: float = 0.0,
        mass_error_ppm: float = 0.0,
        intensity_variation: float = 0.0,

        # Charge state distribution
        charge_distribution: Optional[dict[int, float]] = None,
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.max_peaks = max_peaks
        self.max_seq_len = max_seq_len
        self.ion_types = ion_types or ['b', 'y']
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
