"""
ProteomeTools dataset for training with synthetic peptide library.

ProteomeTools provides comprehensive coverage of the human proteome with
>1M synthetic tryptic peptides and 21M high-quality MS/MS spectra.
This dataset was used to train Prosit.

Dataset: https://doi.org/10.5281/zenodo.15705607
Paper: https://doi.org/10.1038/s41592-019-0426-7
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict
import numpy as np
import torch
from torch.utils.data import Dataset

from ..constants import (
    AA_TO_IDX,
    AMINO_ACID_MASSES,
    WATER_MASS,
    PROTON_MASS,
)


@dataclass
class ProteomeToolsSample:
    """Sample from ProteomeTools (matches interface)."""
    spectrum_masses: torch.Tensor  # (max_peaks,)
    spectrum_intensities: torch.Tensor  # (max_peaks,)
    spectrum_mask: torch.Tensor  # (max_peaks,) bool
    precursor_mass: torch.Tensor  # scalar
    precursor_charge: torch.Tensor  # scalar
    sequence: torch.Tensor  # (max_seq_len,) token IDs
    sequence_mask: torch.Tensor  # (max_seq_len,) bool
    # Metadata (optional)
    collision_energy: Optional[float] = None
    scan_id: Optional[str] = None


def parse_msp_file(msp_path: Path):
    """
    Parse an MSP (NIST spectral library) file as a generator.

    Yields spectrum dictionaries one at a time, allowing early stopping
    without loading the entire file into memory.

    Yields:
        Dict with: peptide, precursor_charge, peaks, precursor_mw, etc.
    """
    current_spectrum = None
    reading_peaks = False
    peak_count = 0
    peaks_to_read = 0

    with open(msp_path, 'r') as f:
        for line in f:
            line = line.strip()

            if not line:
                # Empty line may signal end of spectrum
                if current_spectrum is not None and peak_count >= peaks_to_read:
                    yield current_spectrum
                    current_spectrum = None
                    reading_peaks = False
                    peak_count = 0
                    peaks_to_read = 0
                continue

            if line.startswith('Name:'):
                # Start new spectrum - yield previous if exists
                if current_spectrum is not None:
                    yield current_spectrum

                name = line.split(':', 1)[1].strip()
                parts = name.split('/')
                if len(parts) >= 2:
                    peptide = parts[0]
                    charge_str = parts[1].split('_')[0]
                    try:
                        charge = int(charge_str)
                    except ValueError:
                        charge = 2

                    current_spectrum = {
                        'peptide': peptide,
                        'precursor_charge': charge,
                        'peaks': [],
                        'name': name
                    }
                reading_peaks = False
                peak_count = 0

            elif line.startswith('MW:') and current_spectrum is not None:
                mw_str = line.split(':', 1)[1].strip()
                try:
                    current_spectrum['precursor_mw'] = float(mw_str)
                except ValueError:
                    pass

            elif line.startswith('Comment:') and current_spectrum is not None:
                comment = line.split(':', 1)[1].strip()
                for part in comment.split():
                    if '=' in part:
                        key, val = part.split('=', 1)
                        if key == 'Parent':
                            try:
                                current_spectrum['precursor_mz'] = float(val)
                            except ValueError:
                                pass
                        elif key == 'NCE' or key == 'CollisionEnergy':
                            try:
                                current_spectrum['collision_energy'] = float(val)
                            except ValueError:
                                pass

            elif line.startswith('Num peaks:') and current_spectrum is not None:
                num_str = line.split(':', 1)[1].strip()
                try:
                    peaks_to_read = int(num_str)
                except ValueError:
                    peaks_to_read = 0
                reading_peaks = True

            elif reading_peaks and current_spectrum is not None:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        mz = float(parts[0])
                        intensity = float(parts[1])
                        current_spectrum['peaks'].append((mz, intensity))
                        peak_count += 1

                        if peak_count >= peaks_to_read:
                            reading_peaks = False
                    except ValueError:
                        continue

        # Don't forget last spectrum
        if current_spectrum is not None:
            yield current_spectrum


class ProteomeToolsDataset(Dataset):
    """
    Dataset for ProteomeTools synthetic peptide library.

    This is a large-scale synthetic library with comprehensive coverage
    of tryptic peptides. Unlike Nine-Species (real data), this provides
    high-quality, consistent spectra for a large variety of peptides.
    """

    def __init__(
        self,
        data_dir: Path,
        split: str = 'train',
        max_peaks: int = 100,
        max_seq_len: int = 35,
        min_peaks: int = 10,
        min_length: int = 7,
        max_length: int = 30,
        normalize_intensities: bool = True,
        collision_energy: Optional[int] = None,  # Filter by NCE if desired
        val_fraction: float = 0.1,  # Fraction for validation
        random_seed: int = 42,
        max_samples: Optional[int] = None,
    ):
        """
        Args:
            data_dir: Directory containing extracted .msp files
            split: 'train' or 'val'
            max_peaks: Maximum peaks to keep
            max_seq_len: Maximum sequence length
            min_peaks: Minimum peaks required
            min_length: Minimum peptide length
            max_length: Maximum peptide length
            normalize_intensities: Normalize to [0, 1]
            collision_energy: Filter by collision energy (e.g., 28)
            val_fraction: Fraction of data for validation
            random_seed: Seed for reproducible splits
            max_samples: Maximum samples to load (None = all). Use for fast validation.
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_peaks = max_peaks
        self.max_seq_len = max_seq_len
        self.min_peaks = min_peaks
        self.min_length = min_length
        self.max_length = max_length
        self.normalize_intensities = normalize_intensities
        self.collision_energy = collision_energy
        self.val_fraction = val_fraction
        self.random_seed = random_seed
        self.max_samples = max_samples

        # Load spectra
        self.samples = self._load_data()

    def _load_data(self) -> List[ProteomeToolsSample]:
        """Load and preprocess spectra from .msp files."""
        samples = []

        # Find all .msp files
        msp_files = list(self.data_dir.glob("**/*.msp"))

        if not msp_files:
            raise FileNotFoundError(
                f"No .msp files found in {self.data_dir}\n"
                f"Please download ProteomeTools data from:\n"
                f"https://zenodo.org/records/15705607"
            )

        # Early stopping: load 2x max_samples to have buffer for filtering
        target_samples = self.max_samples * 2 if self.max_samples else None

        # Parse all MSP files
        for msp_file in msp_files:
            # Early stopping check
            if target_samples and len(samples) >= target_samples:
                break

            spectra = parse_msp_file(msp_file)

            for spectrum in spectra:
                # Filter by collision energy if specified
                if self.collision_energy is not None:
                    spec_nce = spectrum.get('collision_energy')
                    if spec_nce is not None and abs(spec_nce - self.collision_energy) > 1:
                        continue

                # Process spectrum
                sample = self._process_spectrum(spectrum)
                if sample is not None:
                    samples.append(sample)
                    # Early stopping within file
                    if target_samples and len(samples) >= target_samples:
                        break

        # Split into train/val
        np.random.seed(self.random_seed)
        indices = np.random.permutation(len(samples))
        val_size = int(self.val_fraction * len(samples))

        if self.split == 'val':
            samples = [samples[i] for i in indices[:val_size]]
        else:  # train
            samples = [samples[i] for i in indices[val_size:]]

        # Limit samples for fast validation
        if self.max_samples is not None and len(samples) > self.max_samples:
            np.random.seed(self.random_seed)
            indices = np.random.permutation(len(samples))[:self.max_samples]
            samples = [samples[i] for i in indices]

        return samples

    def _process_spectrum(
        self,
        spectrum: Dict
    ) -> Optional[ProteomeToolsSample]:
        """
        Process a single spectrum into a training sample.

        Returns None if spectrum should be filtered.
        """
        # Get peptide sequence
        if 'peptide' not in spectrum:
            return None

        peptide = spectrum['peptide'].upper()

        # Remove modifications (ProteomeTools uses format like C[+57.021])
        # Simple approach: remove anything in brackets
        import re
        peptide = re.sub(r'\[.*?\]', '', peptide)
        # Also handle common modification notations
        peptide = peptide.replace('(ox)', '').replace('[ox]', '')

        # Filter by length
        if len(peptide) < self.min_length or len(peptide) > self.max_length:
            return None

        # Check vocabulary
        if not all(aa in AA_TO_IDX for aa in peptide):
            return None

        # Get peaks
        peaks = spectrum.get('peaks', [])
        if len(peaks) < self.min_peaks:
            return None

        masses = np.array([p[0] for p in peaks], dtype=np.float32)
        intensities = np.array([p[1] for p in peaks], dtype=np.float32)

        # Normalize
        if self.normalize_intensities and intensities.max() > 0:
            intensities = intensities / intensities.max()

        # Sort by mass
        sort_idx = np.argsort(masses)
        masses = masses[sort_idx]
        intensities = intensities[sort_idx]

        # Keep top-k most intense if too many
        if len(masses) > self.max_peaks:
            top_idx = np.argsort(intensities)[-self.max_peaks:]
            top_idx = np.sort(top_idx)
            masses = masses[top_idx]
            intensities = intensities[top_idx]

        # Pad
        num_peaks = len(masses)
        peak_mask = np.zeros(self.max_peaks, dtype=bool)
        peak_mask[:num_peaks] = True

        spectrum_masses = np.zeros(self.max_peaks, dtype=np.float32)
        spectrum_intensities = np.zeros(self.max_peaks, dtype=np.float32)
        spectrum_masses[:num_peaks] = masses
        spectrum_intensities[:num_peaks] = intensities

        # Precursor info
        precursor_mz = spectrum.get('precursor_mz', spectrum.get('precursor_mw', 0.0))
        precursor_charge = spectrum.get('precursor_charge', 2)

        # Calculate precursor mass
        if 'precursor_mw' in spectrum:
            precursor_mass = spectrum['precursor_mw']
        else:
            precursor_mass = (precursor_mz * precursor_charge) - (precursor_charge * PROTON_MASS)

        # Encode sequence
        seq_tokens = [AA_TO_IDX[aa] for aa in peptide]
        seq_len = len(seq_tokens)

        sequence = np.zeros(self.max_seq_len, dtype=np.int64)
        sequence_mask = np.zeros(self.max_seq_len, dtype=bool)
        sequence[:seq_len] = seq_tokens
        sequence_mask[:seq_len] = True

        return ProteomeToolsSample(
            spectrum_masses=torch.from_numpy(spectrum_masses),
            spectrum_intensities=torch.from_numpy(spectrum_intensities),
            spectrum_mask=torch.from_numpy(peak_mask),
            precursor_mass=torch.tensor(precursor_mass, dtype=torch.float32),
            precursor_charge=torch.tensor(precursor_charge, dtype=torch.int64),
            sequence=torch.from_numpy(sequence),
            sequence_mask=torch.from_numpy(sequence_mask),
            collision_energy=spectrum.get('collision_energy'),
            scan_id=spectrum.get('name', 'unknown'),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> ProteomeToolsSample:
        return self.samples[idx]

    def set_difficulty(
        self,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        noise_peaks: Optional[int] = None,
        peak_dropout: Optional[float] = None,
        mass_error_ppm: Optional[float] = None,
        intensity_variation: Optional[float] = None,
        clean_data_ratio: Optional[float] = None,
    ):
        """Update difficulty parameters (for curriculum compatibility)."""
        if min_length is not None:
            self.min_length = min_length
        if max_length is not None:
            self.max_length = max_length
        # Note: Length changes require reloading


def create_proteometools_dataloader(
    data_dir: Path,
    batch_size: int,
    split: str = 'train',
    num_workers: int = 4,
    **dataset_kwargs
):
    """Create DataLoader for ProteomeTools dataset."""
    from torch.utils.data import DataLoader

    dataset = ProteomeToolsDataset(
        data_dir=data_dir,
        split=split,
        **dataset_kwargs
    )

    # Same collate function as other datasets (must match MS2PIP key names)
    def collate_fn(batch):
        return {
            'spectrum_masses': torch.stack([s.spectrum_masses for s in batch]),
            'spectrum_intensities': torch.stack([s.spectrum_intensities for s in batch]),
            'spectrum_mask': torch.stack([s.spectrum_mask for s in batch]),
            'precursor_mass': torch.stack([s.precursor_mass for s in batch]),
            'precursor_charge': torch.stack([s.precursor_charge for s in batch]),
            'sequence': torch.stack([s.sequence for s in batch]),
            'sequence_mask': torch.stack([s.sequence_mask for s in batch]),
        }

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=(split == 'train'),
        collate_fn=collate_fn,
        pin_memory=True,
    )
