"""
Nine-Species benchmark dataset for real MS/MS data training.

This dataset provides access to the multi-species proteomics benchmark
published in Scientific Data (2024), containing 2.8M high-confidence
peptide-spectrum matches from 9 species.

Dataset: https://doi.org/10.5281/zenodo.13685813
Paper: https://doi.org/10.1038/s41597-024-04068-4
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import re
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
class RealSpectrumSample:
    """Sample from real MS/MS data (matches MS2PIPSample interface)."""
    spectrum_masses: torch.Tensor  # (max_peaks,)
    spectrum_intensities: torch.Tensor  # (max_peaks,)
    spectrum_mask: torch.Tensor  # (max_peaks,) bool
    precursor_mass: torch.Tensor  # scalar
    precursor_charge: torch.Tensor  # scalar
    sequence: torch.Tensor  # (max_seq_len,) token IDs
    sequence_mask: torch.Tensor  # (max_seq_len,) bool
    # Metadata (optional, for debugging/analysis)
    species: Optional[str] = None
    scan_id: Optional[str] = None


def parse_mgf_file(mgf_path: Path):
    """
    Parse an MGF (Mascot Generic Format) file as a generator.

    Yields spectrum dictionaries one at a time, allowing early stopping
    without loading the entire file into memory.

    Yields:
        Dict with: peaks, precursor_mz, precursor_charge, peptide, scan_id, etc.
    """
    current_spectrum = None
    current_peaks = []

    with open(mgf_path, 'r') as f:
        for line in f:
            line = line.strip()

            if line == 'BEGIN IONS':
                current_spectrum = {}
                current_peaks = []

            elif line == 'END IONS':
                if current_spectrum is not None:
                    current_spectrum['peaks'] = current_peaks
                    yield current_spectrum
                    current_spectrum = None
                    current_peaks = []

            elif '=' in line and current_spectrum is not None:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()

                if key == 'TITLE':
                    current_spectrum['scan_id'] = value
                elif key == 'PEPMASS':
                    parts = value.split()
                    current_spectrum['precursor_mz'] = float(parts[0])
                elif key == 'CHARGE':
                    charge_str = value.rstrip('+').rstrip('-')
                    current_spectrum['precursor_charge'] = int(charge_str)
                elif key == 'SEQ':
                    current_spectrum['peptide'] = value
                elif key == 'SCANS':
                    current_spectrum['scan_number'] = value
                else:
                    current_spectrum[key.lower()] = value

            elif line and not line.startswith('#') and current_spectrum is not None:
                try:
                    parts = line.split()
                    if len(parts) >= 2:
                        mz = float(parts[0])
                        intensity = float(parts[1])
                        current_peaks.append((mz, intensity))
                except ValueError:
                    continue


class NineSpeciesDataset(Dataset):
    """
    Dataset for the Nine-Species proteomics benchmark.

    Provides access to real MS/MS spectra with ground-truth peptide sequences.
    Supports species-based cross-validation (train on 8 species, test on 1).
    """

    def __init__(
        self,
        data_dir: Path,
        split: str = 'train',
        test_species: Optional[str] = None,
        max_peaks: int = 100,
        max_seq_len: int = 35,
        min_peaks: int = 10,
        min_length: int = 7,
        max_length: int = 30,
        normalize_intensities: bool = True,
        use_balanced: bool = False,
        max_samples: Optional[int] = None,
    ):
        """
        Args:
            data_dir: Root directory containing Nine-Species MGF files
            split: One of 'train', 'val', 'test'
            test_species: Species to hold out for testing (for CV). If None, uses all species.
            max_peaks: Maximum number of peaks to keep
            max_seq_len: Maximum sequence length (with padding)
            min_peaks: Minimum peaks required (filter out spectra with fewer)
            min_length: Minimum peptide length
            max_length: Maximum peptide length
            normalize_intensities: Whether to normalize intensities to [0, 1]
            use_balanced: Whether to use balanced version (780K PSMs) vs main (2.8M PSMs)
            max_samples: Maximum samples to load (None = all). Use for fast validation.
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.test_species = test_species
        self.max_peaks = max_peaks
        self.max_seq_len = max_seq_len
        self.min_peaks = min_peaks
        self.min_length = min_length
        self.max_length = max_length
        self.normalize_intensities = normalize_intensities
        self.use_balanced = use_balanced
        self.max_samples = max_samples

        # Species list and file name mapping
        self.species_list = [
            'vigna_mungo',           # plant
            'mus_musculus',          # mouse
            'methanosarcina_mazei',  # archaeon
            'bacillus_subtilis',     # bacterium
            'candidatus_endoloripes',# bacterium
            'solanum_lycopersicum',  # tomato
            'saccharomyces_cerevisiae', # yeast
            'apis_mellifera',        # honeybee
            'homo_sapiens',          # human
        ]

        # Map species names to actual MGF file patterns
        self.species_to_file = {
            'vigna_mungo': 'Vigna-mungo.mgf',
            'mus_musculus': 'Mus-musculus.mgf',
            'methanosarcina_mazei': 'Methanosarcina-mazei.mgf',
            'bacillus_subtilis': 'Bacillus-subtilis.mgf',
            'candidatus_endoloripes': 'Candidatus-endoloripes.mgf',
            'solanum_lycopersicum': 'Solanum-lycopersicum.mgf',
            'saccharomyces_cerevisiae': 'Saccharomyces-cerevisiae.mgf',
            'apis_mellifera': 'Apis-mellifera.mgf',
            'homo_sapiens': 'H.-sapiens.mgf',
        }

        # Load spectra
        self.samples = self._load_data()

    def _load_data(self) -> List[RealSpectrumSample]:
        """Load and preprocess spectra from MGF files."""
        samples = []

        # Determine which MGF files to load
        # The extracted structure is: data_dir/nine-species-{version}/*.mgf
        version = 'balanced' if self.use_balanced else 'main'
        mgf_dir = self.data_dir / f'nine-species-{version}'

        if not mgf_dir.exists():
            raise FileNotFoundError(
                f"MGF directory not found: {mgf_dir}\n"
                f"Please download the Nine-Species benchmark from:\n"
                f"https://doi.org/10.5281/zenodo.13685813\n"
                f"and extract to: {self.data_dir}"
            )

        # For cross-validation: determine which species to use
        if self.test_species is not None:
            if self.split == 'test':
                species_to_load = [self.test_species]
            else:  # train or val
                species_to_load = [s for s in self.species_list if s != self.test_species]
        else:
            species_to_load = self.species_list

        # Load MGF files for each species
        # Early stopping: load 2x max_samples to have buffer for filtering
        target_samples = self.max_samples * 2 if self.max_samples else None

        for species in species_to_load:
            # Early stopping check
            if target_samples and len(samples) >= target_samples:
                break

            # Get the actual MGF filename for this species
            mgf_filename = self.species_to_file[species]
            mgf_file = mgf_dir / mgf_filename

            if not mgf_file.exists():
                print(f"Warning: MGF file not found for {species}: {mgf_file}")
                continue

            spectra = parse_mgf_file(mgf_file)

            for spectrum in spectra:
                # Filter and preprocess
                sample = self._process_spectrum(spectrum, species)
                if sample is not None:
                    samples.append(sample)
                    # Early stopping within file
                    if target_samples and len(samples) >= target_samples:
                        break

        # For train/val split, we'll use a simple 90/10 split based on index
        if self.test_species is not None and self.split != 'test':
            # Split train species into train/val
            np.random.seed(42)  # Reproducible splits
            indices = np.random.permutation(len(samples))
            val_size = int(0.1 * len(samples))

            if self.split == 'val':
                samples = [samples[i] for i in indices[:val_size]]
            else:  # train
                samples = [samples[i] for i in indices[val_size:]]

        # Limit samples for fast validation
        if self.max_samples is not None and len(samples) > self.max_samples:
            np.random.seed(42)
            indices = np.random.permutation(len(samples))[:self.max_samples]
            samples = [samples[i] for i in indices]

        return samples

    def _process_spectrum(
        self,
        spectrum: Dict,
        species: str
    ) -> Optional[RealSpectrumSample]:
        """
        Process a single spectrum into a training sample.

        Returns None if spectrum should be filtered out.
        """
        # Check if peptide annotation exists
        if 'peptide' not in spectrum:
            return None

        peptide = spectrum['peptide'].upper()

        # Filter by length
        if len(peptide) < self.min_length or len(peptide) > self.max_length:
            return None

        # Check if all amino acids are in vocabulary
        if not all(aa in AA_TO_IDX for aa in peptide):
            return None

        # Get peaks
        peaks = spectrum.get('peaks', [])
        if len(peaks) < self.min_peaks:
            return None

        # Extract masses and intensities
        masses = np.array([p[0] for p in peaks], dtype=np.float32)
        intensities = np.array([p[1] for p in peaks], dtype=np.float32)

        # Normalize intensities
        if self.normalize_intensities and intensities.max() > 0:
            intensities = intensities / intensities.max()

        # Sort by mass
        sort_idx = np.argsort(masses)
        masses = masses[sort_idx]
        intensities = intensities[sort_idx]

        # Keep top-k most intense peaks if we have too many
        if len(masses) > self.max_peaks:
            top_idx = np.argsort(intensities)[-self.max_peaks:]
            top_idx = np.sort(top_idx)  # Keep sorted by mass
            masses = masses[top_idx]
            intensities = intensities[top_idx]

        # Pad to max_peaks
        num_peaks = len(masses)
        peak_mask = np.zeros(self.max_peaks, dtype=bool)
        peak_mask[:num_peaks] = True

        spectrum_masses = np.zeros(self.max_peaks, dtype=np.float32)
        spectrum_intensities = np.zeros(self.max_peaks, dtype=np.float32)
        spectrum_masses[:num_peaks] = masses
        spectrum_intensities[:num_peaks] = intensities

        # Get precursor info
        precursor_mz = spectrum.get('precursor_mz', 0.0)
        precursor_charge = spectrum.get('precursor_charge', 2)

        # Calculate precursor mass from m/z and charge
        # precursor_mass = (precursor_mz * charge) - (charge * PROTON_MASS)
        precursor_mass = (precursor_mz * precursor_charge) - (precursor_charge * PROTON_MASS)

        # Encode sequence
        seq_tokens = [AA_TO_IDX[aa] for aa in peptide]
        seq_len = len(seq_tokens)

        sequence = np.zeros(self.max_seq_len, dtype=np.int64)
        sequence_mask = np.zeros(self.max_seq_len, dtype=bool)
        sequence[:seq_len] = seq_tokens
        sequence_mask[:seq_len] = True

        return RealSpectrumSample(
            spectrum_masses=torch.from_numpy(spectrum_masses),
            spectrum_intensities=torch.from_numpy(spectrum_intensities),
            spectrum_mask=torch.from_numpy(peak_mask),
            precursor_mass=torch.tensor(precursor_mass, dtype=torch.float32),
            precursor_charge=torch.tensor(precursor_charge, dtype=torch.int64),
            sequence=torch.from_numpy(sequence),
            sequence_mask=torch.from_numpy(sequence_mask),
            species=species,
            scan_id=spectrum.get('scan_id', 'unknown'),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> RealSpectrumSample:
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
        """
        Update difficulty parameters (for curriculum compatibility).

        Note: Real data doesn't support dynamic difficulty like synthetic data.
        Length changes would require reloading the dataset.
        """
        if min_length is not None:
            self.min_length = min_length
        if max_length is not None:
            self.max_length = max_length

        # Note: Other parameters (noise, dropout, etc.) don't apply to real data
        # as we're loading ground-truth spectra. If curriculum is needed,
        # we could apply augmentation, but that's a future enhancement.


def create_nine_species_dataloader(
    data_dir: Path,
    batch_size: int,
    split: str = 'train',
    test_species: Optional[str] = None,
    num_workers: int = 4,
    **dataset_kwargs
):
    """Create DataLoader for Nine-Species dataset."""
    from torch.utils.data import DataLoader

    dataset = NineSpeciesDataset(
        data_dir=data_dir,
        split=split,
        test_species=test_species,
        **dataset_kwargs
    )

    # Custom collate function (must match MS2PIP key names for trainer compatibility)
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
