"""
Deep inspection of data formats across all datasets.

This script compares ACTUAL data from:
1. MS2PIP synthetic (training)
2. ProteomeTools (real)
3. Nine-Species (real)

Looking for distribution mismatches, format bugs, or preprocessing issues.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from torch.utils.data import DataLoader

from src.data.ms2pip_dataset import MS2PIPSyntheticDataset
from src.data.proteometools_dataset import ProteomeToolsDataset
from src.data.nine_species_dataset import NineSpeciesDataset
from src.constants import IDX_TO_AA, AA_TO_IDX, AMINO_ACID_MASSES, WATER_MASS, PROTON_MASS


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


def decode_sequence(seq_tensor, mask_tensor):
    """Convert token tensor to string."""
    seq = []
    for i, (idx, valid) in enumerate(zip(seq_tensor.tolist(), mask_tensor.tolist())):
        if not valid:
            break
        if idx < len(IDX_TO_AA):
            aa = IDX_TO_AA[idx]
            if aa not in ['<PAD>', '<SOS>', '<EOS>', '<UNK>']:
                seq.append(aa)
    return ''.join(seq)


def compute_theoretical_mass(sequence: str) -> float:
    """Compute theoretical precursor mass from sequence."""
    return sum(AMINO_ACID_MASSES.get(aa, 0) for aa in sequence) + WATER_MASS


def inspect_sample(sample_dict, idx, dataset_name):
    """Detailed inspection of a single sample."""
    print(f"\n{'='*60}")
    print(f"Sample {idx} from {dataset_name}")
    print('='*60)

    # Sequence info
    seq = decode_sequence(sample_dict['sequence'][idx], sample_dict['sequence_mask'][idx])
    seq_len = sample_dict['sequence_mask'][idx].sum().item()
    print(f"\nSequence: {seq}")
    print(f"Sequence length: {seq_len}")
    print(f"Sequence tokens: {sample_dict['sequence'][idx][:seq_len].tolist()}")

    # Precursor info
    precursor_mass = sample_dict['precursor_mass'][idx].item()
    precursor_charge = sample_dict['precursor_charge'][idx].item()

    # Handle different tensor shapes
    if isinstance(precursor_mass, (list, tuple)):
        precursor_mass = precursor_mass[0] if precursor_mass else 0
    if hasattr(sample_dict['precursor_mass'][idx], 'shape'):
        if len(sample_dict['precursor_mass'][idx].shape) > 0:
            precursor_mass = sample_dict['precursor_mass'][idx].squeeze().item()

    theoretical_mass = compute_theoretical_mass(seq)
    mass_error = abs(precursor_mass - theoretical_mass)

    print(f"\nPrecursor mass (from data): {precursor_mass:.4f} Da")
    print(f"Theoretical mass (computed): {theoretical_mass:.4f} Da")
    print(f"Mass error: {mass_error:.4f} Da ({mass_error/theoretical_mass*1e6:.1f} ppm)")
    print(f"Precursor charge: {precursor_charge}")

    # What the model receives as precursor m/z
    precursor_mz = (precursor_mass + precursor_charge * PROTON_MASS) / precursor_charge
    print(f"Precursor m/z (model input): {precursor_mz:.4f}")

    # Spectrum info
    num_peaks = sample_dict['spectrum_mask'][idx].sum().item()
    masses = sample_dict['spectrum_masses'][idx][:int(num_peaks)].numpy()
    intensities = sample_dict['spectrum_intensities'][idx][:int(num_peaks)].numpy()

    print(f"\nSpectrum: {num_peaks} peaks")
    print(f"Mass range: {masses.min():.2f} - {masses.max():.2f} Da")
    print(f"Intensity range: {intensities.min():.4f} - {intensities.max():.4f}")
    print(f"Mean intensity: {intensities.mean():.4f}")

    # Show top 10 peaks
    top_idx = np.argsort(intensities)[-10:][::-1]
    print(f"\nTop 10 peaks (m/z, intensity):")
    for i in top_idx:
        print(f"  {masses[i]:.4f}, {intensities[i]:.4f}")

    # Check for potential issues
    print(f"\n--- Sanity Checks ---")

    # 1. Mass range sanity
    if masses.max() > precursor_mass + 100:
        print(f"  ⚠️  WARNING: Max peak mass ({masses.max():.1f}) > precursor mass ({precursor_mass:.1f})")
    else:
        print(f"  ✓ Peak masses within expected range")

    # 2. Intensity normalization
    if intensities.max() > 1.1:
        print(f"  ⚠️  WARNING: Intensities not normalized (max={intensities.max():.2f})")
    elif intensities.max() < 0.5:
        print(f"  ⚠️  WARNING: Max intensity very low ({intensities.max():.2f})")
    else:
        print(f"  ✓ Intensities normalized to ~1.0")

    # 3. Precursor mass sanity
    if mass_error > 5:  # More than 5 Da error
        print(f"  ⚠️  WARNING: Large precursor mass error ({mass_error:.1f} Da)")
        print(f"       This could indicate modification or wrong sequence")
    else:
        print(f"  ✓ Precursor mass matches theoretical")

    return {
        'seq_len': seq_len,
        'num_peaks': num_peaks,
        'precursor_mass': precursor_mass,
        'theoretical_mass': theoretical_mass,
        'mass_error': mass_error,
        'charge': precursor_charge,
        'max_peak_mass': masses.max(),
        'max_intensity': intensities.max(),
    }


def compare_distributions(stats_list, names):
    """Compare statistics across datasets."""
    print("\n" + "=" * 70)
    print("DISTRIBUTION COMPARISON")
    print("=" * 70)

    metrics = ['seq_len', 'num_peaks', 'precursor_mass', 'mass_error', 'charge', 'max_intensity']

    for metric in metrics:
        print(f"\n{metric}:")
        for name, stats in zip(names, stats_list):
            values = [s[metric] for s in stats]
            print(f"  {name:20s}: mean={np.mean(values):.2f}, std={np.std(values):.2f}, "
                  f"min={np.min(values):.2f}, max={np.max(values):.2f}")


def main():
    data_dir = project_root / 'data'

    print("=" * 70)
    print("DATA FORMAT INSPECTION")
    print("=" * 70)

    # Load datasets
    datasets = []
    names = []

    # 1. MS2PIP (training data)
    print("\nLoading MS2PIP synthetic dataset...")
    ms2pip = MS2PIPSyntheticDataset(
        min_length=10,
        max_length=20,
        noise_peaks=0,
        peak_dropout=0.0,
        mass_error_ppm=0.0,
        ms2pip_model='HCDch2',
    )
    datasets.append(('MS2PIP', ms2pip))
    names.append('MS2PIP')

    # 2. ProteomeTools
    pt_dir = data_dir / 'proteometools'
    if pt_dir.exists():
        print("Loading ProteomeTools dataset...")
        try:
            pt = ProteomeToolsDataset(
                data_dir=pt_dir,
                split='val',
                max_peaks=100,
                max_seq_len=35,
                max_samples=500,
            )
            datasets.append(('ProteomeTools', pt))
            names.append('ProteomeTools')
            print(f"  Loaded {len(pt)} samples")
        except Exception as e:
            print(f"  Failed: {e}")

    # 3. Nine-Species
    ns_dir = data_dir / 'nine_species'
    if ns_dir.exists():
        print("Loading Nine-Species dataset...")
        try:
            ns = NineSpeciesDataset(
                data_dir=ns_dir,
                split='val',
                max_peaks=100,
                max_seq_len=35,
                use_balanced=True,
                max_samples=500,
            )
            datasets.append(('Nine-Species', ns))
            names.append('Nine-Species')
            print(f"  Loaded {len(ns)} samples")
        except Exception as e:
            print(f"  Failed: {e}")

    # Inspect samples from each dataset
    all_stats = []

    for dataset_name, dataset in datasets:
        print(f"\n\n{'#'*70}")
        print(f"# {dataset_name}")
        print('#'*70)

        loader = DataLoader(dataset, batch_size=10, collate_fn=collate_fn)
        batch = next(iter(loader))

        stats = []
        # Inspect first 3 samples in detail
        for i in range(min(3, len(batch['sequence']))):
            s = inspect_sample(batch, i, dataset_name)
            stats.append(s)

        # Collect stats for more samples
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= 5:  # 50 samples total
                break
            for i in range(len(batch['sequence'])):
                seq = decode_sequence(batch['sequence'][i], batch['sequence_mask'][i])
                precursor_mass = batch['precursor_mass'][i]
                if hasattr(precursor_mass, 'squeeze'):
                    precursor_mass = precursor_mass.squeeze().item()
                else:
                    precursor_mass = float(precursor_mass)

                theoretical = compute_theoretical_mass(seq)
                num_peaks = batch['spectrum_mask'][i].sum().item()
                masses = batch['spectrum_masses'][i][:int(num_peaks)].numpy()
                intensities = batch['spectrum_intensities'][i][:int(num_peaks)].numpy()

                stats.append({
                    'seq_len': batch['sequence_mask'][i].sum().item(),
                    'num_peaks': num_peaks,
                    'precursor_mass': precursor_mass,
                    'theoretical_mass': theoretical,
                    'mass_error': abs(precursor_mass - theoretical),
                    'charge': batch['precursor_charge'][i].item(),
                    'max_peak_mass': masses.max() if len(masses) > 0 else 0,
                    'max_intensity': intensities.max() if len(intensities) > 0 else 0,
                })

        all_stats.append(stats)

    # Compare distributions
    compare_distributions(all_stats, names)

    print("\n" + "=" * 70)
    print("INSPECTION COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
