"""
Analyze signal-to-noise ratio in real vs synthetic data.

Questions:
1. What % of observed peaks are explainable (b/y ions)?
2. What % are unexplained (noise)?
3. How does this differ from MS2PIP training data?
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
from src.constants import AMINO_ACID_MASSES, WATER_MASS, PROTON_MASS, IDX_TO_AA


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
    seq = []
    for idx, valid in zip(seq_tensor.tolist(), mask_tensor.tolist()):
        if not valid:
            break
        if idx < len(IDX_TO_AA):
            aa = IDX_TO_AA[idx]
            if aa not in ['<PAD>', '<SOS>', '<EOS>', '<UNK>']:
                seq.append(aa)
    return ''.join(seq)


def compute_all_theoretical_mz(sequence: str):
    """Compute all theoretical b/y ion m/z values (singly and doubly charged)."""
    n = len(sequence)
    all_mz = []

    # b-ions (singly charged)
    prefix_mass = 0.0
    for i in range(n - 1):
        prefix_mass += AMINO_ACID_MASSES.get(sequence[i], 0.0)
        b_mz = prefix_mass + PROTON_MASS
        all_mz.append(b_mz)
        all_mz.append((b_mz + PROTON_MASS) / 2)  # b++

    # y-ions (singly charged)
    suffix_mass = WATER_MASS
    for i in range(n - 1, 0, -1):
        suffix_mass += AMINO_ACID_MASSES.get(sequence[i], 0.0)
        y_mz = suffix_mass + PROTON_MASS
        all_mz.append(y_mz)
        all_mz.append((y_mz + PROTON_MASS) / 2)  # y++

    return all_mz


def count_explainable_peaks(observed_masses, theoretical_mz, tolerance=0.5):
    """Count how many observed peaks match any theoretical ion."""
    explained = 0
    for obs in observed_masses:
        if any(abs(obs - theo) < tolerance for theo in theoretical_mz):
            explained += 1
    return explained


def analyze_dataset(dataset, name, num_samples=100):
    """Analyze signal-to-noise for a dataset."""
    print(f"\n{'='*60}")
    print(f"SIGNAL-TO-NOISE ANALYSIS: {name}")
    print('='*60)

    loader = DataLoader(dataset, batch_size=20, collate_fn=collate_fn)

    total_peaks = []
    explained_peaks = []
    unexplained_peaks = []
    pct_explained = []

    samples_processed = 0

    for batch in loader:
        for i in range(len(batch['sequence'])):
            if samples_processed >= num_samples:
                break

            seq = decode_sequence(batch['sequence'][i], batch['sequence_mask'][i])
            num_peaks = int(batch['spectrum_mask'][i].sum().item())
            observed_masses = batch['spectrum_masses'][i][:num_peaks].numpy()

            theoretical = compute_all_theoretical_mz(seq)
            n_explained = count_explainable_peaks(observed_masses, theoretical)
            n_unexplained = num_peaks - n_explained

            total_peaks.append(num_peaks)
            explained_peaks.append(n_explained)
            unexplained_peaks.append(n_unexplained)
            if num_peaks > 0:
                pct_explained.append(n_explained / num_peaks)

            samples_processed += 1

        if samples_processed >= num_samples:
            break

    print(f"\nSamples analyzed: {samples_processed}")
    print(f"\nPeaks per spectrum:")
    print(f"  Total:      {np.mean(total_peaks):.1f} ± {np.std(total_peaks):.1f}")
    print(f"  Explained:  {np.mean(explained_peaks):.1f} ± {np.std(explained_peaks):.1f}")
    print(f"  Unexplained: {np.mean(unexplained_peaks):.1f} ± {np.std(unexplained_peaks):.1f}")
    print(f"\n% of peaks explainable by b/y ions:")
    print(f"  {np.mean(pct_explained)*100:.1f}% ± {np.std(pct_explained)*100:.1f}%")

    return {
        'total_peaks': np.mean(total_peaks),
        'explained_peaks': np.mean(explained_peaks),
        'pct_explained': np.mean(pct_explained),
    }


def main():
    data_dir = project_root / 'data'

    print("=" * 70)
    print("SIGNAL-TO-NOISE ANALYSIS")
    print("What % of observed peaks are explainable b/y ions?")
    print("=" * 70)

    results = {}

    # MS2PIP clean (no noise)
    print("\nLoading MS2PIP (clean)...")
    ms2pip_clean = MS2PIPSyntheticDataset(
        min_length=10, max_length=20,
        noise_peaks=0, peak_dropout=0.0,
        ms2pip_model='HCDch2',
    )
    results['MS2PIP (clean)'] = analyze_dataset(ms2pip_clean, "MS2PIP (clean)", 100)

    # MS2PIP with curriculum noise
    print("\nLoading MS2PIP (with noise)...")
    ms2pip_noisy = MS2PIPSyntheticDataset(
        min_length=10, max_length=20,
        noise_peaks=30, peak_dropout=0.45,
        ms2pip_model='HCDch2',
    )
    ms2pip_noisy.set_difficulty(clean_data_ratio=0.0)  # Force all noisy
    results['MS2PIP (noisy)'] = analyze_dataset(ms2pip_noisy, "MS2PIP (curriculum noise)", 100)

    # ProteomeTools
    pt_dir = data_dir / 'proteometools'
    if pt_dir.exists():
        print("\nLoading ProteomeTools...")
        try:
            pt = ProteomeToolsDataset(
                data_dir=pt_dir, split='val',
                max_peaks=100, max_seq_len=35,
                max_samples=500,
            )
            results['ProteomeTools'] = analyze_dataset(pt, "ProteomeTools", 100)
        except Exception as e:
            print(f"  Failed: {e}")

    # Nine-Species
    ns_dir = data_dir / 'nine_species'
    if ns_dir.exists():
        print("\nLoading Nine-Species...")
        try:
            ns = NineSpeciesDataset(
                data_dir=ns_dir, split='val',
                max_peaks=100, max_seq_len=35,
                use_balanced=True, max_samples=500,
            )
            results['Nine-Species'] = analyze_dataset(ns, "Nine-Species", 100)
        except Exception as e:
            print(f"  Failed: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: SIGNAL-TO-NOISE COMPARISON")
    print("=" * 70)
    print(f"\n{'Dataset':<25} {'Total Peaks':<15} {'Explained':<15} {'% Explained':<15}")
    print("-" * 70)
    for name, r in results.items():
        print(f"{name:<25} {r['total_peaks']:<15.1f} {r['explained_peaks']:<15.1f} {r['pct_explained']*100:<15.1f}%")

    print("\n" + "=" * 70)
    print("KEY INSIGHT")
    print("=" * 70)
    print("""
The model is trained on MS2PIP where:
- Clean: 100% of peaks are explainable b/y ions
- With curriculum noise: ~50% explainable + ~50% random noise

Real data has:
- Many peaks that are NOT b/y ions (internal fragments, neutral losses, etc.)
- These are not random noise - they have structure the model doesn't understand
- Missing many expected b/y ions

This is a DISTRIBUTION SHIFT problem, not just noise robustness.
""")


if __name__ == '__main__':
    main()
