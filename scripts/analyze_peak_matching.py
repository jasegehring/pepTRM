"""
Deep analysis of peak matching between real data and theoretical spectra.

Key question: Do the observed peaks in real data correspond to the
expected b/y ion masses for the sequence?

If the peaks don't match the expected ions, that would explain why
the model can't learn anything from real data.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from torch.utils.data import DataLoader
from collections import Counter

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
    """Convert token tensor to string."""
    seq = []
    for idx, valid in zip(seq_tensor.tolist(), mask_tensor.tolist()):
        if not valid:
            break
        if idx < len(IDX_TO_AA):
            aa = IDX_TO_AA[idx]
            if aa not in ['<PAD>', '<SOS>', '<EOS>', '<UNK>']:
                seq.append(aa)
    return ''.join(seq)


def compute_theoretical_ions(sequence: str):
    """
    Compute theoretical b and y ion m/z values for a peptide sequence.

    IMPORTANT: Returns m/z, not neutral mass!
    - Singly charged: m/z = M_neutral + proton
    - Doubly charged: m/z = (M_neutral + 2*proton) / 2

    Returns:
        b_ions: list of b-ion m/z (singly charged)
        y_ions: list of y-ion m/z (singly charged)
    """
    n = len(sequence)

    # b-ions: cumulative mass from N-terminus + proton for [M+H]+
    prefix_mass = 0.0
    b_ions = []
    for i in range(n - 1):  # b1 to b_{n-1}
        prefix_mass += AMINO_ACID_MASSES.get(sequence[i], 0.0)
        b_ions.append(prefix_mass + PROTON_MASS)  # [M+H]+

    # y-ions: cumulative mass from C-terminus + H2O + proton for [M+H]+
    suffix_mass = WATER_MASS
    y_ions = []
    for i in range(n - 1, 0, -1):  # y1 to y_{n-1}
        suffix_mass += AMINO_ACID_MASSES.get(sequence[i], 0.0)
        y_ions.append(suffix_mass + PROTON_MASS)  # [M+H]+

    return b_ions, y_ions


def compute_doubly_charged(ions):
    """
    Compute doubly-charged (++/2+) m/z from singly-charged m/z.
    ions are already [M+H]+, so: [M+2H]2+ = (M + 2H) / 2 = ([M+H] + H) / 2
    """
    return [(m + PROTON_MASS) / 2 for m in ions]


def find_matching_peaks(observed_masses, theoretical_masses, tolerance_da=0.5):
    """
    Find how many theoretical masses have a matching observed peak.

    Returns:
        num_matched: number of theoretical masses with a nearby observed peak
        total_theoretical: total number of theoretical masses
    """
    num_matched = 0
    for theo_mass in theoretical_masses:
        # Check if any observed mass is within tolerance
        min_diff = min(abs(obs - theo_mass) for obs in observed_masses) if len(observed_masses) > 0 else float('inf')
        if min_diff <= tolerance_da:
            num_matched += 1
    return num_matched, len(theoretical_masses)


def analyze_dataset(dataset, name, num_samples=100):
    """Analyze peak matching for a dataset."""
    print(f"\n{'='*60}")
    print(f"PEAK MATCHING ANALYSIS: {name}")
    print('='*60)

    loader = DataLoader(dataset, batch_size=20, collate_fn=collate_fn)

    # Track statistics
    b_match_rates = []
    y_match_rates = []
    b2_match_rates = []
    y2_match_rates = []
    total_match_rates = []

    precursor_mass_errors = []
    has_modification = []

    samples_processed = 0

    for batch in loader:
        for i in range(len(batch['sequence'])):
            if samples_processed >= num_samples:
                break

            # Get sequence and spectrum
            seq = decode_sequence(batch['sequence'][i], batch['sequence_mask'][i])
            num_peaks = int(batch['spectrum_mask'][i].sum().item())
            observed_masses = batch['spectrum_masses'][i][:num_peaks].numpy()

            # Get precursor info
            precursor_mass = batch['precursor_mass'][i]
            if hasattr(precursor_mass, 'squeeze'):
                precursor_mass = precursor_mass.squeeze().item()
            else:
                precursor_mass = float(precursor_mass)

            theoretical_mass = sum(AMINO_ACID_MASSES.get(aa, 0) for aa in seq) + WATER_MASS
            mass_error = abs(precursor_mass - theoretical_mass)
            precursor_mass_errors.append(mass_error)
            has_modification.append(mass_error > 5)  # More than 5 Da suggests modification

            # Compute theoretical ions
            b_ions, y_ions = compute_theoretical_ions(seq)
            b2_ions = compute_doubly_charged(b_ions)
            y2_ions = compute_doubly_charged(y_ions)

            # Find matches
            b_matched, b_total = find_matching_peaks(observed_masses, b_ions, tolerance_da=0.5)
            y_matched, y_total = find_matching_peaks(observed_masses, y_ions, tolerance_da=0.5)
            b2_matched, b2_total = find_matching_peaks(observed_masses, b2_ions, tolerance_da=0.5)
            y2_matched, y2_total = find_matching_peaks(observed_masses, y2_ions, tolerance_da=0.5)

            if b_total > 0:
                b_match_rates.append(b_matched / b_total)
            if y_total > 0:
                y_match_rates.append(y_matched / y_total)
            if b2_total > 0:
                b2_match_rates.append(b2_matched / b2_total)
            if y2_total > 0:
                y2_match_rates.append(y2_matched / y2_total)

            # Total match rate
            total_theoretical = b_total + y_total + b2_total + y2_total
            total_matched = b_matched + y_matched + b2_matched + y2_matched
            if total_theoretical > 0:
                total_match_rates.append(total_matched / total_theoretical)

            samples_processed += 1

        if samples_processed >= num_samples:
            break

    # Report results
    print(f"\nSamples analyzed: {samples_processed}")
    print(f"\nPrecursor mass errors:")
    print(f"  Mean: {np.mean(precursor_mass_errors):.2f} Da")
    print(f"  Median: {np.median(precursor_mass_errors):.2f} Da")
    print(f"  Max: {np.max(precursor_mass_errors):.2f} Da")
    print(f"  Samples with likely modifications (>5 Da): {sum(has_modification)} ({100*sum(has_modification)/len(has_modification):.1f}%)")

    print(f"\nTheoretical ion coverage (how many expected peaks are found):")
    print(f"  b-ions:   {np.mean(b_match_rates)*100:.1f}% ± {np.std(b_match_rates)*100:.1f}%")
    print(f"  y-ions:   {np.mean(y_match_rates)*100:.1f}% ± {np.std(y_match_rates)*100:.1f}%")
    print(f"  b++-ions: {np.mean(b2_match_rates)*100:.1f}% ± {np.std(b2_match_rates)*100:.1f}%")
    print(f"  y++-ions: {np.mean(y2_match_rates)*100:.1f}% ± {np.std(y2_match_rates)*100:.1f}%")
    print(f"  TOTAL:    {np.mean(total_match_rates)*100:.1f}% ± {np.std(total_match_rates)*100:.1f}%")

    # Compare modified vs unmodified
    if sum(has_modification) > 0 and sum(has_modification) < len(has_modification):
        mod_rates = [r for r, m in zip(total_match_rates, has_modification) if m]
        unmod_rates = [r for r, m in zip(total_match_rates, has_modification) if not m]
        print(f"\nBreakdown by modification status:")
        print(f"  Unmodified: {np.mean(unmod_rates)*100:.1f}% coverage ({len(unmod_rates)} samples)")
        print(f"  Modified:   {np.mean(mod_rates)*100:.1f}% coverage ({len(mod_rates)} samples)")

    return {
        'b_match': np.mean(b_match_rates),
        'y_match': np.mean(y_match_rates),
        'total_match': np.mean(total_match_rates),
        'mass_error': np.mean(precursor_mass_errors),
        'pct_modified': sum(has_modification) / len(has_modification),
    }


def main():
    data_dir = project_root / 'data'

    print("=" * 70)
    print("PEAK MATCHING ANALYSIS")
    print("Checking if observed peaks match expected b/y ions")
    print("=" * 70)

    results = {}

    # 1. MS2PIP (baseline - should be nearly perfect)
    print("\nLoading MS2PIP...")
    ms2pip = MS2PIPSyntheticDataset(
        min_length=10,
        max_length=20,
        noise_peaks=0,
        peak_dropout=0.0,
        mass_error_ppm=0.0,
        ms2pip_model='HCDch2',
    )
    results['MS2PIP'] = analyze_dataset(ms2pip, "MS2PIP (Training Data)", num_samples=100)

    # 2. ProteomeTools
    pt_dir = data_dir / 'proteometools'
    if pt_dir.exists():
        print("\nLoading ProteomeTools...")
        try:
            pt = ProteomeToolsDataset(
                data_dir=pt_dir,
                split='val',
                max_peaks=100,
                max_seq_len=35,
                max_samples=500,
            )
            results['ProteomeTools'] = analyze_dataset(pt, "ProteomeTools (Real Data)", num_samples=100)
        except Exception as e:
            print(f"  Failed: {e}")

    # 3. Nine-Species
    ns_dir = data_dir / 'nine_species'
    if ns_dir.exists():
        print("\nLoading Nine-Species...")
        try:
            ns = NineSpeciesDataset(
                data_dir=ns_dir,
                split='val',
                max_peaks=100,
                max_seq_len=35,
                use_balanced=True,
                max_samples=500,
            )
            results['Nine-Species'] = analyze_dataset(ns, "Nine-Species (Real Data)", num_samples=100)
        except Exception as e:
            print(f"  Failed: {e}")

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)
    print(f"\n{'Dataset':<20} {'Total Coverage':<15} {'Mass Error':<12} {'% Modified':<12}")
    print("-" * 60)
    for name, r in results.items():
        print(f"{name:<20} {r['total_match']*100:>6.1f}%        {r['mass_error']:>6.1f} Da    {r['pct_modified']*100:>6.1f}%")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if 'MS2PIP' in results and 'ProteomeTools' in results:
        ms2pip_cov = results['MS2PIP']['total_match']
        pt_cov = results['ProteomeTools']['total_match']

        if pt_cov < 0.5 * ms2pip_cov:
            print("\n⚠️  ProteomeTools has much lower ion coverage than MS2PIP!")
            print("    This suggests the real spectra have different peak patterns.")
            print("    The model trained on MS2PIP won't recognize these patterns.")
        else:
            print("\n✓ ProteomeTools ion coverage is comparable to MS2PIP")

    if 'Nine-Species' in results:
        ns_cov = results['Nine-Species']['total_match']
        if ns_cov < 0.3:
            print("\n⚠️  Nine-Species has very low ion coverage (<30%)!")
            print("    Real biological data is very different from training data.")


if __name__ == '__main__':
    main()
