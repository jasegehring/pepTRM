"""
Debug script to understand why peak matching is so low.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from src.data.ms2pip_dataset import MS2PIPSyntheticDataset
from src.constants import AMINO_ACID_MASSES, WATER_MASS, PROTON_MASS


def compute_theoretical_ions(sequence: str):
    """
    Compute theoretical b and y ion m/z values.

    Note: MS2PIP outputs m/z, not neutral mass!
    For singly charged ions: m/z = M_neutral + proton
    For doubly charged ions: m/z = (M_neutral + 2*proton) / 2
    """
    n = len(sequence)

    # b-ions: cumulative mass from N-terminus (neutral)
    prefix_mass = 0.0
    b_ions = []
    for i in range(n - 1):  # b1 to b_{n-1}
        prefix_mass += AMINO_ACID_MASSES.get(sequence[i], 0.0)
        # Singly charged: [M+H]+
        b_ions.append(prefix_mass + PROTON_MASS)

    # y-ions: cumulative mass from C-terminus + H2O (neutral)
    suffix_mass = WATER_MASS
    y_ions = []
    for i in range(n - 1, 0, -1):  # y1 to y_{n-1}
        suffix_mass += AMINO_ACID_MASSES.get(sequence[i], 0.0)
        # Singly charged: [M+H]+
        y_ions.append(suffix_mass + PROTON_MASS)

    # Doubly charged: [M+2H]2+
    b2_ions = [(m - PROTON_MASS + 2*PROTON_MASS) / 2 for m in b_ions]  # Convert back to neutral then add 2H and divide
    y2_ions = [(m - PROTON_MASS + 2*PROTON_MASS) / 2 for m in y_ions]

    return b_ions, y_ions, b2_ions, y2_ions


def main():
    # Generate a single MS2PIP sample
    dataset = MS2PIPSyntheticDataset(
        min_length=10,
        max_length=10,  # Fixed length for clarity
        noise_peaks=0,
        peak_dropout=0.0,
        mass_error_ppm=0.0,
        ms2pip_model='HCDch2',
    )

    sample = next(iter(dataset))

    # Decode sequence
    from src.constants import IDX_TO_AA
    seq_tokens = sample.sequence[:sample.sequence_mask.sum().item()].tolist()
    sequence = ''.join(IDX_TO_AA.get(t, '?') for t in seq_tokens if IDX_TO_AA.get(t, '?') not in ['<PAD>', '<SOS>', '<EOS>', '<UNK>'])

    print(f"Sequence: {sequence}")
    print(f"Length: {len(sequence)}")

    # Get observed peaks
    num_peaks = sample.spectrum_mask.sum().item()
    observed_masses = sample.spectrum_masses[:num_peaks].numpy()
    observed_intensities = sample.spectrum_intensities[:num_peaks].numpy()

    print(f"\nObserved peaks: {num_peaks}")

    # Compute theoretical ions
    b_ions, y_ions, b2_ions, y2_ions = compute_theoretical_ions(sequence)

    print(f"\nTheoretical b-ions: {len(b_ions)}")
    for i, m in enumerate(b_ions):
        print(f"  b{i+1}: {m:.4f}")

    print(f"\nTheoretical y-ions: {len(y_ions)}")
    for i, m in enumerate(y_ions):
        print(f"  y{i+1}: {m:.4f}")

    print(f"\nTheoretical b++-ions:")
    for i, m in enumerate(b2_ions):
        print(f"  b{i+1}++: {m:.4f}")

    print(f"\nTheoretical y++-ions:")
    for i, m in enumerate(y2_ions):
        print(f"  y{i+1}++: {m:.4f}")

    # Print observed masses sorted
    print(f"\nObserved masses (sorted):")
    sorted_idx = np.argsort(observed_masses)
    for i in sorted_idx:
        m = observed_masses[i]
        intensity = observed_intensities[i]

        # Check if matches any theoretical
        all_theo = b_ions + y_ions + b2_ions + y2_ions
        min_diff = min(abs(m - t) for t in all_theo) if all_theo else float('inf')

        match_str = ""
        if min_diff < 0.5:
            match_str = f" <- MATCH (diff={min_diff:.4f})"

        print(f"  {m:.4f} (int={intensity:.4f}){match_str}")

    # Count matches
    print("\n" + "="*60)
    print("MATCHING ANALYSIS")
    print("="*60)

    all_theo = b_ions + y_ions + b2_ions + y2_ions
    matched = 0
    for theo in all_theo:
        diffs = [abs(obs - theo) for obs in observed_masses]
        min_diff = min(diffs) if diffs else float('inf')
        if min_diff < 0.5:
            matched += 1

    print(f"Theoretical ions: {len(all_theo)}")
    print(f"Matched within 0.5 Da: {matched}")
    print(f"Coverage: {matched/len(all_theo)*100:.1f}%")


if __name__ == '__main__':
    main()
