"""
Diagnostic script for real MS/MS datasets (ProteomeTools, Nine-Species).

This script investigates:
1. Data format and structure
2. Modification patterns and filtering statistics
3. Spectrum characteristics (peak counts, mass ranges)
4. Compare with MS2PIP training distribution

Usage:
    python scripts/diagnose_real_data.py
"""

import sys
from pathlib import Path
import re
from collections import Counter

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from src.constants import AA_TO_IDX, AMINO_ACID_MASSES, WATER_MASS


def parse_modification(seq_with_mods: str):
    """
    Parse a modified peptide sequence to extract:
    1. Base sequence (AA only)
    2. Modifications (position, delta mass)

    Common modification patterns:
    - M+15.995 = Oxidized methionine
    - C+57.021 = Carbamidomethyl cysteine
    - Q+0.984 = Deamidated glutamine
    - N+0.984 = Deamidated asparagine
    - -17.027Q = Pyroglutamate (N-term)
    - +43.006 = N-term acetylation/carbamylation
    """
    # Pattern: AA followed by +/- number
    # Or: +/- number at start (N-term mod)

    # Remove N-terminal mods (start with + or -)
    if seq_with_mods and seq_with_mods[0] in '+-':
        match = re.match(r'^[+-][\d.]+', seq_with_mods)
        if match:
            seq_with_mods = seq_with_mods[match.end():]

    # Remove inline mods (AA+delta or AA-delta)
    base_seq = re.sub(r'[+-][\d.]+', '', seq_with_mods)

    # Check if base sequence is valid
    is_valid = all(aa in AA_TO_IDX for aa in base_seq)

    return base_seq, is_valid


def analyze_nine_species(data_dir: Path):
    """Analyze Nine-Species MGF files."""
    print("\n" + "=" * 70)
    print("NINE-SPECIES DATASET ANALYSIS")
    print("=" * 70)

    mgf_dir = data_dir / 'nine_species' / 'nine-species-balanced'
    if not mgf_dir.exists():
        print(f"  Directory not found: {mgf_dir}")
        return

    # Find all MGF files
    mgf_files = list(mgf_dir.glob("*.mgf"))
    print(f"\nFound {len(mgf_files)} MGF files")

    total_spectra = 0
    total_modified = 0
    total_valid_after_strip = 0
    total_invalid = 0
    mod_types = Counter()
    seq_lengths = []
    peak_counts = []
    charges = Counter()
    precursor_mzs = []

    for mgf_file in mgf_files:
        species_name = mgf_file.stem
        print(f"\n--- {species_name} ---")

        file_spectra = 0
        file_modified = 0
        file_valid = 0

        with open(mgf_file, 'r') as f:
            current_peptide = None
            current_charge = None
            current_peaks = 0
            current_mz = None

            for line in f:
                line = line.strip()

                if line == 'BEGIN IONS':
                    current_peptide = None
                    current_charge = None
                    current_peaks = 0
                    current_mz = None

                elif line.startswith('SEQ='):
                    current_peptide = line[4:]

                elif line.startswith('CHARGE='):
                    try:
                        current_charge = int(line[7:].rstrip('+').rstrip('-'))
                    except:
                        current_charge = 2

                elif line.startswith('PEPMASS='):
                    try:
                        current_mz = float(line.split()[0][8:])
                    except:
                        current_mz = None

                elif line == 'END IONS':
                    if current_peptide:
                        file_spectra += 1

                        # Check for modifications
                        has_mod = '+' in current_peptide or (current_peptide and current_peptide[0] == '-')
                        if has_mod:
                            file_modified += 1
                            # Extract modification types
                            mods = re.findall(r'[A-Z]?([+-][\d.]+)', current_peptide)
                            for mod in mods:
                                mod_types[mod] += 1

                        # Parse and validate
                        base_seq, is_valid = parse_modification(current_peptide)
                        if is_valid:
                            file_valid += 1
                            seq_lengths.append(len(base_seq))

                        if current_charge:
                            charges[current_charge] += 1

                        if current_mz:
                            precursor_mzs.append(current_mz)

                elif line and not line.startswith('#') and '=' not in line:
                    # Peak line
                    current_peaks += 1

            # Record peak count for last spectrum
            if current_peaks > 0:
                peak_counts.append(current_peaks)

        print(f"  Total spectra: {file_spectra:,}")
        print(f"  Modified spectra: {file_modified:,} ({100*file_modified/max(file_spectra,1):.1f}%)")
        print(f"  Valid after stripping mods: {file_valid:,} ({100*file_valid/max(file_spectra,1):.1f}%)")

        total_spectra += file_spectra
        total_modified += file_modified
        total_valid_after_strip += file_valid
        total_invalid += (file_spectra - file_valid)

    # Summary
    print("\n" + "-" * 40)
    print("SUMMARY")
    print("-" * 40)
    print(f"Total spectra: {total_spectra:,}")
    print(f"Modified spectra: {total_modified:,} ({100*total_modified/total_spectra:.1f}%)")
    print(f"Valid after mod stripping: {total_valid_after_strip:,} ({100*total_valid_after_strip/total_spectra:.1f}%)")
    print(f"Invalid (contain unknown chars): {total_invalid:,} ({100*total_invalid/total_spectra:.1f}%)")

    print(f"\nTop 10 modification types:")
    for mod, count in mod_types.most_common(10):
        # Identify modification
        mass = float(mod.replace('+', '').replace('-', '-'))
        if abs(mass - 15.995) < 0.1:
            name = "Oxidation (M)"
        elif abs(mass - 57.021) < 0.1:
            name = "Carbamidomethyl (C)"
        elif abs(mass - 0.984) < 0.1:
            name = "Deamidation (N/Q)"
        elif abs(mass + 17.027) < 0.1:
            name = "Pyroglutamate (-Q)"
        elif abs(mass - 43.006) < 0.1:
            name = "Carbamyl/Acetyl (N-term)"
        else:
            name = "Unknown"
        print(f"  {mod}: {count:,} ({name})")

    print(f"\nCharge distribution:")
    for charge, count in sorted(charges.items()):
        print(f"  {charge}+: {count:,} ({100*count/total_spectra:.1f}%)")

    if seq_lengths:
        print(f"\nSequence length distribution (unmodified):")
        print(f"  Min: {min(seq_lengths)}, Max: {max(seq_lengths)}, Mean: {np.mean(seq_lengths):.1f}")

    if precursor_mzs:
        print(f"\nPrecursor m/z distribution:")
        print(f"  Min: {min(precursor_mzs):.1f}, Max: {max(precursor_mzs):.1f}, Mean: {np.mean(precursor_mzs):.1f}")


def analyze_proteometools(data_dir: Path):
    """Analyze ProteomeTools MSP file."""
    print("\n" + "=" * 70)
    print("PROTEOMETOOLS DATASET ANALYSIS")
    print("=" * 70)

    msp_dir = data_dir / 'proteometools'
    if not msp_dir.exists():
        print(f"  Directory not found: {msp_dir}")
        return

    msp_files = list(msp_dir.glob("*.msp"))
    if not msp_files:
        print("  No .msp files found")
        return

    print(f"\nFound {len(msp_files)} MSP files")

    total_spectra = 0
    total_modified = 0
    total_valid = 0
    mod_counts = Counter()
    seq_lengths = []
    peak_counts = []
    charges = Counter()

    for msp_file in msp_files:
        print(f"\n--- {msp_file.name} ---")

        file_spectra = 0
        file_modified = 0
        file_valid = 0

        current_peptide = None
        current_mods = None
        current_charge = None
        current_peaks = 0

        with open(msp_file, 'r') as f:
            for line in f:
                line = line.strip()

                if line.startswith('Name:'):
                    # Save previous spectrum info
                    if current_peptide:
                        file_spectra += 1
                        if current_mods and current_mods != '0':
                            file_modified += 1
                        if all(aa in AA_TO_IDX for aa in current_peptide):
                            file_valid += 1
                            seq_lengths.append(len(current_peptide))
                        if current_charge:
                            charges[current_charge] += 1
                        if current_peaks > 0:
                            peak_counts.append(current_peaks)

                    # Parse new spectrum name
                    name = line.split(':', 1)[1].strip()
                    parts = name.split('/')
                    if len(parts) >= 2:
                        current_peptide = parts[0]
                        # Remove any bracket modifications
                        current_peptide = re.sub(r'\[.*?\]', '', current_peptide)
                        try:
                            current_charge = int(parts[1].split('_')[0])
                        except:
                            current_charge = 2
                    current_mods = None
                    current_peaks = 0

                elif line.startswith('Comment:'):
                    # Parse Mods=N from comment
                    match = re.search(r'Mods=(\d+)', line)
                    if match:
                        current_mods = match.group(1)
                        if current_mods != '0':
                            mod_counts[current_mods] += 1

                elif line.startswith('Num peaks:'):
                    try:
                        current_peaks = int(line.split(':')[1].strip())
                    except:
                        pass

            # Don't forget last spectrum
            if current_peptide:
                file_spectra += 1
                if current_mods and current_mods != '0':
                    file_modified += 1
                if all(aa in AA_TO_IDX for aa in current_peptide):
                    file_valid += 1
                    seq_lengths.append(len(current_peptide))
                if current_peaks > 0:
                    peak_counts.append(current_peaks)

        print(f"  Total spectra: {file_spectra:,}")
        print(f"  Modified spectra (Mods>0): {file_modified:,} ({100*file_modified/max(file_spectra,1):.1f}%)")
        print(f"  Valid unmodified: {file_valid:,} ({100*file_valid/max(file_spectra,1):.1f}%)")

        total_spectra += file_spectra
        total_modified += file_modified
        total_valid += file_valid

        # Stop after first file for large datasets
        if file_spectra > 100000:
            print(f"  (Sampled first {file_spectra:,} spectra)")
            break

    print("\n" + "-" * 40)
    print("SUMMARY")
    print("-" * 40)
    print(f"Total spectra sampled: {total_spectra:,}")
    print(f"Modified spectra: {total_modified:,} ({100*total_modified/total_spectra:.1f}%)")
    print(f"Valid unmodified: {total_valid:,} ({100*total_valid/total_spectra:.1f}%)")

    print(f"\nCharge distribution:")
    for charge, count in sorted(charges.items()):
        print(f"  {charge}+: {count:,} ({100*count/total_spectra:.1f}%)")

    if seq_lengths:
        print(f"\nSequence length distribution:")
        print(f"  Min: {min(seq_lengths)}, Max: {max(seq_lengths)}, Mean: {np.mean(seq_lengths):.1f}")

    if peak_counts:
        print(f"\nPeak count distribution:")
        print(f"  Min: {min(peak_counts)}, Max: {max(peak_counts)}, Mean: {np.mean(peak_counts):.1f}")


def compare_with_ms2pip():
    """Compare real data characteristics with MS2PIP training distribution."""
    print("\n" + "=" * 70)
    print("COMPARISON: REAL DATA vs MS2PIP TRAINING")
    print("=" * 70)

    print("""
MS2PIP Training Data (what we train on):
  - Clean theoretical spectra from MS2PIP
  - Only b, y, b++, y++ ions
  - No modifications (M, C are unmodified)
  - Length 7-25 (curriculum controlled)
  - Charge 2+ (70%) and 3+ (30%)
  - Noise added via curriculum (up to 30 peaks, 45% dropout)

ProteomeTools:
  - Real HCD spectra from synthetic peptide library
  - Includes all ion types + neutral losses + internal fragments
  - Low modification rate (mostly unmodified by design)
  - Clean, high-quality spectra

Nine-Species:
  - Real biological data
  - ~30% modified (Oxidation M, Carbamidomethyl C, Deamidation N/Q)
  - Complex spectra with many unexplained peaks
  - Variable quality, charge states 2-6+

Key Gaps:
  1. MODIFICATIONS: Model doesn't know about C+57, M+16 etc.
     - MS2PIP uses unmodified masses
     - Real data has modified masses in spectra
     - This creates systematic mass errors

  2. ION TYPES: Model expects b/y/b++/y++
     - Real spectra have neutral losses (H2O, NH3)
     - Internal fragments, a-ions, immonium ions
     - These appear as "noise" to our model

  3. INTENSITY DISTRIBUTION: MS2PIP != Reality
     - MS2PIP trained on ProteomeTools, so PT should be similar
     - Nine-Species from different instruments/methods

RECOMMENDATIONS:
  1. For ProteomeTools: Should work reasonably (same training data as MS2PIP)
  2. For Nine-Species: Need modification-aware model or filter to unmodified only
  3. Consider adding modification support to model vocabulary
""")


def main():
    data_dir = project_root / 'data'

    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return

    analyze_nine_species(data_dir)
    analyze_proteometools(data_dir)
    compare_with_ms2pip()

    print("\n" + "=" * 70)
    print("DIAGNOSIS COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
