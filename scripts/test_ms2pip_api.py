"""Quick test to understand MS2PIP v4.1.0 API."""

from ms2pip import predict_single
import numpy as np

# Test with charge 2
print("=" * 80)
print("TEST 1: PEPTIDE/2")
print("=" * 80)

result = predict_single('PEPTIDE/2', model='HCD2021')

print(f"\nIon types: {list(result.theoretical_mz.keys())}")

for ion_type in result.theoretical_mz.keys():
    mz_vals = result.theoretical_mz[ion_type]
    int_vals = result.predicted_intensity[ion_type]
    print(f"\n{ion_type} ions:")
    print(f"  Count: {len(mz_vals)}")
    print(f"  mz range: {mz_vals.min():.2f} - {mz_vals.max():.2f}")
    print(f"  intensity range: {int_vals.min():.4f} - {int_vals.max():.4f}")
    print(f"  First 3 mz: {mz_vals[:3]}")
    print(f"  First 3 int: {int_vals[:3]}")

# Test with charge 3 to see if we get doubly-charged fragments
print("\n" + "=" * 80)
print("TEST 2: PEPTIDE/3")
print("=" * 80)

result = predict_single('PEPTIDE/3', model='HCD2021')

print(f"\nIon types: {list(result.theoretical_mz.keys())}")

for ion_type in result.theoretical_mz.keys():
    mz_vals = result.theoretical_mz[ion_type]
    int_vals = result.predicted_intensity[ion_type]
    print(f"\n{ion_type} ions:")
    print(f"  Count: {len(mz_vals)}")
    print(f"  mz range: {mz_vals.min():.2f} - {mz_vals.max():.2f}")
    print(f"  intensity range: {int_vals.min():.4f} - {int_vals.max():.4f}")

# Check if intensities are log-space
print("\n" + "=" * 80)
print("TEST 3: INTENSITY TRANSFORMATION")
print("=" * 80)

result = predict_single('PEPTIDE/2', model='HCD2021')
b_int = result.predicted_intensity['b']

print(f"\nRaw intensities (first 3): {b_int[:3]}")
print(f"After exp() (first 3): {np.exp(b_int[:3])}")
print(f"After exp() normalized: {np.exp(b_int[:3]) / np.exp(b_int).max()}")

# Try longer peptide
print("\n" + "=" * 80)
print("TEST 4: LONGER PEPTIDE")
print("=" * 80)

result = predict_single('SEQUENCEPEPTIDE/2', model='HCD2021')

total_fragments = sum(len(result.theoretical_mz[k]) for k in result.theoretical_mz)
print(f"\nPeptide: SEQUENCEPEPTIDE (15 amino acids)")
print(f"Total fragments predicted: {total_fragments}")
for ion_type in result.theoretical_mz.keys():
    print(f"  {ion_type}: {len(result.theoretical_mz[ion_type])} fragments")
