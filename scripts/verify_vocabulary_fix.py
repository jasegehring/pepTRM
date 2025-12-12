"""
Verify that the vocabulary indexing fix resolves the precursor mass issue.

This script tests that:
1. Dataset encodes sequences with correct token indices
2. Loss functions decode those indices to correct masses
3. Precursor mass calculation is accurate
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from src.data.ms2pip_dataset import MS2PIPSyntheticDataset
from src.constants import AMINO_ACID_MASSES, WATER_MASS, VOCAB, AA_TO_IDX

def test_vocabulary_consistency():
    """Test that dataset uses same vocabulary as loss functions."""
    print("Testing vocabulary consistency...")

    # Create dataset
    dataset = MS2PIPSyntheticDataset(
        min_length=7,
        max_length=10,
        ms2pip_model='HCDch2',
    )

    # Check that dataset uses correct token mapping
    assert dataset.token_to_idx == AA_TO_IDX, "Dataset token_to_idx doesn't match AA_TO_IDX!"
    print("✓ Dataset uses correct token_to_idx mapping")

    # Generate a sample
    sample = next(iter(dataset))

    # Decode the sequence
    sequence = sample.sequence
    sequence_mask = sample.sequence_mask

    # Get non-padding tokens
    valid_tokens = sequence[sequence_mask]

    # Decode to amino acids
    decoded_aas = [dataset.idx_to_token[idx.item()] for idx in valid_tokens]
    print(f"  Sample sequence: {''.join(decoded_aas)}")

    # Calculate mass using dataset's understanding
    expected_mass = sum(AMINO_ACID_MASSES[aa] for aa in decoded_aas) + WATER_MASS
    print(f"  Expected precursor mass: {expected_mass:.4f} Da")
    print(f"  Dataset precursor mass:  {sample.precursor_mass.item():.4f} Da")

    # Verify they match
    mass_diff = abs(expected_mass - sample.precursor_mass.item())
    assert mass_diff < 0.001, f"Mass mismatch: {mass_diff:.6f} Da"
    print("✓ Dataset precursor mass is correct")

    # Now test what the loss function will see
    print("\nTesting loss function calculation...")

    # Create aa_masses tensor like in the loss function
    aa_masses = torch.tensor([
        AMINO_ACID_MASSES.get(aa, 0.0) for aa in VOCAB
    ])

    # Simulate what loss function does:
    # Create one-hot encoding (perfect predictions)
    batch_size = 1
    max_seq_len = sequence.shape[0]
    vocab_size = len(VOCAB)

    # Create probability distribution (one-hot for correct tokens)
    sequence_probs = torch.zeros(batch_size, max_seq_len, vocab_size)
    for i, (token_id, valid) in enumerate(zip(sequence, sequence_mask)):
        if valid:
            sequence_probs[0, i, token_id] = 1.0

    # Calculate expected masses like in PrecursorMassLoss
    expected_masses = torch.einsum('bsv,v->bs', sequence_probs, aa_masses)
    masked_masses = expected_masses * sequence_mask.unsqueeze(0).float()
    predicted_peptide_mass = masked_masses.sum(dim=1).item()
    predicted_precursor_mass = predicted_peptide_mass + WATER_MASS

    print(f"  Loss function predicted mass: {predicted_precursor_mass:.4f} Da")
    print(f"  Target mass: {sample.precursor_mass.item():.4f} Da")

    # Check if they match
    loss_diff = abs(predicted_precursor_mass - sample.precursor_mass.item())
    print(f"  Error: {loss_diff:.6f} Da ({loss_diff / sample.precursor_mass.item() * 1e6:.2f} ppm)")

    if loss_diff < 0.001:
        print("✓ Loss function calculates correct mass!")
    else:
        print("✗ Loss function mass calculation is wrong!")
        return False

    return True

def test_amino_acid_masses():
    """Test that all amino acids have correct mass lookups."""
    print("\nTesting amino acid mass lookups...")

    # Create aa_masses tensor like in loss function
    aa_masses = torch.tensor([
        AMINO_ACID_MASSES.get(aa, 0.0) for aa in VOCAB
    ])

    # Check special tokens have 0 mass
    for special_token in ['<PAD>', '<SOS>', '<EOS>', '<UNK>']:
        idx = AA_TO_IDX[special_token]
        assert aa_masses[idx] == 0.0, f"{special_token} should have 0 mass"
    print("✓ Special tokens have 0 mass")

    # Check amino acids have correct masses
    for aa in AMINO_ACID_MASSES.keys():
        idx = AA_TO_IDX[aa]
        expected_mass = AMINO_ACID_MASSES[aa]
        actual_mass = aa_masses[idx].item()
        assert abs(expected_mass - actual_mass) < 0.0001, \
            f"Mass mismatch for {aa}: expected {expected_mass}, got {actual_mass}"
    print("✓ All amino acid masses are correct")

    return True

if __name__ == '__main__':
    print("=" * 60)
    print("Vocabulary Fix Verification")
    print("=" * 60)

    try:
        # Test amino acid mass lookups
        if not test_amino_acid_masses():
            print("\n✗ FAILED: Amino acid mass lookup test")
            sys.exit(1)

        # Test vocabulary consistency
        if not test_vocabulary_consistency():
            print("\n✗ FAILED: Vocabulary consistency test")
            sys.exit(1)

        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("The vocabulary fix is working correctly.")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
