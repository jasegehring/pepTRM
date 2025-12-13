"""
Diagnostic script to debug precursor mass loss.

This script tests the precursor loss computation step-by-step to identify bugs.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn.functional as F
from src.constants import (
    VOCAB, AA_TO_IDX, AMINO_ACID_MASSES, WATER_MASS, PAD_IDX
)
from src.training.losses import PrecursorMassLoss
from src.data.ms2pip_dataset import MS2PIPSyntheticDataset


def test_precursor_loss_basic():
    """Test basic precursor loss computation."""
    print("=" * 70)
    print("TEST 1: Basic Precursor Loss Computation")
    print("=" * 70)

    # Create a simple test case
    batch_size = 2
    seq_len = 5
    vocab_size = len(VOCAB)

    # Create ground truth sequences: "ALA" and "GLYK"
    # A = index 4, L = 11, G = 8, Y = 23, K = 10
    sequences = torch.tensor([
        [AA_TO_IDX['A'], AA_TO_IDX['L'], AA_TO_IDX['A'], PAD_IDX, PAD_IDX],
        [AA_TO_IDX['G'], AA_TO_IDX['L'], AA_TO_IDX['Y'], AA_TO_IDX['K'], PAD_IDX],
    ])
    sequence_mask = torch.tensor([
        [True, True, True, False, False],
        [True, True, True, True, False],
    ])

    # Calculate true precursor masses
    true_masses = torch.tensor([
        AMINO_ACID_MASSES['A'] + AMINO_ACID_MASSES['L'] + AMINO_ACID_MASSES['A'] + WATER_MASS,
        AMINO_ACID_MASSES['G'] + AMINO_ACID_MASSES['L'] + AMINO_ACID_MASSES['Y'] + AMINO_ACID_MASSES['K'] + WATER_MASS,
    ])

    print("\nGround Truth:")
    print(f"  Sequence 1: ALA")
    print(f"    Masses: {AMINO_ACID_MASSES['A']:.2f} + {AMINO_ACID_MASSES['L']:.2f} + {AMINO_ACID_MASSES['A']:.2f} + {WATER_MASS:.2f}")
    print(f"    True precursor mass: {true_masses[0]:.2f} Da")
    print(f"  Sequence 2: GLYK")
    print(f"    Masses: {AMINO_ACID_MASSES['G']:.2f} + {AMINO_ACID_MASSES['L']:.2f} + {AMINO_ACID_MASSES['Y']:.2f} + {AMINO_ACID_MASSES['K']:.2f} + {WATER_MASS:.2f}")
    print(f"    True precursor mass: {true_masses[1]:.2f} Da")

    # Create perfect predictions (one-hot)
    probs_perfect = torch.zeros(batch_size, seq_len, vocab_size)
    for b in range(batch_size):
        for s in range(seq_len):
            if sequence_mask[b, s]:
                probs_perfect[b, s, sequences[b, s]] = 1.0

    # Test with perfect predictions
    loss_fn = PrecursorMassLoss()
    loss, metrics = loss_fn(probs_perfect, true_masses, sequence_mask)

    print("\n--- Test with PERFECT predictions (should be ~0 error) ---")
    print(f"  Loss: {loss.item():.6f}")
    print(f"  Mass error (Da): {metrics['mass_error_da']:.2f}")
    print(f"  PPM error: {metrics['ppm_error']:.2f}")
    print(f"  AA error: {metrics['aa_error']:.6f}")

    if metrics['ppm_error'] > 1.0:
        print("  ❌ ERROR: Perfect predictions should have ~0 ppm error!")
        return False
    else:
        print("  ✓ Perfect predictions work correctly")

    # Create predictions with 50% PAD probability at all positions (BUG TEST)
    probs_with_pad = torch.zeros(batch_size, seq_len, vocab_size)
    for b in range(batch_size):
        for s in range(seq_len):
            if sequence_mask[b, s]:
                probs_with_pad[b, s, sequences[b, s]] = 0.5  # 50% correct token
                probs_with_pad[b, s, PAD_IDX] = 0.5           # 50% PAD (mass = 0!)

    loss, metrics = loss_fn(probs_with_pad, true_masses, sequence_mask)

    print("\n--- Test with 50% PAD probability (simulates potential bug) ---")
    print(f"  Loss: {loss.item():.6f}")
    print(f"  Predicted mass: {metrics['predicted_peptide_mass']:.2f} Da")
    print(f"  Mass error (Da): {metrics['mass_error_da']:.2f}")
    print(f"  PPM error: {metrics['ppm_error']:.2f}")
    print(f"  AA error: {metrics['aa_error']:.6f}")

    expected_error_pct = 50.0
    if abs(metrics['ppm_error'] / 1e4 - expected_error_pct) < 10:
        print(f"  ❌ WARNING: 50% PAD causes ~50% mass error! This could be the bug!")

    return True


def test_with_real_model():
    """Test with actual model predictions."""
    print("\n" + "=" * 70)
    print("TEST 2: Actual Model Predictions")
    print("=" * 70)

    # Create dataset
    dataset = MS2PIPSyntheticDataset(
        min_length=7,
        max_length=10,
        noise_peaks=0,
        peak_dropout=0.0,
    )

    # Get a few samples
    samples = [dataset._generate_sample() for _ in range(3)]

    # Stack into batch
    batch = {
        'sequence': torch.stack([s.sequence for s in samples]),
        'sequence_mask': torch.stack([s.sequence_mask for s in samples]),
        'precursor_mass': torch.stack([s.precursor_mass for s in samples]),
    }

    print("\nBatch info:")
    for i in range(len(samples)):
        seq_tokens = batch['sequence'][i][batch['sequence_mask'][i]].tolist()
        seq_str = ''.join([VOCAB[t] for t in seq_tokens])
        print(f"  Sample {i}: {seq_str} (mass: {batch['precursor_mass'][i]:.2f} Da)")

    # Create random "predictions" to simulate model output
    batch_size, seq_len = batch['sequence'].shape
    vocab_size = len(VOCAB)

    # Simulate model predictions with random noise
    # Start with correct predictions and add noise
    probs = torch.zeros(batch_size, seq_len, vocab_size)
    for b in range(batch_size):
        for s in range(seq_len):
            if batch['sequence_mask'][b, s]:
                # 80% correct token, 20% distributed randomly
                correct_token = batch['sequence'][b, s].item()
                probs[b, s, correct_token] = 0.8
                probs[b, s] += torch.rand(vocab_size) * 0.02  # Add small random noise
                probs[b, s] = F.softmax(probs[b, s], dim=0)  # Renormalize

    # Compute loss
    loss_fn = PrecursorMassLoss()
    loss, metrics = loss_fn(probs, batch['precursor_mass'], batch['sequence_mask'])

    print("\n--- Simulated Model Predictions (80% accuracy per token) ---")
    print(f"  Loss: {loss.item():.6f}")
    print(f"  Mass error (Da): {metrics['mass_error_da']:.2f}")
    print(f"  PPM error: {metrics['ppm_error']:.2f}")
    print(f"  AA error: {metrics['aa_error']:.6f}")

    if metrics['ppm_error'] > 1000:
        print(f"  ❌ WARNING: Very high PPM error for 80% accuracy!")
    else:
        print(f"  ✓ Reasonable error for 80% accuracy")


def inspect_vocab_alignment():
    """Inspect vocabulary and mass alignment."""
    print("\n" + "=" * 70)
    print("TEST 3: Vocabulary and Mass Alignment")
    print("=" * 70)

    loss_fn = PrecursorMassLoss()
    aa_masses = loss_fn.aa_masses

    print("\nVocab alignment check:")
    print(f"{'Index':<6} {'Token':<8} {'AA Mass':<12} {'True Mass':<12} {'Match':<6}")
    print("-" * 60)

    all_match = True
    for i, token in enumerate(VOCAB):
        aa_mass = aa_masses[i].item()
        true_mass = AMINO_ACID_MASSES.get(token, 0.0)
        match = "✓" if abs(aa_mass - true_mass) < 1e-6 else "✗"
        if aa_mass != true_mass:
            all_match = False
        print(f"{i:<6} {token:<8} {aa_mass:<12.5f} {true_mass:<12.5f} {match:<6}")

    if all_match:
        print("\n✓ All vocabulary masses are correctly aligned!")
    else:
        print("\n❌ ERROR: Vocabulary mass misalignment detected!")
        return False

    return True


def main():
    """Run all diagnostic tests."""
    print("\n")
    print("=" * 70)
    print("PRECURSOR MASS LOSS DIAGNOSTIC")
    print("=" * 70)

    # Run tests
    test1_pass = test_precursor_loss_basic()
    test2_pass = test_with_real_model()
    test3_pass = inspect_vocab_alignment()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if test1_pass and test3_pass:
        print("✓ All basic tests passed")
        print("\nPossible causes of high PPM error in training:")
        print("  1. Model is predicting PAD/SOS/EOS tokens with high probability")
        print("  2. Model accuracy is genuinely very low (~10-20%)")
        print("  3. sequence_mask is incorrect (masking valid positions)")
        print("  4. Model output logits are not properly normalized before loss")
        print("\nRecommendation: Add logging to check model prediction distribution")
    else:
        print("❌ Some tests failed - check output above")


if __name__ == '__main__':
    main()
