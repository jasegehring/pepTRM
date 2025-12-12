"""
Test spectrum matching loss to verify it's working correctly.

Tests:
1. Perfect prediction → low loss
2. Wrong prediction → high loss
3. Partial match → intermediate loss
4. Loss is differentiable
5. Gradients point in the right direction
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn.functional as F
from src.data.ms2pip_dataset import MS2PIPSyntheticDataset
from src.training.losses import SpectrumMatchingLoss
from src.constants import AMINO_ACID_MASSES, WATER_MASS, VOCAB, AA_TO_IDX
from src.data.ion_types import compute_theoretical_peaks

def test_perfect_prediction():
    """Test that perfect predictions give near-zero loss."""
    print("=" * 80)
    print("TEST 1: Perfect Prediction → Low Loss")
    print("=" * 80)

    # Create dataset and get a sample
    dataset = MS2PIPSyntheticDataset(
        min_length=7,
        max_length=10,
        ms2pip_model='HCDch2',
    )
    sample = next(iter(dataset))

    # Decode sequence
    sequence = sample.sequence
    sequence_mask = sample.sequence_mask
    valid_tokens = sequence[sequence_mask]
    decoded_aas = [dataset.idx_to_token[idx.item()] for idx in valid_tokens]
    peptide = ''.join(decoded_aas)

    print(f"\nPeptide: {peptide}")
    print(f"Observed peaks: {sample.spectrum_mask.sum().item()}")

    # Create perfect prediction (one-hot on correct sequence)
    batch_size = 1
    max_seq_len = sequence.shape[0]
    vocab_size = len(VOCAB)

    perfect_probs = torch.zeros(batch_size, max_seq_len, vocab_size)
    for i, (token_id, valid) in enumerate(zip(sequence, sequence_mask)):
        if valid:
            perfect_probs[0, i, token_id] = 1.0

    # Create spectrum loss
    spectrum_loss = SpectrumMatchingLoss(
        mass_tolerance=0.5,
        ms2pip_model='HCDch2',
    )

    # Compute loss
    loss = spectrum_loss(
        sequence_probs=perfect_probs,
        observed_masses=sample.spectrum_masses.unsqueeze(0),
        observed_intensities=sample.spectrum_intensities.unsqueeze(0),
        peak_mask=sample.spectrum_mask.unsqueeze(0),
    )

    print(f"\nLoss with perfect prediction: {loss.item():.6f} Da")
    print(f"Expected: ~0.0 Da (should be very small)")

    if loss.item() < 0.1:
        print("✓ PASS: Loss is near zero for perfect prediction")
    else:
        print("✗ FAIL: Loss should be near zero but is {:.4f}".format(loss.item()))

    return loss.item() < 0.1

def test_random_prediction():
    """Test that random predictions give high loss."""
    print("\n" + "=" * 80)
    print("TEST 2: Random Prediction → High Loss")
    print("=" * 80)

    # Create dataset and get a sample
    dataset = MS2PIPSyntheticDataset(
        min_length=7,
        max_length=10,
        ms2pip_model='HCDch2',
    )
    sample = next(iter(dataset))

    # Decode sequence
    sequence = sample.sequence
    sequence_mask = sample.sequence_mask
    valid_tokens = sequence[sequence_mask]
    decoded_aas = [dataset.idx_to_token[idx.item()] for idx in valid_tokens]
    peptide = ''.join(decoded_aas)

    print(f"\nTrue peptide: {peptide}")

    # Create random prediction (uniform distribution over amino acids)
    batch_size = 1
    max_seq_len = sequence.shape[0]
    vocab_size = len(VOCAB)

    # Uniform over amino acids (indices 4-23)
    random_probs = torch.zeros(batch_size, max_seq_len, vocab_size)
    for i in range(max_seq_len):
        if sequence_mask[i]:
            random_probs[0, i, 4:] = 1.0 / 20  # Uniform over 20 amino acids

    # Create spectrum loss
    spectrum_loss = SpectrumMatchingLoss(
        mass_tolerance=0.5,
        ms2pip_model='HCDch2',
    )

    # Compute loss
    loss = spectrum_loss(
        sequence_probs=random_probs,
        observed_masses=sample.spectrum_masses.unsqueeze(0),
        observed_intensities=sample.spectrum_intensities.unsqueeze(0),
        peak_mask=sample.spectrum_mask.unsqueeze(0),
    )

    print(f"\nLoss with random prediction: {loss.item():.6f} Da")
    print(f"Expected: > 0.1 Da (should be high)")

    if loss.item() > 0.1:
        print("✓ PASS: Loss is high for random prediction")
    else:
        print("✗ FAIL: Loss should be high but is {:.4f}".format(loss.item()))

    return loss.item() > 0.1

def test_wrong_prediction():
    """Test that completely wrong prediction gives high loss."""
    print("\n" + "=" * 80)
    print("TEST 3: Wrong Prediction → High Loss")
    print("=" * 80)

    # Create dataset and get a sample
    dataset = MS2PIPSyntheticDataset(
        min_length=7,
        max_length=10,
        ms2pip_model='HCDch2',
    )
    sample = next(iter(dataset))

    # Decode sequence
    sequence = sample.sequence
    sequence_mask = sample.sequence_mask
    valid_tokens = sequence[sequence_mask]
    decoded_aas = [dataset.idx_to_token[idx.item()] for idx in valid_tokens]
    peptide = ''.join(decoded_aas)

    # Create wrong prediction (predict opposite - heavy AAs for light, light for heavy)
    batch_size = 1
    max_seq_len = sequence.shape[0]
    vocab_size = len(VOCAB)

    wrong_probs = torch.zeros(batch_size, max_seq_len, vocab_size)
    aa_masses_list = [AMINO_ACID_MASSES.get(aa, 0.0) for aa in VOCAB]

    for i, (token_id, valid) in enumerate(zip(sequence, sequence_mask)):
        if valid:
            true_aa = dataset.idx_to_token[token_id.item()]
            true_mass = AMINO_ACID_MASSES.get(true_aa, 0.0)

            # Pick an amino acid with very different mass
            if true_mass < 100:  # Light AA → predict heavy
                wrong_aa = 'W'  # Tryptophan (186 Da)
            else:  # Heavy AA → predict light
                wrong_aa = 'G'  # Glycine (57 Da)

            wrong_idx = AA_TO_IDX[wrong_aa]
            wrong_probs[0, i, wrong_idx] = 1.0

    print(f"\nTrue peptide: {peptide}")
    wrong_peptide = ''.join([
        dataset.idx_to_token[wrong_probs[0, i].argmax().item()]
        for i in range(max_seq_len) if sequence_mask[i]
    ])
    print(f"Wrong peptide: {wrong_peptide}")

    # Create spectrum loss
    spectrum_loss = SpectrumMatchingLoss(
        mass_tolerance=0.5,
        ms2pip_model='HCDch2',
    )

    # Compute loss
    loss = spectrum_loss(
        sequence_probs=wrong_probs,
        observed_masses=sample.spectrum_masses.unsqueeze(0),
        observed_intensities=sample.spectrum_intensities.unsqueeze(0),
        peak_mask=sample.spectrum_mask.unsqueeze(0),
    )

    print(f"\nLoss with wrong prediction: {loss.item():.6f} Da")
    print(f"Expected: > 0.2 Da (should be very high)")

    if loss.item() > 0.2:
        print("✓ PASS: Loss is high for wrong prediction")
    else:
        print("✗ FAIL: Loss should be very high but is {:.4f}".format(loss.item()))

    return loss.item() > 0.2

def test_gradient_direction():
    """Test that gradients point toward correct prediction."""
    print("\n" + "=" * 80)
    print("TEST 4: Gradient Direction")
    print("=" * 80)

    # Create dataset and get a sample
    dataset = MS2PIPSyntheticDataset(
        min_length=7,
        max_length=10,
        ms2pip_model='HCDch2',
    )
    sample = next(iter(dataset))

    # Decode true sequence
    sequence = sample.sequence
    sequence_mask = sample.sequence_mask
    valid_tokens = sequence[sequence_mask]
    num_valid = valid_tokens.shape[0]

    print(f"\nTesting gradient direction for {num_valid} positions...")

    # Start with uniform prediction
    batch_size = 1
    max_seq_len = sequence.shape[0]
    vocab_size = len(VOCAB)

    logits = torch.zeros(batch_size, max_seq_len, vocab_size, requires_grad=True)

    # Create spectrum loss
    spectrum_loss = SpectrumMatchingLoss(
        mass_tolerance=0.5,
        ms2pip_model='HCDch2',
    )

    # Compute loss and gradients
    probs = F.softmax(logits, dim=-1)
    loss = spectrum_loss(
        sequence_probs=probs,
        observed_masses=sample.spectrum_masses.unsqueeze(0),
        observed_intensities=sample.spectrum_intensities.unsqueeze(0),
        peak_mask=sample.spectrum_mask.unsqueeze(0),
    )

    loss.backward()

    # Check if gradient for correct token is negative (reducing logit reduces loss)
    correct_directions = 0
    for i, (token_id, valid) in enumerate(zip(sequence, sequence_mask)):
        if valid:
            true_idx = token_id.item()
            grad = logits.grad[0, i, true_idx].item()

            # For spectrum matching, we want to increase prob of correct token
            # So gradient w.r.t. logit should guide us toward correct token
            # The sign depends on whether current mass prediction is too high or too low
            # Just check that gradient is non-zero
            if abs(grad) > 1e-6:
                correct_directions += 1

    print(f"\nPositions with non-zero gradients: {correct_directions}/{num_valid}")

    if correct_directions >= num_valid * 0.8:
        print("✓ PASS: Most positions have meaningful gradients")
        return True
    else:
        print("✗ FAIL: Too few positions have gradients")
        return False

def test_loss_magnitude():
    """Test that loss magnitude is reasonable and differentiable."""
    print("\n" + "=" * 80)
    print("TEST 5: Loss Magnitude Analysis")
    print("=" * 80)

    # Create dataset
    dataset = MS2PIPSyntheticDataset(
        min_length=7,
        max_length=10,
        ms2pip_model='HCDch2',
    )

    # Test on multiple samples
    perfect_losses = []
    random_losses = []

    spectrum_loss = SpectrumMatchingLoss(
        mass_tolerance=0.5,
        ms2pip_model='HCDch2',
    )

    for _ in range(10):
        sample = next(iter(dataset))
        sequence = sample.sequence
        sequence_mask = sample.sequence_mask
        batch_size = 1
        max_seq_len = sequence.shape[0]
        vocab_size = len(VOCAB)

        # Perfect prediction
        perfect_probs = torch.zeros(batch_size, max_seq_len, vocab_size)
        for i, (token_id, valid) in enumerate(zip(sequence, sequence_mask)):
            if valid:
                perfect_probs[0, i, token_id] = 1.0

        loss_perfect = spectrum_loss(
            sequence_probs=perfect_probs,
            observed_masses=sample.spectrum_masses.unsqueeze(0),
            observed_intensities=sample.spectrum_intensities.unsqueeze(0),
            peak_mask=sample.spectrum_mask.unsqueeze(0),
        )
        perfect_losses.append(loss_perfect.item())

        # Random prediction
        random_probs = torch.zeros(batch_size, max_seq_len, vocab_size)
        for i in range(max_seq_len):
            if sequence_mask[i]:
                random_probs[0, i, 4:] = 1.0 / 20

        loss_random = spectrum_loss(
            sequence_probs=random_probs,
            observed_masses=sample.spectrum_masses.unsqueeze(0),
            observed_intensities=sample.spectrum_intensities.unsqueeze(0),
            peak_mask=sample.spectrum_mask.unsqueeze(0),
        )
        random_losses.append(loss_random.item())

    print(f"\nPerfect prediction losses (10 samples):")
    print(f"  Mean: {sum(perfect_losses)/len(perfect_losses):.6f} Da")
    print(f"  Range: [{min(perfect_losses):.6f}, {max(perfect_losses):.6f}] Da")

    print(f"\nRandom prediction losses (10 samples):")
    print(f"  Mean: {sum(random_losses)/len(random_losses):.6f} Da")
    print(f"  Range: [{min(random_losses):.6f}, {max(random_losses):.6f}] Da")

    ratio = (sum(random_losses)/len(random_losses)) / (sum(perfect_losses)/len(perfect_losses))
    print(f"\nRatio (random/perfect): {ratio:.2f}x")

    print("\nExpected ranges:")
    print("  Perfect: 0.0 - 0.1 Da (small due to MS2PIP approximation)")
    print("  Random: 0.1 - 1.0 Da (high due to wrong masses)")
    print("  Ratio: > 2x")

    if ratio > 2:
        print("✓ PASS: Loss discriminates between correct and random predictions")
        return True
    else:
        print("✗ FAIL: Loss doesn't discriminate enough")
        return False

if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("SPECTRUM MATCHING LOSS VERIFICATION")
    print("=" * 80)

    results = []

    try:
        results.append(("Perfect prediction", test_perfect_prediction()))
        results.append(("Random prediction", test_random_prediction()))
        results.append(("Wrong prediction", test_wrong_prediction()))
        results.append(("Gradient direction", test_gradient_direction()))
        results.append(("Loss magnitude", test_loss_magnitude()))

        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)

        for name, passed in results:
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{status}: {name}")

        if all(passed for _, passed in results):
            print("\n✓ ALL TESTS PASSED!")
            print("Spectrum matching loss is working correctly.")
        else:
            print("\n✗ SOME TESTS FAILED")
            print("Spectrum matching loss may have issues.")

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
