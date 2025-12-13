"""
Test spectrum loss behavior at 48% token accuracy (actual training state).

This simulates what the gradient signal looks like when training accuracy is 48%,
not the 71.9% validation accuracy I measured before.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn.functional as F

from src.data.ms2pip_dataset import MS2PIPSyntheticDataset
from src.model.trm import TRMConfig
from src.training.losses import SpectrumMatchingLoss
from src.constants import VOCAB
from omegaconf import OmegaConf


def test_at_accuracy(target_accuracy=0.48, num_samples=20):
    """
    Test spectrum loss with simulated predictions at target accuracy.

    This shows what gradients look like when model is at 48% token accuracy
    during training (not the easier 71.9% on validation).
    """

    print("=" * 70)
    print(f"SPECTRUM LOSS AT {target_accuracy:.0%} TOKEN ACCURACY")
    print("=" * 70)
    print(f"\nSimulating ACTUAL training state (not validation)")

    device = 'cpu'

    # Load config
    config_path = project_root / 'configs' / 'optimized_extended.yaml'
    cfg = OmegaConf.load(config_path)
    model_config = TRMConfig(**cfg.model)

    # Create dataset
    dataset = MS2PIPSyntheticDataset(
        max_peaks=model_config.max_peaks,
        max_seq_len=model_config.max_seq_len,
        ms2pip_model=cfg.data.ms2pip_model,
        min_length=7,
        max_length=10,  # Stage 3 curriculum
    )

    # Create spectrum loss
    spectrum_loss = SpectrumMatchingLoss(
        sigma=0.2,
        ms2pip_model=cfg.data.ms2pip_model,
    ).to(device)

    vocab_size = len(VOCAB)

    print(f"Testing {num_samples} samples at {target_accuracy:.0%} accuracy...")

    dataset_iter = iter(dataset)
    all_losses = []
    all_coverages = []

    with torch.no_grad():
        for i in range(num_samples):
            sample = next(dataset_iter)

            # Move to device
            spectrum_masses = sample.spectrum_masses.unsqueeze(0).to(device)
            spectrum_intensities = sample.spectrum_intensities.unsqueeze(0).to(device)
            spectrum_mask = sample.spectrum_mask.unsqueeze(0).to(device)
            sequence = sample.sequence.unsqueeze(0).to(device)
            sequence_mask = sample.sequence_mask.unsqueeze(0).to(device)

            # Create probabilities that simulate target_accuracy
            # Mix ground truth with random noise to achieve target accuracy

            # Ground truth one-hot
            correct_probs = F.one_hot(sequence, vocab_size).float()

            # Random uniform distribution
            random_probs = torch.ones_like(correct_probs) / vocab_size

            # Mix to achieve target accuracy
            # When we sample from this distribution, we'll get target_accuracy correct
            mixed_probs = target_accuracy * correct_probs + (1 - target_accuracy) * random_probs
            mixed_probs = mixed_probs / mixed_probs.sum(dim=-1, keepdim=True)

            # Compute spectrum loss
            loss = spectrum_loss(
                sequence_probs=mixed_probs,
                observed_masses=spectrum_masses,
                observed_intensities=spectrum_intensities,
                peak_mask=spectrum_mask,
                sequence_mask=sequence_mask,
            )

            coverage = 1.0 - loss.item()
            all_losses.append(loss.item())
            all_coverages.append(coverage)

    # Summary
    avg_coverage = sum(all_coverages) / len(all_coverages)
    avg_loss = sum(all_losses) / len(all_losses)

    print(f"\n{'=' * 70}")
    print(f"RESULTS AT {target_accuracy:.0%} TOKEN ACCURACY")
    print(f"{'=' * 70}")
    print(f"\nSpectrum Loss: {avg_loss:.4f} (range: [{min(all_losses):.4f}, {max(all_losses):.4f}])")
    print(f"Coverage:      {avg_coverage:.1%} (range: [{min(all_coverages):.1%}, {max(all_coverages):.1%}])")

    # Compare to different accuracies
    print(f"\n{'=' * 70}")
    print("COMPARISON ACROSS ACCURACIES")
    print(f"{'=' * 70}")
    print("\nExpected coverage at different token accuracies:")
    print("  48% token acc → ??? coverage (current training state)")
    print("  72% token acc → ??? coverage (current validation state)")
    print(" 100% token acc → 64% coverage (ground truth)")

    # Test at different accuracies
    test_accuracies = [0.25, 0.48, 0.60, 0.72, 0.85, 1.0]
    print(f"\nActual measured coverage:")

    for test_acc in test_accuracies:
        coverages = []
        for _ in range(5):  # Quick test
            sample = next(iter(dataset))
            spectrum_masses = sample.spectrum_masses.unsqueeze(0)
            spectrum_intensities = sample.spectrum_intensities.unsqueeze(0)
            spectrum_mask = sample.spectrum_mask.unsqueeze(0)
            sequence = sample.sequence.unsqueeze(0)
            sequence_mask = sample.sequence_mask.unsqueeze(0)

            correct_probs = F.one_hot(sequence, vocab_size).float()
            random_probs = torch.ones_like(correct_probs) / vocab_size
            mixed_probs = test_acc * correct_probs + (1 - test_acc) * random_probs
            mixed_probs = mixed_probs / mixed_probs.sum(dim=-1, keepdim=True)

            loss = spectrum_loss(
                mixed_probs, spectrum_masses, spectrum_intensities,
                spectrum_mask, sequence_mask
            )
            coverages.append(1.0 - loss.item())

        marker = " ← TRAINING" if abs(test_acc - 0.48) < 0.01 else ""
        marker = " ← VALIDATION" if abs(test_acc - 0.72) < 0.01 else marker
        print(f"  {test_acc:>3.0%} token acc → {sum(coverages)/len(coverages):5.1%} coverage{marker}")

    print(f"\n{'=' * 70}")
    print("WHY SPECTRUM LOSS MIGHT HURT TOKEN ACCURACY")
    print(f"{'=' * 70}")

    if avg_coverage < 0.05:
        print(f"\n⚠️  Coverage is VERY low ({avg_coverage:.1%}) at 48% token accuracy!")
        print(f"\nThis creates a problem:")
        print(f"  1. CE loss says: 'Predict token X'")
        print(f"  2. Spectrum loss says: 'Your peaks are wrong' (but signal is weak/noisy)")
        print(f"  3. The tiny spectrum gradient might push in WRONG directions")
        print(f"  4. This can interfere with CE learning, even if gradient is small")
        print(f"\nAnalogy: Like trying to navigate with a broken compass.")
        print(f"         Even if you mostly follow the map (CE), the bad compass")
        print(f"         (spectrum) occasionally steers you wrong.")
    elif avg_coverage < 0.15:
        print(f"\nCoverage is low but not terrible ({avg_coverage:.1%}).")
        print(f"Spectrum loss might provide weak but useful signal.")
    else:
        print(f"\nCoverage is reasonable ({avg_coverage:.1%}).")
        print(f"Spectrum loss should provide useful gradient signal.")

    print(f"{'=' * 70}")

    return avg_coverage


if __name__ == '__main__':
    test_at_accuracy(target_accuracy=0.48, num_samples=20)
