"""
Test spectrum coverage with GROUND TRUTH sequences.

If ground truth sequences give high coverage, the implementation is correct.
If ground truth sequences give low coverage, there's still a bug.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from src.data.ms2pip_dataset import MS2PIPSyntheticDataset
from src.model.trm import TRMConfig
from src.training.losses import SpectrumMatchingLoss
from src.data.ion_types import get_ion_types_for_model, compute_theoretical_peaks
from src.constants import AMINO_ACID_MASSES, VOCAB


def test_ground_truth_coverage(num_samples=20):
    """Test what coverage we get with perfect sequence predictions."""

    print("=" * 70)
    print("GROUND TRUTH SPECTRUM COVERAGE TEST")
    print("=" * 70)
    print("\nThis tests what coverage we SHOULD get if predictions were perfect.")

    device = 'cpu'

    # Load config
    config_path = project_root / 'configs' / 'optimized_extended.yaml'
    cfg = OmegaConf.load(config_path)

    model_config = TRMConfig(**cfg.model)

    # Create dataset (same as training)
    dataset = MS2PIPSyntheticDataset(
        max_peaks=model_config.max_peaks,
        max_seq_len=model_config.max_seq_len,
        ms2pip_model=cfg.data.ms2pip_model,
        min_length=10,
        max_length=15,
    )

    # Create spectrum loss
    spectrum_loss = SpectrumMatchingLoss(
        sigma=0.2,
        ms2pip_model=cfg.data.ms2pip_model,
    ).to(device)

    print(f"Ion types: {get_ion_types_for_model(cfg.data.ms2pip_model)}")
    print(f"Sigma (matching tolerance): {spectrum_loss.sigma} Da")

    # Test on multiple samples
    print(f"\nTesting {num_samples} samples with GROUND TRUTH sequences...")

    dataset_iter = iter(dataset)
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

            # Create PERFECT sequence probabilities (one-hot of ground truth)
            vocab_size = len(VOCAB)
            perfect_probs = F.one_hot(sequence, vocab_size).float()

            # Compute spectrum loss with GROUND TRUTH
            loss = spectrum_loss(
                sequence_probs=perfect_probs,
                observed_masses=spectrum_masses,
                observed_intensities=spectrum_intensities,
                peak_mask=spectrum_mask,
                sequence_mask=sequence_mask,
            )

            coverage = 1.0 - loss.item()
            all_coverages.append(coverage)

            if i < 5:  # Show first 5 samples in detail
                print(f"\nSample {i+1}:")
                print(f"  Sequence length: {sequence_mask.sum().item()}")
                print(f"  Num observed peaks: {spectrum_mask.sum().item()}")
                print(f"  Coverage: {coverage:.1%}")
                print(f"  Loss: {loss.item():.4f}")

    # Summary
    print(f"\n{'=' * 70}")
    print("RESULTS WITH GROUND TRUTH SEQUENCES")
    print(f"{'=' * 70}")
    print(f"\nCoverage Statistics:")
    print(f"  Mean: {sum(all_coverages)/len(all_coverages):.1%}")
    print(f"  Min:  {min(all_coverages):.1%}")
    print(f"  Max:  {max(all_coverages):.1%}")

    avg_coverage = sum(all_coverages) / len(all_coverages)

    print(f"\n{'=' * 70}")
    print("INTERPRETATION")
    print(f"{'=' * 70}")

    if avg_coverage > 0.50:
        print(f"✓✓✓ EXCELLENT: {avg_coverage:.1%} coverage with ground truth!")
        print(f"    Implementation is CORRECT.")
        print(f"    Low coverage in training is due to wrong predictions.")
    elif avg_coverage > 0.20:
        print(f"✓ GOOD: {avg_coverage:.1%} coverage with ground truth.")
        print(f"   Implementation seems correct.")
        print(f"   Consider:")
        print(f"   - Is sigma={spectrum_loss.sigma} too strict?")
        print(f"   - Are we missing some ion types?")
    elif avg_coverage > 0.05:
        print(f"⚠️  MODERATE: {avg_coverage:.1%} coverage with ground truth.")
        print(f"   This is better than training (0.9%) but still low.")
        print(f"   Possible issues:")
        print(f"   - Sigma too strict (try 0.3-0.5 Da)")
        print(f"   - Missing ion types in prediction")
        print(f"   - Data has noise/dropout that we're not matching")
    else:
        print(f"❌ FAIL: Only {avg_coverage:.1%} coverage even with ground truth!")
        print(f"   There may still be a bug in the implementation.")
        print(f"   Check:")
        print(f"   - Are ion types matching between data and loss?")
        print(f"   - Is sigma too strict?")
        print(f"   - Are we computing theoretical peaks correctly?")

    print(f"{'=' * 70}")


if __name__ == '__main__':
    test_ground_truth_coverage(num_samples=20)
