"""
Diagnostic script to measure actual precursor loss magnitudes.

This helps us understand what loss_scale and curriculum weights should be.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from src.data.dataset import SyntheticPeptideDataset
from src.training.losses import PrecursorMassLoss
from src.constants import VOCAB

def main():
    print("=" * 60)
    print("Precursor Loss Magnitude Diagnostic")
    print("=" * 60)

    # Create dataset (infinite iterable)
    dataset = SyntheticPeptideDataset(
        min_length=7,
        max_length=10,
        noise_peaks=0,  # Clean data
    )

    # Create different precursor loss configurations
    configs = [
        {"loss_scale": 100.0, "name": "Current (scale=100)"},
        {"loss_scale": 1000.0, "name": "Larger (scale=1000)"},
        {"loss_scale": 10000.0, "name": "Much larger (scale=10000)"},
    ]

    batch_size = 32
    vocab_size = len(VOCAB)

    # Simulate random predictions at different training stages
    training_stages = [
        {"name": "Random initialization", "accuracy": 0.05},
        {"name": "Early training (step 1000)", "accuracy": 0.15},
        {"name": "Mid training (step 5000)", "accuracy": 0.30},
        {"name": "Before precursor loss (step 10k)", "accuracy": 0.45},
    ]

    for stage_info in training_stages:
        print(f"\n{'='*60}")
        print(f"Stage: {stage_info['name']}")
        print(f"{'='*60}")

        # Get a batch from iterator
        dataset_iter = iter(dataset)
        batch = [next(dataset_iter) for _ in range(batch_size)]

        # Extract targets and precursor masses (batch items are dataclass objects)
        targets = torch.stack([b.sequence for b in batch])
        precursor_masses = torch.stack([b.precursor_mass for b in batch])
        target_mask = torch.stack([b.sequence_mask for b in batch])

        seq_len = targets.shape[1]

        # Simulate predictions at this accuracy level
        # Higher accuracy = more peaked distribution toward correct token
        accuracy = stage_info['accuracy']

        # Create soft predictions
        sequence_probs = torch.zeros(batch_size, seq_len, vocab_size)
        for b in range(batch_size):
            for s in range(seq_len):
                if target_mask[b, s]:
                    correct_token = targets[b, s].item()
                    # Put accuracy% probability on correct token, rest uniform
                    sequence_probs[b, s] = (1 - accuracy) / vocab_size
                    sequence_probs[b, s, correct_token] = accuracy + (1 - accuracy) / vocab_size
                else:
                    sequence_probs[b, s] = 1.0 / vocab_size

        # Test each configuration
        for config in configs:
            loss_fn = PrecursorMassLoss(
                use_relative=True,
                loss_scale=config["loss_scale"],
                use_log_loss=True,
            )

            loss, metrics = loss_fn(sequence_probs, precursor_masses, target_mask)

            print(f"\n{config['name']}:")
            print(f"  Raw loss value: {loss.item():.2f}")
            print(f"  PPM error: {metrics['ppm_error']:.1f}")
            print(f"  Mass error (Da): {metrics['mass_error_da']:.4f}")

            # Show what this contributes with different curriculum weights
            for weight in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]:
                contribution = weight * loss.item()
                print(f"  With weight={weight:.3f}: contributes {contribution:.3f} to total loss")

    print("\n" + "=" * 60)
    print("Recommendations:")
    print("=" * 60)
    print("1. If PPM errors are 10,000+ early in training:")
    print("   - Use loss_scale=10000 or higher")
    print("   - Use curriculum weights < 0.001 initially")
    print()
    print("2. Target contribution to total loss:")
    print("   - CE loss is ~2-3")
    print("   - Precursor contribution should be ~0.01-0.1 initially")
    print("   - This means: weight * raw_loss should be ~0.01-0.1")
    print()
    print("3. For loss_scale=10000 with weight=0.001:")
    print("   - If raw loss is ~10, contribution is 0.01 ✓")
    print("   - If raw loss is ~100, contribution is 0.1 ✓")

if __name__ == "__main__":
    main()
