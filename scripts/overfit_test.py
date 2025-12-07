"""
Overfit test: Train on a single batch to verify model can learn.

This is a critical checkpoint - if the model can't overfit a single batch,
something is fundamentally wrong.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import DataLoader

from src.model.trm import TRMConfig, create_model
from src.data.dataset import SyntheticPeptideDataset, collate_peptide_samples
from src.training.losses import DeepSupervisionLoss
from src.training.metrics import compute_metrics


def overfit_test():
    print("=" * 60)
    print("Overfit Test - Single Batch")
    print("=" * 60)

    # Small model for faster testing
    config = TRMConfig(
        hidden_dim=128,
        num_encoder_layers=2,
        num_decoder_layers=2,
        num_heads=4,
        num_supervision_steps=8,
        num_latent_steps=4,
    )

    model = create_model(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Device: {device}")

    # Create dataset and get a single batch
    dataset = SyntheticPeptideDataset(
        min_length=7,
        max_length=10,
        max_peaks=100,
        max_seq_len=25,
    )

    loader = DataLoader(dataset, batch_size=16, collate_fn=collate_peptide_samples)
    batch = next(iter(loader))
    batch = {k: v.to(device) for k, v in batch.items()}

    print(f"Batch size: {batch['spectrum_masses'].shape[0]}")

    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = DeepSupervisionLoss()

    print("\nTraining on single batch...")
    print("Target: Loss should decrease to near zero\n")

    # Train for 500 steps (increased from 200)
    for step in range(500):
        model.train()

        # Forward
        all_logits, _ = model(
            spectrum_masses=batch['spectrum_masses'],
            spectrum_intensities=batch['spectrum_intensities'],
            spectrum_mask=batch['spectrum_mask'],
            precursor_mass=batch['precursor_mass'],
            precursor_charge=batch['precursor_charge'],
        )

        # Loss
        loss, metrics = loss_fn(
            all_logits=all_logits,
            targets=batch['sequence'],
            target_mask=batch['sequence_mask'],
        )

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Metrics
        if step % 50 == 0 or step == 499:
            with torch.no_grad():
                acc_metrics = compute_metrics(
                    all_logits[-1],
                    batch['sequence'],
                    batch['sequence_mask'],
                )

            print(
                f"Step {step:3d} | "
                f"Loss: {loss.item():.4f} | "
                f"Token Acc: {acc_metrics['token_accuracy']:.3f} | "
                f"Seq Acc: {acc_metrics['sequence_accuracy']:.3f}"
            )

    print("\n" + "=" * 60)

    # Final check
    final_loss = loss.item()
    final_acc = acc_metrics['token_accuracy']

    if final_loss < 0.1 and final_acc > 0.95:
        print("✓ OVERFIT TEST PASSED!")
        print(f"  Final loss: {final_loss:.4f} (target: <0.1)")
        print(f"  Final accuracy: {final_acc:.3f} (target: >0.95)")
        print("\nModel can learn - ready to proceed with full training!")
    elif final_acc > 0.8:
        print("⚠ OVERFIT TEST PARTIAL SUCCESS")
        print(f"  Final loss: {final_loss:.4f}")
        print(f"  Final accuracy: {final_acc:.3f}")
        print("\nModel is learning but might need more steps or tuning.")
    else:
        print("✗ OVERFIT TEST FAILED")
        print(f"  Final loss: {final_loss:.4f}")
        print(f"  Final accuracy: {final_acc:.3f}")
        print("\nModel is not learning - check implementation!")

    print("=" * 60)


if __name__ == '__main__':
    overfit_test()
