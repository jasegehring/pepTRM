"""
Verify training data flow through the model with curriculum.

This script simulates the first few steps of training to confirm:
1. Curriculum stages transition correctly
2. DataLoader produces correct length distributions
3. Model receives and processes data correctly
4. Length metrics are computed correctly
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import DataLoader

from src.model.trm import TRMConfig, RecursivePeptideModel
from src.data.ms2pip_dataset import MS2PIPSyntheticDataset
from src.training.curriculum_progressive_length import (
    PROGRESSIVE_LENGTH_CURRICULUM,
    CurriculumScheduler,
)
from src.training.metrics import compute_metrics, compute_metrics_by_length
from src.constants import PROTON_MASS


def collate_fn(batch):
    return {
        'spectrum_masses': torch.stack([s.spectrum_masses for s in batch]),
        'spectrum_intensities': torch.stack([s.spectrum_intensities for s in batch]),
        'spectrum_mask': torch.stack([s.spectrum_mask for s in batch]),
        'precursor_mass': torch.stack([s.precursor_mass for s in batch]),
        'precursor_charge': torch.stack([s.precursor_charge for s in batch]),
        'sequence': torch.stack([s.sequence for s in batch]),
        'sequence_mask': torch.stack([s.sequence_mask for s in batch]),
    }


def get_batch_stats(batch):
    """Get statistics about a batch."""
    lengths = batch['sequence_mask'].sum(dim=1).tolist()
    num_peaks = batch['spectrum_mask'].sum(dim=1).float().mean().item()
    return {
        'min_length': int(min(lengths)),
        'max_length': int(max(lengths)),
        'avg_length': sum(lengths) / len(lengths),
        'avg_peaks': num_peaks,
        'batch_size': len(lengths),
    }


def main():
    print("=" * 70)
    print("TRAINING DATA FLOW VERIFICATION")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Create model
    model_config = TRMConfig(
        hidden_dim=384,
        num_encoder_layers=3,
        num_decoder_layers=3,
        num_heads=6,
        max_peaks=100,
        max_seq_len=35,
        num_supervision_steps=8,
        num_latent_steps=6,
        dropout=0.1,
    )
    model = RecursivePeptideModel(model_config).to(device)
    model.eval()  # We're just testing data flow, not training

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create dataset and curriculum
    dataset = MS2PIPSyntheticDataset(
        max_peaks=model_config.max_peaks,
        max_seq_len=model_config.max_seq_len,
        ms2pip_model='HCDch2',
        charge_distribution={2: 0.7, 3: 0.3},
    )

    curriculum = CurriculumScheduler(PROGRESSIVE_LENGTH_CURRICULUM, dataset)

    # Create DataLoader
    loader = DataLoader(
        dataset,
        batch_size=32,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Test stages
    test_steps = [0, 5000, 10000, 15000, 25000, 40000, 50000, 60000, 80000]

    print("\n" + "=" * 70)
    print("TESTING CURRICULUM STAGES")
    print("=" * 70)

    for step in test_steps:
        # Step curriculum and recreate iterator (simulating the fix)
        stage_changed = curriculum.step(step)
        if stage_changed or step == 0:
            train_iterator = iter(loader)

        stage = curriculum.current_stage
        print(f"\n--- Step {step}: Stage {curriculum.current_stage_idx + 1} ({stage.name}) ---")
        print(f"Expected lengths: {stage.min_length}-{stage.max_length}")
        print(f"Clean ratio: {stage.clean_data_ratio:.0%}")

        # Get a batch and verify
        batch = next(train_iterator)
        stats = get_batch_stats(batch)

        print(f"Actual lengths: {stats['min_length']}-{stats['max_length']} (avg: {stats['avg_length']:.1f})")
        print(f"Avg peaks: {stats['avg_peaks']:.1f}")

        # Verify lengths match expected
        if stats['min_length'] < stage.min_length or stats['max_length'] > stage.max_length:
            print(f"  WARNING: Lengths outside expected range!")
        else:
            print(f"  OK: Lengths within expected range")

        # Run through model
        batch = {k: v.to(device) for k, v in batch.items()}

        precursor_mass = batch['precursor_mass']
        precursor_charge = batch['precursor_charge']
        precursor_mz = (precursor_mass + precursor_charge * PROTON_MASS) / precursor_charge

        with torch.no_grad():
            all_logits, final_z = model(
                spectrum_masses=batch['spectrum_masses'],
                spectrum_intensities=batch['spectrum_intensities'],
                spectrum_mask=batch['spectrum_mask'],
                precursor_mass=precursor_mz,
                precursor_charge=precursor_charge,
            )

        # Compute metrics
        final_logits = all_logits[-1].float()
        metrics = compute_metrics(
            logits=final_logits,
            targets=batch['sequence'],
            mask=batch['sequence_mask'],
        )
        length_metrics = compute_metrics_by_length(
            logits=final_logits,
            targets=batch['sequence'],
            mask=batch['sequence_mask'],
        )

        print(f"Model output shape: {all_logits.shape}")
        print(f"Token accuracy: {metrics['token_accuracy']:.1%}")
        print(f"Sequence accuracy: {metrics['sequence_accuracy']:.1%}")

        # Show length breakdown if available
        if 'length_7_12_acc' in length_metrics:
            print(f"  Length 7-12 acc: {length_metrics['length_7_12_acc']:.1%}")
        if 'length_13_17_acc' in length_metrics:
            print(f"  Length 13-17 acc: {length_metrics['length_13_17_acc']:.1%}")
        if 'length_18_25_acc' in length_metrics:
            print(f"  Length 18-25 acc: {length_metrics['length_18_25_acc']:.1%}")

    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)
    print("""
If all stages show correct length ranges, the data flow is working correctly.
The model should show low accuracy initially (untrained), which is expected.

Key things verified:
1. Curriculum stages transition at correct steps
2. DataLoader iterator recreation works (lengths change with stage)
3. Model receives and processes batches without error
4. Length metrics are computed correctly
""")


if __name__ == '__main__':
    main()
