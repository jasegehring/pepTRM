"""
Verify the DataLoader bug - simulating actual training loop behavior.

The actual training loop does:
    for batch in self.train_loader:
        # curriculum.step() is called here

This creates ONE iterator and uses it continuously. The workers are
spawned when the iterator is created, with their own copies of the dataset.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import DataLoader

from src.data.ms2pip_dataset import MS2PIPSyntheticDataset
from src.training.curriculum_aggressive_noise import AGGRESSIVE_NOISE_CURRICULUM, CurriculumScheduler


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


def get_lengths_from_batch(batch):
    """Get sequence lengths from a batch."""
    lengths = []
    for i in range(len(batch['sequence'])):
        seq_len = batch['sequence_mask'][i].sum().item()
        lengths.append(int(seq_len))
    return lengths


def main():
    print("=" * 70)
    print("DATALOADER BUG V2 - SIMULATING ACTUAL TRAINING LOOP")
    print("=" * 70)

    # Create dataset exactly as in training
    dataset = MS2PIPSyntheticDataset(
        max_peaks=100,
        max_seq_len=35,
        ms2pip_model='HCDch2',
        charge_distribution={2: 0.7, 3: 0.3},
    )

    # Create curriculum (this sets initial parameters to Stage 1)
    curriculum = CurriculumScheduler(AGGRESSIVE_NOISE_CURRICULUM, dataset)

    # Create DataLoader WITH WORKERS (like in training)
    loader = DataLoader(
        dataset,
        batch_size=32,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    print("\n--- SIMULATING TRAINING LOOP ---")
    print("Creating iterator (this spawns worker processes with Stage 1 settings)")

    # This is what happens in: for batch in self.train_loader:
    iterator = iter(loader)

    print(f"\nDataset in main process: min_length={dataset.min_length}, max_length={dataset.max_length}")

    # Get a few batches with initial settings
    print("\n--- INITIAL BATCHES (before curriculum step) ---")
    for i in range(3):
        batch = next(iterator)
        lengths = get_lengths_from_batch(batch)
        print(f"Batch {i+1}: lengths {min(lengths)}-{max(lengths)}")

    # Now step curriculum to Stage 6
    print("\n--- STEPPING CURRICULUM TO STAGE 6 ---")
    curriculum.step(65000)
    print(f"Dataset in main process: min_length={dataset.min_length}, max_length={dataset.max_length}")

    # Continue getting batches from the SAME iterator
    print("\n--- BATCHES AFTER CURRICULUM STEP (same iterator) ---")
    for i in range(5):
        batch = next(iterator)
        lengths = get_lengths_from_batch(batch)
        print(f"Batch {i+4}: lengths {min(lengths)}-{max(lengths)}")

    # Check if we're still getting Stage 1 data
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # Get more batches to check
    all_lengths = []
    for i in range(10):
        batch = next(iterator)
        all_lengths.extend(get_lengths_from_batch(batch))

    if max(all_lengths) <= 12:
        print("\n🚨 BUG CONFIRMED!")
        print("   After stepping curriculum to Stage 6 (lengths 12-25),")
        print("   DataLoader is STILL producing lengths 7-12!")
        print("")
        print("   This explains why train/token_accuracy is 96% at step 73K!")
        print("   The model has been training on Stage 1 data THE ENTIRE TIME!")
    elif min(all_lengths) < 12:
        print("\n⚠️  PARTIAL BUG: Still seeing some short sequences")
        print(f"   Lengths observed: {min(all_lengths)}-{max(all_lengths)}")
    else:
        print("\n✓ Curriculum change propagated to workers")
        print(f"   Lengths observed: {min(all_lengths)}-{max(all_lengths)}")


if __name__ == '__main__':
    main()
