"""
Verify the DataLoader + IterableDataset curriculum bug.

HYPOTHESIS: When using DataLoader with num_workers > 0, each worker gets
its own COPY of the IterableDataset. When curriculum.step() calls
dataset.set_difficulty(), it only updates the main process's copy,
NOT the worker processes' copies.

This would mean training data NEVER changes from the initial settings!
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


def get_batch_lengths(loader):
    """Get sequence lengths from one batch."""
    batch = next(iter(loader))
    lengths = []
    for i in range(len(batch['sequence'])):
        seq_len = batch['sequence_mask'][i].sum().item()
        lengths.append(int(seq_len))
    return lengths


def main():
    print("=" * 70)
    print("DATALOADER + ITERABLEDATASET CURRICULUM BUG VERIFICATION")
    print("=" * 70)

    # Create dataset exactly as in training
    dataset = MS2PIPSyntheticDataset(
        max_peaks=100,
        max_seq_len=35,
        ms2pip_model='HCDch2',
        charge_distribution={2: 0.7, 3: 0.3},
    )

    # Create curriculum (this sets initial parameters)
    curriculum = CurriculumScheduler(AGGRESSIVE_NOISE_CURRICULUM, dataset)

    print("\n--- INITIAL STATE ---")
    print(f"Dataset min_length: {dataset.min_length}")
    print(f"Dataset max_length: {dataset.max_length}")

    # Create DataLoader WITH WORKERS (like in training)
    loader_with_workers = DataLoader(
        dataset,
        batch_size=32,
        num_workers=4,  # Same as training script
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Create DataLoader WITHOUT WORKERS
    loader_no_workers = DataLoader(
        dataset,
        batch_size=32,
        num_workers=0,  # Main process only
        collate_fn=collate_fn,
    )

    print("\n--- TEST 1: Initial state (Stage 1: lengths 7-12) ---")

    lengths_workers = get_batch_lengths(loader_with_workers)
    lengths_no_workers = get_batch_lengths(loader_no_workers)

    print(f"With workers (num_workers=4): min={min(lengths_workers)}, max={max(lengths_workers)}")
    print(f"Without workers (num_workers=0): min={min(lengths_no_workers)}, max={max(lengths_no_workers)}")

    # Now simulate stepping to Stage 6 (step 60000+)
    print("\n--- Stepping curriculum to Stage 6 (step 65000) ---")
    curriculum.step(65000)

    print(f"\nDataset min_length after step: {dataset.min_length}")
    print(f"Dataset max_length after step: {dataset.max_length}")
    print("Expected: 12-25 (Stage 6)")

    print("\n--- TEST 2: After curriculum step (Stage 6: lengths 12-25) ---")

    # Get new batches
    lengths_workers_after = get_batch_lengths(loader_with_workers)
    lengths_no_workers_after = get_batch_lengths(loader_no_workers)

    print(f"With workers (num_workers=4): min={min(lengths_workers_after)}, max={max(lengths_workers_after)}")
    print(f"Without workers (num_workers=0): min={min(lengths_no_workers_after)}, max={max(lengths_no_workers_after)}")

    # Check for bug
    print("\n" + "=" * 70)
    print("BUG DETECTION")
    print("=" * 70)

    if max(lengths_workers_after) <= 12:
        print("\n🚨 BUG CONFIRMED!")
        print("   DataLoader with workers is STILL producing lengths 7-12")
        print("   even though curriculum stepped to Stage 6 (lengths 12-25)!")
        print("")
        print("   ROOT CAUSE: Worker processes have their own copy of the dataset")
        print("   and don't see the set_difficulty() changes made in main process.")
        print("")
        print("   This means the model has been training on Stage 1 data")
        print("   (lengths 7-12, 80% clean) for the ENTIRE training run!")
    else:
        print("\n✓ No bug detected - workers are producing correct lengths")

    if max(lengths_no_workers_after) > 12:
        print("\n✓ Main process (num_workers=0) correctly produces Stage 6 lengths")
    else:
        print("\n⚠️ Even main process has wrong lengths - different bug!")


if __name__ == '__main__':
    main()
