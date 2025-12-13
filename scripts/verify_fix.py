"""
Verify that the DataLoader iterator recreation fix works.
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
    lengths = []
    for i in range(len(batch['sequence'])):
        seq_len = batch['sequence_mask'][i].sum().item()
        lengths.append(int(seq_len))
    return lengths


def main():
    print("=" * 70)
    print("VERIFYING DATALOADER ITERATOR RECREATION FIX")
    print("=" * 70)

    dataset = MS2PIPSyntheticDataset(
        max_peaks=100,
        max_seq_len=35,
        ms2pip_model='HCDch2',
        charge_distribution={2: 0.7, 3: 0.3},
    )

    curriculum = CurriculumScheduler(AGGRESSIVE_NOISE_CURRICULUM, dataset)

    loader = DataLoader(
        dataset,
        batch_size=32,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    print("\n--- SIMULATING FIXED TRAINING LOOP ---")

    # Create initial iterator
    train_iterator = iter(loader)

    print(f"\nInitial state: min_length={dataset.min_length}, max_length={dataset.max_length}")

    # Get a few batches with initial settings
    print("\n--- INITIAL BATCHES (Stage 1: lengths 7-12) ---")
    for i in range(3):
        batch = next(train_iterator)
        lengths = get_lengths_from_batch(batch)
        print(f"Batch {i+1}: lengths {min(lengths)}-{max(lengths)}")

    # Step curriculum to Stage 6 AND RECREATE ITERATOR (the fix)
    print("\n--- STEPPING CURRICULUM TO STAGE 6 AND RECREATING ITERATOR ---")
    curriculum.step(65000)
    train_iterator = iter(loader)  # <-- THE FIX: recreate iterator

    print(f"After step: min_length={dataset.min_length}, max_length={dataset.max_length}")

    # Get batches from NEW iterator
    print("\n--- BATCHES FROM NEW ITERATOR (Stage 6: lengths 12-25) ---")
    for i in range(5):
        batch = next(train_iterator)
        lengths = get_lengths_from_batch(batch)
        print(f"Batch {i+1}: lengths {min(lengths)}-{max(lengths)}")

    # Verify
    all_lengths = []
    for i in range(10):
        batch = next(train_iterator)
        all_lengths.extend(get_lengths_from_batch(batch))

    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)

    if min(all_lengths) >= 12 and max(all_lengths) >= 20:
        print("\n FIX VERIFIED!")
        print(f"   After recreating iterator, lengths are now {min(all_lengths)}-{max(all_lengths)}")
        print("   (Expected: 12-25 for Stage 6)")
    else:
        print("\n FIX NOT WORKING!")
        print(f"   Lengths are {min(all_lengths)}-{max(all_lengths)}")
        print("   (Expected: 12-25 for Stage 6)")


if __name__ == '__main__':
    main()
