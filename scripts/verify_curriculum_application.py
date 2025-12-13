"""
Verify that the curriculum is actually being applied during training.

Check if there's a bug where dataset parameters aren't being updated.
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


def check_batch_lengths(loader, num_batches=5):
    """Check actual sequence lengths in batches."""
    lengths = []
    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        for j in range(len(batch['sequence'])):
            seq_len = batch['sequence_mask'][j].sum().item()
            lengths.append(int(seq_len))
    return lengths


def main():
    print("=" * 70)
    print("CURRICULUM APPLICATION VERIFICATION")
    print("=" * 70)

    # Create dataset and curriculum exactly as in training
    dataset = MS2PIPSyntheticDataset(
        max_peaks=100,
        max_seq_len=35,
        ms2pip_model='HCDch2',
        charge_distribution={2: 0.7, 3: 0.3},
    )

    curriculum = CurriculumScheduler(AGGRESSIVE_NOISE_CURRICULUM, dataset)
    loader = DataLoader(dataset, batch_size=80, collate_fn=collate_fn)

    print("\n--- INITIAL STATE (Step 0) ---")
    print(f"Dataset min_length: {dataset.min_length}")
    print(f"Dataset max_length: {dataset.max_length}")
    print(f"Dataset clean_data_ratio: {dataset.clean_data_ratio}")
    print(f"Dataset noise_peaks: {dataset.noise_peaks}")

    lengths = check_batch_lengths(loader)
    print(f"Actual lengths in batches: min={min(lengths)}, max={max(lengths)}")
    print(f"Length distribution: {sorted(set(lengths))}")

    # Simulate stepping through curriculum
    test_steps = [0, 5000, 10000, 20000, 30000, 45000, 60000, 73000]

    print("\n" + "=" * 70)
    print("CURRICULUM PROGRESSION")
    print("=" * 70)

    for step in test_steps:
        # Step curriculum to this point
        curriculum.step(step)

        print(f"\n--- Step {step} (Stage {curriculum.current_stage_idx + 1}: {curriculum.current_stage.name}) ---")
        print(f"Expected length range: {curriculum.current_stage.min_length}-{curriculum.current_stage.max_length}")
        print(f"Dataset min_length: {dataset.min_length}")
        print(f"Dataset max_length: {dataset.max_length}")
        print(f"Dataset clean_data_ratio: {dataset.clean_data_ratio}")

        # Check if dataset parameters match curriculum
        if dataset.min_length != curriculum.current_stage.min_length:
            print(f"  WARNING: min_length mismatch!")
        if dataset.max_length != curriculum.current_stage.max_length:
            print(f"  WARNING: max_length mismatch!")

        # Sample actual batches to verify
        lengths = check_batch_lengths(loader, num_batches=3)
        print(f"Actual lengths in batches: min={min(lengths)}, max={max(lengths)}")

        if min(lengths) < curriculum.current_stage.min_length:
            print(f"  BUG: Got length {min(lengths)} but min should be {curriculum.current_stage.min_length}!")
        if max(lengths) > curriculum.current_stage.max_length:
            print(f"  BUG: Got length {max(lengths)} but max should be {curriculum.current_stage.max_length}!")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
If the dataset parameters match the curriculum at each step, then the
curriculum IS being applied correctly.

If there are mismatches, there's a bug in curriculum application.
""")


if __name__ == '__main__':
    main()
