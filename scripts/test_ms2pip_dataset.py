"""
Quick validation test for MS2PIP dataset integration.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.ms2pip_dataset import MS2PIPSyntheticDataset
import torch

def test_dataset_generation():
    """Test that we can generate samples from MS2PIP dataset."""
    print("=" * 80)
    print("TESTING MS2PIP DATASET")
    print("=" * 80)

    # Create dataset with simple config
    print("\n1. Creating MS2PIP dataset...")
    dataset = MS2PIPSyntheticDataset(
        min_length=7,
        max_length=15,
        max_peaks=100,
        max_seq_len=20,
        charge_distribution={2: 0.7, 3: 0.3},
        # No curriculum noise for this test
        noise_peaks=0,
        peak_dropout=0.0,
        mass_error_ppm=0.0,
        intensity_variation=0.0,
        ms2pip_model="HCD2021",
        top_k_peaks=50,  # Limit to top 50 peaks
    )
    print(f"   ✓ Dataset created")
    print(f"   Vocab size: {dataset.vocab_size}")
    print(f"   Amino acids: {''.join(dataset.amino_acids)}")

    # Generate a few samples
    print("\n2. Generating samples...")
    iterator = iter(dataset)

    for i in range(3):
        print(f"\n   Sample {i+1}:")
        try:
            sample = next(iterator)

            # Decode sequence
            seq_tokens = sample.sequence[sample.sequence_mask].tolist()
            sequence = ''.join([dataset.idx_to_token[tok] for tok in seq_tokens])

            # Count real peaks
            num_peaks = sample.spectrum_mask.sum().item()

            # Get charge
            charge = sample.precursor_charge.item()

            # Get precursor mass
            prec_mass = sample.precursor_mass.item()

            print(f"      Sequence: {sequence} (length {len(sequence)})")
            print(f"      Charge: +{charge}")
            print(f"      Precursor m/z: {prec_mass:.2f}")
            print(f"      Number of peaks: {num_peaks}")

            # Check peak statistics
            real_masses = sample.spectrum_masses[sample.spectrum_mask]
            real_intensities = sample.spectrum_intensities[sample.spectrum_mask]

            if len(real_masses) > 0:
                print(f"      m/z range: {real_masses.min():.2f} - {real_masses.max():.2f}")
                print(f"      Intensity range: {real_intensities.min():.4f} - {real_intensities.max():.4f}")
                print(f"      Top 3 peaks (m/z, intensity):")
                top_3_idx = torch.argsort(real_intensities, descending=True)[:3]
                for idx in top_3_idx:
                    print(f"         {real_masses[idx]:.2f} m/z, {real_intensities[idx]:.4f}")

            print(f"      ✓ Sample generated successfully")

        except Exception as e:
            print(f"      ✗ Error generating sample: {e}")
            import traceback
            traceback.print_exc()
            return False

    return True


def test_curriculum_noise():
    """Test that curriculum noise is applied correctly."""
    print("\n" + "=" * 80)
    print("TESTING CURRICULUM NOISE")
    print("=" * 80)

    # Create dataset with realistic noise
    print("\n1. Creating dataset with curriculum noise...")
    dataset = MS2PIPSyntheticDataset(
        min_length=7,
        max_length=15,
        max_peaks=100,
        noise_peaks=10,
        peak_dropout=0.15,
        mass_error_ppm=15.0,
        intensity_variation=0.1,
        ms2pip_model="HCD2021",
        top_k_peaks=50,
    )
    print(f"   ✓ Dataset created with noise parameters:")
    print(f"      Noise peaks: {dataset.noise_peaks}")
    print(f"      Peak dropout: {dataset.peak_dropout}")
    print(f"      Mass error: {dataset.mass_error_ppm} ppm")
    print(f"      Intensity variation: {dataset.intensity_variation}")

    # Generate sample
    print("\n2. Generating noisy sample...")
    sample = next(iter(dataset))

    num_peaks = sample.spectrum_mask.sum().item()
    print(f"   Number of peaks (with noise): {num_peaks}")

    # Decode sequence
    seq_tokens = sample.sequence[sample.sequence_mask].tolist()
    sequence = ''.join([dataset.idx_to_token[tok] for tok in seq_tokens])
    print(f"   Sequence: {sequence}")
    print(f"   ✓ Noisy sample generated successfully")

    return True


def test_dataloader():
    """Test DataLoader creation."""
    print("\n" + "=" * 80)
    print("TESTING DATALOADER")
    print("=" * 80)

    from src.data.ms2pip_dataset import create_ms2pip_dataloader

    print("\n1. Creating DataLoader...")
    dataloader = create_ms2pip_dataloader(
        batch_size=4,
        num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
        min_length=7,
        max_length=15,
        max_peaks=100,
        top_k_peaks=50,
    )
    print(f"   ✓ DataLoader created")

    print("\n2. Loading a batch...")
    batch = next(iter(dataloader))

    print(f"   Batch keys: {list(batch.keys())}")
    print(f"   Batch size: {batch['sequence'].shape[0]}")
    print(f"   Sequence shape: {batch['sequence'].shape}")
    print(f"   Peak masses shape: {batch['peak_masses'].shape}")
    print(f"   Peak intensities shape: {batch['peak_intensities'].shape}")

    # Check each sample in batch
    for i in range(batch['sequence'].shape[0]):
        seq_mask = batch['sequence_mask'][i]
        seq_tokens = batch['sequence'][i][seq_mask].tolist()

        # Decode using first dataset's vocab
        dataset = MS2PIPSyntheticDataset()
        sequence = ''.join([dataset.idx_to_token[tok] for tok in seq_tokens])

        num_peaks = batch['peak_mask'][i].sum().item()
        charge = batch['precursor_charge'][i].item()

        print(f"\n   Sample {i+1}: {sequence} (charge +{charge}, {num_peaks} peaks)")

    print(f"\n   ✓ Batch loaded successfully")

    return True


if __name__ == '__main__':
    print("\nStarting MS2PIP dataset validation...\n")

    success = True

    # Run tests
    if not test_dataset_generation():
        success = False
        print("\n✗ Dataset generation test FAILED")

    if not test_curriculum_noise():
        success = False
        print("\n✗ Curriculum noise test FAILED")

    if not test_dataloader():
        success = False
        print("\n✗ DataLoader test FAILED")

    if success:
        print("\n" + "=" * 80)
        print("✓ ALL TESTS PASSED!")
        print("=" * 80)
        print("\nMS2PIP dataset is ready to use for training.")
    else:
        print("\n" + "=" * 80)
        print("✗ SOME TESTS FAILED")
        print("=" * 80)
        print("\nPlease fix the errors above before using the dataset.")
