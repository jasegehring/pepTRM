"""
Test if precursor mass is causing the length generalization failure.

Hypothesis: Model uses precursor mass to infer sequence length. When precursor
mass indicates a long sequence (14+), the model gets confused even though the
spectrum itself contains enough information.

Test:
1. Feed length-14 spectrum with CORRECT precursor mass → expect ~40% accuracy
2. Feed length-14 spectrum with FAKE precursor mass (pretend it's length 10) → what happens?
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader

from src.model.trm import TRMConfig, RecursivePeptideModel
from src.data.ms2pip_dataset import MS2PIPSyntheticDataset
from src.training.metrics import compute_metrics
from src.constants import PROTON_MASS, AMINO_ACID_MASSES, WATER_MASS


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


def load_model(checkpoint_path: str, device: str = 'cuda'):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

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

    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()

    return model


@torch.no_grad()
def test_with_precursor_modification(model, length, device, num_batches=20, precursor_factor=1.0, fake_length=None):
    """
    Test model with modified precursor mass.

    Args:
        precursor_factor: Multiply true precursor by this factor
        fake_length: If set, use the average mass of this length peptide instead
    """
    dataset = MS2PIPSyntheticDataset(
        min_length=length,
        max_length=length,
        noise_peaks=0,
        peak_dropout=0.0,
        mass_error_ppm=0.0,
        ms2pip_model='HCDch2',
    )
    loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)

    all_acc = []

    for i, batch in enumerate(loader):
        if i >= num_batches:
            break

        batch = {k: v.to(device) for k, v in batch.items()}

        # Original precursor mass
        precursor_mass = batch['precursor_mass']
        precursor_charge = batch['precursor_charge']

        # Modify precursor mass
        if fake_length is not None:
            # Use average amino acid mass * fake_length + water
            avg_aa_mass = np.mean(list(AMINO_ACID_MASSES.values()))
            fake_mass = avg_aa_mass * fake_length + WATER_MASS
            precursor_mass = torch.full_like(precursor_mass, fake_mass)
        else:
            precursor_mass = precursor_mass * precursor_factor

        precursor_mz = (precursor_mass + precursor_charge * PROTON_MASS) / precursor_charge

        all_logits, _ = model(
            spectrum_masses=batch['spectrum_masses'],
            spectrum_intensities=batch['spectrum_intensities'],
            spectrum_mask=batch['spectrum_mask'],
            precursor_mass=precursor_mz,
            precursor_charge=precursor_charge,
        )

        metrics = compute_metrics(
            logits=all_logits[-1],
            targets=batch['sequence'],
            mask=batch['sequence_mask'],
        )
        all_acc.append(metrics['token_accuracy'])

    return np.mean(all_acc), np.std(all_acc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default='cuda')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    model = load_model(args.checkpoint, device)

    print("=" * 70)
    print("PRECURSOR MASS IMPACT TEST")
    print("=" * 70)
    print("\nHypothesis: Model uses precursor mass to infer sequence length,")
    print("and gets confused when precursor mass indicates a longer sequence.")

    # Test 1: Baseline with correct precursor
    print("\n" + "-" * 70)
    print("TEST 1: Baseline accuracy with correct precursor mass")
    print("-" * 70)

    for length in [10, 12, 14, 16, 18]:
        acc, std = test_with_precursor_modification(model, length, device, precursor_factor=1.0)
        print(f"  Length {length}: {acc*100:.1f}% ± {std*100:.1f}%")

    # Test 2: Length 14 spectrum with fake precursor mass pretending to be length 10
    print("\n" + "-" * 70)
    print("TEST 2: Length 14 spectrum with precursor mass for different lengths")
    print("-" * 70)

    length = 14
    print(f"\nActual sequence length: {length}")
    print("Testing with fake precursor masses:")

    for fake_len in [10, 12, 14, 16, 18]:
        acc, std = test_with_precursor_modification(
            model, length, device, fake_length=fake_len
        )
        label = "REAL" if fake_len == length else ""
        print(f"  Fake length {fake_len}: {acc*100:.1f}% ± {std*100:.1f}% {label}")

    # Test 3: Length 10 spectrum with fake precursor mass pretending to be length 14
    print("\n" + "-" * 70)
    print("TEST 3: Length 10 spectrum with precursor mass for different lengths")
    print("-" * 70)

    length = 10
    print(f"\nActual sequence length: {length}")
    print("Testing with fake precursor masses:")

    for fake_len in [10, 12, 14, 16, 18]:
        acc, std = test_with_precursor_modification(
            model, length, device, fake_length=fake_len
        )
        label = "REAL" if fake_len == length else ""
        print(f"  Fake length {fake_len}: {acc*100:.1f}% ± {std*100:.1f}% {label}")

    # Test 4: Scaling precursor mass
    print("\n" + "-" * 70)
    print("TEST 4: Length 14 spectrum with scaled precursor mass")
    print("-" * 70)

    length = 14
    print(f"\nActual sequence length: {length}")
    print("Testing with scaled precursor masses:")

    for factor in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]:
        acc, std = test_with_precursor_modification(
            model, length, device, precursor_factor=factor
        )
        label = "REAL" if factor == 1.0 else ""
        print(f"  Factor {factor:.1f}: {acc*100:.1f}% ± {std*100:.1f}% {label}")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("""
If accuracy improves when we give length-14 spectra a fake length-10 precursor:
  → Model is over-relying on precursor mass as a "length hint"
  → Model has NOT learned to use spectrum peaks for longer sequences

If accuracy stays ~40% regardless of precursor mass:
  → Model has fundamentally not learned the peak patterns for longer sequences
  → The spectrum encoder is the bottleneck, not precursor interpretation
""")


if __name__ == '__main__':
    main()
