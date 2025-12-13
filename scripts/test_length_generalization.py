"""
Test model accuracy vs sequence length.

Hypothesis: Model fails on longer sequences, even without noise.
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
def test_length(model, length, device, num_batches=10, with_noise=False):
    """Test model on specific length, optionally with noise."""
    dataset = MS2PIPSyntheticDataset(
        min_length=length,
        max_length=length,
        noise_peaks=20 if with_noise else 0,
        peak_dropout=0.25 if with_noise else 0.0,
        mass_error_ppm=15.0 if with_noise else 0.0,
        ms2pip_model='HCDch2',
    )
    if with_noise:
        dataset.set_difficulty(clean_data_ratio=0.0)

    loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)

    all_acc = []
    for i, batch in enumerate(loader):
        if i >= num_batches:
            break

        batch = {k: v.to(device) for k, v in batch.items()}

        precursor_mass = batch['precursor_mass']
        precursor_charge = batch['precursor_charge']
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
    print("SEQUENCE LENGTH GENERALIZATION TEST")
    print("=" * 70)

    # Test lengths from 7 to 25
    lengths = [7, 8, 9, 10, 12, 14, 16, 18, 20, 22, 25]

    print("\n--- CLEAN DATA (no noise) ---")
    print(f"{'Length':<10} {'Accuracy':<15} {'Std':<10}")
    print("-" * 35)

    clean_results = []
    for length in lengths:
        acc, std = test_length(model, length, device, num_batches=15, with_noise=False)
        clean_results.append((length, acc))
        print(f"{length:<10} {acc*100:.1f}%{'':<8} ±{std*100:.1f}%")

    print("\n--- WITH NOISE (20 peaks, 25% dropout) ---")
    print(f"{'Length':<10} {'Accuracy':<15} {'Std':<10}")
    print("-" * 35)

    noisy_results = []
    for length in lengths:
        acc, std = test_length(model, length, device, num_batches=15, with_noise=True)
        noisy_results.append((length, acc))
        print(f"{length:<10} {acc*100:.1f}%{'':<8} ±{std*100:.1f}%")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Check where accuracy drops below 90%
    clean_threshold = None
    for length, acc in clean_results:
        if acc < 0.9 and clean_threshold is None:
            clean_threshold = length

    noisy_threshold = None
    for length, acc in noisy_results:
        if acc < 0.9 and noisy_threshold is None:
            noisy_threshold = length

    print(f"\nClean data accuracy drops below 90% at length: {clean_threshold}")
    print(f"Noisy data accuracy drops below 90% at length: {noisy_threshold}")

    # Check curriculum coverage
    print("\n--- CURRICULUM ANALYSIS ---")
    print("Training curriculum stages:")
    print("  Stage 1 (0-10K): length 7-12")
    print("  Stage 2 (10K-20K): length 10-15")
    print("  Stage 3 (20K-30K): length 12-18")
    print("  Stage 4 (30K-45K): length 12-20")
    print("  Stage 5 (45K-60K): length 12-22")
    print("  Stage 6-7 (60K-100K): length 12-25")

    # Find max length trained on
    print(f"\nModel trained up to step 73000 → Stage 6 (length 12-25)")
    print(f"But curriculum starts with short sequences, may not see enough long ones")


if __name__ == '__main__':
    main()
