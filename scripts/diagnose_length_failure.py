"""
Diagnose WHY the model fails at sequence length 14+.

Key questions:
1. Does model get positions 1-12 correct even in length 14+ sequences?
2. Or does the presence of longer sequence confuse the ENTIRE prediction?
3. Is there something special about the 12/13 boundary?
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
from src.constants import PROTON_MASS, IDX_TO_AA


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


def decode_sequence(seq_tensor, mask_tensor):
    seq = []
    for idx, valid in zip(seq_tensor.tolist(), mask_tensor.tolist()):
        if not valid:
            break
        if idx < len(IDX_TO_AA):
            aa = IDX_TO_AA[idx]
            if aa not in ['<PAD>', '<SOS>', '<EOS>', '<UNK>']:
                seq.append(aa)
    return ''.join(seq)


@torch.no_grad()
def analyze_position_accuracy(model, length, device, num_batches=20):
    """Analyze per-position accuracy for a given sequence length."""
    dataset = MS2PIPSyntheticDataset(
        min_length=length,
        max_length=length,
        noise_peaks=0,
        peak_dropout=0.0,
        mass_error_ppm=0.0,
        ms2pip_model='HCDch2',
    )
    loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)

    # Track accuracy at each position
    position_correct = np.zeros(length)
    position_total = np.zeros(length)

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

        predictions = all_logits[-1].argmax(dim=-1)  # (batch, seq_len)
        targets = batch['sequence']
        mask = batch['sequence_mask']

        # Per-position accuracy
        for b in range(predictions.shape[0]):
            seq_len = mask[b].sum().item()
            for pos in range(int(seq_len)):
                position_total[pos] += 1
                if predictions[b, pos] == targets[b, pos]:
                    position_correct[pos] += 1

    position_acc = position_correct / np.maximum(position_total, 1)
    return position_acc


@torch.no_grad()
def show_sample_predictions(model, length, device, num_samples=5):
    """Show sample predictions for visual inspection."""
    dataset = MS2PIPSyntheticDataset(
        min_length=length,
        max_length=length,
        noise_peaks=0,
        peak_dropout=0.0,
        mass_error_ppm=0.0,
        ms2pip_model='HCDch2',
    )
    loader = DataLoader(dataset, batch_size=num_samples, collate_fn=collate_fn)

    batch = next(iter(loader))
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

    predictions = all_logits[-1].argmax(dim=-1)

    print(f"\nSample predictions for length {length}:")
    print("-" * 60)
    for i in range(num_samples):
        true_seq = decode_sequence(batch['sequence'][i], batch['sequence_mask'][i])
        pred_seq = decode_sequence(predictions[i], batch['sequence_mask'][i])

        # Show character-by-character comparison
        match_str = ""
        for t, p in zip(true_seq, pred_seq):
            if t == p:
                match_str += "✓"
            else:
                match_str += "✗"

        correct = sum(1 for t, p in zip(true_seq, pred_seq) if t == p)
        print(f"  True: {true_seq}")
        print(f"  Pred: {pred_seq}")
        print(f"  Match: {match_str} ({correct}/{len(true_seq)} = {100*correct/len(true_seq):.0f}%)")
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default='cuda')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    model = load_model(args.checkpoint, device)

    print("=" * 70)
    print("POSITION-LEVEL ACCURACY ANALYSIS")
    print("=" * 70)
    print("\nHypothesis: Model may fail uniformly or only at later positions")

    # Test multiple lengths
    test_lengths = [10, 12, 14, 16, 18, 20]

    all_position_acc = {}
    for length in test_lengths:
        print(f"\nAnalyzing length {length}...")
        position_acc = analyze_position_accuracy(model, length, device)
        all_position_acc[length] = position_acc

    # Display results
    print("\n" + "=" * 70)
    print("PER-POSITION ACCURACY")
    print("=" * 70)

    # Print table header
    header = "Pos " + " ".join([f"L={l:2d}" for l in test_lengths])
    print(header)
    print("-" * len(header))

    # Print each position's accuracy across lengths
    max_len = max(test_lengths)
    for pos in range(max_len):
        row = f"{pos+1:3d} "
        for length in test_lengths:
            if pos < length:
                acc = all_position_acc[length][pos]
                row += f" {acc*100:5.1f}"
            else:
                row += "     -"
        print(row)

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for length in test_lengths:
        accs = all_position_acc[length]
        overall = np.mean(accs)
        first_half = np.mean(accs[:length//2])
        second_half = np.mean(accs[length//2:])

        print(f"\nLength {length}:")
        print(f"  Overall accuracy:    {overall*100:.1f}%")
        print(f"  First half (1-{length//2}):   {first_half*100:.1f}%")
        print(f"  Second half ({length//2+1}-{length}): {second_half*100:.1f}%")

        # Check if there's a cliff
        if len(accs) >= 14:
            before_12 = np.mean(accs[:12])
            after_12 = np.mean(accs[12:])
            print(f"  Positions 1-12:      {before_12*100:.1f}%")
            print(f"  Positions 13+:       {after_12*100:.1f}%")

    # Show sample predictions
    print("\n" + "=" * 70)
    print("SAMPLE PREDICTIONS")
    print("=" * 70)

    for length in [10, 14, 18]:
        show_sample_predictions(model, length, device, num_samples=3)

    # Final diagnosis
    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)

    # Check if model fails everywhere or just at later positions
    length_14_acc = all_position_acc[14]
    first_10 = np.mean(length_14_acc[:10])
    last_4 = np.mean(length_14_acc[10:])

    if first_10 > 0.9 and last_4 < 0.5:
        print("\nFinding: Model gets first 10 positions correct but fails on 11-14")
        print("Likely cause: Model doesn't generalize to later positions")
    elif first_10 < 0.5 and last_4 < 0.5:
        print("\nFinding: Model fails on ALL positions for length 14")
        print("Likely cause: Longer sequences confuse the entire model")
        print("This suggests the model is using sequence LENGTH as a feature")
    else:
        print(f"\nFinding: First 10 positions: {first_10*100:.1f}%, Last 4: {last_4*100:.1f}%")
        print("Pattern needs further investigation")


if __name__ == '__main__':
    main()
