"""
Debug the contradiction between high training accuracy and low validation accuracy.

Training shows ~93% accuracy, but validation on length 14+ shows ~40%.
This is a contradiction that needs explanation.

Hypotheses:
1. Training batches are dominated by short sequences (length 12 or less)
2. Training accuracy is computed differently than validation
3. Clean_data_ratio during training makes training easier
4. There's a bug in training accuracy computation
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from torch.utils.data import DataLoader

from src.model.trm import TRMConfig, RecursivePeptideModel
from src.data.ms2pip_dataset import MS2PIPSyntheticDataset
from src.training.curriculum_aggressive_noise import AGGRESSIVE_NOISE_CURRICULUM, CurriculumScheduler
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

    return model, checkpoint.get('step', 0)


@torch.no_grad()
def analyze_batch_by_length(model, batch, device):
    """Analyze accuracy broken down by sequence length within a batch."""
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
    targets = batch['sequence']
    mask = batch['sequence_mask']

    # Group by length
    length_results = {}
    for i in range(len(predictions)):
        seq_len = int(mask[i].sum().item())
        correct = ((predictions[i] == targets[i]) & mask[i]).sum().item()
        total = seq_len

        if seq_len not in length_results:
            length_results[seq_len] = {'correct': 0, 'total': 0, 'count': 0}
        length_results[seq_len]['correct'] += correct
        length_results[seq_len]['total'] += total
        length_results[seq_len]['count'] += 1

    return length_results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default='cuda')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    model, step = load_model(args.checkpoint, device)

    print("=" * 70)
    print("TRAINING ACCURACY CONTRADICTION ANALYSIS")
    print("=" * 70)
    print(f"\nCheckpoint from step: {step}")

    # Determine which curriculum stage this checkpoint is from
    cumulative = 0
    current_stage = None
    for i, stage in enumerate(AGGRESSIVE_NOISE_CURRICULUM):
        if step < cumulative + stage.steps:
            current_stage = stage
            stage_idx = i
            break
        cumulative += stage.steps
    else:
        current_stage = AGGRESSIVE_NOISE_CURRICULUM[-1]
        stage_idx = len(AGGRESSIVE_NOISE_CURRICULUM) - 1

    print(f"Curriculum stage: {stage_idx + 1} ({current_stage.name})")
    print(f"  Length range: {current_stage.min_length}-{current_stage.max_length}")
    print(f"  Clean ratio: {current_stage.clean_data_ratio:.0%}")
    print(f"  Noise: {current_stage.noise_peaks} peaks, {current_stage.peak_dropout:.0%} dropout")

    # TEST 1: Simulate training batch distribution
    print("\n" + "-" * 70)
    print("TEST 1: Simulated Training Batch (with curriculum settings)")
    print("-" * 70)

    train_dataset = MS2PIPSyntheticDataset(
        min_length=current_stage.min_length,
        max_length=current_stage.max_length,
        noise_peaks=current_stage.noise_peaks,
        peak_dropout=current_stage.peak_dropout,
        mass_error_ppm=current_stage.mass_error_ppm,
        intensity_variation=current_stage.intensity_variation,
        ms2pip_model='HCDch2',
    )
    train_dataset.set_difficulty(clean_data_ratio=current_stage.clean_data_ratio)

    train_loader = DataLoader(train_dataset, batch_size=80, collate_fn=collate_fn)

    # Collect stats over multiple batches
    all_length_results = {}
    overall_correct = 0
    overall_total = 0

    print("\nAnalyzing 20 training batches...")
    for i, batch in enumerate(train_loader):
        if i >= 20:
            break

        length_results = analyze_batch_by_length(model, batch, device)

        for length, stats in length_results.items():
            if length not in all_length_results:
                all_length_results[length] = {'correct': 0, 'total': 0, 'count': 0}
            all_length_results[length]['correct'] += stats['correct']
            all_length_results[length]['total'] += stats['total']
            all_length_results[length]['count'] += stats['count']
            overall_correct += stats['correct']
            overall_total += stats['total']

    print(f"\nOverall training accuracy: {overall_correct/overall_total*100:.1f}%")
    print(f"\nBreakdown by length in training batch:")
    print(f"{'Length':<10} {'Samples':<10} {'Accuracy':<15} {'% of Batch':<15}")
    print("-" * 50)

    total_samples = sum(r['count'] for r in all_length_results.values())
    for length in sorted(all_length_results.keys()):
        r = all_length_results[length]
        acc = r['correct'] / r['total'] * 100
        pct = r['count'] / total_samples * 100
        print(f"{length:<10} {r['count']:<10} {acc:>6.1f}%        {pct:>5.1f}%")

    # TEST 2: Same data but 100% clean
    print("\n" + "-" * 70)
    print("TEST 2: Same length range, but 100% CLEAN data")
    print("-" * 70)

    clean_dataset = MS2PIPSyntheticDataset(
        min_length=current_stage.min_length,
        max_length=current_stage.max_length,
        noise_peaks=0,
        peak_dropout=0.0,
        mass_error_ppm=0.0,
        intensity_variation=0.0,
        ms2pip_model='HCDch2',
    )

    clean_loader = DataLoader(clean_dataset, batch_size=80, collate_fn=collate_fn)

    clean_length_results = {}
    clean_overall_correct = 0
    clean_overall_total = 0

    for i, batch in enumerate(clean_loader):
        if i >= 20:
            break

        length_results = analyze_batch_by_length(model, batch, device)

        for length, stats in length_results.items():
            if length not in clean_length_results:
                clean_length_results[length] = {'correct': 0, 'total': 0, 'count': 0}
            clean_length_results[length]['correct'] += stats['correct']
            clean_length_results[length]['total'] += stats['total']
            clean_length_results[length]['count'] += stats['count']
            clean_overall_correct += stats['correct']
            clean_overall_total += stats['total']

    print(f"\nOverall accuracy (clean): {clean_overall_correct/clean_overall_total*100:.1f}%")
    print(f"\nBreakdown by length (clean data):")
    print(f"{'Length':<10} {'Samples':<10} {'Accuracy':<15}")
    print("-" * 35)

    for length in sorted(clean_length_results.keys()):
        r = clean_length_results[length]
        acc = r['correct'] / r['total'] * 100
        print(f"{length:<10} {r['count']:<10} {acc:>6.1f}%")

    # TEST 3: Check if clean_data_ratio affects which samples are "easy"
    print("\n" + "-" * 70)
    print("TEST 3: Accuracy on CLEAN vs NOISY samples within mixed batch")
    print("-" * 70)

    # We can't easily separate clean vs noisy samples after generation,
    # but we can compare 100% clean vs 100% noisy
    noisy_dataset = MS2PIPSyntheticDataset(
        min_length=current_stage.min_length,
        max_length=current_stage.max_length,
        noise_peaks=current_stage.noise_peaks,
        peak_dropout=current_stage.peak_dropout,
        mass_error_ppm=current_stage.mass_error_ppm,
        intensity_variation=current_stage.intensity_variation,
        ms2pip_model='HCDch2',
    )
    noisy_dataset.set_difficulty(clean_data_ratio=0.0)  # Force ALL noisy

    noisy_loader = DataLoader(noisy_dataset, batch_size=80, collate_fn=collate_fn)

    noisy_length_results = {}
    noisy_overall_correct = 0
    noisy_overall_total = 0

    for i, batch in enumerate(noisy_loader):
        if i >= 20:
            break

        length_results = analyze_batch_by_length(model, batch, device)

        for length, stats in length_results.items():
            if length not in noisy_length_results:
                noisy_length_results[length] = {'correct': 0, 'total': 0, 'count': 0}
            noisy_length_results[length]['correct'] += stats['correct']
            noisy_length_results[length]['total'] += stats['total']
            noisy_length_results[length]['count'] += stats['count']
            noisy_overall_correct += stats['correct']
            noisy_overall_total += stats['total']

    print(f"\nOverall accuracy (100% noisy): {noisy_overall_correct/noisy_overall_total*100:.1f}%")
    print(f"\nBreakdown by length (noisy data):")
    print(f"{'Length':<10} {'Samples':<10} {'Accuracy':<15}")
    print("-" * 35)

    for length in sorted(noisy_length_results.keys()):
        r = noisy_length_results[length]
        acc = r['correct'] / r['total'] * 100
        print(f"{length:<10} {r['count']:<10} {acc:>6.1f}%")

    # SUMMARY
    print("\n" + "=" * 70)
    print("SUMMARY: EXPLAINING THE CONTRADICTION")
    print("=" * 70)

    # Calculate weighted average accuracy based on length distribution
    print("\nTraining batch composition (curriculum stage settings):")
    print(f"  Length range: {current_stage.min_length}-{current_stage.max_length}")
    print(f"  Clean ratio: {current_stage.clean_data_ratio:.0%}")

    print(f"\nAccuracy by condition:")
    print(f"  Training batch (mixed):     {overall_correct/overall_total*100:.1f}%")
    print(f"  Same lengths (100% clean):  {clean_overall_correct/clean_overall_total*100:.1f}%")
    print(f"  Same lengths (100% noisy):  {noisy_overall_correct/noisy_overall_total*100:.1f}%")

    # Check if short sequences dominate
    short_correct = sum(all_length_results.get(l, {'correct': 0})['correct'] for l in range(7, 13))
    short_total = sum(all_length_results.get(l, {'total': 1})['total'] for l in range(7, 13))
    long_correct = sum(all_length_results.get(l, {'correct': 0})['correct'] for l in range(13, 26))
    long_total = sum(all_length_results.get(l, {'total': 1})['total'] for l in range(13, 26))

    if short_total > 0 and long_total > 0:
        print(f"\n  Short sequences (7-12): {short_correct/short_total*100:.1f}%")
        print(f"  Long sequences (13+):   {long_correct/long_total*100:.1f}%")


if __name__ == '__main__':
    main()
