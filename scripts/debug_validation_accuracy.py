"""
Debug script to understand why validation accuracy is capped at 40%.

This script systematically investigates potential causes:
1. Is val_easy also at 40%? (would indicate fundamental issue)
2. Is there a train/val distribution shift?
3. Is there a bug in how validation data is generated?
4. Is the model's accuracy correctly computed?

Usage:
    python scripts/debug_validation_accuracy.py --checkpoint path/to/checkpoint.pt
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
from src.training.metrics import compute_metrics, decode_sequence
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
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

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

    return model, checkpoint.get('step', 'unknown')


@torch.no_grad()
def detailed_evaluation(model, dataloader, device, num_batches=10, verbose=True):
    """
    Detailed evaluation showing individual predictions.
    """
    all_token_acc = []
    all_seq_acc = []

    for i, batch in enumerate(dataloader):
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

        final_logits = all_logits[-1].float()
        metrics = compute_metrics(
            logits=final_logits,
            targets=batch['sequence'],
            mask=batch['sequence_mask'],
        )

        all_token_acc.append(metrics['token_accuracy'])
        all_seq_acc.append(metrics['sequence_accuracy'])

        # Show first few predictions on first batch
        if verbose and i == 0:
            print("\nSample predictions (first batch):")
            predictions = final_logits.argmax(dim=-1)
            for j in range(min(5, len(predictions))):
                pred_seq = decode_sequence(predictions[j], batch['sequence_mask'][j])
                true_seq = decode_sequence(batch['sequence'][j], batch['sequence_mask'][j])
                seq_len = batch['sequence_mask'][j].sum().item()
                num_peaks = batch['spectrum_mask'][j].sum().item()

                # Per-token accuracy for this sequence
                correct = (predictions[j] == batch['sequence'][j]) & batch['sequence_mask'][j]
                token_acc = correct.sum().item() / batch['sequence_mask'][j].sum().item()

                print(f"  [{j}] Pred: {pred_seq[:30]}{'...' if len(pred_seq) > 30 else ''}")
                print(f"       True: {true_seq[:30]}{'...' if len(true_seq) > 30 else ''}")
                print(f"       Length: {seq_len}, Peaks: {num_peaks}, Token Acc: {token_acc*100:.1f}%")

    return {
        'token_accuracy': np.mean(all_token_acc),
        'sequence_accuracy': np.mean(all_seq_acc),
        'token_acc_std': np.std(all_token_acc),
        'seq_acc_std': np.std(all_seq_acc),
    }


def main():
    parser = argparse.ArgumentParser(description="Debug validation accuracy")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--device", type=str, default='cuda', help="Device to use")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load model
    print(f"\nLoading checkpoint: {args.checkpoint}")
    model, step = load_model(args.checkpoint, device)
    print(f"Model from step: {step}")

    print("\n" + "=" * 70)
    print("VALIDATION ACCURACY DEBUGGING")
    print("=" * 70)

    # Test 1: Val Easy (should be near-perfect)
    print("\n--- TEST 1: Val Easy (clean, short peptides) ---")
    print("Expected: ~95%+ accuracy if model works at all")
    val_easy = MS2PIPSyntheticDataset(
        min_length=7,
        max_length=10,
        noise_peaks=0,
        peak_dropout=0.0,
        mass_error_ppm=0.0,
        ms2pip_model='HCDch2',
    )
    loader_easy = DataLoader(val_easy, batch_size=32, collate_fn=collate_fn)
    metrics_easy = detailed_evaluation(model, loader_easy, device, num_batches=20)
    print(f"\nVal Easy Results:")
    print(f"  Token Accuracy: {metrics_easy['token_accuracy']*100:.1f}% (±{metrics_easy['token_acc_std']*100:.1f}%)")
    print(f"  Sequence Accuracy: {metrics_easy['sequence_accuracy']*100:.1f}% (±{metrics_easy['seq_acc_std']*100:.1f}%)")

    # Test 2: Val Hard (as configured in training script)
    print("\n--- TEST 2: Val Hard (as currently configured) ---")
    print("This is what training script uses for hard validation")
    val_hard = MS2PIPSyntheticDataset(
        min_length=8,
        max_length=25,
        noise_peaks=20,
        peak_dropout=0.25,
        mass_error_ppm=15.0,
        intensity_variation=0.3,
        ms2pip_model='HCDch2',
    )
    loader_hard = DataLoader(val_hard, batch_size=32, collate_fn=collate_fn)
    metrics_hard = detailed_evaluation(model, loader_hard, device, num_batches=20)
    print(f"\nVal Hard Results:")
    print(f"  Token Accuracy: {metrics_hard['token_accuracy']*100:.1f}% (±{metrics_hard['token_acc_std']*100:.1f}%)")
    print(f"  Sequence Accuracy: {metrics_hard['sequence_accuracy']*100:.1f}% (±{metrics_hard['seq_acc_std']*100:.1f}%)")

    # Test 3: Training-like data (curriculum final stage)
    print("\n--- TEST 3: Curriculum Final Stage Data ---")
    print("What training looks like at end of curriculum")
    val_curriculum = MS2PIPSyntheticDataset(
        min_length=12,
        max_length=25,
        noise_peaks=30,
        peak_dropout=0.45,
        mass_error_ppm=20.0,
        intensity_variation=0.7,
        ms2pip_model='HCDch2',
    )
    val_curriculum.set_difficulty(clean_data_ratio=0.0)  # Force noisy
    loader_curriculum = DataLoader(val_curriculum, batch_size=32, collate_fn=collate_fn)
    metrics_curriculum = detailed_evaluation(model, loader_curriculum, device, num_batches=20)
    print(f"\nCurriculum Final Stage Results:")
    print(f"  Token Accuracy: {metrics_curriculum['token_accuracy']*100:.1f}% (±{metrics_curriculum['token_acc_std']*100:.1f}%)")
    print(f"  Sequence Accuracy: {metrics_curriculum['sequence_accuracy']*100:.1f}% (±{metrics_curriculum['seq_acc_std']*100:.1f}%)")

    # Test 4: Check if clean_data_ratio matters
    print("\n--- TEST 4: Effect of clean_data_ratio ---")
    print("Default val_hard doesn't set clean_data_ratio (defaults to 1.0 = all clean!)")

    val_hard_clean_ratio_default = MS2PIPSyntheticDataset(
        min_length=8,
        max_length=25,
        noise_peaks=20,
        peak_dropout=0.25,
        mass_error_ppm=15.0,
        intensity_variation=0.3,
        # clean_data_ratio defaults to 1.0
        ms2pip_model='HCDch2',
    )
    loader_hard_default = DataLoader(val_hard_clean_ratio_default, batch_size=32, collate_fn=collate_fn)

    # Check what clean_data_ratio is
    print(f"  clean_data_ratio default value: {val_hard_clean_ratio_default.clean_data_ratio}")

    if val_hard_clean_ratio_default.clean_data_ratio == 1.0:
        print("\n  ⚠️  FOUND THE BUG!")
        print("  Val Hard has clean_data_ratio=1.0 by default!")
        print("  This means 100% of samples get clean data (noise NOT applied!)")
        print("  But wait... that should make it EASIER, not harder...")

    # Test 5: Force noisy validation
    print("\n--- TEST 5: Val Hard with clean_data_ratio=0.0 (force noisy) ---")
    val_hard_noisy = MS2PIPSyntheticDataset(
        min_length=8,
        max_length=25,
        noise_peaks=20,
        peak_dropout=0.25,
        mass_error_ppm=15.0,
        intensity_variation=0.3,
        ms2pip_model='HCDch2',
    )
    val_hard_noisy.set_difficulty(clean_data_ratio=0.0)  # Force ALL samples to be noisy
    loader_hard_noisy = DataLoader(val_hard_noisy, batch_size=32, collate_fn=collate_fn)
    metrics_hard_noisy = detailed_evaluation(model, loader_hard_noisy, device, num_batches=20)
    print(f"\nVal Hard (forced noisy) Results:")
    print(f"  Token Accuracy: {metrics_hard_noisy['token_accuracy']*100:.1f}% (±{metrics_hard_noisy['token_acc_std']*100:.1f}%)")
    print(f"  Sequence Accuracy: {metrics_hard_noisy['sequence_accuracy']*100:.1f}% (±{metrics_hard_noisy['seq_acc_std']*100:.1f}%)")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Val Easy (clean):        {metrics_easy['token_accuracy']*100:.1f}%")
    print(f"  Val Hard (as config):    {metrics_hard['token_accuracy']*100:.1f}%")
    print(f"  Val Hard (force noisy):  {metrics_hard_noisy['token_accuracy']*100:.1f}%")
    print(f"  Curriculum Final Stage:  {metrics_curriculum['token_accuracy']*100:.1f}%")

    if metrics_easy['token_accuracy'] < 0.7:
        print("\n  🚨 CRITICAL: Even val_easy is below 70%!")
        print("     This suggests a fundamental model or data issue")
    elif metrics_hard['token_accuracy'] < 0.5:
        print("\n  ⚠️  Val Hard below 50% while Val Easy is good")
        print("     The noise parameters in val_hard may be out of distribution")
        print("     Or there's an issue with how noise is applied")

    print("\nRecommendations based on results:")
    if metrics_curriculum['token_accuracy'] > metrics_hard['token_accuracy']:
        print("  - Curriculum final stage is easier than val_hard")
        print("  - Consider aligning val_hard parameters with curriculum")
    else:
        print("  - Model handles curriculum data better than val_hard")
        print("  - There may be a distribution shift in val_hard")


if __name__ == '__main__':
    main()
