"""
Verify the spectrum loss fix using the ACTUAL checkpoint from step 15000.

This script loads your real trained model and tests the spectrum loss
to confirm the fix works with the actual training state.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from src.model.trm import TRMConfig, create_model
from src.data.ms2pip_dataset import MS2PIPSyntheticDataset
from src.training.losses import CombinedLoss
from src.data.ion_types import get_ion_types_for_model, compute_theoretical_peaks
from src.constants import AMINO_ACID_MASSES, VOCAB, PROTON_MASS


def load_checkpoint(checkpoint_path, model, device='cpu'):
    """Load model weights from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle compiled models
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        # Remove _orig_mod. prefix from compiled model
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    return checkpoint


def test_with_real_checkpoint(checkpoint_path, num_samples=10):
    """Test spectrum loss with actual trained model."""

    print("=" * 70)
    print("SPECTRUM LOSS FIX VERIFICATION")
    print("=" * 70)
    print(f"\nCheckpoint: {checkpoint_path}")

    device = 'cpu'  # Use CPU to avoid memory issues

    # Load config
    config_path = project_root / 'configs' / 'optimized_extended.yaml'
    cfg = OmegaConf.load(config_path)

    # Create model
    model_config = TRMConfig(**cfg.model)
    model = create_model(model_config).to(device)

    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path, model, device)
    model.eval()

    print(f"\nLoaded checkpoint from step: {checkpoint['step']}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create dataset (same as training)
    dataset = MS2PIPSyntheticDataset(
        max_peaks=model_config.max_peaks,
        max_seq_len=model_config.max_seq_len,
        ms2pip_model=cfg.data.ms2pip_model,
        min_length=10,
        max_length=15,
    )

    # Create loss (same as training)
    loss_fn = CombinedLoss(
        ce_weight=1.0,
        spectrum_weight=1.0,
        precursor_weight=0.0,  # Test spectrum loss only
        ms2pip_model=cfg.data.ms2pip_model,
    ).to(device)

    print(f"Ion types: {get_ion_types_for_model(cfg.data.ms2pip_model)}")

    # Test on multiple samples
    print("\n" + "=" * 70)
    print("TESTING WITH ACTUAL TRAINED MODEL")
    print("=" * 70)

    dataset_iter = iter(dataset)
    all_losses = []
    all_coverages = []
    all_token_accs = []
    predicted_mass_ranges = []
    observed_mass_ranges = []

    with torch.no_grad():
        for i in range(num_samples):
            sample = next(dataset_iter)

            # Move to device
            spectrum_masses = sample.spectrum_masses.unsqueeze(0).to(device)
            spectrum_intensities = sample.spectrum_intensities.unsqueeze(0).to(device)
            spectrum_mask = sample.spectrum_mask.unsqueeze(0).to(device)
            precursor_mass = sample.precursor_mass.unsqueeze(0).to(device)
            precursor_charge = sample.precursor_charge.unsqueeze(0).to(device)
            sequence = sample.sequence.unsqueeze(0).to(device)
            sequence_mask = sample.sequence_mask.unsqueeze(0).to(device)

            # Calculate precursor m/z
            precursor_mz = (precursor_mass + precursor_charge * PROTON_MASS) / precursor_charge

            # Forward pass (ACTUAL TRAINING PATH)
            all_logits, _ = model(
                spectrum_masses=spectrum_masses,
                spectrum_intensities=spectrum_intensities,
                spectrum_mask=spectrum_mask,
                precursor_mass=precursor_mz,
                precursor_charge=precursor_charge,
            )

            # Get final probabilities (ACTUAL TRAINING STATE)
            final_probs = F.softmax(all_logits[-1], dim=-1)

            # Compute token accuracy
            predictions = final_probs.argmax(dim=-1)
            correct = (predictions == sequence) & sequence_mask
            token_acc = correct.sum().float() / sequence_mask.sum().float()
            all_token_accs.append(token_acc.item())

            # Compute spectrum loss (SAME AS TRAINING)
            spec_loss = loss_fn.spectrum_loss(
                final_probs,
                spectrum_masses,
                spectrum_intensities,
                spectrum_mask,
                sequence_mask=sequence_mask,  # THE FIX!
            )

            # Compute theoretical peaks to check mass range
            aa_masses = torch.tensor([AMINO_ACID_MASSES.get(aa, 0.0) for aa in VOCAB]).to(device)
            predicted_masses = compute_theoretical_peaks(
                sequence_probs=final_probs,
                aa_masses=aa_masses,
                ion_type_names=get_ion_types_for_model(cfg.data.ms2pip_model),
                sequence_mask=sequence_mask,  # THE FIX!
            )

            # Get observed peaks
            obs_masses = spectrum_masses[0][spectrum_mask[0].bool()]

            # Record stats
            all_losses.append(spec_loss.item())
            all_coverages.append(1.0 - spec_loss.item())
            predicted_mass_ranges.append((predicted_masses.min().item(), predicted_masses.max().item()))
            observed_mass_ranges.append((obs_masses.min().item(), obs_masses.max().item()))

    # Summary statistics
    print(f"\n{'─' * 70}")
    print("RESULTS SUMMARY")
    print(f"{'─' * 70}")
    print(f"\nTested {num_samples} samples from ACTUAL training distribution")
    print(f"\nToken Accuracy (from checkpoint):")
    print(f"  Mean: {sum(all_token_accs)/len(all_token_accs):.1%}")
    print(f"  Range: [{min(all_token_accs):.1%}, {max(all_token_accs):.1%}]")

    print(f"\nSpectrum Loss:")
    print(f"  Mean: {sum(all_losses)/len(all_losses):.4f}")
    print(f"  Range: [{min(all_losses):.4f}, {max(all_losses):.4f}]")

    print(f"\nSpectrum Coverage (1 - loss):")
    print(f"  Mean: {sum(all_coverages)/len(all_coverages):.1%}")
    print(f"  Range: [{min(all_coverages):.1%}, {max(all_coverages):.1%}]")

    print(f"\nPredicted Peak Mass Ranges:")
    for i, (min_m, max_m) in enumerate(predicted_mass_ranges):
        print(f"  Sample {i+1}: [{min_m:.1f}, {max_m:.1f}] Da")

    print(f"\nObserved Peak Mass Ranges:")
    for i, (min_m, max_m) in enumerate(observed_mass_ranges):
        print(f"  Sample {i+1}: [{min_m:.1f}, {max_m:.1f}] Da")

    # Check if ranges overlap properly
    print(f"\n{'─' * 70}")
    print("VERIFICATION CHECKS")
    print(f"{'─' * 70}")

    max_predicted = max(r[1] for r in predicted_mass_ranges)
    max_observed = max(r[1] for r in observed_mass_ranges)

    if max_predicted > 2000:
        print(f"❌ FAIL: Predicted masses too high ({max_predicted:.1f} Da > 2000 Da)")
        print(f"   This suggests PAD tokens are still being included!")
    else:
        print(f"✓ PASS: Predicted masses in reasonable range (max: {max_predicted:.1f} Da)")

    if max_predicted > max_observed * 1.5:
        print(f"❌ FAIL: Predicted masses much higher than observed")
        print(f"   Predicted max: {max_predicted:.1f} Da")
        print(f"   Observed max: {max_observed:.1f} Da")
    else:
        print(f"✓ PASS: Predicted and observed mass ranges are compatible")

    avg_coverage = sum(all_coverages) / len(all_coverages)
    if avg_coverage < 0.05:
        print(f"⚠️  WARNING: Coverage very low ({avg_coverage:.1%})")
        print(f"   This is expected if model predictions are still random")
    elif avg_coverage > 0.10:
        print(f"✓ GOOD: Coverage is {avg_coverage:.1%} (model making reasonable predictions)")
    else:
        print(f"✓ PASS: Coverage is {avg_coverage:.1%} (low but model has some signal)")

    print(f"\n{'=' * 70}")
    if max_predicted < 2000 and max_predicted < max_observed * 1.5:
        print("✓✓✓ FIX VERIFIED: Spectrum loss is computing correctly!")
        print("    Predicted peaks are in the right mass range.")
        print("    PAD tokens are being masked out properly.")
    else:
        print("❌❌❌ FIX NOT WORKING: Predicted peaks still out of range!")
    print("=" * 70)


if __name__ == '__main__':
    checkpoint_path = project_root / 'checkpoints_optimized' / 'checkpoint_step_15000.pt'
    test_with_real_checkpoint(checkpoint_path, num_samples=20)
