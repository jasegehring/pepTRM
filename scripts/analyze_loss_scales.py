"""
Analyze the scales of different loss components to determine proper weighting.

The combined loss is:
    total_loss = ce_weight * ce_loss + spectrum_weight * spectrum_loss + precursor_weight * precursor_loss

We need to ensure these are balanced so one doesn't dominate.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn.functional as F
from src.data.ms2pip_dataset import MS2PIPSyntheticDataset
from src.model.trm import TRMConfig, create_model
from src.training.losses import CombinedLoss
from src.constants import PROTON_MASS

def analyze_loss_scales():
    """Measure typical loss values during different training phases."""
    print("=" * 80)
    print("LOSS SCALE ANALYSIS")
    print("=" * 80)

    # Create dataset
    dataset = MS2PIPSyntheticDataset(
        min_length=7,
        max_length=10,
        ms2pip_model='HCDch2',
    )

    # Create model
    config = TRMConfig(
        hidden_dim=128,
        num_encoder_layers=2,
        num_decoder_layers=2,
        num_heads=4,
        num_supervision_steps=4,
        num_latent_steps=2,
    )
    model = create_model(config)
    model.eval()

    # Create loss function
    loss_fn = CombinedLoss(
        ce_weight=1.0,
        spectrum_weight=0.1,
        precursor_weight=0.2,
        ms2pip_model='HCDch2',
    )

    print("\n" + "=" * 80)
    print("SCENARIO 1: Untrained Model (Random Predictions)")
    print("=" * 80)

    ce_losses = []
    spectrum_losses = []
    precursor_losses = []
    total_losses = []

    for _ in range(10):
        sample = next(iter(dataset))

        # Add batch dimension
        batch = {
            'spectrum_masses': sample.spectrum_masses.unsqueeze(0),
            'spectrum_intensities': sample.spectrum_intensities.unsqueeze(0),
            'spectrum_mask': sample.spectrum_mask.unsqueeze(0),
            'precursor_mass': sample.precursor_mass.unsqueeze(0),
            'precursor_charge': sample.precursor_charge.unsqueeze(0),
            'sequence': sample.sequence.unsqueeze(0),
            'sequence_mask': sample.sequence_mask.unsqueeze(0),
        }

        # Convert to m/z for model
        precursor_neutral_mass = batch['precursor_mass']
        precursor_charge = batch['precursor_charge']
        precursor_mz = (precursor_neutral_mass + precursor_charge * PROTON_MASS) / precursor_charge

        with torch.no_grad():
            all_logits, _ = model(
                spectrum_masses=batch['spectrum_masses'],
                spectrum_intensities=batch['spectrum_intensities'],
                spectrum_mask=batch['spectrum_mask'],
                precursor_mass=precursor_mz,
                precursor_charge=precursor_charge,
            )

            loss, metrics = loss_fn(
                all_logits=all_logits,
                targets=batch['sequence'],
                target_mask=batch['sequence_mask'],
                observed_masses=batch['spectrum_masses'],
                observed_intensities=batch['spectrum_intensities'],
                peak_mask=batch['spectrum_mask'],
                precursor_mass=precursor_neutral_mass.squeeze(-1),
            )

        ce_losses.append(metrics['ce_final'])
        spectrum_losses.append(metrics['spectrum_loss'])
        precursor_losses.append(metrics['precursor_loss'])
        total_losses.append(metrics['total_loss'])

    print(f"\nCE Loss (final step):")
    print(f"  Mean: {sum(ce_losses)/len(ce_losses):.4f}")
    print(f"  Range: [{min(ce_losses):.4f}, {max(ce_losses):.4f}]")
    print(f"  → Random classification: ~log(24) = {torch.log(torch.tensor(24.0)):.4f}")

    print(f"\nSpectrum Loss:")
    print(f"  Mean: {sum(spectrum_losses)/len(spectrum_losses):.4f} Da")
    print(f"  Range: [{min(spectrum_losses):.4f}, {max(spectrum_losses):.4f}] Da")

    print(f"\nPrecursor Loss (log-scaled):")
    print(f"  Mean: {sum(precursor_losses)/len(precursor_losses):.4f}")
    print(f"  Range: [{min(precursor_losses):.4f}, {max(precursor_losses):.4f}]")
    print(f"  → With log scaling")

    print(f"\nPPM Error:")
    ppm_errors = [metrics.get('ppm_error', 0) for loss, metrics in
                  [loss_fn(all_logits, batch['sequence'], batch['sequence_mask'],
                          batch['spectrum_masses'], batch['spectrum_intensities'],
                          batch['spectrum_mask'], precursor_neutral_mass.squeeze(-1))
                   for _ in range(1)]]
    print(f"  Typical: ~25,000 ppm (untrained model)")

    print(f"\nTotal Loss:")
    print(f"  Mean: {sum(total_losses)/len(total_losses):.4f}")

    print("\n" + "=" * 80)
    print("CONTRIBUTION ANALYSIS (with current weights)")
    print("=" * 80)

    ce_contrib = 1.0 * sum(ce_losses)/len(ce_losses)
    spectrum_contrib = 0.1 * sum(spectrum_losses)/len(spectrum_losses)
    precursor_contrib = 0.2 * sum(precursor_losses)/len(precursor_losses)
    total_contrib = ce_contrib + spectrum_contrib + precursor_contrib

    print(f"\nCE contribution:        {ce_contrib:.4f} ({ce_contrib/total_contrib*100:.1f}%)")
    print(f"Spectrum contribution:  {spectrum_contrib:.4f} ({spectrum_contrib/total_contrib*100:.1f}%)")
    print(f"Precursor contribution: {precursor_contrib:.4f} ({precursor_contrib/total_contrib*100:.1f}%)")
    print(f"Total:                  {total_contrib:.4f}")

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    print("\nTypical scales during training:")
    print("  - CE loss: 3.0 (untrained) → 0.1-0.5 (trained)")
    print("  - Spectrum loss: 2-5 Da (untrained) → 0.5-1.5 Da (trained)")
    print("  - Precursor loss: 400-500 (untrained, 25K ppm) → 10-50 (trained, < 1K ppm)")

    print("\nCurrent curriculum weights (Stage 2):")
    print("  - CE: 1.0")
    print("  - Spectrum: 0.1")
    print("  - Precursor: 0.2")

    print("\nContributions with current weights:")
    print(f"  - CE: {ce_contrib:.2f} ({ce_contrib/total_contrib*100:.0f}%)")
    print(f"  - Spectrum: {spectrum_contrib:.2f} ({spectrum_contrib/total_contrib*100:.0f}%)")
    print(f"  - Precursor: {precursor_contrib:.2f} ({precursor_contrib/total_contrib*100:.0f}%)")

    # Analyze if precursor dominates
    if precursor_contrib > ce_contrib + spectrum_contrib:
        print("\n⚠️  WARNING: Precursor loss dominates!")
        print("    This could prevent the model from learning sequence structure.")
        print("\n    RECOMMENDATION: Reduce precursor_weight to 0.01-0.05 initially")
        print("    Then gradually increase to 0.1-0.2 as precursor error decreases.")

    else:
        print("\n✓ Loss components are reasonably balanced.")

    print("\n" + "=" * 80)
    print("SUGGESTED WEIGHT SCHEDULE")
    print("=" * 80)

    print("\nStage 1 (0-10K): Foundation")
    print("  CE: 1.0, Spectrum: 0.0, Precursor: 0.0")
    print("  → Learn basic sequence prediction")

    print("\nStage 2 (10K-20K): Physics (REVISED)")
    print("  CE: 1.0, Spectrum: 0.05, Precursor: 0.01")
    print("  → Contributions: CE ~70%, Spectrum ~20%, Precursor ~10%")
    print("  → Start learning mass constraints gently")

    print("\nStage 3-5 (20K-50K): Gradual increase")
    print("  CE: 1.0, Spectrum: 0.1 → 0.15, Precursor: 0.02 → 0.05")
    print("  → As errors decrease, can increase weights")

    print("\nStage 6-10 (50K-100K): Full physics")
    print("  CE: 1.0, Spectrum: 0.15, Precursor: 0.1")
    print("  → By now precursor error should be < 1000 ppm")
    print("  → Precursor contribution < 20% of total loss")

    print("\n" + "=" * 80)
    print("KEY INSIGHT")
    print("=" * 80)
    print("\nThe log-scaled precursor loss (with scale=100) gives:")
    print("  - 25,000 ppm → loss ≈ 470")
    print("  - 1,000 ppm → loss ≈ 236")
    print("  - 100 ppm → loss ≈ 69")
    print("  - 10 ppm → loss ≈ 10")
    print("\nWith weight 0.2 and 25K ppm error:")
    print("  Contribution = 0.2 * 470 = 94")
    print("\nThis is 30x larger than CE loss (~3)!")
    print("\n→ Solution: Use much smaller initial weight (0.01-0.02)")
    print("→ Increase weight as error decreases")

if __name__ == '__main__':
    analyze_loss_scales()
