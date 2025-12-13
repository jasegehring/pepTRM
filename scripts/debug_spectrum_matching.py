"""
Debug script to analyze spectrum matching loss behavior.

This script helps diagnose why spectrum loss gets stuck at 96-98%:
1. Checks if theoretical peaks are being computed correctly
2. Analyzes peak matching scores
3. Visualizes observed vs predicted peak distributions
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
from src.constants import AMINO_ACID_MASSES, VOCAB, WATER_MASS, PROTON_MASS


def analyze_single_sample(model, loss_fn, sample, device='cuda'):
    """Analyze spectrum matching for a single sample."""

    # Move to device
    spectrum_masses = sample.spectrum_masses.unsqueeze(0).to(device)
    spectrum_intensities = sample.spectrum_intensities.unsqueeze(0).to(device)
    spectrum_mask = sample.spectrum_mask.unsqueeze(0).to(device)
    precursor_mass = sample.precursor_mass.unsqueeze(0).to(device)
    precursor_charge = sample.precursor_charge.unsqueeze(0).to(device)
    sequence = sample.sequence.unsqueeze(0).to(device)
    sequence_mask = sample.sequence_mask.unsqueeze(0).to(device)

    # Calculate precursor m/z for model input
    precursor_mz = (precursor_mass + precursor_charge * PROTON_MASS) / precursor_charge

    # Forward pass
    with torch.no_grad():
        all_logits, _ = model(
            spectrum_masses=spectrum_masses,
            spectrum_intensities=spectrum_intensities,
            spectrum_mask=spectrum_mask,
            precursor_mass=precursor_mz,
            precursor_charge=precursor_charge,
        )

        # Get final probabilities
        final_probs = F.softmax(all_logits[-1], dim=-1)

        # Compute theoretical peaks
        aa_masses = torch.tensor([AMINO_ACID_MASSES.get(aa, 0.0) for aa in VOCAB]).to(device)
        ion_type_names = get_ion_types_for_model('HCDch2')

        predicted_masses = compute_theoretical_peaks(
            sequence_probs=final_probs,
            aa_masses=aa_masses,
            ion_type_names=ion_type_names,
            sequence_mask=sequence_mask,
        )

        # Get observed peaks
        obs_masses = spectrum_masses[0][spectrum_mask[0].bool()]
        obs_intensities = spectrum_intensities[0][spectrum_mask[0].bool()]

        # Compute matching scores (same as in loss)
        sigma = 0.2
        mass_diff = obs_masses.unsqueeze(-1) - predicted_masses.unsqueeze(1)
        match_scores = torch.exp(-0.5 * (mass_diff / sigma) ** 2)
        peak_coverage = match_scores.max(dim=-1)[0]

        # Weighted coverage
        obs_weights = obs_intensities / obs_intensities.sum()
        weighted_coverage = (peak_coverage * obs_weights).sum()

    return {
        'observed_masses': obs_masses.cpu().numpy(),
        'observed_intensities': obs_intensities.cpu().numpy(),
        'predicted_masses': predicted_masses[0].cpu().numpy(),
        'peak_coverage': peak_coverage.cpu().numpy(),
        'weighted_coverage': weighted_coverage.item(),
        'spectrum_loss': 1.0 - weighted_coverage.item(),
        'num_observed': len(obs_masses),
        'num_predicted': predicted_masses.shape[1],
        'sequence_probs_shape': final_probs.shape,
        'true_sequence': sequence[0].cpu().numpy(),
        'sequence_mask': sequence_mask[0].cpu().numpy(),
    }


def main():
    print("=" * 70)
    print("Spectrum Matching Loss Debugger")
    print("=" * 70)

    # Use CPU for debugging to avoid CUDA OOM
    device = 'cpu'
    print(f"\nDevice: {device}")

    # Load config
    config_path = project_root / 'configs' / 'optimized_extended.yaml'
    cfg = OmegaConf.load(config_path)

    # Create model
    model_config = TRMConfig(**cfg.model)
    model = create_model(model_config).to(device)
    print(f"\nModel created: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create dataset
    dataset = MS2PIPSyntheticDataset(
        max_peaks=model_config.max_peaks,
        max_seq_len=model_config.max_seq_len,
        ms2pip_model=cfg.data.ms2pip_model,
        min_length=10,
        max_length=15,
    )

    # Create loss
    loss_fn = CombinedLoss(
        ce_weight=1.0,
        spectrum_weight=1.0,
        ms2pip_model=cfg.data.ms2pip_model,
    ).to(device)

    print(f"\nIon types: {get_ion_types_for_model(cfg.data.ms2pip_model)}")

    # Analyze several samples
    print("\n" + "=" * 70)
    print("ANALYZING SAMPLES")
    print("=" * 70)

    dataset_iter = iter(dataset)
    for i in range(5):
        sample = next(dataset_iter)
        result = analyze_single_sample(model, loss_fn, sample, device)

        print(f"\n{'─' * 70}")
        print(f"Sample {i+1}")
        print(f"{'─' * 70}")
        print(f"True sequence shape:     {result['sequence_probs_shape']}")
        print(f"Sequence mask sum:       {result['sequence_mask'].sum()}")
        print(f"Observed peaks:          {result['num_observed']}")
        print(f"Predicted peaks:         {result['num_predicted']}")
        print(f"Weighted coverage:       {result['weighted_coverage']:.4f} (0=bad, 1=perfect)")
        print(f"Spectrum loss:           {result['spectrum_loss']:.4f}")

        # Analyze coverage distribution
        coverage = result['peak_coverage']
        print(f"\nCoverage statistics:")
        print(f"  Min:    {coverage.min():.4f}")
        print(f"  Mean:   {coverage.mean():.4f}")
        print(f"  Max:    {coverage.max():.4f}")
        print(f"  Peaks with >0.5 coverage: {(coverage > 0.5).sum()} / {len(coverage)}")
        print(f"  Peaks with >0.1 coverage: {(coverage > 0.1).sum()} / {len(coverage)}")

        # Show mass range
        print(f"\nMass ranges:")
        print(f"  Observed:  [{result['observed_masses'].min():.1f}, {result['observed_masses'].max():.1f}] Da")
        print(f"  Predicted: [{result['predicted_masses'].min():.1f}, {result['predicted_masses'].max():.1f}] Da")

        # Check if predicted masses are reasonable
        if result['predicted_masses'].min() < 0:
            print(f"  ⚠️  WARNING: Negative predicted masses detected!")
        if result['predicted_masses'].max() > 3000:
            print(f"  ⚠️  WARNING: Very high predicted masses (>3000 Da)!")

    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)
    print("""
If spectrum loss is stuck at ~0.96-0.98 (4% coverage), check:

1. Are predicted peaks in the right mass range?
   → If predicted peaks are all very different from observed, there's a problem

2. Are there enough predicted peaks?
   → Should be ~2-4x the sequence length (b, y, b++, y++ ions)

3. Is the model predicting random sequences?
   → At 50% token accuracy, sequences should be somewhat reasonable

4. Is the sigma parameter too strict?
   → Current sigma=0.2 Da means peaks must be within ~0.4 Da to match

5. Are SOS/EOS tokens being handled correctly?
   → Check sequence_probs_shape and whether slicing [1:-1] is correct
    """)


if __name__ == '__main__':
    main()
