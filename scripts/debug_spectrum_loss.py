"""
Debug script to diagnose Gaussian spectrum loss issues.

Run this to see what's happening with predicted vs observed spectra.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from src.data.ms2pip_dataset import create_ms2pip_dataloader
from src.training.losses import SpectrumMatchingLoss
from src.constants import AA_TO_IDX, VOCAB


def debug_spectrum_loss():
    """Debug the spectrum loss with a single batch."""

    print("="*60)
    print("Spectrum Loss Diagnostic")
    print("="*60)

    # Create a small dataloader
    dataloader = create_ms2pip_dataloader(
        batch_size=4,
        min_length=7,
        max_length=12,
        ms2pip_model='HCDch2',
        num_workers=0,
    )

    # Get one batch
    batch = next(iter(dataloader))

    # Create spectrum loss
    spectrum_loss = SpectrumMatchingLoss(
        bin_size=0.1,
        max_mz=2000.0,
        sigma=0.2,  # INCREASED: More forgiving matching (was 0.05)
        ms2pip_model='HCDch2',
    )

    print(f"\nBatch info:")
    print(f"  Sequences: {batch['sequence'].shape}")
    print(f"  Peak masses: {batch['peak_masses'].shape}")
    print(f"  Peak intensities: {batch['peak_intensities'].shape}")

    # Create fake sequence probabilities at different accuracy levels
    for accuracy in [0.0, 0.25, 0.5, 0.75, 1.0]:
        print(f"\n--- Testing with {accuracy*100:.0f}% accuracy ---")

        # Create sequence probs that simulate different accuracy levels
        batch_size, seq_len = batch['sequence'].shape
        vocab_size = len(VOCAB)

        # Start with uniform distribution
        sequence_probs = torch.ones(batch_size, seq_len, vocab_size) / vocab_size

        # Mix in correct answers based on accuracy
        if accuracy > 0:
            correct_probs = F.one_hot(batch['sequence'], vocab_size).float()
            sequence_probs = accuracy * correct_probs + (1 - accuracy) * sequence_probs
            sequence_probs = sequence_probs / sequence_probs.sum(dim=-1, keepdim=True)

        # Compute loss
        with torch.no_grad():
            loss = spectrum_loss(
                sequence_probs=sequence_probs,
                observed_masses=batch['peak_masses'],
                observed_intensities=batch['peak_intensities'],
                peak_mask=batch['peak_mask'],
                sequence_mask=batch['sequence_mask'],
            )

        print(f"  Spectrum loss: {loss.item():.4f}")

        # Check predicted peaks
        from src.data.ion_types import compute_theoretical_peaks
        predicted_masses = compute_theoretical_peaks(
            sequence_probs=sequence_probs,
            aa_masses=spectrum_loss.aa_masses,
            ion_type_names=spectrum_loss.ion_type_names,
            sequence_mask=batch['sequence_mask'],
        )

        print(f"  Predicted peaks shape: {predicted_masses.shape}")
        print(f"  Predicted peaks range: [{predicted_masses.min():.1f}, {predicted_masses.max():.1f}]")
        print(f"  Observed peaks range: [{batch['peak_masses'][batch['peak_mask']].min():.1f}, {batch['peak_masses'][batch['peak_mask']].max():.1f}]")

        # Check if predicted spectrum is all zeros
        pred_spectrum = spectrum_loss._gaussian_render(predicted_masses)
        print(f"  Predicted spectrum sum: {pred_spectrum.sum().item():.4f}")
        print(f"  Predicted spectrum max: {pred_spectrum.max().item():.4f}")

        # Check observed spectrum
        obs_masses_masked = batch['peak_masses'] * batch['peak_mask'].float()
        obs_intens_masked = batch['peak_intensities'] * batch['peak_mask'].float()
        target_spectrum = spectrum_loss._gaussian_render(obs_masses_masked, obs_intens_masked)
        print(f"  Observed spectrum sum: {target_spectrum.sum().item():.4f}")
        print(f"  Observed spectrum max: {target_spectrum.max().item():.4f}")

        # Check peak coverage (new metric)
        mass_diff = batch['peak_masses'].unsqueeze(-1) - predicted_masses.unsqueeze(1)
        match_scores = torch.exp(-0.5 * (mass_diff / 0.2) ** 2)  # Use same sigma as loss
        peak_coverage = match_scores.max(dim=-1)[0]

        # Weighted coverage
        obs_intens_norm = batch['peak_intensities'] * batch['peak_mask'].float()
        intens_sum = obs_intens_norm.sum(dim=1, keepdim=True).clamp(min=1e-8)
        obs_weights = obs_intens_norm / intens_sum
        weighted_coverage = (peak_coverage * obs_weights).sum(dim=1)

        print(f"  Peak coverage (mean): {peak_coverage.mean().item():.4f}")
        print(f"  Peak coverage (min): {peak_coverage.min().item():.4f}")
        print(f"  Weighted coverage: {weighted_coverage.mean().item():.4f}")

        # Count well-matched peaks (coverage > 0.5)
        well_matched = (peak_coverage > 0.5).float() * batch['peak_mask'].float()
        print(f"  Well-matched peaks: {well_matched.sum(dim=1).mean().item():.1f}/{batch['peak_mask'].sum(dim=1).float().mean().item():.1f}")

    print("\n" + "="*60)
    print("Potential issues to check:")
    print("="*60)
    print("1. If loss is ~1.0 even at 100% accuracy → implementation bug")
    print("2. If predicted spectrum sum is very small → peaks not rendering")
    print("3. If cosine similarity is ~0 at high accuracy → spectral mismatch")
    print("4. Loss should decrease as accuracy increases")


if __name__ == '__main__':
    debug_spectrum_loss()
