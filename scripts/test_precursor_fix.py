"""
Quick test to verify precursor loss fix works end-to-end.
"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from omegaconf import OmegaConf

from src.model.trm import TRMConfig, RecursivePeptideModel
from src.data.ms2pip_dataset import MS2PIPSyntheticDataset
from src.training.losses import CombinedLoss
from torch.utils.data import DataLoader


def test_end_to_end():
    """Test precursor loss with actual model."""
    print("=" * 70)
    print("END-TO-END PRECURSOR LOSS TEST")
    print("=" * 70)

    # Load model config
    config_path = project_root / 'configs' / 'aggressive_noise_test.yaml'
    cfg = OmegaConf.load(config_path)

    # Create small model (use config defaults, just reduce layers)
    model_config = TRMConfig(**cfg.model)
    model_config.num_encoder_layers = 2
    model_config.num_decoder_layers = 2
    model = RecursivePeptideModel(model_config)

    print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create dataset
    dataset = MS2PIPSyntheticDataset(
        min_length=7,
        max_length=10,
        noise_peaks=0,
        peak_dropout=0.0,
        max_peaks=model_config.max_peaks,
        max_seq_len=model_config.max_seq_len,
    )

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

    loader = DataLoader(dataset, batch_size=4, num_workers=0, collate_fn=collate_fn)

    # Create loss function with precursor loss enabled
    loss_fn = CombinedLoss(
        ce_weight=1.0,
        spectrum_weight=0.0,
        precursor_weight=0.1,  # Enable precursor loss
        iteration_weights='exponential',
    )

    print("\nRunning inference on batch...")
    model.eval()

    batch = next(iter(loader))

    with torch.no_grad():
        # Forward pass
        all_logits, _ = model(
            spectrum_masses=batch['spectrum_masses'],
            spectrum_intensities=batch['spectrum_intensities'],
            spectrum_mask=batch['spectrum_mask'],
            precursor_mass=batch['precursor_mass'],
            precursor_charge=batch['precursor_charge'],
        )

        # Compute loss
        loss, metrics = loss_fn(
            all_logits=all_logits,
            targets=batch['sequence'],
            target_mask=batch['sequence_mask'],
            observed_masses=batch['spectrum_masses'],
            observed_intensities=batch['spectrum_intensities'],
            peak_mask=batch['spectrum_mask'],
            precursor_mass=batch['precursor_mass'],
        )

    print("\n--- Results ---")
    print(f"Total loss: {loss.item():.4f}")
    print(f"\nCross-entropy metrics:")
    for k, v in metrics.items():
        if 'step' in k or 'ce' in k:
            print(f"  {k}: {v:.4f}")

    print(f"\nPrecursor mass metrics:")
    if 'ppm_error' in metrics:
        print(f"  PPM error: {metrics['ppm_error']:.1f}")
        print(f"  AA error: {metrics['aa_error']:.1f}%")
        print(f"  Mass error: {metrics['mass_error_da']:.2f} Da")
        print(f"  Special token prob: {metrics.get('special_token_prob', 0.0):.3f}")

        # Check if fix is working
        if metrics['special_token_prob'] < 0.05:
            print(f"\n✓ Special token probability is low ({metrics['special_token_prob']:.3f})")
            print("✓ Fix is working - model not putting significant mass on PAD/SOS/EOS")
        else:
            print(f"\n⚠ Special token probability is high ({metrics['special_token_prob']:.3f})")
            print("⚠ Model may need training to avoid special tokens")

        # With random initialization, expect high errors but they should be reasonable
        if metrics['ppm_error'] < 500000:  # Less than 50% error
            print(f"✓ PPM error is reasonable for random model ({metrics['ppm_error']:.0f} ppm)")
        else:
            print(f"❌ PPM error is too high ({metrics['ppm_error']:.0f} ppm) - possible bug")

    else:
        print("  (No precursor metrics - precursor loss weight is 0)")

    print("\n" + "=" * 70)
    print("Test complete!")
    print("=" * 70)


if __name__ == '__main__':
    test_end_to_end()
