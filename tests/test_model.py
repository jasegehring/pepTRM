"""Test model forward pass."""

import torch
from src.model.trm import TRMConfig, create_model


def test_model_forward_pass():
    """Test that model forward pass produces correct shapes."""
    # Create small model for testing
    config = TRMConfig(
        hidden_dim=64,
        num_encoder_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        max_peaks=50,
        max_seq_len=15,
        num_supervision_steps=4,
        num_latent_steps=2,
    )

    model = create_model(config)
    model.eval()

    # Create dummy input
    batch_size = 2
    spectrum_masses = torch.randn(batch_size, config.max_peaks).abs() * 1000
    spectrum_intensities = torch.rand(batch_size, config.max_peaks)
    spectrum_mask = torch.ones(batch_size, config.max_peaks, dtype=torch.bool)
    precursor_mass = torch.tensor([800.0, 900.0])
    precursor_charge = torch.tensor([2, 3], dtype=torch.long)

    # Forward pass
    with torch.no_grad():
        all_logits, final_z = model(
            spectrum_masses,
            spectrum_intensities,
            spectrum_mask,
            precursor_mass,
            precursor_charge,
        )

    # Check shapes
    assert all_logits.shape == (config.num_supervision_steps, batch_size, config.max_seq_len, config.vocab_size)
    assert final_z.shape == (batch_size, config.max_seq_len, config.hidden_dim)

    print("✓ Model forward pass test passed!")
    print(f"  Output shape: {all_logits.shape}")
    print(f"  Latent state shape: {final_z.shape}")


if __name__ == '__main__':
    test_model_forward_pass()
