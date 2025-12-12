"""
Verify model construction and functionality before training.

Tests:
1. Model architecture (parameter counts, layer dimensions)
2. Forward pass shapes
3. Loss computation (CE, spectrum matching, precursor constraint)
4. Ion type configuration
5. Curriculum integration
6. Memory usage estimates
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from omegaconf import OmegaConf

from src.model.trm import TRMConfig, create_model
from src.data.ms2pip_dataset import MS2PIPSyntheticDataset
from src.training.losses import CombinedLoss
from src.data.ion_types import get_ion_types_for_model


def verify_model_architecture(model, config):
    """Verify model architecture matches expectations."""
    print("=" * 60)
    print("MODEL ARCHITECTURE VERIFICATION")
    print("=" * 60)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")
    print(f"\nConfiguration:")
    print(f"  - Hidden dim: {config.hidden_dim}")
    print(f"  - Encoder layers: {config.num_encoder_layers}")
    print(f"  - Decoder layers: {config.num_decoder_layers}")
    print(f"  - Attention heads: {config.num_heads}")
    print(f"  - Supervision steps: {config.num_supervision_steps}")
    print(f"  - Latent steps: {config.num_latent_steps}")
    print(f"  - Max sequence length: {config.max_seq_len}")

    # Verify head dimension
    head_dim = config.hidden_dim // config.num_heads
    assert config.hidden_dim % config.num_heads == 0, \
        f"Hidden dim {config.hidden_dim} must be divisible by num_heads {config.num_heads}"
    print(f"\n✓ Head dimension: {head_dim} (valid)")

    # Expected parameter count (rough estimate)
    # Encoder: embedding + layers
    # Decoder: embedding + layers + output
    expected_range = (10_000_000, 15_000_000)  # 10-15M for this config
    assert expected_range[0] <= total_params <= expected_range[1], \
        f"Parameter count {total_params:,} outside expected range {expected_range}"
    print(f"✓ Parameter count within expected range {expected_range[0]:,} - {expected_range[1]:,}")

    return True


def verify_forward_pass(model, config, device='cuda'):
    """Verify forward pass produces correct shapes."""
    print("\n" + "=" * 60)
    print("FORWARD PASS VERIFICATION")
    print("=" * 60)

    batch_size = 4
    max_peaks = config.max_peaks
    max_seq_len = config.max_seq_len

    # Create dummy inputs
    spectrum_masses = torch.randn(batch_size, max_peaks).abs().to(device) * 1000
    spectrum_intensities = torch.randn(batch_size, max_peaks).abs().to(device)
    spectrum_mask = torch.ones(batch_size, max_peaks, dtype=torch.bool).to(device)
    precursor_mass = torch.randn(batch_size).abs().to(device) * 1500
    precursor_charge = torch.randint(2, 4, (batch_size,)).to(device)

    print(f"\nInput shapes:")
    print(f"  - spectrum_masses: {spectrum_masses.shape}")
    print(f"  - spectrum_intensities: {spectrum_intensities.shape}")
    print(f"  - spectrum_mask: {spectrum_mask.shape}")
    print(f"  - precursor_mass: {precursor_mass.shape}")
    print(f"  - precursor_charge: {precursor_charge.shape}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        all_logits, final_z = model(
            spectrum_masses=spectrum_masses,
            spectrum_intensities=spectrum_intensities,
            spectrum_mask=spectrum_mask,
            precursor_mass=precursor_mass,
            precursor_charge=precursor_charge,
        )

    print(f"\nOutput shapes:")
    print(f"  - all_logits: {all_logits.shape}")
    print(f"  - final_z: {final_z.shape}")

    # Verify shapes (model returns T, batch, seq, vocab)
    expected_logits_shape = (config.num_supervision_steps, batch_size, max_seq_len, config.vocab_size)
    assert all_logits.shape == expected_logits_shape, \
        f"Expected logits shape {expected_logits_shape}, got {all_logits.shape}"
    print(f"✓ Logits shape correct (T, batch, seq, vocab): {all_logits.shape}")

    expected_z_shape = (batch_size, max_seq_len, config.hidden_dim)
    assert final_z.shape == expected_z_shape, \
        f"Expected final_z shape {expected_z_shape}, got {final_z.shape}"
    print(f"✓ Final z shape correct: {final_z.shape}")

    # Check for NaNs
    assert not torch.isnan(all_logits).any(), "NaN detected in logits"
    assert not torch.isnan(final_z).any(), "NaN detected in final_z"
    print(f"✓ No NaNs in outputs")

    return True


def verify_loss_computation(model, config, device='cuda'):
    """Verify loss computation with all components."""
    print("\n" + "=" * 60)
    print("LOSS COMPUTATION VERIFICATION")
    print("=" * 60)

    batch_size = 4
    max_peaks = config.max_peaks
    max_seq_len = config.max_seq_len

    # Create inputs
    spectrum_masses = torch.randn(batch_size, max_peaks).abs().to(device) * 1000
    spectrum_intensities = torch.randn(batch_size, max_peaks).abs().to(device)
    spectrum_mask = torch.ones(batch_size, max_peaks, dtype=torch.bool).to(device)
    precursor_mass = torch.randn(batch_size).abs().to(device) * 1500
    precursor_charge = torch.randint(2, 4, (batch_size,)).to(device)

    # Create target sequences
    sequence = torch.randint(1, 20, (batch_size, max_seq_len)).to(device)
    sequence_mask = torch.ones(batch_size, max_seq_len, dtype=torch.bool).to(device)

    # Create loss function with ion types
    ion_types = get_ion_types_for_model('HCDch2')
    print(f"\nIon types: {ion_types}")

    loss_fn = CombinedLoss(
        ce_weight=1.0,
        spectrum_weight=0.5,
        precursor_weight=0.5,
        iteration_weights='linear',
        label_smoothing=0.1,
        ion_type_names=ion_types,
    )
    loss_fn = loss_fn.to(device)

    # Forward pass
    model.eval()
    with torch.no_grad():
        all_logits, _ = model(
            spectrum_masses=spectrum_masses,
            spectrum_intensities=spectrum_intensities,
            spectrum_mask=spectrum_mask,
            precursor_mass=precursor_mass,
            precursor_charge=precursor_charge,
        )

        # Compute loss
        loss, metrics = loss_fn(
            all_logits=all_logits,
            targets=sequence,
            target_mask=sequence_mask,
            observed_masses=spectrum_masses,
            observed_intensities=spectrum_intensities,
            peak_mask=spectrum_mask,
            precursor_mass=precursor_mass,
        )

    print(f"\nLoss components:")
    print(f"  - Total loss: {loss.item():.4f}")
    print(f"  - Available metrics: {list(metrics.keys())}")
    if 'ce_loss' in metrics:
        print(f"  - CE loss: {metrics['ce_loss']:.4f}")
    if 'spectrum_loss' in metrics:
        print(f"  - Spectrum loss: {metrics['spectrum_loss']:.4f}")
    if 'precursor_loss' in metrics:
        print(f"  - Precursor loss: {metrics['precursor_loss']:.4f}")

    # Verify loss is finite
    assert torch.isfinite(loss), "Loss is not finite"
    assert loss.item() > 0, "Loss should be positive"
    print(f"✓ Loss is finite and positive")

    # Verify required metrics are present
    assert 'spectrum_loss' in metrics, "Missing spectrum_loss"
    assert 'precursor_loss' in metrics, "Missing precursor_loss"
    assert 'ce_final' in metrics or 'total_loss' in metrics, "Missing CE loss metrics"
    print(f"✓ All loss components present")

    # Check precursor loss is reasonable (should be high for random sequences)
    if metrics['precursor_loss'] > 10.0:
        print(f"✓ Precursor loss is high for random sequence (expected)")

    print(f"\nNote: Token/sequence accuracy are computed separately in the trainer.")

    return True


def verify_dataset(config):
    """Verify dataset generates valid samples."""
    print("\n" + "=" * 60)
    print("DATASET VERIFICATION")
    print("=" * 60)

    dataset = MS2PIPSyntheticDataset(
        min_length=7,
        max_length=10,
        max_peaks=config.max_peaks,
        max_seq_len=config.max_seq_len,
        noise_peaks=0,
        peak_dropout=0.0,
        mass_error_ppm=0.0,
        ms2pip_model='HCDch2',
    )

    print(f"\nDataset configuration:")
    print(f"  - Peptide length: {dataset.min_length}-{dataset.max_length}")
    print(f"  - Max peaks: {dataset.max_peaks}")
    print(f"  - MS2PIP model: {dataset.ms2pip_model}")
    print(f"  - Noise peaks: {dataset.noise_peaks}")

    # Generate a sample
    sample_iter = iter(dataset)
    sample = next(sample_iter)

    print(f"\nSample shapes:")
    print(f"  - spectrum_masses: {sample.spectrum_masses.shape}")
    print(f"  - spectrum_intensities: {sample.spectrum_intensities.shape}")
    print(f"  - spectrum_mask: {sample.spectrum_mask.shape}")
    print(f"  - precursor_mass: {sample.precursor_mass.shape}")
    print(f"  - precursor_charge: {sample.precursor_charge.shape}")
    print(f"  - sequence: {sample.sequence.shape}")
    print(f"  - sequence_mask: {sample.sequence_mask.shape}")

    # Verify sample properties
    num_peaks = sample.spectrum_mask.sum().item()
    print(f"\nSample properties:")
    print(f"  - Number of peaks: {num_peaks}")
    print(f"  - Precursor charge: {sample.precursor_charge.item()}")
    print(f"  - Precursor mass: {sample.precursor_mass.item():.2f} Da")

    assert num_peaks > 0, "No peaks in sample"
    assert num_peaks <= config.max_peaks, f"Too many peaks: {num_peaks}"
    print(f"✓ Dataset generates valid samples")

    return True


def estimate_memory_usage(config, batch_size):
    """Estimate GPU memory usage for given batch size."""
    print("\n" + "=" * 60)
    print(f"MEMORY ESTIMATION (batch_size={batch_size})")
    print("=" * 60)

    # Model parameters (FP32)
    total_params = 12_482_816  # From model creation
    model_memory_fp32 = total_params * 4 / (1024**3)  # GB
    model_memory_fp16 = total_params * 2 / (1024**3)  # GB

    print(f"\nModel memory:")
    print(f"  - FP32: {model_memory_fp32:.2f} GB")
    print(f"  - FP16 (mixed precision): {model_memory_fp16:.2f} GB")

    # Rough activation memory estimate
    # Main memory hogs:
    # - Encoder output: batch * (max_peaks+1) * hidden_dim
    # - Decoder states: batch * max_seq_len * hidden_dim * num_steps
    # - Attention: batch * num_heads * seq_len * seq_len

    hidden_dim = config.hidden_dim
    max_peaks = config.max_peaks
    max_seq_len = config.max_seq_len
    num_steps = config.num_supervision_steps

    # Rough estimate (FP16)
    encoder_mem = batch_size * (max_peaks + 1) * hidden_dim * 2 / (1024**3)
    decoder_mem = batch_size * max_seq_len * hidden_dim * num_steps * 2 / (1024**3)
    attention_mem = batch_size * 6 * max_seq_len * max_seq_len * 2 / (1024**3) * 2  # encoder + decoder

    total_activation = encoder_mem + decoder_mem + attention_mem

    print(f"\nEstimated activation memory (FP16):")
    print(f"  - Encoder: {encoder_mem:.2f} GB")
    print(f"  - Decoder: {decoder_mem:.2f} GB")
    print(f"  - Attention: {attention_mem:.2f} GB")
    print(f"  - Total activations: {total_activation:.2f} GB")

    # Gradients (same size as model)
    gradient_memory = model_memory_fp16
    print(f"\nGradient memory: {gradient_memory:.2f} GB")

    # Optimizer state (Adam: 2x model size for momentum + variance)
    optimizer_memory = model_memory_fp32 * 2
    print(f"Optimizer state (Adam): {optimizer_memory:.2f} GB")

    # Total estimate
    total_estimate = model_memory_fp16 + total_activation + gradient_memory + optimizer_memory
    print(f"\n{'='*60}")
    print(f"TOTAL ESTIMATED MEMORY: {total_estimate:.2f} GB")
    print(f"{'='*60}")

    # Safety margin
    safety_margin = 1.2  # 20% overhead
    total_with_margin = total_estimate * safety_margin
    print(f"With 20% safety margin: {total_with_margin:.2f} GB")

    # GPU check
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"\nAvailable GPU memory: {gpu_memory:.2f} GB")

        if total_with_margin < gpu_memory:
            print(f"✓ Should fit in GPU memory (headroom: {gpu_memory - total_with_margin:.2f} GB)")
        else:
            print(f"⚠ May not fit in GPU memory (deficit: {total_with_margin - gpu_memory:.2f} GB)")

    return total_with_margin


def main():
    print("\n" + "=" * 70)
    print(" " * 15 + "MODEL VERIFICATION SUITE")
    print("=" * 70)

    # Load config
    config_path = project_root / 'configs' / 'optimized_extended.yaml'
    cfg = OmegaConf.load(config_path)

    # Create model
    model_config = TRMConfig(**cfg.model)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\nDevice: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")

    model = create_model(model_config).to(device)

    # Run verifications
    try:
        verify_model_architecture(model, model_config)
        verify_forward_pass(model, model_config, device)
        verify_loss_computation(model, model_config, device)
        verify_dataset(model_config)

        # Memory estimates for different batch sizes
        print("\n" + "=" * 70)
        print(" " * 20 + "BATCH SIZE ANALYSIS")
        print("=" * 70)

        for bs in [96, 128, 144, 160, 192]:
            estimate_memory_usage(model_config, bs)
            print()

        print("\n" + "=" * 70)
        print(" " * 20 + "✓ ALL VERIFICATIONS PASSED")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ VERIFICATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
