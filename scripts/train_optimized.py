"""
Optimized training entry point for high-performance GPUs (RTX 4090, A100).

Features:
- Mixed precision training (AMP) - 2-3x speedup
- torch.compile() - 1.2-1.5x speedup
- Extended curriculum (100K steps, 10 stages)
- Larger batch size (192 vs 64)

Expected training time: ~2.5 hours for 100K steps (vs ~7 hours without optimizations)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from omegaconf import OmegaConf

from src.model.trm import TRMConfig, create_model
from src.data.dataset import SyntheticPeptideDataset
from src.training.trainer_optimized import OptimizedTrainer, TrainingConfig


def main():
    # Load config
    config_path = project_root / 'configs' / 'optimized_extended.yaml'
    cfg = OmegaConf.load(config_path)

    print("=" * 60)
    print("Recursive Peptide Model - Optimized Training")
    print("=" * 60)

    # Create model
    model_config = TRMConfig(**cfg.model)
    model = create_model(model_config)

    print(f"\nModel created:")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Hidden dim: {model_config.hidden_dim}")
    print(f"  Supervision steps: {model_config.num_supervision_steps}")
    print(f"  Latent steps: {model_config.num_latent_steps}")

    # Create datasets
    train_dataset = SyntheticPeptideDataset(
        min_length=cfg.data.min_length,
        max_length=cfg.data.max_length,
        max_peaks=model_config.max_peaks,
        max_seq_len=model_config.max_seq_len,
        ion_types=cfg.data.ion_types,
        include_neutral_losses=cfg.data.include_neutral_losses,
        noise_peaks=cfg.data.noise_peaks,
        peak_dropout=cfg.data.peak_dropout,
        mass_error_ppm=cfg.data.mass_error_ppm,
        intensity_variation=cfg.data.intensity_variation,
        charge_distribution=dict(cfg.data.charge_distribution),
    )

    val_dataset = SyntheticPeptideDataset(
        min_length=cfg.data.min_length,
        max_length=cfg.data.max_length,
        max_peaks=model_config.max_peaks,
        max_seq_len=model_config.max_seq_len,
        ion_types=cfg.data.ion_types,
    )

    print(f"\nDataset created:")
    print(f"  Peptide length: {cfg.data.min_length}-{cfg.data.max_length}")
    print(f"  Ion types: {cfg.data.ion_types}")
    print(f"  Initial difficulty: Clean synthetic (curriculum will adjust)")

    # Create optimized trainer
    training_config = TrainingConfig(**cfg.training)

    # Auto-detect device
    if torch.cuda.is_available():
        training_config.device = 'cuda'
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\n✓ CUDA available - using GPU")
        print(f"  GPU: {gpu_name}")
        print(f"  VRAM: {gpu_memory:.1f} GB")
    elif torch.backends.mps.is_available():
        training_config.device = 'mps'
        training_config.use_amp = False  # MPS doesn't support AMP yet
        training_config.use_compile = False  # MPS doesn't support compile yet
        print(f"\n✓ MPS available - using Apple Silicon GPU")
        print(f"  Note: AMP and compile disabled (not supported on MPS)")
    else:
        training_config.device = 'cpu'
        training_config.use_amp = False
        training_config.use_compile = False
        print(f"\n! No GPU available - using CPU")

    # Create trainer
    trainer = OptimizedTrainer(
        model=model,
        train_dataset=train_dataset,
        config=training_config,
        val_dataset=val_dataset,
        use_wandb=True,
    )

    print(f"\nOptimized training configuration:")
    print(f"  Batch size: {training_config.batch_size}")
    print(f"  Mixed precision: {training_config.use_amp}")
    print(f"  Model compilation: {training_config.use_compile}")
    print(f"  Learning rate: {training_config.learning_rate}")
    print(f"  Max steps: {training_config.max_steps}")
    print(f"  Extended curriculum: {training_config.use_curriculum}")
    print(f"  Device: {training_config.device}")

    # Train
    trainer.train()


if __name__ == '__main__':
    main()
