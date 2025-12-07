"""
Training entry point for peptide TRM.
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
from src.training.trainer import Trainer, TrainingConfig


def main():
    # Load config
    config_path = project_root / 'configs' / 'default.yaml'
    cfg = OmegaConf.load(config_path)

    print("=" * 60)
    print("Recursive Peptide Model - Training")
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

    # Validation dataset (same config for now)
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
    print(f"  Difficulty: Clean synthetic (no noise)")

    # Create trainer
    training_config = TrainingConfig(**cfg.training)

    # Auto-detect best available device
    if torch.cuda.is_available():
        training_config.device = 'cuda'
        print(f"\n✓ CUDA available - using GPU")
    elif torch.backends.mps.is_available():
        training_config.device = 'mps'
        print(f"\n✓ MPS available - using Apple Silicon GPU")
    else:
        training_config.device = 'cpu'
        print(f"\n! No GPU available - using CPU (training will be slow)")

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        config=training_config,
        val_dataset=val_dataset,
        use_wandb=True,  # W&B logging enabled
    )

    print(f"\nTraining configuration:")
    print(f"  Batch size: {training_config.batch_size}")
    print(f"  Learning rate: {training_config.learning_rate}")
    print(f"  Max steps: {training_config.max_steps}")
    print(f"  Device: {training_config.device}")
    print(f"  Checkpoint dir: {training_config.checkpoint_dir}")

    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")

    # Train
    trainer.train()

    print("\n" + "=" * 60)
    print("Training finished!")
    print("=" * 60)


if __name__ == '__main__':
    main()
