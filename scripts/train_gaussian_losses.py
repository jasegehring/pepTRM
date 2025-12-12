"""
Training script using Gaussian Spectral Rendering losses.

Tests the new loss formulation with:
- Gaussian rendering for spectrum matching (smooth gradients)
- Scaled L1 for precursor mass (constant gradients)
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.model.trm import TRMConfig, create_model
from src.data.ms2pip_dataset import MS2PIPSyntheticDataset
from src.training.trainer_optimized import OptimizedTrainer, TrainingConfig
from src.training.curriculum_extended import CurriculumScheduler, EXTENDED_CURRICULUM

# Import the NEW Gaussian losses
from src.training.gaussian_spectral_rendering_losses import CombinedLoss


def main():
    # Clear CUDA cache
    if torch.cuda.is_available():
        print("Clearing CUDA cache...")
        torch.cuda.empty_cache()
        print("✓ CUDA cache cleared.")

    # Load base config
    config_path = project_root / 'configs' / 'optimized_extended.yaml'
    cfg = OmegaConf.load(config_path)

    print("=" * 80)
    print("Training with Gaussian Spectral Rendering Losses")
    print("=" * 80)
    print("\nNew loss components:")
    print("  - Gaussian rendering for spectrum matching")
    print("  - Scaled L1 for precursor mass constraint")
    print("  - Tuned for gradient stability\n")

    # Create model
    model_cfg = TRMConfig(
        vocab_size=cfg.model.vocab_size,
        d_model=cfg.model.d_model,
        nhead=cfg.model.nhead,
        num_layers=cfg.model.num_layers,
        dim_feedforward=cfg.model.dim_feedforward,
        num_iterations=cfg.model.num_iterations,
        dropout=cfg.model.dropout,
    )
    model = create_model(model_cfg)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create datasets
    train_dataset = MS2PIPSyntheticDataset(
        min_length=cfg.data.min_length,
        max_length=cfg.data.max_length,
        max_peaks=cfg.data.max_peaks,
        max_seq_len=cfg.data.max_seq_len,
        noise_peaks=0,  # Curriculum will adjust
        ms2pip_model=cfg.data.ms2pip_model,
    )

    val_dataset_easy = MS2PIPSyntheticDataset(
        min_length=7,
        max_length=10,
        max_peaks=cfg.data.max_peaks,
        max_seq_len=cfg.data.max_seq_len,
        noise_peaks=0,
        ms2pip_model=cfg.data.ms2pip_model,
    )

    val_dataset_hard = MS2PIPSyntheticDataset(
        min_length=15,
        max_length=20,
        max_peaks=cfg.data.max_peaks,
        max_seq_len=cfg.data.max_seq_len,
        noise_peaks=10,
        peak_dropout=0.15,
        mass_error_ppm=15.0,
        ms2pip_model=cfg.data.ms2pip_model,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        pin_memory=True,
        num_workers=4,
    )

    val_loader_easy = DataLoader(val_dataset_easy, batch_size=cfg.training.batch_size, pin_memory=True)
    val_loader_hard = DataLoader(val_dataset_hard, batch_size=cfg.training.batch_size, pin_memory=True)

    # Create curriculum
    curriculum = CurriculumScheduler(
        stages=EXTENDED_CURRICULUM,
        dataset=train_dataset,
    )

    print(f"\nCurriculum: {len(curriculum.stages)} stages, {curriculum.total_steps:,} total steps")

    # Create training config
    training_cfg = TrainingConfig(
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        batch_size=cfg.training.batch_size,
        max_steps=cfg.training.max_steps,
        use_amp=cfg.training.use_amp,
        use_compile=cfg.training.get('use_compile', False),  # Disable compilation for first test
        use_curriculum=True,
        log_interval=100,
        eval_interval=1000,
        save_interval=5000,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )

    print(f"\nTraining configuration:")
    print(f"  Batch size: {training_cfg.batch_size}")
    print(f"  Learning rate: {training_cfg.learning_rate}")
    print(f"  Max steps: {training_cfg.max_steps:,}")
    print(f"  Mixed precision: {training_cfg.use_amp}")
    print(f"  Device: {training_cfg.device}")

    # Create Gaussian loss (NEW!)
    print("\n" + "=" * 80)
    print("Creating Gaussian Spectral Rendering Loss")
    print("=" * 80)

    loss_fn = CombinedLoss(
        ce_weight=1.0,
        spectrum_weight=0.0,  # Will be controlled by curriculum
        precursor_weight=0.0,  # Will be controlled by curriculum
        iteration_weights='linear',
        label_smoothing=0.1,
        ion_type_names=None,
        ms2pip_model=cfg.data.ms2pip_model,  # Auto-select ion types
    )

    print(f"\nLoss components:")
    print(f"  - Deep supervision: ✓")
    print(f"  - Gaussian spectrum rendering: bin_size=0.1 Da, sigma=0.05 Da")
    print(f"  - Scaled L1 precursor: scale_factor=0.004")

    # Create trainer
    trainer = OptimizedTrainer(
        model=model,
        train_loader=train_loader,
        config=training_cfg,
        val_loader_easy=val_loader_easy,
        val_loader_hard=val_loader_hard,
        curriculum=curriculum,
        use_wandb=False,  # Disable wandb for testing
    )

    # Override the loss function to use Gaussian losses
    trainer.criterion = loss_fn

    print("\n" + "=" * 80)
    print("Starting training with Gaussian losses...")
    print("=" * 80)
    print("\nMonitoring for:")
    print("  ✓ Stable gradients (no NaN)")
    print("  ✓ Smooth loss curves")
    print("  ✓ Auxiliary losses provide signal without dominating")
    print()

    # Train
    try:
        trainer.train()
        print("\n" + "=" * 80)
        print("Training completed successfully! ✓")
        print("=" * 80)
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"Training failed with error: {e}")
        print("=" * 80)
        raise


if __name__ == "__main__":
    main()
