"""
Training script using PHASED curriculum (delayed auxiliary loss introduction).

Phased Introduction Schedule:
- Steps 0-15k:   Pure CE (build foundation to ~55% accuracy)
- Steps 15k-30k: Add spectrum loss (gradient strength 0.25+)
- Steps 30k+:    Add precursor loss (gradient strength 0.59+)

Compare against early introduction to see which gives better results.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.model.trm import TRMConfig, create_model
from src.data.ms2pip_dataset import MS2PIPSyntheticDataset
from src.training.trainer_optimized import OptimizedTrainer, TrainingConfig
from src.training.curriculum_phased import CurriculumScheduler, PHASED_CURRICULUM


def main():
    parser = argparse.ArgumentParser(description="Training with phased curriculum")
    parser.add_argument("--resume_from", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--max_steps", type=int, default=None, help="Override max steps")
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    args = parser.parse_args()

    # Clear CUDA cache
    if torch.cuda.is_available():
        print("Clearing CUDA cache...")
        torch.cuda.empty_cache()
        print("✓ CUDA cache cleared.")

    # Load config
    config_path = project_root / 'configs' / 'optimized_extended.yaml'
    cfg = OmegaConf.load(config_path)

    print("=" * 80)
    print("Training with PHASED Curriculum (Delayed Auxiliary Loss Introduction)")
    print("=" * 80)
    print("\nCurriculum schedule:")
    print("  Phase 1 (0-15k):   Pure CE, build foundation")
    print("  Phase 2 (15k-30k): Add spectrum loss (55% acc, gradient=0.25)")
    print("  Phase 3 (30k+):    Add precursor loss (70% acc, gradient=0.59)")
    print()
    print("Hypothesis: Stronger gradients when introduced → better training")
    print()

    # Create model
    model_config = TRMConfig(**cfg.model)
    model = create_model(model_config)

    print(f"\nModel created:")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Hidden dim: {model_config.hidden_dim}")
    print(f"  Supervision steps: {model_config.num_supervision_steps}")
    print(f"  Latent steps: {model_config.num_latent_steps}")

    # Create datasets
    train_dataset = MS2PIPSyntheticDataset(
        max_peaks=model_config.max_peaks,
        max_seq_len=model_config.max_seq_len,
        ms2pip_model=cfg.data.ms2pip_model,
        charge_distribution=dict(cfg.data.charge_distribution),
    )

    val_dataset_easy = MS2PIPSyntheticDataset(
        min_length=7,
        max_length=10,
        noise_peaks=0,
        peak_dropout=0.0,
        mass_error_ppm=0.0,
        max_peaks=model_config.max_peaks,
        max_seq_len=model_config.max_seq_len,
        ms2pip_model=cfg.data.ms2pip_model,
    )

    val_dataset_hard = MS2PIPSyntheticDataset(
        min_length=12,
        max_length=18,
        noise_peaks=10,
        peak_dropout=0.15,
        mass_error_ppm=15.0,
        max_peaks=model_config.max_peaks,
        max_seq_len=model_config.max_seq_len,
        ms2pip_model=cfg.data.ms2pip_model,
    )

    # Custom collate function for MS2PIPSample dataclasses
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

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=4,
    )

    val_loader_easy = DataLoader(
        val_dataset_easy,
        batch_size=cfg.training.batch_size,
        collate_fn=collate_fn,
        pin_memory=True
    )
    val_loader_hard = DataLoader(
        val_dataset_hard,
        batch_size=cfg.training.batch_size,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # Create PHASED curriculum
    curriculum = CurriculumScheduler(
        stages=PHASED_CURRICULUM,
        dataset=train_dataset,
    )

    print(f"Curriculum: {len(curriculum.stages)} stages, {curriculum.total_steps:,} total steps")
    print("\nKey milestones:")
    print(f"  Step 15k:  Spectrum loss introduced (Phase 2 start)")
    print(f"  Step 30k:  Precursor loss introduced (Phase 3 start)")
    print(f"  Step 100k: Full physics-aware training complete")
    print()

    # Create training config
    max_steps = args.max_steps if args.max_steps is not None else cfg.training.max_steps

    training_cfg = TrainingConfig(
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        batch_size=cfg.training.batch_size,
        max_steps=max_steps,
        use_amp=cfg.training.use_amp,
        use_compile=cfg.training.get('use_compile', False),
        use_curriculum=True,
        log_interval=100,
        eval_interval=1000,
        save_interval=5000,
        checkpoint_dir='checkpoints_phased',  # Different directory!
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )

    print(f"Training configuration:")
    print(f"  Batch size: {training_cfg.batch_size}")
    print(f"  Learning rate: {training_cfg.learning_rate}")
    print(f"  Max steps: {training_cfg.max_steps:,}")
    print(f"  Mixed precision: {training_cfg.use_amp}")
    print(f"  Checkpoint dir: {training_cfg.checkpoint_dir}")
    print(f"  Device: {training_cfg.device}")

    # Create trainer
    trainer = OptimizedTrainer(
        model=model,
        train_loader=train_loader,
        config=training_cfg,
        val_loader_easy=val_loader_easy,
        val_loader_hard=val_loader_hard,
        curriculum=curriculum,
        use_wandb=args.use_wandb,
    )

    print("\n" + "=" * 80)
    print("Starting training with PHASED curriculum...")
    print("=" * 80)
    print("\nMonitoring for:")
    print("  ✓ Smooth loss curves (no sudden jumps)")
    print("  ✓ Stable training at step 15k (spectrum intro)")
    print("  ✓ Stable training at step 30k (precursor intro)")
    print("  ✓ Higher final accuracy vs early introduction")
    print()

    # Resume from checkpoint if provided
    if args.resume_from:
        print(f"Resuming from checkpoint: {args.resume_from}")
        trainer.load_checkpoint(args.resume_from)

    # Train
    try:
        trainer.train()
        print("\n" + "=" * 80)
        print("Training completed successfully! ✓")
        print("=" * 80)
        print("\nCheckpoints saved to:", training_cfg.checkpoint_dir)
        print("\nTo compare against early introduction, run:")
        print("  python scripts/compare_curricula.py")
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"Training failed with error: {e}")
        print("=" * 80)
        raise


if __name__ == "__main__":
    main()
