"""
Resume training from checkpoint with FIXED spectrum loss.

This script resumes from step 15000 checkpoint with the spectrum loss bug fix applied.
It will monitor both token accuracy and spectrum coverage to verify the fix is working.
"""

import sys
from pathlib import Path
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.model.trm import TRMConfig, create_model
from src.data.ms2pip_dataset import MS2PIPSyntheticDataset
from src.training.trainer_optimized import OptimizedTrainer, TrainingConfig
from src.training.curriculum import CurriculumScheduler, DEFAULT_CURRICULUM


def main():
    parser = argparse.ArgumentParser(description="Resume training with fixed spectrum loss.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints_optimized/checkpoint_step_15000.pt",
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default="gallant-water-62-fixed",
        help="W&B run name (will add -fixed suffix automatically)"
    )
    args = parser.parse_args()

    # Clear CUDA cache
    if torch.cuda.is_available():
        print("Clearing CUDA cache...")
        torch.cuda.empty_cache()
        print("✓ CUDA cache cleared.")

    # Load config
    config_path = project_root / 'configs' / 'optimized_extended.yaml'
    cfg = OmegaConf.load(config_path)

    print("=" * 70)
    print("RESUMING TRAINING WITH FIXED SPECTRUM LOSS")
    print("=" * 70)
    print(f"\nCheckpoint: {args.checkpoint}")
    print("\nFIX APPLIED:")
    print("  ✓ Spectrum loss now respects sequence_mask")
    print("  ✓ PAD tokens no longer contribute to predicted peaks")
    print("  ✓ Predicted peaks in correct mass range")
    print("\nEXPECTED BEHAVIOR:")
    print("  - Spectrum loss should start around 0.95-0.99 (low coverage)")
    print("  - As token accuracy improves, spectrum loss should DECREASE")
    print("  - Watch for spectrum_loss dropping below 0.90 (10% coverage)")
    print("  - Token accuracy may initially dip but should recover")
    print("=" * 70)

    # Create model
    model_config = TRMConfig(**cfg.model)
    model = create_model(model_config)

    print(f"\nModel created:")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Hidden dim: {model_config.hidden_dim}")
    print(f"  Supervision steps: {model_config.num_supervision_steps}")
    print(f"  Latent steps: {model_config.num_latent_steps}")

    # Create datasets (same as original training)
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
        ms2pip_model=cfg.data.ms2pip_model,
    )

    val_dataset_hard = MS2PIPSyntheticDataset(
        min_length=12,
        max_length=18,
        noise_peaks=8,
        peak_dropout=0.15,
        mass_error_ppm=15.0,
        ms2pip_model=cfg.data.ms2pip_model,
    )

    print(f"\nDataset created (using MS2PIP):")
    print(f"  Train peptide length: Managed by curriculum")
    print(f"  Val (Easy) length: 7-10 (clean)")
    print(f"  Val (Hard) length: 12-18 (realistic noise)")
    print(f"  MS2PIP model: {cfg.data.ms2pip_model}")

    # Create curriculum scheduler
    curriculum = CurriculumScheduler(DEFAULT_CURRICULUM, train_dataset)

    # Custom collate function
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

    # Create DataLoaders
    training_config = TrainingConfig(**cfg.training)
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader_easy = DataLoader(
        val_dataset_easy,
        batch_size=64,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader_hard = DataLoader(
        val_dataset_hard,
        batch_size=64,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True,
    )

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
        training_config.use_amp = False
        training_config.use_compile = False
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
        train_loader=train_loader,
        config=training_config,
        val_loader_easy=val_loader_easy,
        val_loader_hard=val_loader_hard,
        curriculum=curriculum,
        use_wandb=True,
    )

    # Update wandb config to note this is a continuation
    if trainer.use_wandb:
        import wandb
        wandb.config.update({
            "resumed_from_checkpoint": args.checkpoint,
            "previous_run": "gallant-water-62",
            "spectrum_loss_fix_applied": True,
            "note": "Resumed training with PAD token masking fix for spectrum loss"
        }, allow_val_change=True)

    print(f"\nTraining configuration:")
    print(f"  Batch size: {training_config.batch_size}")
    print(f"  Mixed precision: {training_config.use_amp}")
    print(f"  Model compilation: {training_config.use_compile}")
    print(f"  Learning rate: {training_config.learning_rate}")
    print(f"  Max steps: {training_config.max_steps}")
    print(f"  Device: {training_config.device}")

    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    trainer.load_checkpoint(args.checkpoint, load_curriculum=True)

    current_step = trainer.global_step
    stage = curriculum.current_stage
    print(f"\nResumed training state:")
    print(f"  Current step: {current_step}")
    print(f"  Curriculum stage: {stage.name if stage else 'None'}")
    if stage:
        print(f"  Spectrum loss weight: {stage.spectrum_loss_weight}")
        print(f"  Precursor loss weight: {stage.precursor_loss_weight}")
        print(f"  Clean data ratio: {stage.clean_data_ratio:.0%}")

    print("\n" + "=" * 70)
    print("MONITORING TIPS:")
    print("=" * 70)
    print("\nWatch these metrics in W&B:")
    print("  1. train/spectrum_loss - Should DECREASE from ~0.99")
    print("  2. train/token_accuracy - May dip slightly then recover")
    print("  3. train/total_loss - Combination of CE + spectrum")
    print("\nGood signs:")
    print("  ✓ Spectrum loss drops below 0.90 within 5K steps")
    print("  ✓ Token accuracy recovers to >70% within 5K steps")
    print("  ✓ Total loss continues to decrease")
    print("\nBad signs:")
    print("  ✗ Spectrum loss stuck at 0.96-0.99 (fix didn't work)")
    print("  ✗ Token accuracy drops and doesn't recover")
    print("  ✗ Total loss increases or oscillates wildly")
    print("\nIf you see bad signs, stop training and let me know!")
    print("=" * 70)

    input("\nPress Enter to start training (or Ctrl+C to cancel)...")

    # Train
    trainer.train()


if __name__ == '__main__':
    main()
