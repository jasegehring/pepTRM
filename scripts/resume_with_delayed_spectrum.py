"""
Resume training from step 15K with DELAYED spectrum loss curriculum.

This script:
1. Loads checkpoint from step 15000
2. Uses DELAYED_SPECTRUM_CURRICULUM (no spectrum until 30K steps)
3. Resumes wandb run ID: nbs1e6hk
4. Gives model time to reach 65% accuracy before introducing spectrum loss
"""

import sys
from pathlib import Path
import argparse
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.model.trm import TRMConfig, create_model
from src.data.ms2pip_dataset import MS2PIPSyntheticDataset
from src.training.trainer_optimized import OptimizedTrainer, TrainingConfig
from src.training.curriculum_delayed_spectrum import CurriculumScheduler, DELAYED_SPECTRUM_CURRICULUM


def main():
    parser = argparse.ArgumentParser(description="Resume training with delayed spectrum curriculum.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints_optimized/checkpoint_step_15000.pt",
        help="Path to checkpoint to resume from"
    )
    # No wandb run ID argument - we want to create a NEW run for comparison
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
    print("RESUMING TRAINING WITH DELAYED SPECTRUM CURRICULUM")
    print("=" * 70)
    print(f"\nCheckpoint: {args.checkpoint}")
    print(f"W&B: Will create NEW run for comparison (not resuming nbs1e6hk)")
    print("\n" + "=" * 70)
    print("CURRICULUM CHANGES")
    print("=" * 70)
    print("\nORIGINAL CURRICULUM:")
    print("  Step 15K-22.5K: Introduce spectrum loss (weight 0.08)")
    print("  Step 22.5K-30K: Ramp spectrum loss (weight 0.12)")
    print("\nNEW DELAYED CURRICULUM:")
    print("  Step 15K-22.5K: PURE CE (weight 0.0) ← Still building foundation")
    print("  Step 22.5K-30K: PURE CE (weight 0.0) ← Give time to reach 65% acc")
    print("  Step 30K-37.5K: Introduce spectrum (weight 0.06) ← Delayed!")
    print("  Step 37.5K-45K: Ramp spectrum (weight 0.10)")
    print("\nRATIONALE:")
    print("  - Current accuracy at step 15K: 48% (below expected 55%)")
    print("  - Spectrum needs 60-65% accuracy for good signal (>10% coverage)")
    print("  - Delaying by 15K steps gives model time to mature")
    print("=" * 70)

    # Create model
    model_config = TRMConfig(**cfg.model)
    model = create_model(model_config)

    print(f"\nModel created:")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

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
    print(f"  MS2PIP model: {cfg.data.ms2pip_model}")

    # Create DELAYED curriculum scheduler
    curriculum = CurriculumScheduler(DELAYED_SPECTRUM_CURRICULUM, train_dataset)

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
    else:
        training_config.device = 'cpu'
        training_config.use_amp = False
        training_config.use_compile = False
        print(f"\n! No GPU available - using CPU")

    # DON'T set WANDB_RUN_ID - we want a NEW run for comparison
    # This allows comparing old run (nbs1e6hk with bug) vs new run (with fix)

    # Set W&B grouping for checkpoint branching experiments (BEST PRACTICE)
    os.environ['WANDB_RUN_GROUP'] = 'checkpoint-15k-experiments'
    os.environ['WANDB_JOB_TYPE'] = 'delayed-curriculum-fixed-spectrum'

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

    # Update wandb config and add descriptive tags
    if trainer.use_wandb:
        import wandb

        # Add tags to help find this run
        wandb.run.tags = wandb.run.tags + (
            "spectrum-loss-FIXED",
            "delayed-curriculum",
            "comparison-run",
            "from-step-15k"
        )

        # Add config to explain what this run is
        wandb.config.update({
            # What changed
            "spectrum_loss_bug_fixed": True,
            "bug_fix": "PAD tokens now masked in theoretical peak computation",
            "curriculum_modified": True,
            "curriculum_change": "Delayed spectrum loss from 15K→30K steps",

            # Why we changed it
            "reason": "Model at 48% accuracy needs more time to reach 65% before spectrum loss",
            "expected_coverage_at_30k": "15-20% (vs 2% at 15K)",

            # Comparison info
            "comparison_to_run": "nbs1e6hk",
            "comparison_note": "nbs1e6hk had buggy spectrum loss (PAD tokens included), this run has fix",
            "resumed_from_checkpoint": "checkpoint_step_15000.pt",
            "resumed_from_step": 15000,
        }, allow_val_change=True)

        # Set a descriptive run name
        wandb.run.name = f"fixed-spectrum-delayed-{wandb.run.id}"
        print(f"\n📊 New W&B run created: {wandb.run.name}")
        print(f"   Group: checkpoint-15k-experiments")
        print(f"   Job type: delayed-curriculum-fixed-spectrum")
        print(f"   Compare to old run: nbs1e6hk (gallant-water-62)")
        print(f"\n   In W&B: Filter by group='checkpoint-15k-experiments' to see related runs")

    print(f"\nTraining configuration:")
    print(f"  Batch size: {training_config.batch_size}")
    print(f"  Device: {training_config.device}")
    print(f"  Max steps: {training_config.max_steps}")

    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    # Load curriculum state but DON'T restore it (use fresh curriculum)
    trainer.load_checkpoint(args.checkpoint, load_curriculum=False)

    # Manually set curriculum to step 15000
    curriculum.step(15000)

    current_step = trainer.global_step
    stage = curriculum.current_stage

    print(f"\nResumed training state:")
    print(f"  Current step: {current_step}")
    print(f"  Curriculum stage: {stage.name if stage else 'None'}")
    if stage:
        print(f"  Spectrum loss weight: {stage.spectrum_loss_weight} (should be 0.0)")
        print(f"  Precursor loss weight: {stage.precursor_loss_weight}")
        print(f"  Peptide length range: {stage.min_length}-{stage.max_length}")

    # Verify we're using the right curriculum
    assert stage.spectrum_loss_weight == 0.0, \
        f"ERROR: Expected spectrum weight 0.0 at step {current_step}, got {stage.spectrum_loss_weight}"

    print("\n" + "=" * 70)
    print("TRAINING PLAN")
    print("=" * 70)
    print("\nSteps 15K-30K (Next 15K steps):")
    print("  Goal: Improve token accuracy from 48% → 65%+")
    print("  Strategy: Pure CE loss on clean data")
    print("  Peptide length: Gradually increase 7-10 → 7-12")
    print("  Spectrum weight: 0.0 (no spectrum loss)")
    print("\nSteps 30K-45K:")
    print("  Goal: Introduce spectrum loss with strong gradient signal")
    print("  Expected accuracy: 65-70% (15-20% spectrum coverage)")
    print("  Spectrum weight: 0.06 → 0.10 (gentle ramp)")
    print("\n" + "=" * 70)
    print("MONITORING")
    print("=" * 70)
    print("\nWatch in W&B (run: nbs1e6hk):")
    print("  - train/token_accuracy → Should improve from 48% to 60%+ by step 25K")
    print("  - train/ce_final → Should continue decreasing")
    print("  - train/spectrum_loss → Should stay at 0.0 (not computed)")
    print("\nGood signs by step 30K:")
    print("  ✓ Token accuracy reaches 65%+")
    print("  ✓ CE loss continues to decrease steadily")
    print("  ✓ Ready for spectrum loss introduction")
    print("\n" + "=" * 70)

    input("\nPress Enter to start training (or Ctrl+C to cancel)...")

    # Train
    trainer.train()


if __name__ == '__main__':
    main()
