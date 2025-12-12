"""
Resume from checkpoint and test different auxiliary loss settings.

Use case: CE training works well, now you want to experiment with
different spectrum/precursor loss schedules without retraining from scratch.
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
from src.training.curriculum_extended import CurriculumScheduler, EXTENDED_CURRICULUM
from src.training.curriculum_phased import PHASED_CURRICULUM


def main():
    parser = argparse.ArgumentParser(
        description="Resume training from checkpoint and test different curricula"
    )
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to checkpoint (e.g., checkpoints_optimized/checkpoint_step_15000.pt)"
    )
    parser.add_argument(
        "--curriculum",
        type=str,
        choices=['early', 'phased', 'current'],
        default='current',
        help="Which curriculum to use: early (10k/20k), phased (15k/30k), or current (continue from checkpoint)"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Override max training steps (default: continue from config)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for new checkpoints (default: checkpoints_resume_TIMESTAMP)"
    )
    args = parser.parse_args()

    # Check checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return

    print("=" * 80)
    print("Resume Training with Different Curriculum")
    print("=" * 80)
    print(f"\nCheckpoint: {checkpoint_path}")
    print(f"New curriculum: {args.curriculum}")
    print()

    # Load checkpoint to get step number
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    resume_step = checkpoint['step']
    print(f"Resuming from step: {resume_step:,}")

    # Load config
    config_path = project_root / 'configs' / 'optimized_extended.yaml'
    cfg = OmegaConf.load(config_path)

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

    # Create datasets
    train_dataset = MS2PIPSyntheticDataset(
        min_length=cfg.data.min_length,
        max_length=cfg.data.max_length,
        max_peaks=cfg.data.max_peaks,
        max_seq_len=cfg.data.max_seq_len,
        noise_peaks=0,
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

    # Select curriculum based on argument
    if args.curriculum == 'early':
        print("\nUsing EARLY curriculum (spectrum @ 10k, precursor @ 20k)")
        curriculum_stages = EXTENDED_CURRICULUM
    elif args.curriculum == 'phased':
        print("\nUsing PHASED curriculum (spectrum @ 15k, precursor @ 30k)")
        curriculum_stages = PHASED_CURRICULUM
    else:
        print("\nUsing CURRENT curriculum from checkpoint")
        curriculum_stages = EXTENDED_CURRICULUM  # Will be overridden by checkpoint

    curriculum = CurriculumScheduler(
        stages=curriculum_stages,
        dataset=train_dataset,
    )

    # Create output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"checkpoints_resume_{args.curriculum}_{timestamp}"

    # Create training config
    max_steps = args.max_steps if args.max_steps is not None else cfg.training.max_steps

    training_cfg = TrainingConfig(
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        batch_size=cfg.training.batch_size,
        max_steps=max_steps,
        use_amp=cfg.training.use_amp,
        use_compile=False,  # Disabled for stability
        use_curriculum=True,
        log_interval=100,
        eval_interval=1000,
        save_interval=5000,
        checkpoint_dir=output_dir,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )

    print(f"\nOutput directory: {output_dir}")
    print(f"Training until step: {max_steps:,}")
    print()

    # Create trainer
    trainer = OptimizedTrainer(
        model=model,
        train_loader=train_loader,
        config=training_cfg,
        val_loader_easy=val_loader_easy,
        val_loader_hard=val_loader_hard,
        curriculum=curriculum,
        use_wandb=False,
    )

    # Load checkpoint
    # load_curriculum=False means: use the NEW curriculum starting from current step
    load_curriculum = (args.curriculum == 'current')
    trainer.load_checkpoint(str(checkpoint_path), load_curriculum=load_curriculum)

    if not load_curriculum:
        print(f"\n⚠️  Curriculum RESET to match step {resume_step}")
        print("    This will apply the new curriculum's settings from this step forward.")

    # Show what auxiliary losses will be active
    current_stage = curriculum.current_stage
    if current_stage:
        print(f"\nCurrent curriculum stage: {current_stage.name}")
        print(f"  Spectrum loss weight: {current_stage.spectrum_loss_weight}")
        print(f"  Precursor loss weight: {current_stage.precursor_loss_weight}")
        print()

        # Calculate when next losses will be introduced
        if current_stage.spectrum_loss_weight == 0:
            for i, stage in enumerate(curriculum.stages):
                if stage.spectrum_loss_weight > 0:
                    intro_step = sum(s.steps for s in curriculum.stages[:i])
                    print(f"  → Spectrum loss will be introduced at step {intro_step:,}")
                    break

        if current_stage.precursor_loss_weight == 0:
            for i, stage in enumerate(curriculum.stages):
                if stage.precursor_loss_weight > 0:
                    intro_step = sum(s.steps for s in curriculum.stages[:i])
                    print(f"  → Precursor loss will be introduced at step {intro_step:,}")
                    break

    print("\n" + "=" * 80)
    print("Starting resumed training...")
    print("=" * 80)
    print()

    # Train
    try:
        trainer.train()
        print("\n" + "=" * 80)
        print("Resumed training completed! ✓")
        print("=" * 80)
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"Training failed: {e}")
        print("=" * 80)
        raise


if __name__ == "__main__":
    main()
