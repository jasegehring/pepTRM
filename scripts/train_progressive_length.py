"""
Progressive Length Training - designed to fix length generalization failure.

Key changes from aggressive_noise training:
1. Extends length ONE at a time (not all at once)
2. 100% clean data until all lengths are learned (0-50K)
3. Noise introduced only AFTER length generalization is achieved (50K-100K)

This should produce a model that:
- Achieves high accuracy on ALL lengths (7-25)
- Can then learn noise robustness without losing length generalization
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

from src.model.trm import TRMConfig, RecursivePeptideModel
from src.data.ms2pip_dataset import MS2PIPSyntheticDataset
from src.data.proteometools_dataset import ProteomeToolsDataset
from src.data.nine_species_dataset import NineSpeciesDataset
from src.training.trainer_optimized import OptimizedTrainer, TrainingConfig
from src.training.curriculum_progressive_length import (
    CurriculumScheduler,
    PROGRESSIVE_LENGTH_CURRICULUM,
    print_curriculum_summary,
)


def create_model(config: TRMConfig) -> RecursivePeptideModel:
    """Create model from config."""
    return RecursivePeptideModel(config)


def main():
    parser = argparse.ArgumentParser(description="Progressive length training for peptide TRM.")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume from.")
    parser.add_argument("--dry_run", action="store_true", help="Print curriculum summary and exit.")
    args = parser.parse_args()

    if args.dry_run:
        print_curriculum_summary()
        return

    # Clear CUDA cache
    if torch.cuda.is_available():
        print("Clearing CUDA cache...")
        torch.cuda.empty_cache()
        print("CUDA cache cleared.")

    # Load config (reuse aggressive_noise config for model/training params)
    config_path = project_root / 'configs' / 'aggressive_noise_test.yaml'
    cfg = OmegaConf.load(config_path)

    print("=" * 70)
    print("PROGRESSIVE LENGTH TRAINING")
    print("=" * 70)
    print("\nStrategy:")
    print("  Phase 1 (0-50K): Learn ALL lengths with 100% clean data")
    print("    - Extend length range progressively: 10 -> 12 -> 14 -> ... -> 25")
    print("    - 5K steps per length extension")
    print("    - Model must generalize to all lengths BEFORE noise")
    print("")
    print("  Phase 2 (50K-100K): Introduce noise gradually")
    print("    - Start with 70% clean, end with 0% clean")
    print("    - Maintain full length range (7-25) throughout")
    print("    - Noise robustness built on solid length foundation")
    print("=" * 70)

    # Print curriculum summary
    print_curriculum_summary()

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

    # Validation datasets - test multiple length ranges
    val_dataset_short = MS2PIPSyntheticDataset(
        min_length=7,
        max_length=10,
        noise_peaks=0,
        peak_dropout=0.0,
        mass_error_ppm=0.0,
        ms2pip_model=cfg.data.ms2pip_model,
    )

    val_dataset_medium = MS2PIPSyntheticDataset(
        min_length=12,
        max_length=16,
        noise_peaks=0,
        peak_dropout=0.0,
        mass_error_ppm=0.0,
        ms2pip_model=cfg.data.ms2pip_model,
    )

    val_dataset_long = MS2PIPSyntheticDataset(
        min_length=18,
        max_length=25,
        noise_peaks=0,
        peak_dropout=0.0,
        mass_error_ppm=0.0,
        ms2pip_model=cfg.data.ms2pip_model,
    )

    # Hard validation (with noise, full range)
    val_dataset_hard = MS2PIPSyntheticDataset(
        min_length=8,
        max_length=25,
        noise_peaks=20,
        peak_dropout=0.25,
        mass_error_ppm=15.0,
        intensity_variation=0.3,
        ms2pip_model=cfg.data.ms2pip_model,
    )
    val_dataset_hard.set_difficulty(clean_data_ratio=0.0)  # Force noisy

    print(f"\nDatasets created:")
    print(f"  Val (Short 7-10):  Clean, tests basic learning")
    print(f"  Val (Medium 12-16): Clean, tests length generalization")
    print(f"  Val (Long 18-25):  Clean, tests full generalization")
    print(f"  Val (Hard):        Noisy 8-25, tests noise robustness")

    # Real data validation datasets
    val_dataset_proteometools = None
    val_dataset_nine_species = None

    pt_dir = project_root / 'data' / 'proteometools'
    if pt_dir.exists():
        try:
            val_dataset_proteometools = ProteomeToolsDataset(
                data_dir=pt_dir,
                split='val',
                max_peaks=model_config.max_peaks,
                max_seq_len=model_config.max_seq_len,
                max_samples=5000,
            )
            print(f"  ProteomeTools: {len(val_dataset_proteometools)} validation samples")
        except Exception as e:
            print(f"  ProteomeTools: Failed to load - {e}")

    ns_dir = project_root / 'data' / 'nine_species'
    if ns_dir.exists():
        try:
            val_dataset_nine_species = NineSpeciesDataset(
                data_dir=ns_dir,
                split='val',
                max_peaks=model_config.max_peaks,
                max_seq_len=model_config.max_seq_len,
                use_balanced=True,
                max_samples=5000,
            )
            print(f"  Nine-Species: {len(val_dataset_nine_species)} validation samples")
        except Exception as e:
            print(f"  Nine-Species: Failed to load - {e}")

    # Create curriculum scheduler
    curriculum = CurriculumScheduler(PROGRESSIVE_LENGTH_CURRICULUM, train_dataset)

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

    # Override checkpoint directory
    training_config.checkpoint_dir = 'checkpoints_progressive_length'

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Use medium-length validation as primary "easy" (to track length generalization)
    val_loader_easy = DataLoader(
        val_dataset_medium,
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

    # Additional loaders for comprehensive tracking
    val_loader_short = DataLoader(
        val_dataset_short,
        batch_size=64,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader_long = DataLoader(
        val_dataset_long,
        batch_size=64,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Real data loaders
    val_loader_proteometools = None
    val_loader_nine_species = None

    if val_dataset_proteometools is not None:
        val_loader_proteometools = DataLoader(
            val_dataset_proteometools,
            batch_size=64,
            num_workers=2,
            collate_fn=collate_fn,
            pin_memory=True,
            shuffle=False,
        )

    if val_dataset_nine_species is not None:
        val_loader_nine_species = DataLoader(
            val_dataset_nine_species,
            batch_size=64,
            num_workers=2,
            collate_fn=collate_fn,
            pin_memory=True,
            shuffle=False,
        )

    # Auto-detect device
    if torch.cuda.is_available():
        training_config.device = 'cuda'
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\nCUDA available - using GPU")
        print(f"  GPU: {gpu_name}")
        print(f"  VRAM: {gpu_memory:.1f} GB")
    elif torch.backends.mps.is_available():
        training_config.device = 'mps'
        training_config.use_amp = False
        training_config.use_compile = False
        print(f"\nMPS available - using Apple Silicon GPU")
    else:
        training_config.device = 'cpu'
        training_config.use_amp = False
        training_config.use_compile = False
        print(f"\nNo GPU available - using CPU")

    # Create trainer
    trainer = OptimizedTrainer(
        model=model,
        train_loader=train_loader,
        config=training_config,
        val_loader_easy=val_loader_easy,  # Medium length (12-16) to track generalization
        val_loader_hard=val_loader_hard,
        val_loader_proteometools=val_loader_proteometools,
        val_loader_nine_species=val_loader_nine_species,
        curriculum=curriculum,
        use_wandb=True,
        wandb_project="pepTRM",
        wandb_run_name="progressive-length",
        real_data_eval_interval=2000,
        real_data_num_batches=20,
    )

    # Store additional loaders for custom validation
    trainer.val_loader_short = val_loader_short
    trainer.val_loader_long = val_loader_long

    print(f"\nTraining configuration:")
    print(f"  Batch size: {training_config.batch_size}")
    print(f"  Mixed precision: {training_config.use_amp}")
    print(f"  Model compilation: {training_config.use_compile}")
    print(f"  Learning rate: {training_config.learning_rate}")
    print(f"  Max steps: {training_config.max_steps:,}")
    print(f"  Checkpoint dir: {training_config.checkpoint_dir}")
    print(f"  Device: {training_config.device}")

    if args.resume_from:
        print(f"\nResuming from: {args.resume_from}")
        trainer.load_checkpoint(args.resume_from)

    # Train
    print("\n" + "=" * 70)
    print("STARTING PROGRESSIVE LENGTH TRAINING")
    print("=" * 70)
    print("\nKey milestones to watch:")
    print("  Step 15K: Should see >90% on length 14 (clean)")
    print("  Step 40K: Should see >90% on length 25 (clean)")
    print("  Step 50K: All lengths mastered, noise introduction begins")
    print("  Step 100K: Full noise robustness achieved")
    print("=" * 70)

    trainer.train()


if __name__ == '__main__':
    main()
