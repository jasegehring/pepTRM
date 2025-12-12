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
    parser = argparse.ArgumentParser(description="Optimized training script for peptide TRM.")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to a checkpoint to resume training from.")
    args = parser.parse_args()

    # Clear CUDA cache
    if torch.cuda.is_available():
        print("Clearing CUDA cache...")
        torch.cuda.empty_cache()
        print("✓ CUDA cache cleared.")

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
    # Training dataset uses MS2PIP realistic spectra
    # Curriculum will dynamically adjust its difficulty.
    train_dataset = MS2PIPSyntheticDataset(
        max_peaks=model_config.max_peaks,
        max_seq_len=model_config.max_seq_len,
        ms2pip_model=cfg.data.ms2pip_model,
        charge_distribution=dict(cfg.data.charge_distribution),
    )

    # Easy validation: clean data, short peptides
    val_dataset_easy = MS2PIPSyntheticDataset(
        min_length=7,
        max_length=10,
        noise_peaks=0,
        peak_dropout=0.0,
        mass_error_ppm=0.0,
        ms2pip_model=cfg.data.ms2pip_model,
    )

    # Hard validation: realistic noise, longer peptides
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
        batch_size=64,  # Smaller batch size for validation
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
        train_loader=train_loader,
        config=training_config,
        val_loader_easy=val_loader_easy,
        val_loader_hard=val_loader_hard,
        curriculum=curriculum,
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

    if args.resume_from:
        trainer.load_checkpoint(args.resume_from)

    # Train
    trainer.train()


if __name__ == '__main__':
    main()
