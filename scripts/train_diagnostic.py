"""
Diagnostic training script to debug model instability.

This script runs the DiagnosticTrainer, which logs extensive information
about the internal state of the model and loss functions at each step.
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
from src.training.trainer_diagnostic import DiagnosticTrainer, TrainingConfig
from src.training.curriculum_extended import CurriculumScheduler, EXTENDED_CURRICULUM


def main():
    parser = argparse.ArgumentParser(description="Diagnostic training script for peptide TRM.")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to a checkpoint to resume training from.")
    parser.add_argument("--num_steps", type=int, default=None, help="Number of steps to run for the diagnostic.")
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
    print("Recursive Peptide Model - DIAGNOSTIC Training")
    print("=" * 60)

    # Override max_steps if provided
    if args.num_steps:
        cfg.training.max_steps = args.num_steps
        print(f"Running for a fixed {args.num_steps} steps.")

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

    print(f"\nDataset created (using MS2PIP).")

    # Create curriculum scheduler
    curriculum = CurriculumScheduler(EXTENDED_CURRICULUM, train_dataset)

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
    val_loader_easy = DataLoader(val_dataset_easy, batch_size=64, num_workers=2, collate_fn=collate_fn)
    val_loader_hard = DataLoader(val_dataset_hard, batch_size=64, num_workers=2, collate_fn=collate_fn)

    # Auto-detect device
    # ... (device detection logic is the same)

    # Create trainer
    trainer = DiagnosticTrainer(
        model=model,
        train_loader=train_loader,
        config=training_config,
        val_loader_easy=val_loader_easy,
        val_loader_hard=val_loader_hard,
        curriculum=curriculum,
        use_wandb=True, # Always use wandb for diagnostics
    )

    print(f"\nDiagnostic training configuration:")
    print(f"  Batch size: {training_config.batch_size}")
    print(f"  Learning rate: {training_config.learning_rate}")
    print(f"  Max steps: {training_config.max_steps}")
    print(f"  Device: {training_config.device}")

    if args.resume_from:
        trainer.load_checkpoint(args.resume_from)

    # Train
    trainer.train()

if __name__ == '__main__':
    main()
