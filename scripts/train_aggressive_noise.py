"""
Aggressive noise training with exponential iteration weighting.

Features:
- Aggressive noise curriculum (force recursion from start)
- Exponential iteration weighting (optimize final steps)
- Precursor loss enabled (ramping 0.05 → 0.3)
- Spectrum loss DISABLED (sigma too narrow)
- Mixed precision + torch.compile
- Batch size 80 (confirmed working with compile)

Philosophy: Test if noise alone unlocks multi-step refinement.
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
from src.training.curriculum_complex_noise import CurriculumScheduler, AGGRESSIVE_NOISE_CURRICULUM


def create_model(config: TRMConfig) -> RecursivePeptideModel:
    """Create model from config."""
    return RecursivePeptideModel(config)


def main():
    parser = argparse.ArgumentParser(description="Aggressive noise training for peptide TRM.")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to a checkpoint to resume training from.")
    args = parser.parse_args()

    # Clear CUDA cache
    if torch.cuda.is_available():
        print("Clearing CUDA cache...")
        torch.cuda.empty_cache()
        print("✓ CUDA cache cleared.")

    # Load config
    config_path = project_root / 'configs' / 'aggressive_noise_test.yaml'
    cfg = OmegaConf.load(config_path)

    print("=" * 70)
    print("AGGRESSIVE NOISE TRAINING - FORCE RECURSION")
    print("=" * 70)
    print("\nStrategy:")
    print("  ✓ Exponential iteration weighting (force step 7 optimization)")
    print("  ✓ Aggressive noise from step 0 (5 peaks, 15% dropout)")
    print("  ✓ Precursor loss enabled (0.05 → 0.3)")
    print("  ✓ Spectrum loss DISABLED (not worth debugging)")
    print("  ✓ Batch size 80 + compile (confirmed working)")
    print("=" * 70)

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

    # Easy validation: clean data, short peptides
    val_dataset_easy = MS2PIPSyntheticDataset(
        min_length=7,
        max_length=10,
        noise_peaks=0,
        peak_dropout=0.0,
        mass_error_ppm=0.0,
        ms2pip_model=cfg.data.ms2pip_model,
    )

    # Hard validation: realistic conditions matching real MS/MS data
    # Based on Nine-Species dataset characteristics:
    # - ~100 peaks per spectrum, mass range 100-1600 Da
    # - Real instrument noise, missing peaks, mass calibration errors
    val_dataset_hard = MS2PIPSyntheticDataset(
        min_length=8,
        max_length=25,          # Real peptides can be longer
        noise_peaks=20,         # Real spectra have many unexplained peaks
        peak_dropout=0.25,      # ~25% of theoretical peaks missing
        mass_error_ppm=15.0,    # Typical instrument accuracy
        intensity_variation=0.3,
        ms2pip_model=cfg.data.ms2pip_model,
    )

    print(f"\nDataset created (using MS2PIP):")
    print(f"  Train peptide length: Managed by aggressive curriculum")
    print(f"  Val (Easy) length: 7-10 (clean)")
    print(f"  Val (Hard) length: 8-25 (realistic: 20 noise, 25% dropout, 15ppm)")
    print(f"  MS2PIP model: {cfg.data.ms2pip_model}")

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
    curriculum = CurriculumScheduler(AGGRESSIVE_NOISE_CURRICULUM, train_dataset)

    print("\n📚 Two-Tier Noise Curriculum (Flat Length):")
    print(f"  Total stages: {len(AGGRESSIVE_NOISE_CURRICULUM)}")
    print(f"  Total steps: {sum(s.steps for s in AGGRESSIVE_NOISE_CURRICULUM):,}")
    print(f"  Length: 7-30 from start (no progressive length)")
    print(f"  Stage 0: Pure foundation (100% clean, all lengths)")
    print(f"  Stage 1-2: Introduce grass + spikes noise")
    print(f"  Stage 3-4: Realistic → Extreme stress test")

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

    # Real data validation loaders
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
        val_loader_proteometools=val_loader_proteometools,
        val_loader_nine_species=val_loader_nine_species,
        curriculum=curriculum,
        use_wandb=True,
        real_data_eval_interval=2000,
        real_data_num_batches=20,
    )

    print(f"\nTraining configuration:")
    print(f"  Batch size: {training_config.batch_size}")
    print(f"  Mixed precision: {training_config.use_amp} ({training_config.amp_dtype})")
    print(f"  Model compilation: {training_config.use_compile}")
    print(f"  Learning rate: {training_config.learning_rate}")
    print(f"  Max steps: {training_config.max_steps:,}")
    print(f"  Iteration weighting: {training_config.iteration_weights}")
    print(f"  Spectrum loss: DISABLED (weight={training_config.spectrum_weight})")
    print(f"  Precursor loss: Curriculum-managed")
    print(f"  Device: {training_config.device}")

    if args.resume_from:
        print(f"\n⚠️  Resuming from: {args.resume_from}")
        trainer.load_checkpoint(args.resume_from)

    # Train
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    trainer.train()


if __name__ == '__main__':
    main()
