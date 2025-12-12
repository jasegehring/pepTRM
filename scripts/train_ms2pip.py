"""
Training with MS2PIP-based synthetic data for realistic fragment ion predictions.

MS2PIP provides:
- Realistic fragment intensities (trained on 21M real spectra)
- Doubly-charged ions (b++, y++) automatically
- Better representation of MS/MS physics than simple synthetic generator

This script uses the same optimizations as train_optimized.py:
- Mixed precision training (AMP)
- torch.compile() support
- Extended curriculum learning
- Large batch sizes
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from omegaconf import OmegaConf

from src.model.trm import TRMConfig, create_model
from src.data.ms2pip_dataset import create_ms2pip_dataloader
from src.training.trainer_optimized import OptimizedTrainer, TrainingConfig
from src.training.curriculum import CurriculumScheduler, DEFAULT_CURRICULUM


def main():
    # Load config
    config_path = project_root / 'configs' / 'optimized_extended.yaml'
    cfg = OmegaConf.load(config_path)

    print("=" * 80)
    print("Recursive Peptide Model - Training with MS2PIP")
    print("=" * 80)

    # Create model
    model_config = TRMConfig(**cfg.model)
    model = create_model(model_config)

    print(f"\nModel created:")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Hidden dim: {model_config.hidden_dim}")
    print(f"  Supervision steps: {model_config.num_supervision_steps}")
    print(f"  Latent steps: {model_config.num_latent_steps}")

    # Create MS2PIP dataloaders
    print(f"\nCreating MS2PIP dataloaders...")

    train_loader = create_ms2pip_dataloader(
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        min_length=cfg.data.min_length,
        max_length=cfg.data.max_length,
        max_peaks=model_config.max_peaks,
        max_seq_len=model_config.max_seq_len,
        charge_distribution=dict(cfg.data.charge_distribution),
        noise_peaks=cfg.data.noise_peaks,
        peak_dropout=cfg.data.peak_dropout,
        mass_error_ppm=cfg.data.mass_error_ppm,
        intensity_variation=cfg.data.intensity_variation,
        ms2pip_model=cfg.data.ms2pip_model,
        top_k_peaks=cfg.data.top_k_peaks,
    )

    val_loader = create_ms2pip_dataloader(
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        min_length=cfg.data.min_length,
        max_length=cfg.data.max_length,
        max_peaks=model_config.max_peaks,
        max_seq_len=model_config.max_seq_len,
        charge_distribution=dict(cfg.data.charge_distribution),
        ms2pip_model=cfg.data.ms2pip_model,
        top_k_peaks=cfg.data.top_k_peaks,
    )
    
    train_dataset = train_loader.dataset

    print(f"  ✓ MS2PIP dataloaders created")
    print(f"  Model: {cfg.data.ms2pip_model}")
    print(f"  Top-k peaks: {cfg.data.top_k_peaks}")
    print(f"  Peptide length: {cfg.data.min_length}-{cfg.data.max_length}")
    print(f"  Initial difficulty: Clean data (curriculum will adjust)")

    # Setup curriculum learning
    curriculum = CurriculumScheduler(
        stages=DEFAULT_CURRICULUM,
        dataset=train_dataset,
    )

    print(f"\nCurriculum: {len(DEFAULT_CURRICULUM)} stages, {curriculum.total_steps} total steps")
    for i, stage in enumerate(DEFAULT_CURRICULUM):
        print(f"  Stage {i+1}: {stage.name} ({stage.steps} steps)")

    # Create optimized trainer
    training_config = TrainingConfig(**cfg.training)

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

    # Performance features
    print(f"\nPerformance features:")
    print(f"  Mixed precision (AMP): {training_config.use_amp}")
    print(f"  torch.compile(): {training_config.use_compile}")
    print(f"  Gradient accumulation: {training_config.gradient_accumulation_steps} steps")

    # Create trainer
    trainer = OptimizedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        curriculum=curriculum,
    )

    print(f"\n" + "=" * 80)
    print(f"Starting training with MS2PIP realistic spectra")
    print(f"=" * 80)

    # Train
    trainer.train()

    print("\n" + "=" * 80)
    print("Training completed!")
    print("=" * 80)


if __name__ == '__main__':
    main()
