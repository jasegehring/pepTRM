"""
Optimized training loop for high-performance GPUs (RTX 4090, A100).

Optimizations:
- Mixed precision training (AMP) - 2-3x speedup
- torch.compile() support - 1.2-1.5x speedup
- Pin memory for faster CPU->GPU transfer
- Extended curriculum support
- Gradient accumulation option
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from copy import deepcopy
from tqdm import tqdm

from ..model.trm import RecursivePeptideModel
from ..data.dataset import SyntheticPeptideDataset, collate_peptide_samples
from .losses import DeepSupervisionLoss, CombinedLoss
from .metrics import compute_metrics
from .curriculum_extended import CurriculumScheduler, EXTENDED_CURRICULUM


@dataclass
class TrainingConfig:
    """Optimized training configuration."""
    # Optimization
    learning_rate: float = 1.5e-4  # Slightly higher for larger batch
    weight_decay: float = 0.01
    batch_size: int = 192  # 3x larger for RTX 4090
    max_steps: int = 100000  # Extended training
    warmup_steps: int = 2000

    # Mixed Precision (NEW)
    use_amp: bool = True
    amp_dtype: str = 'float16'  # or 'bfloat16'

    # Gradient Accumulation
    gradient_accumulation_steps: int = 1

    # Model Compilation (NEW)
    use_compile: bool = True
    compile_mode: str = 'max-autotune'  # 'default', 'reduce-overhead', 'max-autotune'

    # Loss weights
    ce_weight: float = 1.0
    spectrum_weight: float = 0.0
    iteration_weights: str = 'linear'
    label_smoothing: float = 0.1  # Increased for robustness

    # Curriculum
    use_curriculum: bool = True

    # EMA
    use_ema: bool = True
    ema_decay: float = 0.999

    # Logging
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 5000

    # Paths
    checkpoint_dir: str = 'checkpoints_optimized'

    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class EMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model: nn.Module):
        for ema_p, model_p in zip(self.shadow.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1 - self.decay)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict):
        self.shadow.load_state_dict(state_dict)


class OptimizedTrainer:
    """
    Optimized trainer with AMP and compilation support.
    """

    def __init__(
        self,
        model: RecursivePeptideModel,
        train_dataset: SyntheticPeptideDataset,
        config: TrainingConfig,
        val_dataset: Optional[SyntheticPeptideDataset] = None,
        use_wandb: bool = False,
    ):
        self.config = config
        self.device = torch.device(config.device)
        self.use_wandb = use_wandb

        # Compile model if requested (BEFORE moving to device)
        if config.use_compile and hasattr(torch, 'compile') and config.device == 'cuda':
            print(f"🔧 Compiling model with mode='{config.compile_mode}'...")
            print("   (First training step will be slow while compiling)")
            try:
                model = torch.compile(
                    model,
                    mode=config.compile_mode,
                    fullgraph=False,  # Allow fallback for dynamic ops
                )
                print("   ✓ Model compiled successfully")
            except Exception as e:
                print(f"   ⚠️  Compilation failed: {e}")
                print("   Continuing without compilation...")
                config.use_compile = False

        self.model = model.to(self.device)

        # Mixed precision setup
        self.use_amp = config.use_amp and config.device == 'cuda'
        self.scaler = GradScaler(enabled=self.use_amp) if self.use_amp else None

        if self.use_amp:
            print(f"⚡ Mixed precision training enabled ({config.amp_dtype})")
            print(f"   Expected speedup: 2-3x faster")

        # Datasets and loaders
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            collate_fn=collate_peptide_samples,
            num_workers=0,
            pin_memory=True,  # Faster CPU->GPU transfer
        )

        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                collate_fn=collate_peptide_samples,
                pin_memory=True,
            )

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.max_steps,
            eta_min=config.learning_rate * 0.01,
        )

        # Loss
        if config.spectrum_weight > 0 or config.use_curriculum:
            self.loss_fn = CombinedLoss(
                ce_weight=config.ce_weight,
                spectrum_weight=config.spectrum_weight,
                iteration_weights=config.iteration_weights,
                label_smoothing=config.label_smoothing,
            ).to(self.device)
            self.use_combined_loss = True
        else:
            self.loss_fn = DeepSupervisionLoss(
                iteration_weights=config.iteration_weights,
                label_smoothing=config.label_smoothing,
            ).to(self.device)
            self.use_combined_loss = False

        # Curriculum scheduler (extended)
        self.curriculum = None
        if config.use_curriculum:
            self.curriculum = CurriculumScheduler(
                stages=EXTENDED_CURRICULUM,
                dataset=train_dataset,
            )
            print(f"📚 Extended curriculum enabled ({len(EXTENDED_CURRICULUM)} stages, {self.curriculum.total_steps} total steps)")

        # EMA
        self.ema = None
        if config.use_ema:
            self.ema = EMA(model, decay=config.ema_decay)

        # Tracking
        self.global_step = 0
        self.best_val_acc = 0.0

        # Checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize wandb
        if self.use_wandb:
            try:
                import wandb
                tags = ['optimized']
                if self.use_amp:
                    tags.append('amp')
                if config.use_compile:
                    tags.append('compiled')
                tags.append(f'batch_{config.batch_size}')

                wandb.init(
                    project="peptide-trm",
                    config=vars(config),
                    tags=tags,
                )
            except ImportError:
                print("⚠️  wandb not installed, logging disabled")
                self.use_wandb = False

    def train_step(self, batch):
        """Single training step with mixed precision."""
        self.model.train()

        # Move to device
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

        # Mixed precision forward pass
        with autocast(enabled=self.use_amp, dtype=torch.float16 if self.use_amp else torch.float32):
            # Forward
            all_logits, _ = self.model(
                spectrum_masses=batch['spectrum_masses'],
                spectrum_intensities=batch['spectrum_intensities'],
                spectrum_mask=batch['spectrum_mask'],
                precursor_mass=batch['precursor_mass'],
                precursor_charge=batch['precursor_charge'],
            )

            # Update curriculum
            if self.curriculum:
                if self.curriculum.step(self.global_step):
                    if self.use_combined_loss:
                        self.loss_fn.spectrum_weight = self.curriculum.get_spectrum_loss_weight()

            # Compute loss
            if self.use_combined_loss:
                loss, metrics = self.loss_fn(
                    all_logits=all_logits,
                    targets=batch['sequence'],
                    target_mask=batch['sequence_mask'],
                    observed_masses=batch['spectrum_masses'],
                    observed_intensities=batch['spectrum_intensities'],
                    peak_mask=batch['spectrum_mask'],
                )
            else:
                loss, metrics = self.loss_fn(
                    all_logits=all_logits,
                    targets=batch['sequence'],
                    target_mask=batch['sequence_mask'],
                )

            # Gradient accumulation scaling
            if self.config.gradient_accumulation_steps > 1:
                loss = loss / self.config.gradient_accumulation_steps

        # Backward pass with gradient scaling
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Optimizer step (only every N accumulation steps)
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            self.optimizer.zero_grad()
            self.scheduler.step()

            # Update EMA
            if self.ema:
                self.ema.update(self.model)

        # Compute accuracy metrics (in FP32 for precision)
        with torch.no_grad():
            final_logits = all_logits[-1].float()  # Last iteration
            acc_metrics = compute_metrics(
                logits=final_logits,
                targets=batch['sequence'],
                mask=batch['sequence_mask'],
            )
            metrics.update(acc_metrics)

        return loss.item(), metrics

    def train(self):
        """Main training loop."""
        print(f"\n{'='*60}")
        print(f"Starting optimized training on {self.device}")
        print(f"{'='*60}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Max steps: {self.config.max_steps}")
        if self.use_amp:
            print(f"Mixed precision: ✓ ENABLED")
        if self.config.use_compile:
            print(f"Compilation: ✓ ENABLED")
        print(f"{'='*60}\n")

        pbar = tqdm(total=self.config.max_steps, desc='Training')

        while self.global_step < self.config.max_steps:
            for batch in self.train_loader:
                # Training step
                loss, metrics = self.train_step(batch)

                # Logging
                if self.global_step % self.config.log_interval == 0:
                    log_str = (
                        f"Step {self.global_step} | "
                        f"Loss: {loss:.4f} | "
                        f"Token Acc: {metrics.get('token_accuracy', 0):.3f} | "
                        f"Seq Acc: {metrics.get('sequence_accuracy', 0):.3f} | "
                        f"LR: {self.scheduler.get_last_lr()[0]:.2e}"
                    )
                    tqdm.write(log_str)

                    if self.use_wandb:
                        import wandb
                        log_dict = {
                            'train/loss': loss,
                            'train/token_accuracy': metrics.get('token_accuracy', 0),
                            'train/sequence_accuracy': metrics.get('sequence_accuracy', 0),
                            'train/learning_rate': self.scheduler.get_last_lr()[0],
                            'train/step': self.global_step,
                        }
                        # Add all other metrics
                        for k, v in metrics.items():
                            if k not in ['token_accuracy', 'sequence_accuracy']:
                                log_dict[f'train/{k}'] = v

                        # Add curriculum stage info
                        if self.curriculum and self.curriculum.current_stage:
                            stage = self.curriculum.current_stage
                            log_dict['curriculum/stage_idx'] = self.curriculum.current_stage_idx
                            log_dict['curriculum/noise_peaks'] = stage.noise_peaks
                            log_dict['curriculum/peak_dropout'] = stage.peak_dropout
                            log_dict['curriculum/mass_error_ppm'] = stage.mass_error_ppm
                            log_dict['curriculum/spectrum_loss_weight'] = stage.spectrum_loss_weight

                        wandb.log(log_dict, step=self.global_step)

                # Validation
                if self.global_step % self.config.eval_interval == 0 and self.val_dataset:
                    val_metrics = self.evaluate()
                    tqdm.write(
                        f"Validation | Token Acc: {val_metrics['token_accuracy']:.3f} | "
                        f"Seq Acc: {val_metrics['sequence_accuracy']:.3f}"
                    )

                    if self.use_wandb:
                        import wandb
                        wandb.log({
                            f'val/{k}': v for k, v in val_metrics.items()
                        }, step=self.global_step)

                    # Save best model
                    if val_metrics['sequence_accuracy'] > self.best_val_acc:
                        self.best_val_acc = val_metrics['sequence_accuracy']
                        self.save_checkpoint('best_model.pt')

                # Checkpointing
                if self.global_step % self.config.save_interval == 0 and self.global_step > 0:
                    self.save_checkpoint(f'checkpoint_step_{self.global_step}.pt')

                self.global_step += 1
                pbar.update(1)

                if self.global_step >= self.config.max_steps:
                    break

        # Final checkpoint
        self.save_checkpoint('final_model.pt')
        pbar.close()

        print(f"\n{'='*60}")
        print("Training complete!")
        print(f"{'='*60}")

    @torch.no_grad()
    def evaluate(self):
        """Evaluate on validation set."""
        self.model.eval()

        total_metrics = {}
        num_batches = 0

        for batch in self.val_loader:
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

            with autocast(enabled=self.use_amp, dtype=torch.float16 if self.use_amp else torch.float32):
                all_logits, _ = self.model(
                    spectrum_masses=batch['spectrum_masses'],
                    spectrum_intensities=batch['spectrum_intensities'],
                    spectrum_mask=batch['spectrum_mask'],
                    precursor_mass=batch['precursor_mass'],
                    precursor_charge=batch['precursor_charge'],
                )

            final_logits = all_logits[-1].float()
            metrics = compute_metrics(
                logits=final_logits,
                targets=batch['sequence'],
                mask=batch['sequence_mask'],
            )

            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0) + v
            num_batches += 1

            if num_batches >= 10:  # Quick validation
                break

        return {k: v / num_batches for k, v in total_metrics.items()}

    def save_checkpoint(self, filename):
        """Save model checkpoint."""
        checkpoint = {
            'step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': vars(self.config),
            'best_val_acc': self.best_val_acc,
        }

        if self.ema:
            checkpoint['ema_state_dict'] = self.ema.state_dict()

        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint, self.checkpoint_dir / filename)
        tqdm.write(f"Saved checkpoint: {filename}")
