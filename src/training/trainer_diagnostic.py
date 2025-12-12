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
from torch.amp import autocast, GradScaler
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from copy import deepcopy
from tqdm import tqdm
import torch.nn.functional as F
import wandb

from ..model.trm import RecursivePeptideModel
from .losses import DeepSupervisionLoss, CombinedLoss
from .metrics import compute_metrics
from .curriculum_extended import CurriculumScheduler, EXTENDED_CURRICULUM
from ..data.ion_types import get_ion_types_for_model
from ..constants import PROTON_MASS


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

    # Ion types (for spectrum matching loss)
    ms2pip_model: str = 'HCDch2'  # Auto-selects ion types: b, y, b++, y++

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


class DiagnosticTrainer:
    """
    Optimized trainer with AMP and compilation support.
    """

    def __init__(
        self,
        model: RecursivePeptideModel,
        train_loader: DataLoader,
        config: TrainingConfig,
        val_loader_easy: Optional[DataLoader] = None,
        val_loader_hard: Optional[DataLoader] = None,
        curriculum: Optional[CurriculumScheduler] = None,
        use_wandb: bool = False,
    ):
        """
        Args:
            model: The RecursivePeptideModel to train
            train_loader: Training data loader
            config: Training configuration
            val_loader_easy: Easy validation set (clean data, short peptides)
            val_loader_hard: Hard validation set (realistic noise, longer peptides)
            curriculum: Curriculum scheduler for progressive difficulty
            use_wandb: Whether to log to Weights & Biases
        """
        self.config = config
        self.device = torch.device(config.device)
        self.use_wandb = use_wandb

        # Force disable compilation for diagnostics to ensure clear stack traces
        config.use_compile = False
        print("--- RUNNING IN DIAGNOSTIC MODE ---")
        print("   - torch.compile disabled")
        print("   - Logging intermediate values every step")
        print("---------------------------------")

        self.model = model.to(self.device)

        # Mixed precision setup
        self.use_amp = config.use_amp and config.device == 'cuda'
        self.scaler = GradScaler('cuda', enabled=self.use_amp) if self.use_amp else None

        # Datasets and loaders
        self.train_loader = train_loader
        self.val_loader_easy = val_loader_easy
        self.val_loader_hard = val_loader_hard

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
        self.loss_fn = CombinedLoss(
            ce_weight=config.ce_weight,
            spectrum_weight=config.spectrum_weight,
            iteration_weights=config.iteration_weights,
            label_smoothing=config.label_smoothing,
            ms2pip_model=config.ms2pip_model,
        ).to(self.device)
        self.use_combined_loss = True

        # Curriculum scheduler
        self.curriculum = curriculum

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
                tags = ['diagnostic'] # Tag this run as a diagnostic run
                wandb.init(
                    project="peptide-trm-diagnostic", # Use a separate project
                    config=vars(config),
                    tags=tags,
                )
            except ImportError:
                print("⚠️  wandb not installed, logging disabled")
                self.use_wandb = False

    def train_step(self, batch):
        """Single training step with extensive diagnostic logging."""
        self.model.train()
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

        # --- DIAGNOSTIC LOGGING: INPUTS ---
        precursor_neutral_mass = batch['precursor_mass']
        precursor_charge = batch['precursor_charge']
        precursor_mz = (precursor_neutral_mass + precursor_charge * PROTON_MASS) / precursor_charge
        
        diag_log = {}
        if self.use_wandb:
            diag_log.update({
                'diag/input_precursor_neutral_mass': precursor_neutral_mass.mean().item(),
                'diag/input_precursor_charge': precursor_charge.float().mean().item(),
                'diag/input_precursor_mz': precursor_mz.mean().item(),
            })

        # --- FORWARD PASS ---
        with autocast('cuda', enabled=self.use_amp, dtype=torch.float16 if self.use_amp else torch.float32):
            all_logits, _ = self.model(
                spectrum_masses=batch['spectrum_masses'],
                spectrum_intensities=batch['spectrum_intensities'],
                spectrum_mask=batch['spectrum_mask'],
                precursor_mass=precursor_mz,
                precursor_charge=precursor_charge,
            )

            # --- CURRICULUM UPDATE ---
            if self.curriculum and self.curriculum.step(self.global_step):
                self.loss_fn.spectrum_weight = self.curriculum.get_spectrum_loss_weight()
                self.loss_fn.precursor_weight = self.curriculum.get_precursor_loss_weight()

            # --- LOSS COMPUTATION ---
            loss, metrics = self.loss_fn(
                all_logits=all_logits,
                targets=batch['sequence'],
                target_mask=batch['sequence_mask'],
                observed_masses=batch['spectrum_masses'],
                observed_intensities=batch['spectrum_intensities'],
                peak_mask=batch['spectrum_mask'],
                precursor_mass=precursor_neutral_mass.squeeze(-1),
            )
            
            # --- DIAGNOSTIC LOGGING: LOSS & METRICS ---
            if self.use_wandb:
                # Log final probabilities
                final_probs = F.softmax(all_logits[-1], dim=-1).float()
                diag_log.update({
                    'diag/final_probs_min': final_probs.min().item(),
                    'diag/final_probs_max': final_probs.max().item(),
                    'diag/final_probs_mean': final_probs.mean().item(),
                })
                # Log all metrics from the loss function
                for k, v in metrics.items():
                    diag_log[f'diag/{k}'] = v
                wandb.log(diag_log, step=self.global_step)
            
            if self.config.gradient_accumulation_steps > 1:
                loss = loss / self.config.gradient_accumulation_steps

        # --- BACKWARD PASS ---
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # --- OPTIMIZER STEP ---
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            # Unscale and clip grads
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # --- DIAGNOSTIC LOGGING: GRADIENTS ---
            if self.use_wandb:
                total_grad_norm = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        total_grad_norm += p.grad.data.norm(2).item() ** 2
                total_grad_norm = total_grad_norm ** 0.5
                wandb.log({'diag/gradient_norm': total_grad_norm}, step=self.global_step)

            # Step optimizer
            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad()
            self.scheduler.step()

            if self.ema:
                self.ema.update(self.model)

        # --- ACCURACY METRICS ---
        with torch.no_grad():
            acc_metrics = compute_metrics(
                logits=all_logits[-1].float(),
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
                            log_dict['curriculum/min_length'] = stage.min_length
                            log_dict['curriculum/max_length'] = stage.max_length
                            log_dict['curriculum/noise_peaks'] = stage.noise_peaks
                            log_dict['curriculum/peak_dropout'] = stage.peak_dropout
                            log_dict['curriculum/mass_error_ppm'] = stage.mass_error_ppm
                            log_dict['curriculum/spectrum_loss_weight'] = stage.spectrum_loss_weight
                            log_dict['curriculum/precursor_loss_weight'] = stage.precursor_loss_weight

                        wandb.log(log_dict, step=self.global_step)

                # Validation (FIXED: was self.val_dataset, now self.val_loader_easy/hard)
                if self.global_step % self.config.eval_interval == 0:
                    # Easy validation (clean data, short peptides)
                    if self.val_loader_easy is not None:
                        val_easy_metrics = self.evaluate(self.val_loader_easy)
                        tqdm.write(
                            f"Val (Easy) | Token Acc: {val_easy_metrics['token_accuracy']:.3f} | "
                            f"Seq Acc: {val_easy_metrics['sequence_accuracy']:.3f}"
                        )

                        if self.use_wandb:
                            import wandb
                            wandb.log({
                                f'val_easy/{k}': v for k, v in val_easy_metrics.items()
                            }, step=self.global_step)

                        # Save best model based on easy validation
                        if val_easy_metrics['sequence_accuracy'] > self.best_val_acc:
                            self.best_val_acc = val_easy_metrics['sequence_accuracy']
                            self.save_checkpoint('best_model.pt')

                    # Hard validation (realistic noise, longer peptides)
                    if self.val_loader_hard is not None:
                        val_hard_metrics = self.evaluate(self.val_loader_hard)
                        tqdm.write(
                            f"Val (Hard) | Token Acc: {val_hard_metrics['token_accuracy']:.3f} | "
                            f"Seq Acc: {val_hard_metrics['sequence_accuracy']:.3f}"
                        )

                        if self.use_wandb:
                            import wandb
                            wandb.log({
                                f'val_hard/{k}': v for k, v in val_hard_metrics.items()
                            }, step=self.global_step)

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
    def evaluate(self, val_loader: DataLoader):
        """
        Evaluate on validation set.

        Args:
            val_loader: Validation data loader (easy or hard)

        Returns:
            Dict of average metrics
        """
        self.model.eval()

        total_metrics = {}
        num_batches = 0

        for batch in val_loader:
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

            # --- FIX: Calculate precursor m/z for model input ---
            precursor_neutral_mass = batch['precursor_mass']
            precursor_charge = batch['precursor_charge']
            precursor_mz = (precursor_neutral_mass + precursor_charge * PROTON_MASS) / precursor_charge

            with autocast('cuda', enabled=self.use_amp, dtype=torch.float16 if self.use_amp else torch.float32):
                all_logits, _ = self.model(
                    spectrum_masses=batch['spectrum_masses'],
                    spectrum_intensities=batch['spectrum_intensities'],
                    spectrum_mask=batch['spectrum_mask'],
                    precursor_mass=precursor_mz,
                    precursor_charge=precursor_charge,
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

    def load_checkpoint(self, checkpoint_path: str):
        """Load training state from a checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
        if self.ema and 'ema_state_dict' in checkpoint:
            self.ema.load_state_dict(checkpoint['ema_state_dict'])
            
        self.global_step = checkpoint['step']
        self.best_val_acc = checkpoint['best_val_acc']
        
        print(f"\n✓ Resumed training from checkpoint: {checkpoint_path}")
        print(f"  Starting from step: {self.global_step}")
        print(f"  Best validation accuracy: {self.best_val_acc:.3f}")
