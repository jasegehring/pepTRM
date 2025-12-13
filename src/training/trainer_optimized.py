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

from ..model.trm import RecursivePeptideModel
from .losses import DeepSupervisionLoss, CombinedLoss
from .metrics import compute_metrics, compute_metrics_by_length
from .curriculum import CurriculumScheduler, DEFAULT_CURRICULUM
from .refinement_tracker import compute_refinement_metrics
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
        # Handle prefix mismatch between compiled/non-compiled models
        checkpoint_is_compiled = any(k.startswith('_orig_mod.') for k in state_dict.keys())
        current_is_compiled = any(k.startswith('_orig_mod.') for k in self.shadow.state_dict().keys())

        if checkpoint_is_compiled and not current_is_compiled:
            # Checkpoint from compiled model, loading into non-compiled → strip prefix
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        elif not checkpoint_is_compiled and current_is_compiled:
            # Checkpoint from non-compiled model, loading into compiled → add prefix
            state_dict = {f'_orig_mod.{k}': v for k, v in state_dict.items()}

        self.shadow.load_state_dict(state_dict)


class OptimizedTrainer:
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
        val_loader_proteometools: Optional[DataLoader] = None,
        val_loader_nine_species: Optional[DataLoader] = None,
        curriculum: Optional[CurriculumScheduler] = None,
        use_wandb: bool = False,
        real_data_eval_interval: int = 2000,
        real_data_num_batches: int = 20,
    ):
        """
        Args:
            model: The RecursivePeptideModel to train
            train_loader: Training data loader
            config: Training configuration
            val_loader_easy: Easy validation set (clean data, short peptides)
            val_loader_hard: Hard validation set (realistic noise, longer peptides)
            val_loader_proteometools: ProteomeTools validation set (synthetic peptide library)
            val_loader_nine_species: Nine-Species validation set (real biological data)
            curriculum: Curriculum scheduler for progressive difficulty
            use_wandb: Whether to log to Weights & Biases
            real_data_eval_interval: How often to evaluate on real data (steps)
            real_data_num_batches: Number of batches to use for real data evaluation
        """
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
        self.scaler = GradScaler('cuda', enabled=self.use_amp) if self.use_amp else None

        if self.use_amp:
            print(f"⚡ Mixed precision training enabled ({config.amp_dtype})")
            print(f"   Expected speedup: 2-3x faster")

        # Datasets and loaders
        self.train_loader = train_loader
        self.val_loader_easy = val_loader_easy
        self.val_loader_hard = val_loader_hard
        self.val_loader_proteometools = val_loader_proteometools
        self.val_loader_nine_species = val_loader_nine_species
        self.real_data_eval_interval = real_data_eval_interval
        self.real_data_num_batches = real_data_num_batches

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

        # Loss (with ion type configuration)
        if config.spectrum_weight > 0 or config.use_curriculum:
            self.loss_fn = CombinedLoss(
                ce_weight=config.ce_weight,
                spectrum_weight=config.spectrum_weight,
                iteration_weights=config.iteration_weights,
                label_smoothing=config.label_smoothing,
                ms2pip_model=config.ms2pip_model,  # Auto-select ion types (e.g., HCDch2)
            ).to(self.device)
            self.use_combined_loss = True
            print(f"📊 Ion types for spectrum matching: {get_ion_types_for_model(config.ms2pip_model)}")
        else:
            self.loss_fn = DeepSupervisionLoss(
                iteration_weights=config.iteration_weights,
                label_smoothing=config.label_smoothing,
            ).to(self.device)
            self.use_combined_loss = False

        # Curriculum scheduler
        self.curriculum = curriculum
        if self.curriculum:
            print(f"📚 Curriculum enabled ({self.curriculum.total_steps} total steps)")

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
                import os

                tags = ['optimized']
                if self.use_amp:
                    tags.append('amp')
                if config.use_compile:
                    tags.append('compiled')
                tags.append(f'batch_{config.batch_size}')

                # Check for resume parameters from environment
                wandb_run_id = os.environ.get('WANDB_RUN_ID', None)
                wandb_resume = os.environ.get('WANDB_RESUME', None)

                # Check for grouping (for checkpoint branching experiments)
                wandb_group = os.environ.get('WANDB_RUN_GROUP', None)
                wandb_job_type = os.environ.get('WANDB_JOB_TYPE', None)

                if wandb_run_id:
                    print(f"📊 Resuming W&B run: {wandb_run_id}")
                    wandb.init(
                        project="peptide-trm",
                        id=wandb_run_id,
                        resume="allow" if wandb_resume == "allow" else "must",
                        group=wandb_group,
                        job_type=wandb_job_type,
                        config=vars(config),
                        tags=tags,
                    )
                else:
                    print(f"📊 Starting new W&B run")
                    wandb.init(
                        project="peptide-trm",
                        group=wandb_group,
                        job_type=wandb_job_type,
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

        # --- FIX: Calculate precursor m/z for model input ---
        # The model encoder expects precursor m/z to be on the same scale as fragment m/z.
        # The loss function still needs the neutral mass.
        precursor_neutral_mass = batch['precursor_mass']
        precursor_charge = batch['precursor_charge']
        # m/z = (M + z*H) / z
        precursor_mz = (precursor_neutral_mass + precursor_charge * PROTON_MASS) / precursor_charge

        # Mixed precision forward pass
        with autocast('cuda', enabled=self.use_amp, dtype=torch.float16 if self.use_amp else torch.float32):
            # Forward
            all_logits, _ = self.model(
                spectrum_masses=batch['spectrum_masses'],
                spectrum_intensities=batch['spectrum_intensities'],
                spectrum_mask=batch['spectrum_mask'],
                precursor_mass=precursor_mz,  # Pass m/z to model
                precursor_charge=precursor_charge,
            )

            # NOTE: Curriculum is now updated in main training loop to allow
            # DataLoader iterator recreation when stage changes

            # Compute loss
            if self.use_combined_loss:
                loss, metrics = self.loss_fn(
                    all_logits=all_logits,
                    targets=batch['sequence'],
                    target_mask=batch['sequence_mask'],
                    observed_masses=batch['spectrum_masses'],
                    observed_intensities=batch['spectrum_intensities'],
                    peak_mask=batch['spectrum_mask'],
                    precursor_mass=precursor_neutral_mass.squeeze(-1),  # Pass neutral mass to loss
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

            # Compute refinement metrics (edit rate, improvement per step)
            # Only compute at log intervals to reduce overhead
            if self.global_step % self.config.log_interval == 0:
                refinement_metrics = compute_refinement_metrics(
                    all_logits=all_logits.float(),
                    targets=batch['sequence'],
                    target_mask=batch['sequence_mask'],
                )
                metrics.update(refinement_metrics)

                # Compute accuracy by length bucket (for debugging length generalization)
                length_metrics = compute_metrics_by_length(
                    logits=final_logits,
                    targets=batch['sequence'],
                    mask=batch['sequence_mask'],
                )
                metrics.update(length_metrics)

        return loss.item(), metrics

    def _create_train_iterator(self):
        """Create a fresh iterator over the training DataLoader.

        This is necessary because when using num_workers > 0 with an IterableDataset,
        the worker processes get their own copy of the dataset. When curriculum.step()
        updates the dataset parameters, the workers don't see those changes.

        By recreating the iterator when the curriculum stage changes, we force
        new worker processes to be spawned with the updated parameters.
        """
        return iter(self.train_loader)

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

        # Create initial iterator
        train_iterator = self._create_train_iterator()

        while self.global_step < self.config.max_steps:
            # Check curriculum and recreate iterator if stage changed
            if self.curriculum:
                stage_changed = self.curriculum.step(self.global_step)
                if stage_changed:
                    # Update loss weights
                    if self.use_combined_loss:
                        self.loss_fn.spectrum_weight = self.curriculum.get_spectrum_loss_weight()
                        self.loss_fn.precursor_weight = self.curriculum.get_precursor_loss_weight()
                    # CRITICAL: Recreate iterator so workers get updated dataset parameters
                    train_iterator = self._create_train_iterator()
                    tqdm.write(f"[Step {self.global_step}] Recreated DataLoader iterator for new curriculum stage")

            # Get next batch
            try:
                batch = next(train_iterator)
            except StopIteration:
                # Iterator exhausted (shouldn't happen with IterableDataset, but handle it)
                train_iterator = self._create_train_iterator()
                batch = next(train_iterator)

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
                    # Add all other metrics - preserve recursion/ prefix for organization
                    for k, v in metrics.items():
                        if k not in ['token_accuracy', 'sequence_accuracy']:
                            # Recursion metrics already have their prefix
                            if k.startswith('recursion/'):
                                log_dict[k] = v
                            else:
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

            # Validation
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

            # Real data validation (less frequent)
            if self.global_step % self.real_data_eval_interval == 0:
                # ProteomeTools validation (synthetic peptide library)
                if self.val_loader_proteometools is not None:
                    pt_metrics = self.evaluate(
                        self.val_loader_proteometools,
                        num_batches=self.real_data_num_batches
                    )
                    tqdm.write(
                        f"Val (ProteomeTools) | Token Acc: {pt_metrics['token_accuracy']:.3f} | "
                        f"Seq Acc: {pt_metrics['sequence_accuracy']:.3f}"
                    )

                    if self.use_wandb:
                        import wandb
                        wandb.log({
                            f'val_proteometools/{k}': v for k, v in pt_metrics.items()
                        }, step=self.global_step)

                # Nine-Species validation (real biological data)
                if self.val_loader_nine_species is not None:
                    ns_metrics = self.evaluate(
                        self.val_loader_nine_species,
                        num_batches=self.real_data_num_batches
                    )
                    tqdm.write(
                        f"Val (Nine-Species) | Token Acc: {ns_metrics['token_accuracy']:.3f} | "
                        f"Seq Acc: {ns_metrics['sequence_accuracy']:.3f}"
                    )

                    if self.use_wandb:
                        import wandb
                        wandb.log({
                            f'val_nine_species/{k}': v for k, v in ns_metrics.items()
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
    def evaluate(self, val_loader: DataLoader, num_batches: int = 10):
        """
        Evaluate on validation set.

        Args:
            val_loader: Validation data loader (easy, hard, or real data)
            num_batches: Maximum number of batches to evaluate (for quick validation)

        Returns:
            Dict of average metrics
        """
        self.model.eval()

        total_metrics = {}
        batch_count = 0

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

            # Compute refinement metrics (edit rate, improvement per step)
            refinement_metrics = compute_refinement_metrics(
                all_logits=all_logits.float(),
                targets=batch['sequence'],
                target_mask=batch['sequence_mask'],
            )
            metrics.update(refinement_metrics)

            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0) + v
            batch_count += 1

            if batch_count >= num_batches:
                break

        if batch_count == 0:
            return {'token_accuracy': 0.0, 'sequence_accuracy': 0.0}

        return {k: v / batch_count for k, v in total_metrics.items()}

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

        # Save curriculum state
        if self.curriculum:
            checkpoint['curriculum_stage_idx'] = self.curriculum.current_stage_idx

        torch.save(checkpoint, self.checkpoint_dir / filename)
        tqdm.write(f"Saved checkpoint: {filename}")

    def load_checkpoint(self, checkpoint_path: str, load_curriculum=True):
        """
        Load training state from a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            load_curriculum: If True, restore curriculum state. If False, start curriculum
                           from current step (useful for testing different curricula)
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Handle prefix mismatch between compiled/non-compiled models
        model_state = checkpoint['model_state_dict']
        checkpoint_is_compiled = any(k.startswith('_orig_mod.') for k in model_state.keys())
        current_is_compiled = any(k.startswith('_orig_mod.') for k in self.model.state_dict().keys())

        if checkpoint_is_compiled and not current_is_compiled:
            # Checkpoint from compiled model, loading into non-compiled → strip prefix
            model_state = {k.replace('_orig_mod.', ''): v for k, v in model_state.items()}
        elif not checkpoint_is_compiled and current_is_compiled:
            # Checkpoint from non-compiled model, loading into compiled → add prefix
            model_state = {f'_orig_mod.{k}': v for k, v in model_state.items()}

        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        if self.ema and 'ema_state_dict' in checkpoint:
            self.ema.load_state_dict(checkpoint['ema_state_dict'])

        self.global_step = checkpoint['step']
        self.best_val_acc = checkpoint['best_val_acc']

        # Optionally restore curriculum state
        if load_curriculum and self.curriculum and 'curriculum_stage_idx' in checkpoint:
            self.curriculum.current_stage_idx = checkpoint['curriculum_stage_idx']
            print(f"  Curriculum stage: {self.curriculum.current_stage_idx}")
        elif not load_curriculum and self.curriculum:
            # Update curriculum based on current step (for testing new curricula)
            self.curriculum.step(self.global_step)
            print(f"  Curriculum reset to step {self.global_step}")

        print(f"\n✓ Resumed training from checkpoint: {checkpoint_path}")
        print(f"  Starting from step: {self.global_step}")
        print(f"  Best validation accuracy: {self.best_val_acc:.3f}")
