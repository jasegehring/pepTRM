"""
Main training loop with logging.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from copy import deepcopy
from tqdm import tqdm

from ..model.trm import RecursivePeptideModel
from ..data.dataset import SyntheticPeptideDataset, collate_peptide_samples
from .losses import DeepSupervisionLoss, CombinedLoss
from .metrics import compute_metrics
from .curriculum import CurriculumScheduler, DEFAULT_CURRICULUM


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 64
    max_steps: int = 50000
    warmup_steps: int = 1000

    # Loss weights
    ce_weight: float = 1.0
    spectrum_weight: float = 0.0  # Will be controlled by curriculum
    iteration_weights: str = 'linear'
    label_smoothing: float = 0.0

    # Curriculum
    use_curriculum: bool = False

    # EMA (important for TRM stability)
    use_ema: bool = True
    ema_decay: float = 0.999

    # Logging
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 5000

    # Paths
    checkpoint_dir: str = 'checkpoints'

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


class Trainer:
    """
    Main trainer class.
    """

    def __init__(
        self,
        model: RecursivePeptideModel,
        train_dataset: SyntheticPeptideDataset,
        config: TrainingConfig,
        val_dataset: Optional[SyntheticPeptideDataset] = None,
        use_wandb: bool = False,
    ):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.use_wandb = use_wandb

        # Move model to device
        self.model = self.model.to(self.device)

        # Datasets and loaders
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            collate_fn=collate_peptide_samples,
            num_workers=0,  # 0 for IterableDataset
        )

        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                collate_fn=collate_peptide_samples,
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

        # Loss - use combined loss if spectrum matching enabled
        if config.spectrum_weight > 0 or config.use_curriculum:
            self.loss_fn = CombinedLoss(
                ce_weight=config.ce_weight,
                spectrum_weight=config.spectrum_weight,
                iteration_weights=config.iteration_weights,
                label_smoothing=config.label_smoothing,
            )
            self.use_combined_loss = True
        else:
            self.loss_fn = DeepSupervisionLoss(
                iteration_weights=config.iteration_weights,
                label_smoothing=config.label_smoothing,
            )
            self.use_combined_loss = False

        # Curriculum scheduler
        self.curriculum = None
        if config.use_curriculum:
            self.curriculum = CurriculumScheduler(
                stages=DEFAULT_CURRICULUM,
                dataset=train_dataset,
            )

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

        # Initialize wandb if requested
        if self.use_wandb:
            try:
                import wandb
                wandb.init(
                    project="peptide-trm",
                    config=vars(config),
                )
            except ImportError:
                print("Warning: wandb not installed, logging disabled")
                self.use_wandb = False

    def train_step(self, batch):
        """Single training step."""
        self.model.train()

        # Move to device
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # Forward pass
        all_logits, _ = self.model(
            spectrum_masses=batch['spectrum_masses'],
            spectrum_intensities=batch['spectrum_intensities'],
            spectrum_mask=batch['spectrum_mask'],
            precursor_mass=batch['precursor_mass'],
            precursor_charge=batch['precursor_charge'],
        )

        # Update curriculum if enabled
        if self.curriculum:
            if self.curriculum.step(self.global_step):
                # Stage changed, update spectrum loss weight
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

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()
        self.scheduler.step()

        # Update EMA
        if self.ema:
            self.ema.update(self.model)

        # Compute accuracy metrics
        with torch.no_grad():
            acc_metrics = compute_metrics(
                all_logits[-1],
                batch['sequence'],
                batch['sequence_mask'],
                batch['precursor_mass'],
            )

        metrics.update(acc_metrics)
        metrics['lr'] = self.scheduler.get_last_lr()[0]

        return metrics

    @torch.no_grad()
    def evaluate(self, num_batches: int = 10):
        """Evaluate on validation set."""
        if self.val_dataset is None:
            return {}

        model = self.ema.shadow if self.ema else self.model
        model.eval()

        total_correct = 0
        total_tokens = 0
        total_sequences = 0
        correct_sequences = 0

        val_iter = iter(self.val_loader)
        for _ in range(num_batches):
            try:
                batch = next(val_iter)
            except StopIteration:
                break

            batch = {k: v.to(self.device) for k, v in batch.items()}

            logits, _ = model(
                spectrum_masses=batch['spectrum_masses'],
                spectrum_intensities=batch['spectrum_intensities'],
                spectrum_mask=batch['spectrum_mask'],
                precursor_mass=batch['precursor_mass'],
                precursor_charge=batch['precursor_charge'],
            )

            # Use final step logits
            final_logits = logits[-1]

            predictions = final_logits.argmax(dim=-1)
            mask = batch['sequence_mask']
            targets = batch['sequence']

            # Token accuracy
            correct = (predictions == targets) & mask
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()

            # Sequence accuracy
            all_correct = ((predictions == targets) | ~mask).all(dim=-1)
            correct_sequences += all_correct.sum().item()
            total_sequences += len(predictions)

        return {
            'val_token_accuracy': total_correct / max(total_tokens, 1),
            'val_sequence_accuracy': correct_sequences / max(total_sequences, 1),
        }

    def save_checkpoint(self, name: str = None):
        """Save model checkpoint."""
        name = name or f'checkpoint_step_{self.global_step}.pt'
        path = self.checkpoint_dir / name

        state = {
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
        }

        if self.ema:
            state['ema_state_dict'] = self.ema.state_dict()

        torch.save(state, path)
        print(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        state = torch.load(path, map_location=self.device)

        self.global_step = state['global_step']
        self.model.load_state_dict(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        self.scheduler.load_state_dict(state['scheduler_state_dict'])
        self.best_val_acc = state.get('best_val_acc', 0.0)

        if self.ema and 'ema_state_dict' in state:
            self.ema.load_state_dict(state['ema_state_dict'])

        print(f"Loaded checkpoint from step {self.global_step}")

    def train(self):
        """Main training loop."""
        print(f"Starting training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        train_iter = iter(self.train_loader)
        pbar = tqdm(total=self.config.max_steps, desc="Training")

        while self.global_step < self.config.max_steps:
            # Get batch
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            # Train step
            metrics = self.train_step(batch)
            self.global_step += 1
            pbar.update(1)

            # Logging
            if self.global_step % self.config.log_interval == 0:
                log_str = (
                    f"Step {self.global_step} | "
                    f"Loss: {metrics['total_loss']:.4f} | "
                    f"Token Acc: {metrics['token_accuracy']:.3f} | "
                    f"Seq Acc: {metrics['sequence_accuracy']:.3f} | "
                    f"LR: {metrics['lr']:.2e}"
                )
                pbar.set_description(log_str)

                if self.use_wandb:
                    import wandb
                    wandb.log(metrics, step=self.global_step)

            # Evaluation
            if self.global_step % self.config.eval_interval == 0:
                val_metrics = self.evaluate()
                if val_metrics:
                    print(
                        f"\nValidation | "
                        f"Token Acc: {val_metrics['val_token_accuracy']:.3f} | "
                        f"Seq Acc: {val_metrics['val_sequence_accuracy']:.3f}"
                    )

                    if self.use_wandb:
                        import wandb
                        wandb.log(val_metrics, step=self.global_step)

                    # Save best model
                    if val_metrics['val_sequence_accuracy'] > self.best_val_acc:
                        self.best_val_acc = val_metrics['val_sequence_accuracy']
                        self.save_checkpoint('best_model.pt')

            # Periodic checkpoint
            if self.global_step % self.config.save_interval == 0:
                self.save_checkpoint()

        # Final save
        pbar.close()
        self.save_checkpoint('final_model.pt')
        print("Training complete!")
