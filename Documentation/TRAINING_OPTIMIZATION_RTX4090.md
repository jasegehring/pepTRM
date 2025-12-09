# Training Optimization Guide for RTX 4090

**Document Version**: 1.0
**Date**: December 8, 2025
**Hardware**: NVIDIA GeForce RTX 4090 (24GB VRAM)
**Current Model**: pepTRM (3.7M parameters)

---

## Executive Summary

Current training performance analysis shows significant underutilization of the RTX 4090:
- **GPU Utilization**: 42-45% (target: 85-95%)
- **VRAM Usage**: 6.1GB / 24.5GB (25% utilization)
- **Training Speed**: ~4.1 it/s
- **Estimated Time**: 3.5 hours for 50K steps

**Optimization Potential**: 2-3x speedup achievable through mixed precision training, larger batch sizes, and model compilation.

**Target Performance**:
- **GPU Utilization**: 85-95%
- **VRAM Usage**: 12-16GB (50-65%)
- **Training Speed**: ~10-12 it/s
- **Estimated Time**: 1-1.5 hours for 50K steps

---

## Current Performance Baseline

### Hardware Profile
```
GPU: NVIDIA GeForce RTX 4090
├── VRAM: 24,564 MB total
├── CUDA Cores: 16,384
├── Tensor Cores: 512 (4th gen)
├── Memory Bandwidth: 1,008 GB/s
├── TDP: 450W
└── CUDA Version: 12.2
```

### Current Training Configuration
```yaml
Model:
  Parameters: 3,721,488
  Hidden Dim: 256
  Encoder Layers: 2
  Decoder Layers: 2
  Supervision Steps: 8
  Latent Steps: 4

Training:
  Batch Size: 64
  Precision: FP32 (32-bit)
  Gradient Accumulation: None
  Compilation: None

DataLoader:
  num_workers: 0
  pin_memory: False
  prefetch_factor: None
```

### Measured Performance
```
Metric                    | Current  | Target   | Gap
--------------------------|----------|----------|--------
GPU Utilization           | 42-45%   | 85-95%   | 2.0x
VRAM Usage                | 6.1 GB   | 12-16 GB | 2.0x
Power Draw                | 179W     | 350-400W | 2.0x
Training Speed            | 4.1 it/s | 10-12 it/s| 2.5x
Time per 50K steps        | 3.5 hrs  | 1.2 hrs  | 2.9x
```

**Diagnosis**: The GPU is severely memory-bandwidth-bound and underutilized due to:
1. Small batch size (only 6GB VRAM used)
2. FP32 precision (no tensor core utilization)
3. Uncompiled model (kernel launch overhead)

---

## Optimization Strategies

### 1. Mixed Precision Training (AMP) 🚀 **Highest Priority**

#### Impact
- **Speed**: 2-3x faster (enables tensor cores)
- **Memory**: 40-50% reduction
- **Quality**: No degradation (validated on transformers)

#### Technical Details

**What is Mixed Precision?**
- Forward/backward passes in FP16 (16-bit)
- Master weights and optimizer state in FP32 (32-bit)
- Automatic loss scaling to prevent gradient underflow

**RTX 4090 Advantages**:
- 4th generation tensor cores optimized for FP16/BF16
- 2-3x higher TFLOPS for FP16 vs FP32
- Dedicated hardware for mixed precision operations

**Implementation** (PyTorch AMP):
```python
from torch.cuda.amp import autocast, GradScaler

class Trainer:
    def __init__(self, ...):
        # ... existing code ...

        # Add AMP scaler
        self.use_amp = True
        self.scaler = GradScaler() if self.use_amp else None

    def train_step(self, batch):
        self.optimizer.zero_grad()

        # Automatic mixed precision context
        with autocast(enabled=self.use_amp):
            # Forward pass in FP16
            outputs = self.model(
                peak_mzs=batch['peak_mzs'],
                peak_intensities=batch['peak_intensities'],
                precursor_mz=batch['precursor_mz'],
                precursor_charge=batch['precursor_charge'],
            )

            # Loss computation in FP16
            loss, metrics = self.loss_fn(
                all_logits=outputs,
                target=batch['sequence'],
                target_mask=batch['mask'],
                peak_mzs=batch['peak_mzs'],
                precursor_mz=batch['precursor_mz'],
                precursor_charge=batch['precursor_charge'],
            )

        # Backward pass with gradient scaling
        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        return loss, metrics
```

**Configuration Changes**:
```python
# In TrainingConfig dataclass
@dataclass
class TrainingConfig:
    # ... existing fields ...

    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = 'float16'  # or 'bfloat16' for RTX 4090
```

**Validation Plan**:
1. Train 1,000 steps with AMP enabled
2. Train 1,000 steps with AMP disabled
3. Compare metrics:
   - Token accuracy
   - Sequence accuracy
   - Mean mass error (ppm)
   - Loss convergence

**Expected Results**: < 0.5% difference in final metrics

---

### 2. Increase Batch Size 💪 **High Priority**

#### Impact
- **Speed**: 1.5-2x faster (better GPU utilization)
- **Memory**: Uses available VRAM efficiently
- **Quality**: More stable gradients, potentially better convergence

#### Current vs Optimized

```
Configuration     | Batch Size | VRAM Usage | It/s  | GPU Util
------------------|------------|------------|-------|----------
Current (FP32)    | 64         | 6.1 GB     | 4.1   | 42%
With AMP (FP16)   | 64         | 3.5 GB     | 8.0   | 65%
With AMP + Larger | 128        | 6.5 GB     | 10.0  | 80%
With AMP + Larger | 192        | 9.5 GB     | 11.0  | 85%
With AMP + Larger | 256        | 12.5 GB    | 12.0  | 90%
With AMP + Max    | 384        | 18.0 GB    | 12.5  | 95%
```

**Recommended**: Batch size 192-256 (sweet spot for speed/memory)

#### Implementation

```yaml
# configs/optimized.yaml
training:
  batch_size: 192  # 3x larger than current
  # ... other settings ...
```

**Gradient Accumulation Alternative** (if single large batch causes issues):
```python
@dataclass
class TrainingConfig:
    batch_size: int = 64  # Physical batch size
    gradient_accumulation_steps: int = 3  # Effective batch = 192
```

```python
def train_step(self, batch, step):
    # Forward + backward
    with autocast(enabled=self.use_amp):
        outputs = self.model(...)
        loss = self.loss_fn(...) / self.config.gradient_accumulation_steps

    if self.use_amp:
        self.scaler.scale(loss).backward()
    else:
        loss.backward()

    # Only update every N steps
    if (step + 1) % self.config.gradient_accumulation_steps == 0:
        if self.use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.optimizer.zero_grad()
```

**Validation**:
- Monitor validation loss every 1K steps
- Ensure convergence rate is similar or better
- May need to adjust learning rate slightly (typically scale by sqrt(batch_ratio))

---

### 3. Model Compilation (torch.compile) ⚡ **Medium Priority**

#### Impact
- **Speed**: 1.2-1.5x faster
- **Memory**: Slightly higher (graph caching)
- **Quality**: Identical (optimization only)

#### Technical Details

PyTorch 2.0+ introduced `torch.compile()` which:
- Fuses operations (reduces kernel launches)
- Optimizes memory access patterns
- Uses TorchInductor backend (CUDA graph optimization)

**Implementation**:
```python
# In train.py
def main():
    # ... create model ...

    model = create_model(model_config)

    # Compile model (PyTorch 2.0+)
    if hasattr(torch, 'compile'):
        print("Compiling model with torch.compile()...")
        model = torch.compile(
            model,
            mode='max-autotune',  # or 'reduce-overhead', 'default'
            fullgraph=True,       # try to compile entire forward pass
        )
        print("✓ Model compiled")

    # ... rest of training ...
```

**Compilation Modes**:
```python
# Option 1: Maximum performance (recommended for production)
model = torch.compile(model, mode='max-autotune')

# Option 2: Faster compilation, good performance
model = torch.compile(model, mode='reduce-overhead')

# Option 3: Balanced (default)
model = torch.compile(model, mode='default')
```

**First Run Behavior**:
- First training step will be slow (~30-60 seconds) while compiling
- Subsequent steps will be fast
- Compiled graph is cached

**Validation**:
- Verify identical outputs with/without compilation
- Check for any dynamic shape warnings
- Monitor memory usage (may increase slightly)

---

### 4. DataLoader Optimizations 🔧 **Low Priority**

#### Impact
- **Speed**: 5-10% faster
- **Memory**: Minimal increase
- **Quality**: No effect

Since you're using synthetic data generation (fast), these have minimal impact but are free optimizations:

```python
self.train_loader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    collate_fn=collate_peptide_samples,
    num_workers=0,        # Keep 0 for IterableDataset
    pin_memory=True,      # Add: faster CPU->GPU transfer
    persistent_workers=False,  # Not needed with num_workers=0
)
```

**Note**: `num_workers > 0` not compatible with IterableDataset, but `pin_memory=True` still helps.

---

### 5. Additional Optimizations (Future)

#### Gradient Checkpointing
For even larger models (not needed now):
```python
# Trade compute for memory
from torch.utils.checkpoint import checkpoint

def forward(self, x):
    # Recompute activations during backward instead of storing
    x = checkpoint(self.layer1, x)
    x = checkpoint(self.layer2, x)
    return x
```

**Impact**: 30-50% memory reduction, 20% slower
**When to use**: When you need to fit a model that doesn't fit in VRAM

#### Flash Attention
For attention-heavy models:
```python
# Requires: pip install flash-attn
from flash_attn import flash_attn_func

# Replace standard attention with Flash Attention 2
# 2-4x faster attention, lower memory
```

**Impact**: 2-4x faster attention operations
**When to use**: When attention is >30% of runtime (check with profiler)

#### Channels Last Memory Format
For conv-heavy models:
```python
model = model.to(memory_format=torch.channels_last)
```

**Impact**: 10-20% faster convolutions
**When to use**: CNNs or models with conv layers (not applicable here)

---

## Complete Optimized Implementation

### File: `src/training/trainer_optimized.py`

```python
"""
Optimized training loop for RTX 4090.
Includes: AMP, larger batches, torch.compile support.
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
from .curriculum import CurriculumScheduler, DEFAULT_CURRICULUM


@dataclass
class TrainingConfig:
    """Optimized training configuration for RTX 4090."""
    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 192  # Increased from 64
    max_steps: int = 50000
    warmup_steps: int = 1000

    # Mixed Precision (NEW)
    use_amp: bool = True
    amp_dtype: str = 'float16'  # or 'bfloat16'

    # Gradient Accumulation (alternative to large batch)
    gradient_accumulation_steps: int = 1

    # Model Compilation (NEW)
    use_compile: bool = True
    compile_mode: str = 'max-autotune'  # 'default', 'reduce-overhead', 'max-autotune'

    # Loss weights
    ce_weight: float = 1.0
    spectrum_weight: float = 0.0
    iteration_weights: str = 'linear'
    label_smoothing: float = 0.0

    # Curriculum
    use_curriculum: bool = False

    # EMA
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


class OptimizedTrainer:
    """
    Optimized trainer for RTX 4090.

    Features:
    - Mixed precision training (AMP)
    - Larger batch sizes
    - Model compilation (torch.compile)
    - Pin memory for faster transfers
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

        # Compile model if requested
        if config.use_compile and hasattr(torch, 'compile'):
            print(f"Compiling model with mode='{config.compile_mode}'...")
            print("(First training step will be slow while compiling)")
            model = torch.compile(
                model,
                mode=config.compile_mode,
                fullgraph=False,  # Allow fallback for dynamic ops
            )
            print("✓ Model compiled")

        self.model = model.to(self.device)

        # Mixed precision setup
        self.use_amp = config.use_amp and config.device == 'cuda'
        self.scaler = GradScaler(enabled=self.use_amp) if self.use_amp else None

        if self.use_amp:
            print(f"✓ Mixed precision training enabled (dtype={config.amp_dtype})")

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

        # Initialize wandb
        if self.use_wandb:
            try:
                import wandb
                wandb.init(
                    project="peptide-trm",
                    config=vars(config),
                    tags=['optimized', 'rtx4090', 'amp'] if self.use_amp else ['optimized', 'rtx4090'],
                )
            except ImportError:
                print("Warning: wandb not installed, logging disabled")
                self.use_wandb = False

    def train_step(self, batch):
        """Single training step with mixed precision."""
        self.model.train()

        # Move batch to device
        batch = {k: v.to(self.device) if torch.is_tensor(v) else v
                 for k, v in batch.items()}

        # Mixed precision forward pass
        with autocast(enabled=self.use_amp, dtype=torch.float16):
            # Forward
            outputs = self.model(
                peak_mzs=batch['peak_mzs'],
                peak_intensities=batch['peak_intensities'],
                precursor_mz=batch['precursor_mz'],
                precursor_charge=batch['precursor_charge'],
            )

            # Loss computation
            if self.use_combined_loss:
                # Update spectrum loss weight from curriculum
                if self.curriculum:
                    self.loss_fn.spectrum_weight = self.curriculum.get_spectrum_loss_weight()

                loss, loss_dict = self.loss_fn(
                    all_logits=outputs,
                    target=batch['sequence'],
                    target_mask=batch['mask'],
                    peak_mzs=batch['peak_mzs'],
                    precursor_mz=batch['precursor_mz'],
                    precursor_charge=batch['precursor_charge'],
                )
            else:
                loss, loss_dict = self.loss_fn(
                    all_logits=outputs,
                    target=batch['sequence'],
                    target_mask=batch['mask'],
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
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            self.optimizer.zero_grad()
            self.scheduler.step()

            # Update EMA
            if self.ema:
                self.ema.update(self.model)

        # Compute metrics (in FP32 for accuracy)
        with torch.no_grad():
            final_logits = outputs[-1].float()  # Last iteration
            metrics = compute_metrics(
                logits=final_logits,
                target=batch['sequence'],
                target_mask=batch['mask'],
            )

        return loss.item(), {**loss_dict, **metrics}

    def train(self):
        """Main training loop."""
        print(f"Starting training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        if self.use_amp:
            print(f"Mixed precision: ENABLED (FP16)")
        if self.config.use_compile:
            print(f"Compilation: ENABLED (mode={self.config.compile_mode})")

        pbar = tqdm(total=self.config.max_steps, desc='Training')

        while self.global_step < self.config.max_steps:
            for batch in self.train_loader:
                # Curriculum update
                if self.curriculum:
                    self.curriculum.step(self.global_step)

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
                        wandb.log({
                            'train/loss': loss,
                            'train/token_accuracy': metrics.get('token_accuracy', 0),
                            'train/sequence_accuracy': metrics.get('sequence_accuracy', 0),
                            'train/learning_rate': self.scheduler.get_last_lr()[0],
                            **{f'train/{k}': v for k, v in metrics.items()},
                        }, step=self.global_step)

                # Validation
                if self.global_step % self.config.eval_interval == 0 and self.val_dataset:
                    val_metrics = self.evaluate()
                    tqdm.write(f"Validation | Token Acc: {val_metrics['token_accuracy']:.3f} | "
                              f"Seq Acc: {val_metrics['sequence_accuracy']:.3f}")

                    if self.use_wandb:
                        import wandb
                        wandb.log({
                            f'val/{k}': v for k, v in val_metrics.items()
                        }, step=self.global_step)

                # Checkpointing
                if self.global_step % self.config.save_interval == 0:
                    self.save_checkpoint(f'checkpoint_step_{self.global_step}.pt')

                self.global_step += 1
                pbar.update(1)

                if self.global_step >= self.config.max_steps:
                    break

        # Final checkpoint
        self.save_checkpoint('final_model.pt')
        pbar.close()

    @torch.no_grad()
    def evaluate(self):
        """Evaluate on validation set."""
        self.model.eval()

        total_metrics = {}
        num_batches = 0

        for batch in self.val_loader:
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v
                     for k, v in batch.items()}

            with autocast(enabled=self.use_amp, dtype=torch.float16):
                outputs = self.model(
                    peak_mzs=batch['peak_mzs'],
                    peak_intensities=batch['peak_intensities'],
                    precursor_mz=batch['precursor_mz'],
                    precursor_charge=batch['precursor_charge'],
                )

            final_logits = outputs[-1].float()
            metrics = compute_metrics(
                logits=final_logits,
                target=batch['sequence'],
                target_mask=batch['mask'],
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
        }

        if self.ema:
            checkpoint['ema_state_dict'] = self.ema.state_dict()

        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint, self.checkpoint_dir / filename)
        print(f"Saved checkpoint: {filename}")
```

### File: `scripts/train_optimized.py`

```python
"""
Optimized training entry point for RTX 4090.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from omegaconf import OmegaConf

from src.model.trm import TRMConfig, create_model
from src.data.dataset import SyntheticPeptideDataset
from src.training.trainer_optimized import OptimizedTrainer, TrainingConfig


def main():
    # Load config
    config_path = project_root / 'configs' / 'optimized.yaml'
    cfg = OmegaConf.load(config_path)

    print("=" * 60)
    print("Recursive Peptide Model - Optimized Training (RTX 4090)")
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
    train_dataset = SyntheticPeptideDataset(
        min_length=cfg.data.min_length,
        max_length=cfg.data.max_length,
        max_peaks=model_config.max_peaks,
        max_seq_len=model_config.max_seq_len,
        ion_types=cfg.data.ion_types,
        include_neutral_losses=cfg.data.include_neutral_losses,
        noise_peaks=cfg.data.noise_peaks,
        peak_dropout=cfg.data.peak_dropout,
        mass_error_ppm=cfg.data.mass_error_ppm,
        intensity_variation=cfg.data.intensity_variation,
        charge_distribution=dict(cfg.data.charge_distribution),
    )

    val_dataset = SyntheticPeptideDataset(
        min_length=cfg.data.min_length,
        max_length=cfg.data.max_length,
        max_peaks=model_config.max_peaks,
        max_seq_len=model_config.max_seq_len,
        ion_types=cfg.data.ion_types,
    )

    print(f"\nDataset created:")
    print(f"  Peptide length: {cfg.data.min_length}-{cfg.data.max_length}")
    print(f"  Ion types: {cfg.data.ion_types}")

    # Create optimized trainer
    training_config = TrainingConfig(**cfg.training)

    # Auto-detect device
    if torch.cuda.is_available():
        training_config.device = 'cuda'
        print(f"\n✓ CUDA available - using GPU")

        # Print GPU info
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU: {gpu_name}")
        print(f"  VRAM: {gpu_memory:.1f} GB")
    else:
        training_config.device = 'cpu'
        print(f"\n! No GPU available - using CPU")

    trainer = OptimizedTrainer(
        model=model,
        train_dataset=train_dataset,
        config=training_config,
        val_dataset=val_dataset,
        use_wandb=True,
    )

    print(f"\nOptimized training configuration:")
    print(f"  Batch size: {training_config.batch_size}")
    print(f"  Mixed precision: {training_config.use_amp}")
    print(f"  Model compilation: {training_config.use_compile}")
    print(f"  Learning rate: {training_config.learning_rate}")
    print(f"  Max steps: {training_config.max_steps}")
    print(f"  Device: {training_config.device}")

    print("\n" + "=" * 60)
    print("Starting optimized training...")
    print("=" * 60 + "\n")

    # Train
    trainer.train()

    print("\n" + "=" * 60)
    print("Training finished!")
    print("=" * 60)


if __name__ == '__main__':
    main()
```

### File: `configs/optimized.yaml`

```yaml
# Optimized configuration for RTX 4090

model:
  hidden_dim: 256
  num_encoder_layers: 2
  num_decoder_layers: 2
  num_heads: 4
  max_peaks: 100
  max_seq_len: 25
  num_supervision_steps: 8
  num_latent_steps: 4
  dropout: 0.1

training:
  # Optimization
  learning_rate: 1.0e-4
  weight_decay: 0.01
  batch_size: 192  # 3x larger (optimized for RTX 4090)
  max_steps: 50000
  warmup_steps: 1000

  # Mixed Precision (NEW)
  use_amp: true
  amp_dtype: 'float16'  # or 'bfloat16'

  # Gradient Accumulation
  gradient_accumulation_steps: 1  # Set to >1 if batch too large

  # Model Compilation (NEW)
  use_compile: true
  compile_mode: 'max-autotune'  # 'default', 'reduce-overhead', 'max-autotune'

  # Loss weights
  ce_weight: 1.0
  spectrum_weight: 0.0
  iteration_weights: 'linear'
  label_smoothing: 0.0

  # Curriculum
  use_curriculum: true

  # EMA
  use_ema: true
  ema_decay: 0.999

  # Logging
  log_interval: 100
  eval_interval: 1000
  save_interval: 5000

  # Paths
  checkpoint_dir: 'checkpoints_optimized'

  # Device (auto-detected)
  device: 'cuda'

data:
  min_length: 7
  max_length: 10
  ion_types: ['b', 'y']
  include_neutral_losses: false
  noise_peaks: 0
  peak_dropout: 0.0
  mass_error_ppm: 0.0
  intensity_variation: 0.0
  charge_distribution:
    2: 0.7
    3: 0.3
```

---

## Migration & Testing Plan

### Phase 1: Validation (No Production Impact)

**Goal**: Verify optimizations don't degrade model quality

```bash
# 1. Create optimized config
cp configs/default.yaml configs/optimized.yaml
# Edit: batch_size=192, use_amp=true, use_compile=true

# 2. Run short training (1000 steps) with optimizations
python scripts/train_optimized.py --max_steps 1000

# 3. Run short training (1000 steps) without optimizations
python scripts/train.py --max_steps 1000

# 4. Compare final metrics
# Expected: <0.5% difference in token/sequence accuracy
```

### Phase 2: Performance Benchmarking

**Goal**: Measure actual speedup

```bash
# Benchmark baseline (FP32, batch=64)
time python scripts/train.py --max_steps 1000

# Benchmark with AMP only (FP16, batch=64)
time python scripts/train_optimized.py --max_steps 1000 --batch_size 64

# Benchmark with AMP + larger batch (FP16, batch=192)
time python scripts/train_optimized.py --max_steps 1000 --batch_size 192

# Benchmark with all optimizations (FP16, batch=192, compile)
time python scripts/train_optimized.py --max_steps 1000
```

**Expected Results**:
```
Configuration              | Time (1000 steps) | Speedup
---------------------------|-------------------|--------
Baseline (FP32, bs=64)     | ~240s            | 1.0x
AMP only (FP16, bs=64)     | ~120s            | 2.0x
AMP + batch (FP16, bs=192) | ~100s            | 2.4x
All optimized (compile)    | ~80s             | 3.0x
```

### Phase 3: Full Training Run

**Goal**: Complete 50K step training with optimizations

```bash
# Run full training
python scripts/train_optimized.py

# Monitor GPU utilization
watch -n 1 nvidia-smi

# Expected completion: 1-1.5 hours (vs 3.5 hours baseline)
```

### Phase 4: Quality Validation

**Goal**: Ensure final model quality matches or exceeds baseline

```python
# scripts/compare_models.py
"""Compare baseline vs optimized model."""

import torch
from src.model.trm import create_model
from src.data.dataset import SyntheticPeptideDataset

# Load both checkpoints
baseline_ckpt = torch.load('checkpoints/final_model.pt')
optimized_ckpt = torch.load('checkpoints_optimized/final_model.pt')

# Evaluate on test set
# ... (implementation)

# Compare metrics:
# - Token accuracy
# - Sequence accuracy
# - Mean mass error
# - Spectrum loss

# Expected: Optimized model should be within 0.5% of baseline
```

---

## Monitoring & Debugging

### GPU Utilization Monitoring

```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Log to file every second
nvidia-smi dmon -s u -c 3600 > gpu_utilization.log &

# Target metrics:
# - GPU utilization: 85-95%
# - Memory usage: 12-16 GB
# - Power draw: 350-400W
# - Temperature: <80°C
```

### Performance Profiling

```python
# Add to training script
import torch.profiler

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profiler'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    for step in range(10):
        train_step(batch)
        prof.step()

# View in TensorBoard
tensorboard --logdir=./log/profiler
```

### Common Issues & Solutions

#### Issue: Out of Memory (OOM)

**Symptoms**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Reduce batch size: 192 → 128 → 96
2. Enable gradient accumulation instead
3. Disable torch.compile temporarily
4. Check for memory leaks (detach intermediate tensors)

```python
# Gradient accumulation config
training:
  batch_size: 64
  gradient_accumulation_steps: 3  # Effective batch = 192
```

#### Issue: Slow Compilation

**Symptoms**: First step takes >2 minutes

**Solutions**:
1. Use `compile_mode='reduce-overhead'` instead of `'max-autotune'`
2. Set `fullgraph=False` to allow partial compilation
3. Skip compilation for debugging

```python
# Faster compilation
model = torch.compile(model, mode='reduce-overhead', fullgraph=False)
```

#### Issue: NaN/Inf Losses

**Symptoms**: Loss becomes NaN during training

**Solutions**:
1. Check gradient scaler is working
2. Add gradient clipping (already in optimized trainer)
3. Reduce learning rate
4. Check for numerical instability in loss function

```python
# Debug NaN
torch.autograd.set_detect_anomaly(True)

# Or add to training loop
if torch.isnan(loss):
    print("NaN detected!")
    torch.save(batch, 'nan_batch.pt')
    break
```

#### Issue: Slower than Expected

**Symptoms**: Speed < 10 it/s with optimizations

**Checklist**:
- [ ] Is AMP actually enabled? Check logs
- [ ] Is model compiled? Check for compilation message
- [ ] Is GPU being used? Check `nvidia-smi`
- [ ] Are you using pin_memory?
- [ ] Is batch size large enough?
- [ ] Check for CPU bottlenecks (profiler)

---

## Performance Expectations

### Baseline (Current)
```
Configuration: FP32, batch=64, no compile
GPU Util: 42-45%
VRAM: 6.1 GB
Speed: 4.1 it/s
Time (50K): 3.5 hours
```

### Optimized (Target)
```
Configuration: FP16, batch=192, compiled
GPU Util: 85-95%
VRAM: 12-16 GB
Speed: 10-12 it/s
Time (50K): 1-1.5 hours
```

### Speedup Breakdown
```
Optimization              | Individual | Cumulative
--------------------------|------------|------------
Baseline                  | 1.0x       | 1.0x
+ Mixed Precision (AMP)   | 2.0x       | 2.0x
+ Larger Batch (192)      | 1.3x       | 2.6x
+ torch.compile()         | 1.2x       | 3.1x
--------------------------|------------|------------
Total Speedup             |            | ~3.0x
```

---

## Cost-Benefit Analysis

### Development Time
- Implementation: 2-3 hours
- Testing: 1-2 hours
- Validation: 1 hour
**Total**: ~5 hours one-time cost

### Time Savings Per Training Run
- Baseline: 3.5 hours
- Optimized: 1.2 hours
**Savings**: 2.3 hours per run

### Break-even Point
- After 3 training runs (~7 hours saved)
- ROI improves with each subsequent run

### Additional Benefits
- Faster iteration cycles
- Can run more experiments in same time
- Lower electricity costs (faster completion)
- Better GPU utilization (more work done)

---

## Future Optimization Opportunities

### 1. Distributed Training (Multi-GPU)
If you add more GPUs:
```python
# DDP (DistributedDataParallel)
from torch.nn.parallel import DistributedDataParallel as DDP

# Expected speedup: ~linear with # GPUs
# 2x RTX 4090: ~6x faster than baseline
```

### 2. Model Architecture Optimizations
- Replace standard attention with Flash Attention 2 (2-4x faster)
- Use rotary positional embeddings (RoPE) instead of learned
- Optimize layer norms (RMSNorm is faster)

### 3. Data Pipeline Optimizations
- Pre-generate spectra (if dataset is fixed)
- Use faster data formats (HDF5, TFRecord)
- Implement data caching

### 4. Hyperparameter Optimization
With faster training, you can afford to:
- Run larger learning rate sweeps
- Test different model sizes
- Optimize curriculum schedule

---

## Appendix: Hardware Specifications

### RTX 4090 Architecture
```
GPU: NVIDIA GeForce RTX 4090 (Ada Lovelace)
├── CUDA Cores: 16,384
├── Tensor Cores: 512 (4th Gen)
├── RT Cores: 128 (3rd Gen)
├── Base Clock: 2.23 GHz
├── Boost Clock: 2.52 GHz
├── Memory: 24 GB GDDR6X
├── Memory Bandwidth: 1,008 GB/s
├── Memory Interface: 384-bit
├── TDP: 450W
├── Process: TSMC 4N (5nm)
└── Compute Capability: 8.9

Performance:
├── FP32 (CUDA): 82.6 TFLOPS
├── FP16 (Tensor): 165.2 TFLOPS (2x)
├── INT8 (Tensor): 330.3 TOPS (4x)
└── Sparsity: 2x additional (with structured sparsity)
```

### Why RTX 4090 is Ideal for ML
1. **Massive Tensor Core throughput** (165 TFLOPS FP16)
2. **Large VRAM** (24GB for large batches)
3. **High memory bandwidth** (1TB/s)
4. **Excellent power efficiency** (165 TFLOPS / 450W = 0.37 TFLOPS/W)

---

## References & Resources

### Documentation
- [PyTorch AMP Tutorial](https://pytorch.org/docs/stable/amp.html)
- [torch.compile() Guide](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
- [NVIDIA Mixed Precision Training](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/)

### Research Papers
- "Mixed Precision Training" (Micikevicius et al., ICLR 2018)
- "Automatic Mixed Precision for Deep Learning" (NVIDIA, 2020)

### Community Resources
- [Hugging Face Performance Guide](https://huggingface.co/docs/transformers/performance)
- [PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)

---

**Document End**

For questions or issues with optimization implementation, please refer to the debugging section or create a GitHub issue.
