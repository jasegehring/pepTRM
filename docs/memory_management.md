# Memory Management Guide

## Current Memory Budget (RTX 4090 - 24GB VRAM)

### Memory Breakdown by Component

| Component | Memory Usage | Notes |
|-----------|--------------|-------|
| **Model Parameters** | ~1.5 GB | 12.5M params × 2 bytes (bfloat16) × 2 (params + gradients) |
| **Optimizer States** | ~3 GB | AdamW keeps 2 states (momentum, variance) per param |
| **EMA Model** | ~750 MB | Full copy of model parameters |
| **Activations (per batch)** | ~2-3 GB | Forward pass intermediate tensors (8 supervision + 6 latent steps) |
| **Gaussian Spectrum Loss** | ~800 MB | Large (batch, peaks, bins) tensors for rendering |
| **Precursor Loss** | ~50 MB | Much lighter - just mass calculations |
| **Buffer/Overhead** | ~1 GB | CUDA overhead, intermediate buffers |
| **TOTAL** | ~9-10 GB base + batch-dependent activations |

### Batch Size Impact

With gradient accumulation, we maintain **effective batch size** while reducing **physical batch size**:

| Config | Physical Batch | Accumulation Steps | Effective Batch | Memory per Forward | Total Memory | Status |
|--------|---------------|-------------------|-----------------|-------------------|--------------|--------|
| **Original** | 96 | 1 | 96 | ~14 GB | ~24 GB | ❌ OOM with spectrum loss |
| **Current (Recommended)** | 48 | 2 | 96 | ~7 GB | ~17 GB | ✅ Safe margin for precursor loss |
| **Conservative** | 32 | 3 | 96 | ~5 GB | ~15 GB | ✅ Maximum safety |
| **Aggressive** | 64 | 2 | 128 | ~9 GB | ~19 GB | ⚠️ Risky, but faster training |

## Why Gradient Accumulation is Ideal

**Advantages:**
- ✅ Maintains **identical** training dynamics (same effective batch size)
- ✅ No quality loss (mathematically equivalent to larger batch)
- ✅ Halves memory usage per forward pass
- ✅ Stable gradients (no noise from smaller batches)

**Trade-offs:**
- ⚠️ Slightly slower (2× forward passes per optimizer step)
- ⚠️ But only ~10-20% slower overall due to reduced memory pressure

## Memory-Saving Alternatives

### 1. torch.compile (Worth Trying)

```yaml
# In config
training:
  use_compile: true
  compile_mode: 'reduce-overhead'  # or 'default'
```

**Pros:**
- Can reduce memory via kernel fusion
- May speed up training 1.1-1.3×

**Cons:**
- Compilation time (2-5 min first run)
- May not help or may use MORE memory
- Mixed results with complex loss functions

**Recommendation**: Try it, but gradient accumulation is more reliable.

---

### 2. Reduce Batch Size (Fallback)

```yaml
# If gradient accumulation isn't enough
training:
  batch_size: 32  # or 24
  gradient_accumulation_steps: 3  # or 4 (to maintain effective_batch=96)
```

**When to use:**
- Gradient accumulation alone isn't sufficient
- Still hitting OOM with batch_size=48

---

### 3. Optimize Spectrum Loss Resolution (Last Resort)

```python
# In trainer or loss initialization
spectrum_loss = SpectrumMatchingLoss(
    bin_size=0.2,    # Increase from 0.1 (half the bins)
    max_mz=1500.0,   # Reduce from 2000 (focus on b/y ions)
    # Result: 7500 bins instead of 20000 → 60% memory reduction
)
```

**Pros:**
- Reduces spectrum loss memory by ~60%

**Cons:**
- ❌ May hurt spectrum matching quality
- ❌ Lower resolution = less precise peak matching

**Only use if other methods fail.**

---

### 4. Gradient Checkpointing (Not Recommended Here)

Good for very deep models (100+ layers), but your model is already recurrent and relatively shallow. Won't help much and adds significant compute overhead.

## Planning for Precursor Loss

**Good news**: Precursor loss is lightweight (~50 MB)!

```python
# From losses.py:268-313
# Just computes: einsum + sum + log - very memory efficient
```

**Recommendation**: With current config (batch_size=48, accumulation=2):
- You have ~7 GB headroom
- Precursor loss needs ~50 MB
- **You're safe!** ✅

If you want extra safety:
```yaml
training:
  batch_size: 40
  gradient_accumulation_steps: 2  # effective_batch=80 (slightly less but safer)
```

## Memory Monitoring Commands

```bash
# Check current GPU usage during training
watch -n 1 nvidia-smi

# Monitor specific process
nvidia-smi --query-gpu=memory.used,memory.free --format=csv -l 1

# Python (in training script)
import torch
allocated = torch.cuda.memory_allocated() / 1e9
reserved = torch.cuda.memory_reserved() / 1e9
print(f"Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
```

## Configuration Reference

### Current (Recommended)
```yaml
training:
  batch_size: 48
  gradient_accumulation_steps: 2
  # Effective batch: 96
  # Memory: ~17 GB
  # Speed: ~10% slower than batch_size=96 without accumulation
```

### If Still OOM (Conservative)
```yaml
training:
  batch_size: 32
  gradient_accumulation_steps: 3
  # Effective batch: 96
  # Memory: ~15 GB
  # Speed: ~15% slower
```

### If You Want Faster Training (Risky)
```yaml
training:
  batch_size: 64
  gradient_accumulation_steps: 2
  # Effective batch: 128 (larger!)
  # Memory: ~19 GB
  # Speed: Faster, but may OOM on complex batches
```

## Troubleshooting OOM

**If you still get OOM after these changes:**

1. **Check curriculum stage**: Some stages (longer peptides, more peaks) use more memory
   - Solution: Reduce `max_length` in curriculum stages
   - Or: Use smaller batch_size for later stages

2. **Check peak count**: More peaks = more memory in Gaussian rendering
   - Current: `max_peaks=100` in model config
   - Could reduce to 80-90 if desperate

3. **Verify AMP is active**:
   ```python
   print(f"AMP enabled: {trainer.use_amp}")
   print(f"AMP dtype: {trainer.amp_dtype}")
   ```

4. **Clear cache before training**:
   ```python
   torch.cuda.empty_cache()
   ```

## Summary

✅ **Current fix (gradient accumulation)**: Maintains quality, safe memory
⚠️ **torch.compile**: Optional, may help speed
❌ **Don't reduce spectrum resolution**: Hurts quality
📊 **Precursor loss**: No problem, lightweight

**You're good to go with the current config!**
