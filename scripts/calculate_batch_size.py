"""
Calculate maximum safe batch size for RTX 4090.
"""

# Model specs
model_params = 12_482_816
param_size = 4  # bytes (fp32)

# RTX 4090 specs
total_vram = 24 * 1024**3  # 24 GB in bytes
safety_margin = 0.15  # Reserve 15% for overhead

usable_vram = total_vram * (1 - safety_margin)

print("=" * 80)
print("Batch Size Calculator for RTX 4090")
print("=" * 80)
print()

# Fixed costs
model_memory = model_params * param_size
optimizer_memory = model_memory * 2  # AdamW stores momentum + variance
gradient_memory = model_memory

fixed_memory = model_memory + optimizer_memory + gradient_memory

print("Fixed memory costs:")
print(f"  Model weights:    {model_memory / 1024**3:.3f} GB")
print(f"  Optimizer states: {optimizer_memory / 1024**3:.3f} GB")
print(f"  Gradients:        {gradient_memory / 1024**3:.3f} GB")
print(f"  TOTAL FIXED:      {fixed_memory / 1024**3:.3f} GB")
print()

# Per-sample costs (estimated)
# These depend on sequence length, but let's use typical values
seq_len = 35  # max_seq_len
d_model = 256
num_layers = 6
num_iterations = 4  # TRM iterations

# Activation memory per sample (rough estimate)
# Encoder: seq_len × d_model × num_layers × num_iterations
# Decoder: seq_len × d_model × num_layers × num_iterations
# Attention: seq_len² × num_heads (stored for backprop)

encoder_activations = seq_len * d_model * num_layers * num_iterations * param_size
decoder_activations = seq_len * d_model * num_layers * num_iterations * param_size
attention_activations = (seq_len ** 2) * 8 * num_layers * param_size  # 8 heads

activations_per_sample = encoder_activations + decoder_activations + attention_activations

# Mixed precision reduces activation memory by ~40%
activations_per_sample_fp16 = activations_per_sample * 0.6

print("Per-sample memory (with mixed precision):")
print(f"  Activations: {activations_per_sample_fp16 / 1024**2:.2f} MB")
print()

# Calculate max batch sizes
remaining_memory = usable_vram - fixed_memory

print(f"Available for activations: {remaining_memory / 1024**3:.2f} GB")
print()

# Test different batch sizes
print("=" * 80)
print("Batch Size Analysis")
print("=" * 80)
print()

batch_sizes = [64, 96, 128, 160, 192, 224, 256, 320, 384]

print(f"{'Batch Size':<12} | {'Activation Memory':<18} | {'Total Memory':<15} | {'Status':<20}")
print("-" * 80)

for batch_size in batch_sizes:
    activation_memory = activations_per_sample_fp16 * batch_size
    total_memory = fixed_memory + activation_memory
    utilization = (total_memory / usable_vram) * 100

    if total_memory < usable_vram * 0.7:
        status = "✓ SAFE"
    elif total_memory < usable_vram * 0.85:
        status = "✓ OK (comfortable)"
    elif total_memory < usable_vram:
        status = "⚠ TIGHT (risky)"
    else:
        status = "✗ OOM (too large)"

    print(f"{batch_size:<12} | {activation_memory / 1024**3:>10.2f} GB      | "
          f"{total_memory / 1024**3:>8.2f} GB     | {status:<20}")

print()
print("=" * 80)
print("Recommendations")
print("=" * 80)
print()

# Find safe max
safe_batch = None
comfortable_batch = None

for batch_size in reversed(batch_sizes):
    activation_memory = activations_per_sample_fp16 * batch_size
    total_memory = fixed_memory + activation_memory

    if total_memory < usable_vram * 0.85 and comfortable_batch is None:
        comfortable_batch = batch_size
    if total_memory < usable_vram * 0.7 and safe_batch is None:
        safe_batch = batch_size

print(f"Conservative (70% VRAM): batch_size = {safe_batch}")
print(f"  - Safest choice, plenty of headroom")
print(f"  - Memory usage: ~{(fixed_memory + activations_per_sample_fp16 * safe_batch) / 1024**3:.1f} GB / 24 GB")
print()

print(f"Recommended (85% VRAM): batch_size = {comfortable_batch}")
print(f"  - Good balance of speed and safety")
print(f"  - Memory usage: ~{(fixed_memory + activations_per_sample_fp16 * comfortable_batch) / 1024**3:.1f} GB / 24 GB")
print()

# Check user's request
user_batch = 192
activation_memory = activations_per_sample_fp16 * user_batch
total_memory = fixed_memory + activation_memory

print(f"Your requested batch_size = 192:")
if total_memory < usable_vram * 0.85:
    print(f"  ✓ SAFE - Memory usage: ~{total_memory / 1024**3:.1f} GB / 24 GB")
    print(f"  Utilization: {(total_memory / usable_vram) * 100:.1f}%")
else:
    print(f"  ⚠ RISKY - Memory usage: ~{total_memory / 1024**3:.1f} GB / 24 GB")
    print(f"  Utilization: {(total_memory / usable_vram) * 100:.1f}%")
    print(f"  Recommend reducing to {comfortable_batch}")

print()
print("=" * 80)
print("Notes")
print("=" * 80)
print("""
- These are estimates; actual usage may vary by ±10%
- Gaussian spectral rendering adds ~1 GB (accounted for in safety margin)
- Longer sequences use more memory (max_seq_len=35 assumed)
- Mixed precision (AMP) saves ~40% activation memory
- PyTorch may reserve extra memory for caching
""")
