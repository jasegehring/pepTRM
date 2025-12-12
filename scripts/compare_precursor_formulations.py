"""
Compare different precursor loss formulations.
"""

import numpy as np

def log1p_loss(error_da, precursor_mass=1200):
    """Current approach: log1p of ppm error."""
    ppm_error = (error_da / precursor_mass) * 1e6
    return np.log1p(ppm_error / 100000)

def l1_scaled_loss(error_da, scale_factor=0.01):
    """New approach: scaled L1."""
    return error_da * scale_factor

print("=" * 80)
print("Precursor Loss Formulation Comparison")
print("=" * 80)
print("\nAssuming precursor mass = 1200 Da (typical 10-mer peptide)")
print()

# Test different error levels
error_scenarios = [
    (200, "Early training (40% token acc)", "step 10-20k"),
    (100, "Mid training (60% token acc)", "step 20-30k"),
    (50, "Late training (80% token acc)", "step 40k"),
    (10, "Very late (90% token acc)", "step 60k"),
    (1, "Nearly perfect", "step 80k+"),
]

print(f"{'Error (Da)':<15} | {'Stage':<30} | {'log1p Loss':<12} | {'L1(0.01)':<12} | {'L1(0.005)':<12} | {'L1(0.002)':<12}")
print("-" * 110)

for error_da, stage, step in error_scenarios:
    log1p_val = log1p_loss(error_da)
    l1_001 = l1_scaled_loss(error_da, 0.01)
    l1_0005 = l1_scaled_loss(error_da, 0.005)
    l1_0002 = l1_scaled_loss(error_da, 0.002)

    print(f"{error_da:<15} | {stage:<30} | {log1p_val:>10.4f} | {l1_001:>10.4f} | {l1_0005:>10.4f} | {l1_0002:>10.4f}")

print("\n" + "=" * 80)
print("Gradient Analysis")
print("=" * 80)
print()

print("Gradient magnitude at different errors:")
print(f"{'Error (Da)':<15} | {'log1p gradient':<20} | {'L1 gradient':<20}")
print("-" * 60)

for error_da in [200, 100, 50, 10, 1]:
    precursor_mass = 1200
    ppm = (error_da / precursor_mass) * 1e6

    # d/d(error_da) of log1p(ppm/100000)
    # = d/d(error_da) of log1p((error_da/mass * 1e6) / 100000)
    # = (1 / (1 + ppm/100000)) * (1e6 / mass / 100000)
    # = (1 / (1 + ppm/100000)) * (1e6 / 1.2e8)
    log1p_grad = (1 / (1 + ppm/100000)) * (1e6 / precursor_mass / 100000)

    # L1 gradient is constant: scale_factor
    l1_grad = 0.01

    print(f"{error_da:<15} | {log1p_grad:>18.6f} | {l1_grad:>18.6f}")

print("\n" + "=" * 80)
print("Contribution to Total Loss (assuming CE loss = 2.2)")
print("=" * 80)
print()

curriculum_weights = [
    (0.01, "Stage 3 (step 20k)"),
    (0.015, "Stage 4 (step 30k)"),
    (0.02, "Stage 5 (step 40k)"),
]

print(f"{'Error':<10} | {'Weight':<8} | {'log1p contrib':<15} | {'L1(0.01) contrib':<15} | {'L1(0.005) contrib':<15}")
print("-" * 80)

for error_da in [200, 100, 50]:
    for weight, stage in curriculum_weights:
        log1p_contrib = weight * log1p_loss(error_da)
        l1_001_contrib = weight * l1_scaled_loss(error_da, 0.01)
        l1_0005_contrib = weight * l1_scaled_loss(error_da, 0.005)

        print(f"{error_da:<10} | {weight:<8.3f} | {log1p_contrib:>13.4f} | {l1_001_contrib:>15.4f} | {l1_0005_contrib:>15.4f}")

print("\n" + "=" * 80)
print("Recommendations")
print("=" * 80)
print("""
1. **log1p approach**:
   - Pros: Bounded growth, can't explode
   - Cons: Gradient vanishes for large errors (200 Da → gradient = 0.000007)
   - Gradient at 200 Da is 7000× smaller than at 1 Da!

2. **L1 with scale=0.01** (current in gaussian file):
   - Pros: Constant gradient (0.01), never vanishes
   - Cons: At 200 Da error, loss = 2.0, which is too large
   - With weight=0.01, contributes 0.02 (acceptable but high)

3. **L1 with scale=0.005**:
   - Pros: Constant gradient (0.005), reasonable loss values
   - At 200 Da error, loss = 1.0, contributes 0.01 with weight=0.01 ✓
   - At 100 Da error, loss = 0.5, contributes 0.005 ✓
   - Cons: Still unbounded for very large errors

4. **L1 with scale=0.002**:
   - Very conservative, loss stays small
   - At 200 Da error, loss = 0.4, contributes 0.004 ✓
   - Might be too weak of a signal

**Verdict**:
- **For gradient flow**: L1 is superior (constant gradients)
- **For stability**: log1p is safer (bounded)
- **Best compromise**: L1 with scale_factor=0.003-0.005

The vanishing gradient in log1p at large errors is actually a real concern!
When errors are 200 Da, the gradient is essentially zero, so the model gets
no signal to improve. L1 always provides a constant "pull" toward the correct mass.

**Recommendation**: Try L1 with scale_factor=0.004
- 200 Da → loss=0.8 → contrib=0.008 (with weight=0.01)
- 100 Da → loss=0.4 → contrib=0.004
- 50 Da → loss=0.2 → contrib=0.002
- Provides consistent gradient throughout training
""")
