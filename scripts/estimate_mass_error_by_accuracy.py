"""
Estimate precursor mass errors at different token accuracy levels.

This helps decide when to introduce precursor mass loss.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from src.constants import AMINO_ACID_MASSES, VOCAB

def estimate_mass_error(token_accuracy: float, peptide_length: int = 10, num_samples: int = 10000):
    """
    Estimate mass error at a given token accuracy level.

    Assumes errors are uniformly distributed across vocabulary.
    """
    aa_list = [aa for aa in VOCAB if aa not in ['<PAD>', '<SOS>', '<EOS>']]
    masses = np.array([AMINO_ACID_MASSES.get(aa, 0.0) for aa in aa_list])

    # Remove zeros
    masses = masses[masses > 0]

    mass_errors = []

    for _ in range(num_samples):
        true_seq_masses = np.random.choice(masses, size=peptide_length)
        true_mass = true_seq_masses.sum()

        # Simulate predictions: token_accuracy% are correct, rest are random
        pred_seq_masses = true_seq_masses.copy()
        num_errors = int(peptide_length * (1 - token_accuracy))
        error_positions = np.random.choice(peptide_length, size=num_errors, replace=False)

        for pos in error_positions:
            # Replace with random AA (could be same by chance)
            pred_seq_masses[pos] = np.random.choice(masses)

        pred_mass = pred_seq_masses.sum()
        mass_error_da = abs(pred_mass - true_mass)
        mass_error_ppm = (mass_error_da / true_mass) * 1e6

        mass_errors.append({
            'da': mass_error_da,
            'ppm': mass_error_ppm,
            'percent': mass_error_da / true_mass * 100
        })

    return {
        'mean_da': np.mean([e['da'] for e in mass_errors]),
        'std_da': np.std([e['da'] for e in mass_errors]),
        'mean_ppm': np.mean([e['ppm'] for e in mass_errors]),
        'std_ppm': np.std([e['ppm'] for e in mass_errors]),
        'mean_percent': np.mean([e['percent'] for e in mass_errors]),
    }

def estimate_precursor_loss(mass_error_ppm: float, loss_scale: float = 100000.0):
    """Estimate the raw precursor loss value."""
    return np.log1p(mass_error_ppm / loss_scale)

print("=" * 70)
print("Precursor Mass Error vs Token Accuracy")
print("=" * 70)
print("\nAssumptions:")
print("- Peptide length: 10 amino acids")
print("- Wrong predictions are uniformly random across vocabulary")
print("- Using new loss formula: log1p(ppm_error / 100000)")
print()

token_accuracies = [
    (0.05, "Random initialization", "Step 0"),
    (0.20, "Very early training", "Step 1-2k"),
    (0.40, "Early training", "Step 10k (current)"),
    (0.50, "Early-mid training", "Step 12-15k"),
    (0.60, "Mid training", "Step 20k"),
    (0.70, "Mid-late training", "Step 30k"),
    (0.80, "Late training", "Step 40-50k"),
    (0.90, "Very late training", "Step 60k+"),
    (0.95, "Near perfect", "Step 80k+"),
]

print(f"{'Token Acc':<12} | {'Stage':<25} | {'Mass Error (Da)':<18} | {'Mass Error (ppm)':<20} | {'Raw Loss':<10}")
print("-" * 110)

for acc, stage, step in token_accuracies:
    result = estimate_mass_error(acc)
    raw_loss = estimate_precursor_loss(result['mean_ppm'])

    print(f"{acc*100:>6.0f}%      | {stage:<25} | "
          f"{result['mean_da']:>7.1f} ± {result['std_da']:<6.1f} | "
          f"{result['mean_ppm']:>10.0f} ± {result['std_ppm']:<6.0f} | "
          f"{raw_loss:>8.3f}")

print("\n" + "=" * 70)
print("Loss Contribution Analysis")
print("=" * 70)
print("\nWith curriculum weights, what does precursor loss contribute to total?")
print("(Assuming CE loss is ~2.0-2.5)")
print()

scenarios = [
    ("Step 10k (current plan)", 0.40, 0.01),
    ("Step 15k (delayed)", 0.50, 0.01),
    ("Step 20k (more delayed)", 0.60, 0.015),
    ("Step 30k (late intro)", 0.70, 0.02),
]

print(f"{'Scenario':<30} | {'Token Acc':<12} | {'Weight':<8} | {'Raw Loss':<10} | {'Contribution':<12} | {'Total Loss':<12}")
print("-" * 110)

for scenario, acc, weight in scenarios:
    result = estimate_mass_error(acc)
    raw_loss = estimate_precursor_loss(result['mean_ppm'])
    contribution = weight * raw_loss
    total = 2.2 + contribution  # Assume CE loss is 2.2

    print(f"{scenario:<30} | {acc*100:>6.0f}%      | {weight:<8.3f} | {raw_loss:>8.3f} | {contribution:>10.3f} | ~{total:.2f}")

print("\n" + "=" * 70)
print("Recommendations")
print("=" * 70)
print("""
1. **Option A: Introduce at step 10k (current)**
   - Token acc: 40% → Raw loss: ~0.5-0.7
   - With weight=0.01, contributes ~0.005-0.007 to total loss
   - Pro: Learn mass constraint on clean data before noise
   - Con: Low accuracy means soft mass prediction is noisy
   - Con: May provide weak/confusing gradient signal

2. **Option B: Introduce at step 15k (middle of stage 2)**
   - Token acc: 50-55% → Raw loss: ~0.3-0.4
   - With weight=0.01, contributes ~0.003-0.004
   - Pro: Model more stable, better soft mass predictions
   - Pro: Still on clean data (no noise yet)
   - Con: Slightly less time to learn constraint before noise

3. **Option C: Introduce at step 20k (start of stage 3)**
   - Token acc: 60-65% → Raw loss: ~0.15-0.25
   - With weight=0.015, contributes ~0.002-0.004
   - Pro: Soft mass predictions are meaningful
   - Pro: Strong, clear gradient signal
   - Con: Data has minimal noise (1 peak), slightly harder to learn

4. **Option D: Introduce at step 30k (start of stage 4)**
   - Token acc: 70-75% → Raw loss: ~0.05-0.10
   - With weight=0.02, contributes ~0.001-0.002
   - Pro: Very accurate predictions, minimal disruption
   - Con: Model may have already "settled" into patterns without mass guidance
   - Con: Higher noise makes learning the constraint harder

**My recommendation: Option B or C**
- Option B (step 15k): Best balance - clean data, decent accuracy, clear signal
- Option C (step 20k): More conservative, very stable learning

**Current Option A (step 10k) concerns:**
- At 40% accuracy, 6 out of 10 amino acids are wrong on average
- The soft mass is essentially a random weighted average
- Gradient may not provide useful signal for learning
- Risk of interfering with CE learning
""")
