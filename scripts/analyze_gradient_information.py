"""
Analyze the information content of auxiliary loss gradients at different accuracy levels.

Key question: When do auxiliary losses provide actionable gradient information?
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import torch.nn.functional as F
from src.constants import VOCAB, AMINO_ACID_MASSES

def simulate_soft_predictions(token_accuracy, seq_length=10, num_samples=1000):
    """
    Simulate soft probability distributions at different accuracy levels.

    Returns:
        entropy: Average entropy of predictions (higher = more uncertain)
        mass_variance: Variance in predicted masses (higher = more unstable)
    """
    aa_list = [aa for aa in VOCAB if aa not in ['<PAD>', '<SOS>', '<EOS>']]
    masses = np.array([AMINO_ACID_MASSES.get(aa, 0.0) for aa in aa_list])
    masses = masses[masses > 0]
    vocab_size = len(masses)

    entropies = []
    mass_variances = []

    for _ in range(num_samples):
        # Generate ground truth sequence
        true_seq_idx = np.random.choice(vocab_size, size=seq_length)

        # Simulate soft predictions at this accuracy level
        # Model is "correct" with probability = token_accuracy
        # Otherwise, outputs softmax over random logits

        probs_list = []
        predicted_masses = []

        for i in range(seq_length):
            if np.random.rand() < token_accuracy:
                # High confidence on correct token
                logits = np.random.randn(vocab_size) * 0.5  # Small noise
                logits[true_seq_idx[i]] += 5.0  # Strong signal for correct token
            else:
                # Random/confused
                logits = np.random.randn(vocab_size) * 2.0

            probs = np.exp(logits) / np.exp(logits).sum()
            probs_list.append(probs)

            # Expected mass at this position
            expected_mass = (probs * masses).sum()
            predicted_masses.append(expected_mass)

            # Entropy (measure of uncertainty)
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            entropies.append(entropy)

        # Variance in predicted mass across sequence
        mass_var = np.var(predicted_masses)
        mass_variances.append(mass_var)

    return {
        'mean_entropy': np.mean(entropies),
        'std_entropy': np.std(entropies),
        'mean_mass_variance': np.mean(mass_variances),
    }

def gradient_signal_strength(token_accuracy):
    """
    Estimate gradient signal strength for spectrum and precursor losses.

    Higher accuracy → stronger, clearer gradients
    """
    # Spectrum loss gradient strength (roughly proportional to confidence)
    # At low accuracy: many possible fragments, gradient is diffuse
    # At high accuracy: few possible fragments, gradient is focused
    spectrum_gradient = token_accuracy ** 2  # Quadratic because depends on fragment pairs

    # Precursor loss gradient strength
    # Depends on how "peaked" the distribution is
    # At low accuracy: uniform distribution, gradient is weak
    # At high accuracy: peaked distribution, gradient is strong
    precursor_gradient = token_accuracy ** 1.5

    return {
        'spectrum_gradient_strength': spectrum_gradient,
        'precursor_gradient_strength': precursor_gradient,
    }

print("=" * 80)
print("Gradient Information Content Analysis")
print("=" * 80)
print("\nQuestion: When do auxiliary losses provide useful gradient information?")
print()

# Test different accuracy levels
accuracy_levels = [
    (0.10, "Very early", "step 2-3k"),
    (0.20, "Early", "step 5-6k"),
    (0.40, "Early-mid", "step 10k (current spectrum intro)"),
    (0.50, "Mid", "step 15k"),
    (0.60, "Mid-late", "step 20k (current precursor intro)"),
    (0.70, "Late-mid", "step 30k"),
    (0.80, "Late", "step 40k"),
    (0.90, "Very late", "step 60k+"),
]

print(f"{'Accuracy':<12} | {'Stage':<15} | {'Entropy':<10} | {'Spec Grad':<12} | {'Prec Grad':<12}")
print("-" * 80)

results = []
for acc, stage, step in accuracy_levels:
    sim = simulate_soft_predictions(acc)
    grad = gradient_signal_strength(acc)

    results.append({
        'accuracy': acc,
        'stage': stage,
        'step': step,
        **sim,
        **grad,
    })

    print(f"{acc*100:>6.0f}%      | {stage:<15} | "
          f"{sim['mean_entropy']:>8.2f} | "
          f"{grad['spectrum_gradient_strength']:>10.3f} | "
          f"{grad['precursor_gradient_strength']:>10.3f}")

print("\n" + "=" * 80)
print("Interpretation")
print("=" * 80)
print("""
**Entropy**: Uncertainty in predictions (lower is better)
  - High entropy (>2.5): Predictions are nearly uniform, gradients are noisy
  - Low entropy (<1.0): Predictions are confident, gradients are clear

**Gradient Strength**: Estimated useful signal (higher is better)
  - <0.2: Very weak, likely to be overwhelmed by noise
  - 0.2-0.5: Moderate, can provide some signal
  - >0.5: Strong, clear directional information
""")

print("\n" + "=" * 80)
print("Key Thresholds")
print("=" * 80)

# Find when gradient becomes "useful"
spectrum_threshold = 0.25
precursor_threshold = 0.30

spectrum_intro_acc = None
precursor_intro_acc = None

for r in results:
    if spectrum_intro_acc is None and r['spectrum_gradient_strength'] > spectrum_threshold:
        spectrum_intro_acc = r
    if precursor_intro_acc is None and r['precursor_gradient_strength'] > precursor_threshold:
        precursor_intro_acc = r

print(f"\nSpectrum loss gradient becomes useful (>{spectrum_threshold}):")
print(f"  At ~{spectrum_intro_acc['accuracy']*100:.0f}% token accuracy ({spectrum_intro_acc['step']})")
print(f"  Gradient strength: {spectrum_intro_acc['spectrum_gradient_strength']:.3f}")

print(f"\nPrecursor loss gradient becomes useful (>{precursor_threshold}):")
print(f"  At ~{precursor_intro_acc['accuracy']*100:.0f}% token accuracy ({precursor_intro_acc['step']})")
print(f"  Gradient strength: {precursor_intro_acc['precursor_gradient_strength']:.3f}")

print("\n" + "=" * 80)
print("Recommendations")
print("=" * 80)
print(f"""
Based on gradient information content:

**Option A: Current curriculum (Early introduction)**
- Spectrum @ step 10k (40% acc)
  - Gradient strength: {results[2]['spectrum_gradient_strength']:.3f} (weak)
  - Entropy: {results[2]['mean_entropy']:.2f} (high uncertainty)
  - Risk: Noisy gradients, potential interference
  - Benefit: Guides learning from the start

**Option B: Delayed introduction (Wait for signal)**
- Spectrum @ step 20k (60% acc)
  - Gradient strength: {results[4]['spectrum_gradient_strength']:.3f} (moderate)
  - Entropy: {results[4]['mean_entropy']:.2f} (lower uncertainty)
  - Risk: May miss early guidance
  - Benefit: Clear, actionable gradients

**Option C: Conservative (Strong signal only)**
- Spectrum @ step 30k (70% acc)
  - Gradient strength: {results[5]['spectrum_gradient_strength']:.3f} (strong)
  - Entropy: {results[5]['mean_entropy']:.2f} (confident predictions)
  - Risk: Model may be in local minimum
  - Benefit: Very strong, clear signal

**Option D: Hybrid approach**
- Spectrum @ step 15k (50% acc) - Fragment matching is easier
- Precursor @ step 30k (70% acc) - Total mass needs high accuracy
- Reasoning: Fragments are fine-grained, total mass is coarse

**My recommendation: Option D (Hybrid)**

Reasoning:
1. Spectrum loss can help even at moderate accuracy (50%)
   - Learning "this peak should match b3 ion" is actionable
   - Fragments are local, less affected by other errors

2. Precursor loss needs high accuracy (70%+)
   - Total mass depends on entire sequence being mostly correct
   - At 40-60% accuracy, soft total mass is too noisy

3. This gives model time to stabilize (15k steps) before adding complexity
   - But not so long that it settles into physics-ignorant patterns
   - Spectrum loss acts as bridge between CE and full physics

4. Matches multitask learning theory
   - Introduce related auxiliary tasks gradually
   - Easier tasks first (fragments), harder tasks later (total mass)
""")

print("\n" + "=" * 80)
print("Proposed New Curriculum")
print("=" * 80)
print("""
Stage 1 (0-15k):   Pure CE, learn token distributions
Stage 2 (15k-25k): Add spectrum loss (weight 0.05-0.10)
Stage 3 (25k-35k): Increase spectrum (0.10-0.15), add precursor (0.01)
Stage 4 (35k+):    Full multi-task learning

This balances:
- Early enough to guide learning (not too late)
- Late enough for clear gradients (not too early)
- Phased introduction (spectrum before precursor)
""")
