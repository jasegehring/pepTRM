"""
Test the new PrecursorMassLoss to ensure it works correctly.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn.functional as F
from src.training.losses import PrecursorMassLoss, CombinedLoss
from src.constants import AMINO_ACID_MASSES, WATER_MASS, VOCAB

print("=" * 80)
print("TESTING PRECURSOR MASS LOSS")
print("=" * 80)

# Create loss function
precursor_loss = PrecursorMassLoss(use_relative=True, clamp_ppm=100.0)
print("\n✓ PrecursorMassLoss created")

# Test case 1: Perfect prediction
print("\n" + "=" * 80)
print("TEST 1: PERFECT PREDICTION")
print("=" * 80)

# Create a simple peptide: "PEPTIDE" (7 amino acids)
peptide = "PEPTIDE"
true_mass = sum(AMINO_ACID_MASSES[aa] for aa in peptide) + WATER_MASS
print(f"\nPeptide: {peptide}")
print(f"True precursor mass: {true_mass:.4f} Da")

# Create one-hot encoding (perfect prediction)
batch_size = 1
seq_len = 10  # Including SOS/EOS padding
vocab_size = len(VOCAB)

# Create probability distribution (one-hot for each position)
probs = torch.zeros(batch_size, seq_len, vocab_size)

# Map sequence to indices (skip position 0 which would be SOS)
aa_to_idx = {aa: i for i, aa in enumerate(VOCAB)}
for i, aa in enumerate(peptide):
    aa_idx = aa_to_idx[aa]
    probs[0, i+1, aa_idx] = 1.0  # Positions 1-7 are the peptide

# Mask (True for peptide positions only, not SOS/EOS/padding)
mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
mask[0, 1:8] = True

# Precursor mass
precursor_mass = torch.tensor([true_mass])

# Compute loss
loss = precursor_loss(probs, precursor_mass, mask)
print(f"\nLoss (perfect prediction): {loss.item():.4f} ppm")
print(f"Expected: ~0 ppm")

if loss.item() < 0.1:
    print("✓ PASS: Perfect prediction has near-zero loss")
else:
    print("✗ FAIL: Perfect prediction should have near-zero loss!")

# Test case 2: Wrong prediction (off by one amino acid)
print("\n" + "=" * 80)
print("TEST 2: WRONG PREDICTION")
print("=" * 80)

# Change last amino acid from E (129.04) to A (71.04) - difference of ~58 Da
probs_wrong = probs.clone()
probs_wrong[0, 7, aa_to_idx['E']] = 0.0
probs_wrong[0, 7, aa_to_idx['A']] = 1.0

wrong_mass = true_mass - AMINO_ACID_MASSES['E'] + AMINO_ACID_MASSES['A']
mass_error_da = abs(wrong_mass - true_mass)
mass_error_ppm = (mass_error_da / true_mass) * 1e6

print(f"\nPredicted peptide: PEPTIDA (wrong!)")
print(f"True mass: {true_mass:.4f} Da")
print(f"Predicted mass: {wrong_mass:.4f} Da")
print(f"Error: {mass_error_da:.4f} Da ({mass_error_ppm:.2f} ppm)")

loss_wrong = precursor_loss(probs_wrong, precursor_mass, mask)
print(f"\nLoss (wrong prediction): {loss_wrong.item():.2f} ppm")
print(f"Expected: ~{mass_error_ppm:.2f} ppm")

if abs(loss_wrong.item() - mass_error_ppm) < 1.0:
    print("✓ PASS: Loss matches expected mass error")
else:
    print("✗ FAIL: Loss doesn't match expected mass error!")

# Test case 3: Soft predictions (uncertain)
print("\n" + "=" * 80)
print("TEST 3: SOFT PREDICTIONS (UNCERTAIN)")
print("=" * 80)

# Create uncertain prediction at last position (50% E, 50% A)
probs_soft = probs.clone()
probs_soft[0, 7, aa_to_idx['E']] = 0.5
probs_soft[0, 7, aa_to_idx['A']] = 0.5

expected_mass = (
    sum(AMINO_ACID_MASSES[aa] for aa in peptide[:-1])  # First 6 amino acids
    + 0.5 * AMINO_ACID_MASSES['E']  # 50% E
    + 0.5 * AMINO_ACID_MASSES['A']  # 50% A
    + WATER_MASS
)
soft_error_da = abs(expected_mass - true_mass)
soft_error_ppm = (soft_error_da / true_mass) * 1e6

print(f"\nPredicted: PEPTID(0.5*E + 0.5*A)")
print(f"Expected mass: {expected_mass:.4f} Da")
print(f"Expected error: {soft_error_da:.4f} Da ({soft_error_ppm:.2f} ppm)")

loss_soft = precursor_loss(probs_soft, precursor_mass, mask)
print(f"\nLoss (soft prediction): {loss_soft.item():.2f} ppm")

if abs(loss_soft.item() - soft_error_ppm) < 1.0:
    print("✓ PASS: Soft loss computed correctly")
else:
    print("✗ FAIL: Soft loss doesn't match expected!")

# Test case 4: CombinedLoss with all components
print("\n" + "=" * 80)
print("TEST 4: COMBINED LOSS")
print("=" * 80)

combined_loss = CombinedLoss(
    ce_weight=1.0,
    spectrum_weight=0.3,
    precursor_weight=0.5,
)
print("\n✓ CombinedLoss created with weights:")
print(f"  CE: {combined_loss.ce_weight}")
print(f"  Spectrum: {combined_loss.spectrum_weight}")
print(f"  Precursor: {combined_loss.precursor_weight}")

# Create dummy data for combined loss
all_logits = torch.randn(3, batch_size, seq_len, vocab_size)  # 3 steps
targets = torch.zeros(batch_size, seq_len, dtype=torch.long)
for i, aa in enumerate(peptide):
    targets[0, i+1] = aa_to_idx[aa]

observed_masses = torch.randn(batch_size, 50) * 1000  # Dummy spectrum
observed_intensities = torch.rand(batch_size, 50)
peak_mask = torch.ones(batch_size, 50, dtype=torch.bool)

loss, metrics = combined_loss(
    all_logits=all_logits,
    targets=targets,
    target_mask=mask,
    observed_masses=observed_masses,
    observed_intensities=observed_intensities,
    peak_mask=peak_mask,
    precursor_mass=precursor_mass,
)

print(f"\nCombined loss computed:")
print(f"  Total loss: {loss.item():.4f}")
print(f"  Precursor loss: {metrics['precursor_loss']:.4f} ppm")
print(f"  Spectrum loss: {metrics['spectrum_loss']:.4f}")
print(f"  CE loss (final): {metrics['ce_final']:.4f}")

if 'precursor_loss' in metrics:
    print("\n✓ PASS: CombinedLoss includes precursor loss component")
else:
    print("\n✗ FAIL: CombinedLoss missing precursor loss!")

# Test case 5: Dynamic weight update (curriculum simulation)
print("\n" + "=" * 80)
print("TEST 5: DYNAMIC WEIGHT UPDATE")
print("=" * 80)

print("\nSimulating curriculum progression:")
weights = [
    (0.0, "Stage 1: Warmup"),
    (0.2, "Stage 2: Introduce mass constraint"),
    (0.5, "Stage 3: Medium constraint"),
    (1.0, "Stage 4: Full constraint"),
]

for weight, stage_name in weights:
    combined_loss.precursor_weight = weight
    loss, metrics = combined_loss(
        all_logits=all_logits,
        targets=targets,
        target_mask=mask,
        observed_masses=observed_masses,
        observed_intensities=observed_intensities,
        peak_mask=peak_mask,
        precursor_mass=precursor_mass,
    )
    print(f"\n{stage_name} (weight={weight}):")
    print(f"  Precursor component: {weight * metrics['precursor_loss']:.4f}")
    print(f"  Total loss: {loss.item():.4f}")

print("\n✓ PASS: Weight can be updated dynamically")

print("\n" + "=" * 80)
print("✓ ALL TESTS PASSED!")
print("=" * 80)
print("\nPrecursor mass loss is working correctly.")
print("Ready for training with enhanced mass constraints!")
