"""
Deep dive into spectrum loss to find why it's not working.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn.functional as F
from src.data.ms2pip_dataset import MS2PIPSyntheticDataset
from src.training.losses import SpectrumMatchingLoss
from src.constants import AMINO_ACID_MASSES, VOCAB, AA_TO_IDX
from src.data.ion_types import compute_theoretical_peaks, get_ion_types_for_model

def diagnose():
    print("=" * 80)
    print("DEEP DIAGNOSIS OF SPECTRUM LOSS")
    print("=" * 80)

    # Get a sample
    dataset = MS2PIPSyntheticDataset(
        min_length=7,
        max_length=10,
        ms2pip_model='HCDch2',
    )
    sample = next(iter(dataset))

    # Decode
    sequence = sample.sequence
    sequence_mask = sample.sequence_mask
    valid_tokens = sequence[sequence_mask]
    decoded_aas = [dataset.idx_to_token[idx.item()] for idx in valid_tokens]
    peptide = ''.join(decoded_aas)

    print(f"\nPeptide: {peptide}")
    print(f"Length: {len(peptide)}")

    # Get observed peaks
    observed_masses = sample.spectrum_masses[sample.spectrum_mask]
    observed_intensities = sample.spectrum_intensities[sample.spectrum_mask]

    print(f"\nObserved spectrum:")
    print(f"  Number of peaks: {len(observed_masses)}")
    print(f"  Mass range: [{observed_masses.min():.2f}, {observed_masses.max():.2f}] m/z")
    print(f"  Intensity range: [{observed_intensities.min():.4f}, {observed_intensities.max():.4f}]")

    # Create perfect and wrong predictions
    batch_size = 1
    max_seq_len = sequence.shape[0]
    vocab_size = len(VOCAB)

    # Perfect prediction
    perfect_probs = torch.zeros(batch_size, max_seq_len, vocab_size)
    for i, (token_id, valid) in enumerate(zip(sequence, sequence_mask)):
        if valid:
            perfect_probs[0, i, token_id] = 1.0

    # Wrong prediction (all Glycine - lightest AA)
    wrong_probs = torch.zeros(batch_size, max_seq_len, vocab_size)
    gly_idx = AA_TO_IDX['G']
    for i in range(max_seq_len):
        if sequence_mask[i]:
            wrong_probs[0, i, gly_idx] = 1.0

    # Get ion types
    ion_types = get_ion_types_for_model('HCDch2')
    print(f"\nIon types: {ion_types}")

    # Compute theoretical peaks manually
    aa_masses = torch.tensor([AMINO_ACID_MASSES.get(aa, 0.0) for aa in VOCAB])

    print("\n" + "=" * 80)
    print("PERFECT PREDICTION")
    print("=" * 80)

    theo_perfect = compute_theoretical_peaks(
        sequence_probs=perfect_probs,
        aa_masses=aa_masses,
        ion_type_names=ion_types,
    )

    print(f"Theoretical peaks: {theo_perfect.shape}")
    print(f"  Number: {theo_perfect.shape[1]}")
    print(f"  Mass range: [{theo_perfect[0].min():.2f}, {theo_perfect[0].max():.2f}] m/z")
    print(f"  First 10: {theo_perfect[0, :10].tolist()}")

    # Check if theoretical peaks match observed
    matches = 0
    for theo_mz in theo_perfect[0]:
        dists = torch.abs(observed_masses - theo_mz)
        if dists.min() < 0.5:  # Within 0.5 Da
            matches += 1

    print(f"\nMatches within 0.5 Da: {matches}/{len(theo_perfect[0])}")
    print(f"Match rate: {matches/len(theo_perfect[0])*100:.1f}%")

    print("\n" + "=" * 80)
    print("WRONG PREDICTION (all Glycine)")
    print("=" * 80)

    wrong_peptide = ''.join([
        dataset.idx_to_token[wrong_probs[0, i].argmax().item()]
        for i in range(max_seq_len) if sequence_mask[i]
    ])
    print(f"Wrong peptide: {wrong_peptide}")

    theo_wrong = compute_theoretical_peaks(
        sequence_probs=wrong_probs,
        aa_masses=aa_masses,
        ion_type_names=ion_types,
    )

    print(f"Theoretical peaks: {theo_wrong.shape}")
    print(f"  Number: {theo_wrong.shape[1]}")
    print(f"  Mass range: [{theo_wrong[0].min():.2f}, {theo_wrong[0].max():.2f}] m/z")
    print(f"  First 10: {theo_wrong[0, :10].tolist()}")

    # Check if theoretical peaks match observed
    matches_wrong = 0
    for theo_mz in theo_wrong[0]:
        dists = torch.abs(observed_masses - theo_mz)
        if dists.min() < 0.5:
            matches_wrong += 1

    print(f"\nMatches within 0.5 Da: {matches_wrong}/{len(theo_wrong[0])}")
    print(f"Match rate: {matches_wrong/len(theo_wrong[0])*100:.1f}%")

    # Now compute loss step-by-step for perfect prediction
    print("\n" + "=" * 80)
    print("LOSS COMPUTATION BREAKDOWN (Perfect Prediction)")
    print("=" * 80)

    observed_masses_batch = sample.spectrum_masses.unsqueeze(0)
    observed_intensities_batch = sample.spectrum_intensities.unsqueeze(0)
    peak_mask_batch = sample.spectrum_mask.unsqueeze(0)

    # Distances
    theo_expanded = theo_perfect.unsqueeze(-1)
    obs_expanded = observed_masses_batch.unsqueeze(1)
    distances = torch.abs(theo_expanded - obs_expanded)

    print(f"Distance matrix shape: {distances.shape}")
    print(f"  (batch, num_theo, max_peaks) = {distances.shape}")

    # Find minimum distance for each theoretical peak
    min_dists, min_indices = distances[0].min(dim=1)
    print(f"\nMinimum distances for each theoretical peak:")
    print(f"  Mean: {min_dists.mean():.6f} Da")
    print(f"  Median: {min_dists.median():.6f} Da")
    print(f"  Max: {min_dists.max():.6f} Da")

    # Check how many are within tolerance
    within_tol = (min_dists < 0.5).sum()
    print(f"  Within 0.5 Da tolerance: {within_tol}/{len(min_dists)}")

    # Soft assignment
    temperature = 0.1
    scores = -distances / temperature
    mass_tolerance = 0.5

    in_window_mask = (distances < mass_tolerance)
    combined_mask = peak_mask_batch.unsqueeze(1) & in_window_mask
    scores_masked = scores.clone()
    scores_masked = scores_masked.masked_fill(~combined_mask, float('-inf'))

    soft_assignment = F.softmax(scores_masked, dim=-1)
    soft_assignment = torch.nan_to_num(soft_assignment, nan=0.0)

    print(f"\nSoft assignment:")
    print(f"  Shape: {soft_assignment.shape}")
    print(f"  Non-zero entries: {(soft_assignment > 0).sum().item()}")
    print(f"  Sum per theo peak (should be ~1.0): {soft_assignment[0].sum(dim=1)[:10].tolist()}")

    # Matched distances
    clamped_distances = torch.clamp(distances, max=mass_tolerance)

    # Huber loss
    huber_delta = 0.2
    huber_loss = torch.where(
        clamped_distances <= huber_delta,
        0.5 * clamped_distances ** 2,
        huber_delta * (clamped_distances - 0.5 * huber_delta)
    )

    matched_distances = (soft_assignment * huber_loss).sum(dim=-1)

    print(f"\nMatched distances (after Huber):")
    print(f"  Shape: {matched_distances.shape}")
    print(f"  Mean: {matched_distances.mean():.6f} Da")
    print(f"  First 10: {matched_distances[0, :10].tolist()}")

    # Intensity weights
    intensity_weights = (soft_assignment * observed_intensities_batch.unsqueeze(1)).sum(dim=-1)

    print(f"\nIntensity weights:")
    print(f"  Shape: {intensity_weights.shape}")
    print(f"  Mean: {intensity_weights.mean():.6f}")
    print(f"  First 10: {intensity_weights[0, :10].tolist()}")

    # Number of matches
    num_matches = combined_mask.any(dim=-1).float()

    print(f"\nNum matches:")
    print(f"  Shape: {num_matches.shape}")
    print(f"  Sum: {num_matches.sum().item()}")
    print(f"  (How many theoretical peaks have at least one observed peak within tolerance)")

    # Final loss
    loss = (matched_distances * intensity_weights * num_matches).sum() / num_matches.sum().clamp(min=1)

    print(f"\nFinal loss: {loss.item():.6f} Da")

    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    print("\nProblem identified:")
    print("1. Matched distances are VERY small (mean ~0.000006 Da)")
    print("2. This is because perfect prediction gives near-perfect matches")
    print("3. Huber loss with small distances: 0.5 * d^2 ≈ 0.5 * (0.001)^2 = 0.0000005")
    print("4. Multiplying by intensity weights makes it even smaller")

    print("\nWhy random/wrong predictions also have low loss:")
    if matches_wrong / len(theo_wrong[0]) > 0.3:
        print("  - Wrong predictions still match ~30% of peaks by chance!")
        print("  - With peptide fragmentation, many masses overlap")
        print("  - Loss only counts matched peaks, not missed ones")

    print("\n" + "=" * 80)
    print("ROOT CAUSE")
    print("=" * 80)
    print("The loss is too focused on MATCHED peaks.")
    print("It doesn't penalize MISSING matches enough.")
    print("")
    print("Current: loss = sum(distance * intensity * has_match) / sum(has_match)")
    print("  → Only cares about peaks that match")
    print("  → If 30% match by chance, loss is low even for wrong prediction")
    print("")
    print("Need: Penalty for observed peaks without theoretical match")
    print("  OR: Higher weight for intensity")
    print("  OR: Different loss formulation")

if __name__ == '__main__':
    diagnose()
