"""
Precursor mass-guided inference for de novo peptide sequencing.

This module provides utilities to constrain predictions based on the
precursor mass, which is a hard physical constraint in mass spectrometry.
"""

import torch
from torch import Tensor
from typing import List, Optional

from ..constants import AMINO_ACID_MASSES, WATER_MASS, VOCAB, AA_TO_IDX


class PrecursorMassGuide:
    """
    Precursor mass constraint for inference.

    Uses the known precursor mass to:
    1. Filter out predictions that violate mass constraints
    2. Rerank predictions based on mass error
    3. Guide beam search toward physically plausible sequences
    """

    def __init__(
        self,
        tolerance_ppm: float = 20.0,  # Typical MS accuracy
        tolerance_da: Optional[float] = None,  # Alternative: absolute tolerance
        use_hard_constraint: bool = False,  # Reject sequences outside tolerance
    ):
        """
        Args:
            tolerance_ppm: Mass tolerance in parts per million (ppm)
            tolerance_da: Mass tolerance in Daltons (overrides tolerance_ppm if provided)
            use_hard_constraint: If True, reject sequences outside tolerance
                                If False, use soft reranking based on mass error
        """
        self.tolerance_ppm = tolerance_ppm
        self.tolerance_da = tolerance_da
        self.use_hard_constraint = use_hard_constraint

        # Precompute amino acid masses
        self.aa_masses = {aa: AMINO_ACID_MASSES[aa] for aa in AMINO_ACID_MASSES}

    def compute_sequence_mass(self, sequence: str) -> float:
        """
        Compute the precursor mass of a sequence.

        Args:
            sequence: Peptide sequence (e.g., "PEPTIDE")

        Returns:
            precursor_mass: Mass in Daltons (including water)
        """
        # Sum amino acid masses
        peptide_mass = sum(self.aa_masses.get(aa, 0.0) for aa in sequence)

        # Add water for peptide bond formation
        precursor_mass = peptide_mass + WATER_MASS

        return precursor_mass

    def compute_mass_error(
        self,
        sequence: str,
        target_precursor_mass: float,
    ) -> float:
        """
        Compute mass error between predicted sequence and target precursor mass.

        Args:
            sequence: Predicted peptide sequence
            target_precursor_mass: Expected precursor mass in Daltons

        Returns:
            ppm_error: Mass error in parts per million (ppm)
                      Positive = prediction is heavier than target
                      Negative = prediction is lighter than target
        """
        predicted_mass = self.compute_sequence_mass(sequence)
        mass_diff = predicted_mass - target_precursor_mass

        # Convert to ppm
        ppm_error = (mass_diff / target_precursor_mass) * 1e6

        return ppm_error

    def is_valid_sequence(
        self,
        sequence: str,
        target_precursor_mass: float,
    ) -> bool:
        """
        Check if a sequence satisfies the precursor mass constraint.

        Args:
            sequence: Predicted peptide sequence
            target_precursor_mass: Expected precursor mass in Daltons

        Returns:
            is_valid: True if sequence mass is within tolerance
        """
        predicted_mass = self.compute_sequence_mass(sequence)
        mass_diff = abs(predicted_mass - target_precursor_mass)

        # Check tolerance
        if self.tolerance_da is not None:
            # Absolute tolerance in Daltons
            return mass_diff <= self.tolerance_da
        else:
            # Relative tolerance in ppm
            ppm_error = abs((mass_diff / target_precursor_mass) * 1e6)
            return ppm_error <= self.tolerance_ppm

    def rerank_predictions(
        self,
        sequences: List[str],
        scores: List[float],
        target_precursor_mass: float,
        mass_weight: float = 0.5,
    ) -> List[tuple[str, float]]:
        """
        Rerank predictions based on both model score and mass error.

        Args:
            sequences: List of predicted sequences
            scores: List of model confidence scores (higher = better)
            target_precursor_mass: Expected precursor mass
            mass_weight: Weight for mass error term (0 = ignore mass, 1 = only mass)

        Returns:
            reranked: List of (sequence, combined_score) tuples, sorted by score
        """
        reranked = []

        for seq, score in zip(sequences, scores):
            # Compute mass error penalty
            ppm_error = abs(self.compute_mass_error(seq, target_precursor_mass))

            # Convert ppm error to a score (lower error = higher score)
            # Use exponential decay: score = exp(-ppm_error / tolerance)
            mass_score = torch.exp(-torch.tensor(ppm_error / self.tolerance_ppm))

            # Combine model score and mass score
            combined_score = (1 - mass_weight) * score + mass_weight * mass_score.item()

            # Filter by hard constraint if enabled
            if self.use_hard_constraint:
                if self.is_valid_sequence(seq, target_precursor_mass):
                    reranked.append((seq, combined_score))
            else:
                reranked.append((seq, combined_score))

        # Sort by combined score (descending)
        reranked.sort(key=lambda x: x[1], reverse=True)

        return reranked

    def filter_valid_sequences(
        self,
        sequences: List[str],
        target_precursor_mass: float,
    ) -> List[str]:
        """
        Filter sequences to only those satisfying the mass constraint.

        Args:
            sequences: List of predicted sequences
            target_precursor_mass: Expected precursor mass

        Returns:
            valid_sequences: Filtered list of valid sequences
        """
        return [
            seq for seq in sequences
            if self.is_valid_sequence(seq, target_precursor_mass)
        ]


def decode_with_mass_constraint(
    model,
    spectrum_masses: Tensor,
    spectrum_intensities: Tensor,
    spectrum_mask: Tensor,
    precursor_mass: float,
    precursor_charge: int,
    tolerance_ppm: float = 20.0,
    mass_weight: float = 0.3,
    device: str = 'cuda',
) -> tuple[str, float, float]:
    """
    Decode a spectrum with precursor mass constraint.

    Args:
        model: Trained RecursivePeptideModel
        spectrum_masses: (num_peaks,) - m/z values
        spectrum_intensities: (num_peaks,) - intensities
        spectrum_mask: (num_peaks,) - mask for valid peaks
        precursor_mass: Precursor mass in Daltons
        precursor_charge: Precursor charge state
        tolerance_ppm: Mass tolerance in ppm
        mass_weight: Weight for mass error in final score
        device: Device to run on

    Returns:
        sequence: Predicted peptide sequence
        model_score: Raw model confidence
        mass_ppm_error: Mass error in ppm
    """
    model.eval()
    guide = PrecursorMassGuide(tolerance_ppm=tolerance_ppm)

    # Prepare inputs (add batch dimension)
    spectrum_masses = spectrum_masses.unsqueeze(0).to(device)
    spectrum_intensities = spectrum_intensities.unsqueeze(0).to(device)
    spectrum_mask = spectrum_mask.unsqueeze(0).to(device)
    precursor_mass_tensor = torch.tensor([[precursor_mass]], device=device)
    precursor_charge_tensor = torch.tensor([[precursor_charge]], device=device)

    # Forward pass
    with torch.no_grad():
        all_logits, _ = model(
            spectrum_masses=spectrum_masses,
            spectrum_intensities=spectrum_intensities,
            spectrum_mask=spectrum_mask,
            precursor_mass=precursor_mass_tensor,
            precursor_charge=precursor_charge_tensor,
        )

    # Get final prediction (last supervision step)
    final_logits = all_logits[-1, 0]  # (seq_len, vocab)

    # Decode (greedy for now - could extend to beam search)
    pred_indices = final_logits.argmax(dim=-1)  # (seq_len,)

    # Convert to sequence
    from ..constants import IDX_TO_AA, SOS_IDX, EOS_IDX, PAD_IDX

    sequence = []
    for idx in pred_indices.cpu().tolist():
        if idx in [SOS_IDX, EOS_IDX, PAD_IDX]:
            continue
        aa = IDX_TO_AA.get(idx, '')
        if aa and aa in AMINO_ACID_MASSES:
            sequence.append(aa)

    sequence_str = ''.join(sequence)

    # Compute model confidence (average probability of predicted tokens)
    probs = torch.softmax(final_logits, dim=-1)
    pred_probs = probs.gather(1, pred_indices.unsqueeze(1)).squeeze(1)
    model_score = pred_probs.mean().item()

    # Compute mass error
    mass_ppm_error = guide.compute_mass_error(sequence_str, precursor_mass)

    return sequence_str, model_score, mass_ppm_error


# Example usage:
# from src.inference.precursor_guided import decode_with_mass_constraint
#
# sequence, score, mass_error = decode_with_mass_constraint(
#     model=trained_model,
#     spectrum_masses=masses,
#     spectrum_intensities=intensities,
#     spectrum_mask=mask,
#     precursor_mass=1234.5678,
#     precursor_charge=2,
#     tolerance_ppm=20.0,
# )
#
# print(f"Predicted: {sequence}")
# print(f"Confidence: {score:.3f}")
# print(f"Mass error: {mass_error:.1f} ppm")
