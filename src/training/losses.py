"""
Loss functions including deep supervision and mass-matching auxiliary loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..constants import AMINO_ACID_MASSES, WATER_MASS, PROTON_MASS, VOCAB, PAD_IDX, CO_MASS


class DeepSupervisionLoss(nn.Module):
    """
    Cross-entropy loss summed over all supervision steps.

    This is the core TRM training signal - computing loss at every
    intermediate step forces the model to learn a trajectory of improvement.
    """

    def __init__(
        self,
        iteration_weights: str = 'linear',
        label_smoothing: float = 0.0,
    ):
        """
        Args:
            iteration_weights: How to weight different steps
                - 'uniform': All steps weighted equally
                - 'linear': Later steps weighted more (1, 2, 3, ...)
                - 'exponential': Exponential weighting toward later steps
            label_smoothing: Label smoothing for cross-entropy
        """
        super().__init__()
        self.iteration_weights = iteration_weights
        self.label_smoothing = label_smoothing

    def _get_weights(self, num_steps: int, device: torch.device) -> Tensor:
        """Get normalized weights for each step."""
        if self.iteration_weights == 'uniform':
            weights = torch.ones(num_steps, device=device)
        elif self.iteration_weights == 'linear':
            weights = torch.arange(1, num_steps + 1, dtype=torch.float, device=device)
        elif self.iteration_weights == 'exponential':
            weights = torch.exp(torch.arange(num_steps, dtype=torch.float, device=device) * 0.5)
        else:
            raise ValueError(f"Unknown weighting: {self.iteration_weights}")

        return weights / weights.sum()

    def forward(
        self,
        all_logits: Tensor,      # (T, batch, seq_len, vocab)
        targets: Tensor,         # (batch, seq_len)
        target_mask: Tensor,     # (batch, seq_len)
    ) -> tuple[Tensor, dict]:
        """
        Compute deep supervision loss.

        Returns:
            loss: Scalar loss value
            metrics: Dict with per-step losses for logging
        """
        num_steps = all_logits.shape[0]
        batch_size = all_logits.shape[1]
        device = all_logits.device

        weights = self._get_weights(num_steps, device)

        total_loss = 0.0
        step_losses = []

        for t in range(num_steps):
            logits = all_logits[t]  # (batch, seq_len, vocab)

            # Reshape for cross entropy
            logits_flat = logits.view(-1, logits.shape[-1])
            targets_flat = targets.view(-1)

            # Compute cross entropy with mask
            ce_loss = F.cross_entropy(
                logits_flat,
                targets_flat,
                ignore_index=PAD_IDX,
                label_smoothing=self.label_smoothing,
                reduction='none',
            )

            # Apply mask and average
            mask_flat = target_mask.view(-1).float()
            masked_loss = (ce_loss * mask_flat).sum() / mask_flat.sum().clamp(min=1)

            step_losses.append(masked_loss.item())
            total_loss = total_loss + weights[t] * masked_loss

        metrics = {
            f'ce_step_{t}': step_losses[t] for t in range(num_steps)
        }
        metrics['ce_final'] = step_losses[-1]

        return total_loss, metrics


class SpectrumMatchingLoss(nn.Module):
    """
    Auxiliary loss that compares predicted peptide's theoretical spectrum
    to the observed spectrum.

    This exploits the unique structure of peptide sequencing: unlike language
    modeling, we have an implicit verification oracle (mass constraints).

    The loss encourages predictions whose theoretical fragment masses
    match observed peaks.
    """

    def __init__(
        self,
        mass_tolerance: float = 0.5,   # Da, for matching window
        temperature: float = 0.1,       # For soft assignment
        use_huber: bool = True,         # Use Huber loss instead of L1
        huber_delta: float = 0.2,       # Da, threshold for Huber loss
    ):
        super().__init__()
        self.mass_tolerance = mass_tolerance
        self.temperature = temperature
        self.use_huber = use_huber
        self.huber_delta = huber_delta

        # Register amino acid masses as buffer
        aa_masses = torch.tensor([
            AMINO_ACID_MASSES.get(aa, 0.0) for aa in VOCAB
        ])
        self.register_buffer('aa_masses', aa_masses)

    def compute_theoretical_peaks(
        self,
        sequence_probs: Tensor,  # (batch, seq_len, vocab)
    ) -> Tensor:
        """
        Compute expected theoretical peak masses from probability distribution.

        This is differentiable because we use expected mass:
        E[mass_i] = sum_aa P(aa_i) * mass(aa)

        Returns:
            (batch, num_theoretical_peaks) expected masses for b, y, and a ions
        """
        # Expected mass at each position
        # (batch, seq_len, vocab) @ (vocab,) -> (batch, seq_len)
        expected_masses = torch.einsum('bsv,v->bs', sequence_probs, self.aa_masses)

        # Skip position 0 (SOS token) and last position (EOS token)
        residue_masses = expected_masses[:, 1:-1]  # (batch, seq_len - 2)

        # Cumulative sum for b-ions, add proton for ionization
        b_ions = torch.cumsum(residue_masses, dim=1)[:, :-1] + PROTON_MASS  # b1 to b_{n-1}

        # Compute y-ions: cumulative mass from C-terminus + H2O + H+ (ionized)
        y_ions = torch.flip(
            torch.cumsum(torch.flip(residue_masses, [1]), dim=1),
            [1]
        )[:, :-1] + WATER_MASS + PROTON_MASS

        # Compute a-ions: b-ions minus CO
        a_ions = b_ions - CO_MASS

        # Concatenate all theoretical peaks
        theoretical_peaks = torch.cat([b_ions, y_ions, a_ions], dim=1)

        return theoretical_peaks

    def forward(
        self,
        sequence_probs: Tensor,       # (batch, seq_len, vocab)
        observed_masses: Tensor,      # (batch, max_peaks)
        observed_intensities: Tensor, # (batch, max_peaks)
        peak_mask: Tensor,            # (batch, max_peaks)
    ) -> Tensor:
        """
        Compute spectrum matching loss.

        For each theoretical peak, find the closest observed peak and penalize
        the distance. Weight by observed peak intensity (more intense = more reliable).
        """
        # Compute theoretical peaks from soft predictions
        theoretical = self.compute_theoretical_peaks(sequence_probs)  # (batch, num_theo)

        # Compute pairwise distances between theoretical and observed peaks
        # theoretical: (batch, num_theo, 1)
        # observed: (batch, 1, max_peaks)
        theo_expanded = theoretical.unsqueeze(-1)
        obs_expanded = observed_masses.unsqueeze(1)

        # Absolute distance: (batch, num_theo, max_peaks)
        distances = torch.abs(theo_expanded - obs_expanded)

        # Soft assignment: which observed peak matches each theoretical peak?
        # Use softmin over distances, restricted to a tolerance window.
        scores = -distances / self.temperature

        # Create a combined mask: must be a real peak (not padding) AND within mass tolerance.
        in_window_mask = (distances < self.mass_tolerance)
        combined_mask = peak_mask.unsqueeze(1) & in_window_mask
        scores = scores.masked_fill(~combined_mask, float('-inf'))

        # Soft assignment weights
        soft_assignment = F.softmax(scores, dim=-1)

        # Replace NaNs with 0 for cases where a theoretical peak has no valid matches
        soft_assignment = torch.nan_to_num(soft_assignment, nan=0.0)

        # Expected distance for each theoretical peak (weighted by soft assignment)
        # We clamp the distances to the tolerance to avoid penalizing matches outside the window
        # that might get a tiny softmax weight due to temperature.
        clamped_distances = torch.clamp(distances, max=self.mass_tolerance)

        # Apply Huber loss if enabled (more robust to outliers)
        if self.use_huber:
            # Huber loss: quadratic for small errors (|error| <= delta), linear for large errors
            # This is less sensitive to outlier peaks than pure L1 or L2
            huber_loss = torch.where(
                clamped_distances <= self.huber_delta,
                0.5 * clamped_distances ** 2,  # Quadratic for small errors
                self.huber_delta * (clamped_distances - 0.5 * self.huber_delta)  # Linear for large errors
            )
            matched_distances = (soft_assignment * huber_loss).sum(dim=-1)
        else:
            # Standard L1 loss (absolute distance)
            matched_distances = (soft_assignment * clamped_distances).sum(dim=-1)

        # Weight by intensity of matched peaks
        intensity_weights = (soft_assignment * observed_intensities.unsqueeze(1)).sum(dim=-1)

        # Average over theoretical peaks
        # We only want to average over theoretical peaks that had at least one match in the window
        num_matches = combined_mask.any(dim=-1).float()
        loss = (matched_distances * intensity_weights * num_matches).sum() / num_matches.sum().clamp(min=1)

        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss with deep supervision and spectrum matching.
    """

    def __init__(
        self,
        ce_weight: float = 1.0,
        spectrum_weight: float = 0.1,
        iteration_weights: str = 'linear',
        label_smoothing: float = 0.0,
        mass_tolerance: float = 0.5,
        use_huber: bool = True,
        huber_delta: float = 0.2,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.spectrum_weight = spectrum_weight

        self.ce_loss = DeepSupervisionLoss(
            iteration_weights=iteration_weights,
            label_smoothing=label_smoothing,
        )

        self.spectrum_loss = SpectrumMatchingLoss(
            mass_tolerance=mass_tolerance,
            use_huber=use_huber,
            huber_delta=huber_delta,
        )

    def forward(
        self,
        all_logits: Tensor,
        targets: Tensor,
        target_mask: Tensor,
        observed_masses: Tensor,
        observed_intensities: Tensor,
        peak_mask: Tensor,
    ) -> tuple[Tensor, dict]:
        """
        Compute combined loss.

        Returns:
            total_loss: Scalar loss
            metrics: Dict with component losses
        """
        # Cross-entropy with deep supervision
        ce_loss, ce_metrics = self.ce_loss(all_logits, targets, target_mask)

        # Spectrum matching on final prediction
        final_probs = F.softmax(all_logits[-1], dim=-1)
        spec_loss = self.spectrum_loss(
            final_probs,
            observed_masses,
            observed_intensities,
            peak_mask,
        )

        # Combine
        total_loss = self.ce_weight * ce_loss + self.spectrum_weight * spec_loss

        metrics = {
            **ce_metrics,
            'spectrum_loss': spec_loss.item(),
            'total_loss': total_loss.item(),
        }

        return total_loss, metrics
