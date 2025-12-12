"""
Loss functions including deep supervision and mass-matching auxiliary loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional

from ..constants import AMINO_ACID_MASSES, WATER_MASS, PROTON_MASS, VOCAB, PAD_IDX, CO_MASS
from ..data.ion_types import compute_theoretical_peaks, get_ion_types_for_model, validate_ion_types


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

    Supports flexible ion types (b, y, doubly charged, neutral losses, etc.)
    """

    def __init__(
        self,
        mass_tolerance: float = 0.5,   # Da, for matching window
        temperature: float = 0.1,       # For soft assignment
        use_huber: bool = True,         # Use Huber loss instead of L1
        huber_delta: float = 0.2,       # Da, threshold for Huber loss
        ion_type_names: Optional[List[str]] = None,  # e.g., ['b', 'y', 'b++', 'y++']
        ms2pip_model: Optional[str] = None,  # e.g., 'HCDch2'
    ):
        """
        Args:
            mass_tolerance: Mass tolerance in Daltons for peak matching
            temperature: Temperature for soft assignment
            use_huber: Use Huber loss instead of L1 for robustness
            huber_delta: Threshold for Huber loss
            ion_type_names: List of ion types to use (e.g., ['b', 'y', 'b++', 'y++'])
                           If None, must provide ms2pip_model
            ms2pip_model: MS2PIP model name (e.g., 'HCDch2') to auto-select ion types
                         Ignored if ion_type_names is provided
        """
        super().__init__()
        self.mass_tolerance = mass_tolerance
        self.temperature = temperature
        self.use_huber = use_huber
        self.huber_delta = huber_delta

        # Determine which ion types to use
        if ion_type_names is not None:
            validate_ion_types(ion_type_names)
            self.ion_type_names = ion_type_names
        elif ms2pip_model is not None:
            self.ion_type_names = get_ion_types_for_model(ms2pip_model)
        else:
            # Default fallback to basic b, y ions
            self.ion_type_names = ['b', 'y']
            print(f"⚠️  Warning: No ion_type_names or ms2pip_model specified for SpectrumMatchingLoss. "
                  f"Using default: {self.ion_type_names}")

        # Register amino acid masses as buffer
        aa_masses = torch.tensor([
            AMINO_ACID_MASSES.get(aa, 0.0) for aa in VOCAB
        ])
        self.register_buffer('aa_masses', aa_masses)

    def compute_theoretical_peaks_batch(
        self,
        sequence_probs: Tensor,  # (batch, seq_len, vocab)
    ) -> Tensor:
        """
        Compute expected theoretical peak masses from probability distribution.

        Uses the flexible ion type system to support arbitrary fragment ion types
        (b, y, doubly charged, neutral losses, etc.)

        This is differentiable because we use expected mass:
        E[mass_i] = sum_aa P(aa_i) * mass(aa)

        Returns:
            (batch, num_theoretical_peaks) expected m/z values for configured ion types
        """
        return compute_theoretical_peaks(
            sequence_probs=sequence_probs,
            aa_masses=self.aa_masses,
            ion_type_names=self.ion_type_names,
        )

    def forward(
        self,
        sequence_probs: Tensor,       # (batch, seq_len, vocab)
        observed_masses: Tensor,      # (batch, max_peaks)
        observed_intensities: Tensor, # (batch, max_peaks)
        peak_mask: Tensor,            # (batch, max_peaks)
    ) -> Tensor:
        """
        Compute bidirectional spectrum matching loss.

        FORWARD: For each observed peak, check if theoretical peak exists nearby
        BACKWARD: Penalize predicted sequences that don't explain observed peaks

        This ensures wrong predictions (which won't generate the observed peaks) get high loss.
        """
        # Compute theoretical peaks from soft predictions
        theoretical = self.compute_theoretical_peaks_batch(sequence_probs)  # (batch, num_theo)

        # Compute pairwise distances between theoretical and observed peaks
        # theoretical: (batch, num_theo, 1)
        # observed: (batch, 1, max_peaks)
        theo_expanded = theoretical.unsqueeze(-1)  # (batch, num_theo, 1)
        obs_expanded = observed_masses.unsqueeze(1)  # (batch, 1, max_peaks)

        # Absolute distance: (batch, num_theo, max_peaks)
        distances = torch.abs(theo_expanded - obs_expanded)

        # --- MAIN LOSS: For each OBSERVED peak, find closest THEORETICAL peak ---
        # This penalizes predictions that don't explain the observed spectrum

        # Transpose to (batch, max_peaks, num_theo)
        distances_transposed = distances.transpose(1, 2)  # (batch, max_peaks, num_theo)

        # For each observed peak, find minimum distance to any theoretical peak
        min_distances, _ = distances_transposed.min(dim=-1)  # (batch, max_peaks)

        # Apply Huber-style loss
        if self.use_huber:
            huber_loss = torch.where(
                min_distances <= self.huber_delta,
                0.5 * min_distances ** 2,
                self.huber_delta * (min_distances - 0.5 * self.huber_delta)
            )
        else:
            huber_loss = min_distances

        # Weight by observed peak intensity (high intensity = more important)
        # Normalize intensities to sum to 1 for each sample
        intensity_normalized = observed_intensities / (observed_intensities.sum(dim=-1, keepdim=True).clamp(min=1e-8))

        # Weighted loss over observed peaks
        weighted_loss = (huber_loss * intensity_normalized * peak_mask.float()).sum(dim=-1)

        # Average over batch
        loss = weighted_loss.mean()

        return loss


class PrecursorMassLoss(nn.Module):
    """
    Direct precursor mass constraint loss.

    Penalizes predictions whose total mass doesn't match the given precursor mass.
    This is the most direct way to enforce mass spectrometry physics:
        sum(amino_acid_masses) + H2O = precursor_mass

    The loss is computed on the probability distribution (soft constraint) to maintain
    differentiability and allow the model to explore.
    """

    def __init__(
        self,
        use_relative: bool = True,     # Use relative error (ppm) instead of absolute (Da)
        loss_scale: float = 100000.0,  # Scale parameter for log loss (100k ppm = 10% error)
        use_log_loss: bool = True,     # Use log(1 + error/scale) for bounded gradients
    ):
        super().__init__()
        self.use_relative = use_relative
        self.loss_scale = loss_scale
        self.use_log_loss = use_log_loss

        # Register amino acid masses as buffer
        aa_masses = torch.tensor([
            AMINO_ACID_MASSES.get(aa, 0.0) for aa in VOCAB
        ])
        self.register_buffer('aa_masses', aa_masses)

    def forward(
        self,
        sequence_probs: Tensor,    # (batch, seq_len, vocab)
        precursor_mass: Tensor,    # (batch,)
        sequence_mask: Tensor,     # (batch, seq_len) - True for real positions
    ) -> tuple[Tensor, dict]:
        """
        Compute precursor mass constraint loss.

        For each position, we compute the expected amino acid mass:
            E[mass_i] = sum_aa P(aa_i) * mass(aa)

        Then sum to get predicted precursor mass and compare to ground truth.
        """
        # Expected mass at each position
        # (batch, seq_len, vocab) @ (vocab,) -> (batch, seq_len)
        expected_masses = torch.einsum('bsv,v->bs', sequence_probs, self.aa_masses)

        # Mask out padding and special tokens (SOS/EOS at positions 0 and last)
        # Only sum masses for actual amino acid positions
        masked_masses = expected_masses * sequence_mask.float()

        # Sum to get predicted peptide mass (backbone only)
        predicted_peptide_mass = masked_masses.sum(dim=1)  # (batch,)

        # Add water mass to get full precursor mass
        predicted_precursor_mass = predicted_peptide_mass + WATER_MASS

        # Compute error
        mass_error = torch.abs(predicted_precursor_mass - precursor_mass)  # (batch,)

        if self.use_relative:
            # Convert to ppm (parts per million)
            # ppm = (error / true_mass) * 1e6
            ppm_error = (mass_error / precursor_mass.clamp(min=1.0)) * 1e6

            if self.use_log_loss:
                # Use logarithmic scaling: loss = log(1 + error/scale)
                # This provides:
                # - Bounded growth for large errors (prevents explosion)
                # - Strong gradients for small errors (log(1+x) ≈ x for small x)
                # - Non-zero gradients even for very large errors
                #
                # With loss_scale=100k ppm:
                # - 100k ppm (10% error) → log(2) ≈ 0.69
                # - 10k ppm (1% error) → log(1.1) ≈ 0.095
                # - 1k ppm (0.1% error) → log(1.01) ≈ 0.01
                # - 100 ppm (0.01% error) → log(1.001) ≈ 0.001
                loss = torch.log1p(ppm_error / self.loss_scale)
                loss = loss.mean()
            else:
                # Legacy: clamp extreme errors (kills gradients!)
                ppm_error_clamped = torch.clamp(ppm_error, max=self.loss_scale)
                loss = ppm_error_clamped.mean()
        else:
            # Use absolute error in Daltons
            ppm_error = (mass_error / precursor_mass.clamp(min=1.0)) * 1e6
            loss = mass_error.mean()

        metrics = {
            'predicted_peptide_mass': predicted_peptide_mass.mean().item(),
            'mass_error_da': mass_error.mean().item(),
            'ppm_error': ppm_error.mean().item(),
        }

        return loss, metrics


class CombinedLoss(nn.Module):
    """
    Combined loss with deep supervision, spectrum matching, and precursor mass constraint.
    """

    def __init__(
        self,
        ce_weight: float = 1.0,
        spectrum_weight: float = 0.1,
        precursor_weight: float = 0.0,
        iteration_weights: str = 'linear',
        label_smoothing: float = 0.0,
        mass_tolerance: float = 0.5,
        use_huber: bool = True,
        huber_delta: float = 0.2,
        ion_type_names: Optional[List[str]] = None,
        ms2pip_model: Optional[str] = None,
    ):
        """
        Args:
            ce_weight: Weight for cross-entropy loss
            spectrum_weight: Weight for spectrum matching loss
            precursor_weight: Weight for precursor mass constraint
            iteration_weights: Weighting scheme for supervision steps
            label_smoothing: Label smoothing for cross-entropy
            mass_tolerance: Mass tolerance in Da for spectrum matching
            use_huber: Use Huber loss for spectrum matching
            huber_delta: Threshold for Huber loss
            ion_type_names: List of ion types (e.g., ['b', 'y', 'b++', 'y++'])
            ms2pip_model: MS2PIP model name (e.g., 'HCDch2') - auto-selects ion types
        """
        super().__init__()
        self.ce_weight = ce_weight
        self.spectrum_weight = spectrum_weight
        self.precursor_weight = precursor_weight

        self.ce_loss = DeepSupervisionLoss(
            iteration_weights=iteration_weights,
            label_smoothing=label_smoothing,
        )

        self.spectrum_loss = SpectrumMatchingLoss(
            mass_tolerance=mass_tolerance,
            use_huber=use_huber,
            huber_delta=huber_delta,
            ion_type_names=ion_type_names,
            ms2pip_model=ms2pip_model,
        )

        self.precursor_loss = PrecursorMassLoss(
            use_relative=True,      # Use ppm for interpretability
            loss_scale=100000.0,    # 100k ppm = 10% error scale
            use_log_loss=True,      # Use log scaling (bounded gradients)
        )

    def forward(
        self,
        all_logits: Tensor,
        targets: Tensor,
        target_mask: Tensor,
        observed_masses: Tensor,
        observed_intensities: Tensor,
        peak_mask: Tensor,
        precursor_mass: Tensor = None,
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

        # --- FIX: Ensure auxiliary losses are computed in float32 for stability ---
        with torch.amp.autocast(enabled=False, device_type=ce_loss.device.type):
            spec_loss = self.spectrum_loss(
                final_probs.float(),
                observed_masses.float(),
                observed_intensities.float(),
                peak_mask,
            )

            # Precursor mass constraint (if weight > 0 and precursor_mass provided)
            prec_loss = torch.tensor(0.0, device=ce_loss.device)
            prec_metrics = {}
            if self.precursor_weight > 0 and precursor_mass is not None:
                prec_loss, prec_metrics = self.precursor_loss(
                    final_probs.float(),
                    precursor_mass.float(),
                    target_mask,
                )

        # Combine
        total_loss = (
            self.ce_weight * ce_loss
            + self.spectrum_weight * spec_loss
            + self.precursor_weight * prec_loss
        )

        metrics = {
            **ce_metrics,
            'spectrum_loss': spec_loss.item(),
            'precursor_loss': prec_loss.item() if isinstance(prec_loss, Tensor) else 0.0,
            **prec_metrics,
            'total_loss': total_loss.item(),
        }

        return total_loss, metrics
