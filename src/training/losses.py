"""
Loss functions for peptide sequencing.

Combines the best approaches:
1. Gaussian Spectrum Rendering - low variance, bounded loss, smooth gradients
2. Log-scaled Precursor Loss - robust gradients, interpretable ppm metrics
3. Deep Supervision - trajectory-based learning across all refinement steps

This implementation merges improvements from previous experiments to create
optimal training signals with low variance and strong convergence properties.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional, Tuple, Dict

from ..constants import AMINO_ACID_MASSES, WATER_MASS, VOCAB, PAD_IDX
from ..data.ion_types import compute_theoretical_peaks, get_ion_types_for_model, validate_ion_types


class DeepSupervisionLoss(nn.Module):
    """
    Cross-entropy loss summed over all supervision steps.

    Forces the model to learn a trajectory of improvement by supervising
    every intermediate recursive step.
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
        """Calculate normalized weights for each timestep."""
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
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Compute deep supervision loss.

        Returns:
            loss: Scalar loss value
            metrics: Dict with per-step losses for logging
        """
        num_steps = all_logits.shape[0]
        weights = self._get_weights(num_steps, all_logits.device)

        total_loss = 0.0
        step_losses = []

        # Flatten targets once for efficiency
        targets_flat = targets.view(-1)
        mask_flat = target_mask.view(-1).float()
        mask_sum = mask_flat.sum().clamp(min=1)

        for t in range(num_steps):
            logits_flat = all_logits[t].view(-1, all_logits[t].shape[-1])

            ce_loss = F.cross_entropy(
                logits_flat,
                targets_flat,
                ignore_index=PAD_IDX,
                label_smoothing=self.label_smoothing,
                reduction='none',
            )

            masked_loss = (ce_loss * mask_flat).sum() / mask_sum

            step_losses.append(masked_loss.item())
            total_loss = total_loss + weights[t] * masked_loss

        metrics = {f'ce_step_{t}': val for t, val in enumerate(step_losses)}
        metrics['ce_final'] = step_losses[-1]

        return total_loss, metrics


class SpectrumMatchingLoss(nn.Module):
    """
    Differentiable Spectrum Matching using Gaussian Kernel Density Estimation.

    Instead of matching peaks by hard distance (which has poor gradients and high variance),
    we 'render' the predicted masses into a spectral bin vector using Gaussian kernels
    and compare it to the observed spectrum using Cosine Similarity.

    Benefits over min-distance matching:
    - Bounded loss [0, 2] (vs unbounded min-distance)
    - Low variance (normalized spectra)
    - Smooth gradients (Gaussian kernels vs hard min operations)
    - Scale-invariant (cosine similarity focuses on spectral shape)
    """

    def __init__(
        self,
        bin_size: float = 0.1,         # Resolution of the spectral grid (Da)
        max_mz: float = 2000.0,        # Maximum m/z to consider
        sigma: float = 0.2,            # Gaussian width for peak matching (Da)
        ion_type_names: Optional[List[str]] = None,
        ms2pip_model: Optional[str] = None,
    ):
        """
        Args:
            bin_size: Resolution of spectral grid in Daltons (0.1 Da = good for MS/MS)
            max_mz: Maximum m/z value to consider
            sigma: Gaussian kernel width (typically bin_size/2 for sharp peaks)
            ion_type_names: List of ion types (e.g., ['b', 'y', 'b++', 'y++'])
            ms2pip_model: MS2PIP model name (e.g., 'HCDch2') to auto-select ion types
        """
        super().__init__()
        self.bin_size = bin_size
        self.max_mz = max_mz
        self.sigma = sigma

        # Create grid buffer
        num_bins = int(max_mz / bin_size)
        self.register_buffer('mz_grid', torch.linspace(0, max_mz, num_bins).view(1, 1, -1))

        # Setup Ion Types
        if ion_type_names is not None:
            validate_ion_types(ion_type_names)
            self.ion_type_names = ion_type_names
        elif ms2pip_model is not None:
            self.ion_type_names = get_ion_types_for_model(ms2pip_model)
        else:
            self.ion_type_names = ['b', 'y']
            print(f"⚠️  SpectrumMatchingLoss: No ion types specified, defaulting to {self.ion_type_names}")

        # AA masses buffer
        aa_masses = torch.tensor([AMINO_ACID_MASSES.get(aa, 0.0) for aa in VOCAB])
        self.register_buffer('aa_masses', aa_masses)

    def _gaussian_render(self, masses: Tensor, intensities: Optional[Tensor] = None) -> Tensor:
        """
        Render masses into a spectrum using Gaussian kernels.

        Args:
            masses: (batch, num_peaks)
            intensities: (batch, num_peaks) or None (assumes 1.0)
        Returns:
            spectrum: (batch, num_bins)
        """
        # Expand dims for broadcasting: (batch, peaks, 1) - (1, 1, bins)
        diff = masses.unsqueeze(-1) - self.mz_grid

        # Compute Gaussian activations: exp(-0.5 * (x - mu)^2 / sigma^2)
        activations = torch.exp(-0.5 * (diff / self.sigma) ** 2)

        # Apply intensities if provided
        if intensities is not None:
            activations = activations * intensities.unsqueeze(-1)

        # Sum over peaks to get total spectrum
        spectrum = activations.sum(dim=1)

        return spectrum

    def forward(
        self,
        sequence_probs: Tensor,       # (batch, seq_len, vocab)
        observed_masses: Tensor,      # (batch, max_peaks)
        observed_intensities: Tensor, # (batch, max_peaks)
        peak_mask: Tensor,            # (batch, max_peaks)
        sequence_mask: Optional[Tensor] = None,  # (batch, seq_len)
    ) -> Tensor:
        """
        Compute spectrum matching loss via soft peak matching.

        Instead of comparing rendered spectra (which fails due to intensity mismatch),
        we directly measure how well predicted peaks explain observed peaks using
        Gaussian kernels. This approach is intensity-agnostic for predicted peaks.

        Args:
            sequence_probs: Probability distribution over sequences
            observed_masses: Observed peak m/z values
            observed_intensities: Observed peak intensities
            peak_mask: Mask for valid observed peaks
            sequence_mask: Mask for valid sequence positions (prevents PAD from contributing)

        Returns:
            loss: Scalar in range [0, 1], where 0 = perfect coverage
        """
        # 1. Compute Predicted Theoretical Peaks (Differentiable)
        # Shape: (batch, num_predicted_peaks)
        predicted_masses = compute_theoretical_peaks(
            sequence_probs=sequence_probs,
            aa_masses=self.aa_masses,
            ion_type_names=self.ion_type_names,
            sequence_mask=sequence_mask,
        )

        # 2. Soft Peak Matching: For each observed peak, compute coverage by predicted peaks
        # Compute pairwise distances: (batch, num_observed, num_predicted)
        # observed: (batch, max_peaks) -> (batch, max_peaks, 1)
        # predicted: (batch, num_predicted) -> (batch, 1, num_predicted)
        mass_diff = observed_masses.unsqueeze(-1) - predicted_masses.unsqueeze(1)

        # Gaussian kernel: how well does each predicted peak explain each observed peak
        # Shape: (batch, num_observed, num_predicted)
        match_scores = torch.exp(-0.5 * (mass_diff / self.sigma) ** 2)

        # For each observed peak, take max match score across all predicted peaks
        # This gives: "how well is this observed peak explained by ANY predicted peak"
        # Shape: (batch, num_observed)
        peak_coverage = match_scores.max(dim=-1)[0]

        # 3. Weight by observed intensities (important peaks should be explained better)
        # Normalize intensities to sum to 1 for each sample
        obs_intens_norm = observed_intensities * peak_mask.float()
        intens_sum = obs_intens_norm.sum(dim=1, keepdim=True).clamp(min=1e-8)
        obs_weights = obs_intens_norm / intens_sum

        # Weighted coverage score: average of how well important peaks are explained
        # Shape: (batch,)
        weighted_coverage = (peak_coverage * obs_weights).sum(dim=1)

        # 4. Loss is 1 - coverage (perfect coverage = 0 loss)
        loss = 1.0 - weighted_coverage.mean()

        return loss


class PrecursorMassLoss(nn.Module):
    """
    Precursor mass constraint scaled in 'Amino Acid Units'.

    Philosophy:
    - CrossEntropy loss penalizes wrong tokens with ~2.0-5.0 loss
    - Normalize mass error by ~110 Da (avg AA mass) so 1 AA error ≈ 1.0 loss
    - This keeps gradients balanced without magic weights
    - Uses Smooth L1 (Huber) for robustness to outliers

    Example:
        - 1 AA wrong (110 Da error) → loss ≈ 1.0 (same magnitude as CE)
        - 2 AA wrong (220 Da error) → loss ≈ 2.0
        - Small errors (< 11 Da) → quadratic (smooth gradients)
        - Large errors (> 11 Da) → linear (bounded gradients)
    """

    def __init__(self, avg_aa_mass: float = 110.0, huber_beta: float = 0.1):
        """
        Args:
            avg_aa_mass: Average amino acid mass for normalization (default: 110 Da)
            huber_beta: Beta parameter for Smooth L1 loss (in AA units)
                       beta=0.1 means errors < 11 Da are quadratic, > 11 Da are linear
        """
        super().__init__()
        self.avg_aa_mass = avg_aa_mass
        self.huber_beta = huber_beta

        # AA masses - special tokens (PAD, SOS, EOS, UNK) have mass 0.0
        # This is intentional: if model predicts EOS, it contributes 0 mass
        aa_masses = torch.tensor([AMINO_ACID_MASSES.get(aa, 0.0) for aa in VOCAB])
        self.register_buffer('aa_masses', aa_masses)

    def forward(
        self,
        sequence_probs: Tensor,    # (batch, seq_len, vocab)
        precursor_mass: Tensor,    # (batch,)
        sequence_mask: Tensor,     # (batch, seq_len) - True for valid positions
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Compute precursor mass constraint loss in AA units.

        Key insight: We do NOT mask special tokens. The aa_masses buffer has
        mass=0 for PAD/SOS/EOS/UNK. If model correctly predicts EOS at end of
        sequence, it contributes 0 mass - which is correct! If model wrongly
        predicts EOS at an AA position, expected mass drops, loss increases.

        Returns:
            loss: Scalar loss value (in AA units, comparable to CE loss)
            metrics: Dict with soft (expected) and hard (argmax) errors
        """
        # === EXPECTED MASS (for loss - gradients flow through this) ===
        # E[mass_i] = sum_token P(token_i) * mass(token)
        # Special tokens have mass=0, so they naturally contribute nothing
        expected_masses = torch.einsum('bsv,v->bs', sequence_probs, self.aa_masses)
        expected_peptide_mass = (expected_masses * sequence_mask.float()).sum(dim=1)
        expected_precursor_mass = expected_peptide_mass + WATER_MASS

        # === PREDICTED MASS (for metrics - what we actually care about) ===
        argmax_indices = sequence_probs.argmax(dim=-1)
        argmax_masses = self.aa_masses[argmax_indices]
        predicted_peptide_mass = (argmax_masses * sequence_mask.float()).sum(dim=1)
        predicted_precursor_mass = predicted_peptide_mass + WATER_MASS

        # === LOSS ===
        error_da = expected_precursor_mass - precursor_mass
        loss = F.smooth_l1_loss(
            error_da / self.avg_aa_mass,
            torch.zeros_like(error_da),
            beta=self.huber_beta,
            reduction='mean'
        )

        # === METRICS ===
        # Soft error: what loss optimizes (high when uncertain)
        soft_error_da = error_da.abs()
        # Hard error: actual prediction quality (what matters)
        hard_error_da = (predicted_precursor_mass - precursor_mass).abs()

        metrics = {
            # Primary: actual prediction error
            'mass_error_da': hard_error_da.mean().item(),
            # Secondary: explains gap between loss and prediction quality
            'expected_mass_error_da': soft_error_da.mean().item(),
        }

        return loss, metrics


class CombinedLoss(nn.Module):
    """
    Combined loss with all components.

    Integrates:
    - Deep Supervision (CE loss at all refinement steps)
    - Gaussian Spectrum Matching (low variance, bounded)
    - Log-scaled Precursor Loss (robust gradients, interpretable metrics)
    """

    def __init__(
        self,
        ce_weight: float = 1.0,
        spectrum_weight: float = 0.1,
        precursor_weight: float = 0.0,
        iteration_weights: str = 'linear',
        label_smoothing: float = 0.0,
        # Spectrum args
        bin_size: float = 0.1,
        max_mz: float = 2000.0,
        sigma: float = 0.2,  # INCREASED: More forgiving peak matching (was 0.05)
        ion_type_names: Optional[List[str]] = None,
        ms2pip_model: Optional[str] = None,
        # Precursor args
        avg_aa_mass: float = 110.0,
        huber_beta: float = 0.1,
    ):
        """
        Args:
            ce_weight: Weight for cross-entropy loss
            spectrum_weight: Weight for spectrum matching loss
            precursor_weight: Weight for precursor mass loss
            iteration_weights: Weighting scheme for supervision steps
            label_smoothing: Label smoothing for cross-entropy
            bin_size: Spectral grid resolution in Da
            max_mz: Maximum m/z for spectrum rendering
            sigma: Gaussian kernel width for peak matching
            ion_type_names: Ion types to use (e.g., ['b', 'y', 'b++', 'y++'])
            ms2pip_model: MS2PIP model name (e.g., 'HCDch2')
            avg_aa_mass: Average AA mass for precursor loss normalization
            huber_beta: Beta parameter for Smooth L1 in precursor loss
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
            bin_size=bin_size,
            max_mz=max_mz,
            sigma=sigma,
            ion_type_names=ion_type_names,
            ms2pip_model=ms2pip_model,
        )

        self.precursor_loss = PrecursorMassLoss(
            avg_aa_mass=avg_aa_mass,
            huber_beta=huber_beta,
        )

    def forward(
        self,
        all_logits: Tensor,
        targets: Tensor,
        target_mask: Tensor,
        observed_masses: Tensor,
        observed_intensities: Tensor,
        peak_mask: Tensor,
        precursor_mass: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Compute combined loss.

        Returns:
            total_loss: Scalar loss
            metrics: Dict with component losses
        """
        # 1. Deep Supervision (Cross Entropy)
        ce_loss, ce_metrics = self.ce_loss(all_logits, targets, target_mask)

        # 2. Spectrum Loss & Precursor Loss on final prediction
        final_probs = F.softmax(all_logits[-1], dim=-1)

        # Ensure float32 for stability in exp() and sum() operations
        with torch.amp.autocast(enabled=False, device_type=ce_loss.device.type):
            final_probs_f32 = final_probs.float()

            # Spectrum Loss
            if self.spectrum_weight > 0:
                spec_loss = self.spectrum_loss(
                    final_probs_f32,
                    observed_masses.float(),
                    observed_intensities.float(),
                    peak_mask,
                    sequence_mask=target_mask  # Pass sequence mask to prevent PAD contribution
                )
            else:
                spec_loss = torch.tensor(0.0, device=ce_loss.device)

            # Precursor Loss
            prec_metrics = {}
            if self.precursor_weight > 0 and precursor_mass is not None:
                prec_loss, prec_metrics = self.precursor_loss(
                    final_probs_f32,
                    precursor_mass.float(),
                    target_mask
                )
            else:
                prec_loss = torch.tensor(0.0, device=ce_loss.device)

        # Combine
        total_loss = (
            self.ce_weight * ce_loss
            + self.spectrum_weight * spec_loss
            + self.precursor_weight * prec_loss
        )

        # Metrics
        metrics = {
            **ce_metrics,
            'spectrum_loss': spec_loss.item(),
            'total_loss': total_loss.item(),
            **prec_metrics
        }

        return total_loss, metrics
