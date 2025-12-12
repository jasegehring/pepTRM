"""
Loss functions for peptide sequencing (TRM).
Includes Deep Supervision, Differentiable Spectrum Matching, and Precursor constraints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional, Tuple, Dict

# Assuming these exist in your project structure
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

    Instead of matching peaks by hard distance (which has poor gradients),
    we 'render' the predicted masses into a spectral bin vector using 
    Gaussian kernels and compare it to the observed spectrum using Cosine Similarity.
    """

    def __init__(
        self,
        bin_size: float = 0.1,         # Resolution of the spectral grid (Da)
        max_mz: float = 2000.0,        # Maximum m/z to consider
        sigma: float = 0.05,           # Gaussian width (standard deviation)
        ion_type_names: Optional[List[str]] = None,
        ms2pip_model: Optional[str] = None,
    ):
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
            print(f"⚠️ SpectrumMatchingLoss: Defaulting to {self.ion_type_names}")

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
        # Note: This creates a large tensor. 
        # For max_mz=2000, bin=0.1 -> 20k bins. 
        # Batch 64 * 100 peaks * 20k bins * 4 bytes ~= 500MB memory. 
        # This is safe for modern GPUs.
        diff = masses.unsqueeze(-1) - self.mz_grid
        
        # Compute Gaussian activations
        # exp(-0.5 * (x - mu)^2 / sigma^2)
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
    ) -> Tensor:
        
        # 1. Compute Predicted Theoretical Peaks (Differentiable)
        # Expectation: E[mass] = sum(P(aa) * mass(aa))
        # This function must return (batch, num_theo_peaks)
        predicted_masses = compute_theoretical_peaks(
            sequence_probs=sequence_probs,
            aa_masses=self.aa_masses,
            ion_type_names=self.ion_type_names,
        )
        
        # 2. Render Predicted Spectrum
        # We assume uniform intensity (1.0) for theoretical peaks for now, 
        # as we are only predicting sequence, not intensity.
        # Ideally, this would use predicted intensities if your model outputs them.
        pred_spectrum = self._gaussian_render(predicted_masses)
        
        # 3. Render Observed Spectrum
        # We render the observed peaks onto the same grid to handle noise/alignment
        with torch.no_grad():
            # Mask out padding in observed data
            obs_masses_masked = observed_masses * peak_mask.float()
            obs_intens_masked = observed_intensities * peak_mask.float()
            
            target_spectrum = self._gaussian_render(obs_masses_masked, obs_intens_masked)
            
            # Normalize target spectrum to max 1.0 for stability
            target_max = target_spectrum.max(dim=1, keepdim=True)[0].clamp(min=1e-8)
            target_spectrum = target_spectrum / target_max

        # 4. Normalize Predicted Spectrum
        # This ensures the loss depends on shape (correlation), not magnitude
        pred_max = pred_spectrum.max(dim=1, keepdim=True)[0].clamp(min=1e-8)
        pred_spectrum = pred_spectrum / pred_max

        # 5. Compute Cosine Similarity Loss
        # 1.0 - CosineSimilarity. Perfect match = 0.0 loss.
        # We use cosine because it focuses on the *alignment* of peaks, not absolute values.
        similarity = F.cosine_similarity(pred_spectrum, target_spectrum, dim=1)
        loss = 1.0 - similarity.mean()
        
        return loss


class PrecursorMassLoss(nn.Module):
    """
    Penalizes predictions where total peptide mass != precursor mass.
    Uses scaled L1 loss for stability.
    """

    def __init__(self, scale_factor: float = 0.004):
        """
        Args:
            scale_factor: Scaling factor for L1 loss.
                          0.004 means 250 Da error = 1.0 loss.
                          Tuned to provide strong gradients without dominating CE loss.

                          Expected contributions (with curriculum weight 0.01):
                          - 200 Da error: 0.008 to total loss
                          - 100 Da error: 0.004 to total loss
                          - 50 Da error: 0.002 to total loss
        """
        super().__init__()
        self.scale_factor = scale_factor
        
        aa_masses = torch.tensor([AMINO_ACID_MASSES.get(aa, 0.0) for aa in VOCAB])
        self.register_buffer('aa_masses', aa_masses)

    def forward(
        self,
        sequence_probs: Tensor,    # (batch, seq_len, vocab)
        precursor_mass: Tensor,    # (batch,)
        sequence_mask: Tensor,     # (batch, seq_len)
    ) -> Tuple[Tensor, Dict[str, float]]:
        
        # Calculate expected mass per position
        # (batch, seq_len, vocab) @ (vocab,) -> (batch, seq_len)
        expected_masses = torch.einsum('bsv,v->bs', sequence_probs, self.aa_masses)

        # Sum valid positions
        predicted_peptide_mass = (expected_masses * sequence_mask.float()).sum(dim=1)
        predicted_total = predicted_peptide_mass + WATER_MASS

        # Absolute Error
        error_da = torch.abs(predicted_total - precursor_mass)
        
        # Scaled Loss
        loss = (error_da * self.scale_factor).mean()

        metrics = {
            'mass_error_da': error_da.mean().item(),
            'precursor_loss': loss.item()
        }

        return loss, metrics


class CombinedLoss(nn.Module):
    """
    Main entry point combining all loss components.
    """

    def __init__(
        self,
        ce_weight: float = 1.0,
        spectrum_weight: float = 0.5,    # Increased slightly as Cosine is [0,1]
        precursor_weight: float = 0.1,
        iteration_weights: str = 'linear',
        label_smoothing: float = 0.0,
        # Spectrum args
        mass_tolerance: float = 0.1,     # Used as bin_size for gaussian
        ion_type_names: Optional[List[str]] = None,
        ms2pip_model: Optional[str] = None,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.spectrum_weight = spectrum_weight
        self.precursor_weight = precursor_weight

        self.ce_loss = DeepSupervisionLoss(
            iteration_weights=iteration_weights,
            label_smoothing=label_smoothing,
        )

        self.spectrum_loss = SpectrumMatchingLoss(
            bin_size=0.1,  # 0.1 Da resolution (good for MS/MS)
            max_mz=2000.0,  # Cover typical peptide m/z range
            sigma=0.05,  # Half the bin size for sharp peaks
            ion_type_names=ion_type_names,
            ms2pip_model=ms2pip_model,
        )

        self.precursor_loss = PrecursorMassLoss(scale_factor=0.004)

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

        # 1. Deep Supervision (Cross Entropy)
        ce_loss, ce_metrics = self.ce_loss(all_logits, targets, target_mask)

        # 2. Spectrum Loss & Precursor Loss
        # We use the final iteration's prediction for auxiliary losses
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
                    peak_mask
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