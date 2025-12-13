#!/usr/bin/env python3
"""
Comprehensive visualization tool for pepTRM predictions.

Generates multi-panel visualizations showing:
1. Input mass spectrum with peak annotations
2. Ground truth vs predicted sequences
3. Step-by-step prediction evolution with highlighted token changes
4. Predicted vs observed spectrum comparison
5. Precursor mass analysis
6. Probability heatmap showing confidence at each position

Usage:
    # Synthetic data (quick test)
    python scripts/visualize_sample.py --checkpoint path/to/checkpoint.pt --synthetic

    # Real data from Nine-Species benchmark
    python scripts/visualize_sample.py --checkpoint path/to/checkpoint.pt --real-data path/to/nine-species

    # Multiple samples
    python scripts/visualize_sample.py --checkpoint path/to/checkpoint.pt --synthetic --num-samples 5

    # Save to file
    python scripts/visualize_sample.py --checkpoint path/to/checkpoint.pt --synthetic --output viz.png --no-show
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass

from src.constants import (
    VOCAB, AA_TO_IDX, IDX_TO_AA, AMINO_ACID_MASSES,
    WATER_MASS, PROTON_MASS, PAD_IDX, SOS_IDX, EOS_IDX
)
from src.model.trm import RecursivePeptideModel, TRMConfig
from src.data.synthetic import generate_theoretical_spectrum, TheoreticalSpectrum
from src.data.dataset import SyntheticPeptideDataset, collate_peptide_samples


@dataclass
class VisualizationData:
    """All data needed for visualization."""
    # Input spectrum
    spectrum_masses: np.ndarray
    spectrum_intensities: np.ndarray
    spectrum_mask: np.ndarray

    # Precursor info
    precursor_mass: float
    precursor_charge: int

    # Ground truth
    target_sequence: str
    target_tokens: List[int]

    # Model predictions at each step
    step_logits: np.ndarray  # (T, seq_len, vocab_size)
    step_predictions: List[str]  # Decoded sequences at each step
    step_tokens: List[List[int]]  # Token indices at each step
    step_probs: np.ndarray  # (T, seq_len, vocab_size) - softmax probabilities

    # Final prediction analysis
    final_sequence: str
    final_theoretical_spectrum: TheoreticalSpectrum
    predicted_precursor_mass: float

    # Data source info
    data_source: str = 'synthetic'  # 'synthetic', 'nine_species', etc.


def load_checkpoint(checkpoint_path: str, device: str = 'cuda') -> Tuple[RecursivePeptideModel, dict]:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Create model with config from checkpoint
    config_dict = checkpoint.get('config', {})

    # Filter to only TRMConfig parameters (exclude training params like learning_rate)
    from dataclasses import fields
    valid_fields = {f.name for f in fields(TRMConfig)}
    filtered_config = {k: v for k, v in config_dict.items() if k in valid_fields}

    config = TRMConfig(**filtered_config) if filtered_config else TRMConfig()

    model = RecursivePeptideModel(config)

    # Handle compiled models (torch.compile adds _orig_mod. prefix)
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        # Strip the _orig_mod. prefix
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, checkpoint


def decode_tokens(tokens: List[int]) -> str:
    """Convert token indices to amino acid string."""
    result = []
    for t in tokens:
        if t == PAD_IDX:
            continue
        if t == SOS_IDX or t == EOS_IDX:
            continue
        result.append(IDX_TO_AA.get(t, '?'))
    return ''.join(result)


def get_sequence_from_indices(indices: np.ndarray, mask: np.ndarray) -> Tuple[str, List[int]]:
    """Extract sequence string and valid token list from indices."""
    valid_tokens = []
    for i, (idx, m) in enumerate(zip(indices, mask)):
        if m and idx not in [PAD_IDX, SOS_IDX, EOS_IDX]:
            valid_tokens.append(int(idx))
    return decode_tokens(valid_tokens), valid_tokens


def compute_precursor_mass(sequence: str) -> float:
    """Compute precursor mass from sequence."""
    return sum(AMINO_ACID_MASSES.get(aa, 0.0) for aa in sequence) + WATER_MASS


def run_inference(
    model: RecursivePeptideModel,
    batch: dict,
    device: str = 'cuda',
    data_source: str = 'synthetic',
) -> VisualizationData:
    """Run model inference and collect all visualization data."""

    # Move batch to device
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    with torch.no_grad():
        # Forward pass to get all steps
        all_logits, _ = model(
            spectrum_masses=batch['spectrum_masses'],
            spectrum_intensities=batch['spectrum_intensities'],
            spectrum_mask=batch['spectrum_mask'],
            precursor_mass=batch['precursor_mass'],
            precursor_charge=batch['precursor_charge'],
        )

    # Extract first sample from batch
    idx = 0
    spectrum_masses = batch['spectrum_masses'][idx].cpu().numpy()
    spectrum_intensities = batch['spectrum_intensities'][idx].cpu().numpy()
    spectrum_mask = batch['spectrum_mask'][idx].cpu().numpy()
    precursor_mass = batch['precursor_mass'][idx].item()
    precursor_charge = batch['precursor_charge'][idx].item()
    target_tokens_raw = batch['sequence'][idx].cpu().numpy()
    target_mask = batch['sequence_mask'][idx].cpu().numpy()

    # Decode target sequence
    target_sequence, target_tokens = get_sequence_from_indices(target_tokens_raw, target_mask)

    # Process predictions at each step
    step_logits = all_logits[:, idx].cpu().numpy()  # (T, seq_len, vocab)
    step_probs = F.softmax(all_logits[:, idx], dim=-1).cpu().numpy()  # (T, seq_len, vocab)
    step_predictions = []
    step_tokens = []

    for t in range(step_logits.shape[0]):
        pred_indices = step_logits[t].argmax(axis=-1)
        pred_seq, pred_toks = get_sequence_from_indices(pred_indices, target_mask)
        step_predictions.append(pred_seq)
        step_tokens.append(pred_toks)

    # Final prediction
    final_sequence = step_predictions[-1]
    predicted_precursor_mass = compute_precursor_mass(final_sequence)

    # Generate theoretical spectrum from final prediction
    if len(final_sequence) >= 2:
        final_theoretical_spectrum = generate_theoretical_spectrum(
            peptide=final_sequence,
            charge=int(precursor_charge),
            ion_types=['b', 'y'],
        )
    else:
        # Handle empty/invalid predictions
        final_theoretical_spectrum = TheoreticalSpectrum(
            peaks=[], precursor_mass=0, precursor_mz=0,
            charge=int(precursor_charge), peptide='', ion_annotations={}
        )

    return VisualizationData(
        spectrum_masses=spectrum_masses,
        spectrum_intensities=spectrum_intensities,
        spectrum_mask=spectrum_mask,
        precursor_mass=precursor_mass,
        precursor_charge=int(precursor_charge),
        target_sequence=target_sequence,
        target_tokens=target_tokens,
        step_logits=step_logits,
        step_predictions=step_predictions,
        step_tokens=step_tokens,
        step_probs=step_probs,
        final_sequence=final_sequence,
        final_theoretical_spectrum=final_theoretical_spectrum,
        predicted_precursor_mass=predicted_precursor_mass,
        data_source=data_source,
    )


def plot_spectrum(
    ax: plt.Axes,
    masses: np.ndarray,
    intensities: np.ndarray,
    mask: np.ndarray,
    color: str = '#2E86AB',
    title: str = 'Mass Spectrum',
    annotations: Optional[Dict[float, str]] = None,
    alpha: float = 1.0,
):
    """Plot mass spectrum as stem plot."""
    valid_masses = masses[mask]
    valid_intensities = intensities[mask]

    # Normalize intensities for display
    if len(valid_intensities) > 0 and valid_intensities.max() > 0:
        valid_intensities = valid_intensities / valid_intensities.max()

    markerline, stemlines, baseline = ax.stem(
        valid_masses, valid_intensities,
        linefmt=color, markerfmt=' ', basefmt=' '
    )
    stemlines.set_alpha(alpha)
    stemlines.set_linewidth(1.2)

    ax.set_xlabel('m/z (Da)', fontsize=10)
    ax.set_ylabel('Relative Intensity', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add annotations if provided
    if annotations:
        for mass, label in annotations.items():
            # Find closest peak
            if len(valid_masses) > 0:
                diffs = np.abs(valid_masses - mass)
                closest_idx = np.argmin(diffs)
                if diffs[closest_idx] < 1.0:  # Within 1 Da
                    ax.annotate(
                        label,
                        (valid_masses[closest_idx], valid_intensities[closest_idx]),
                        textcoords='offset points', xytext=(0, 5),
                        fontsize=6, ha='center', rotation=45
                    )


def plot_spectrum_comparison(
    ax: plt.Axes,
    obs_masses: np.ndarray,
    obs_intensities: np.ndarray,
    obs_mask: np.ndarray,
    pred_spectrum: TheoreticalSpectrum,
    tolerance_da: float = 0.5,
):
    """Plot observed vs predicted spectrum with matching highlights."""
    valid_obs_masses = obs_masses[obs_mask]
    valid_obs_intensities = obs_intensities[obs_mask]

    # Normalize observed intensities
    if len(valid_obs_intensities) > 0 and valid_obs_intensities.max() > 0:
        valid_obs_intensities = valid_obs_intensities / valid_obs_intensities.max()

    # Get predicted peaks
    pred_masses = np.array([p[0] for p in pred_spectrum.peaks]) if pred_spectrum.peaks else np.array([])
    pred_intensities = np.array([p[1] for p in pred_spectrum.peaks]) if pred_spectrum.peaks else np.array([])

    # Normalize predicted intensities
    if len(pred_intensities) > 0 and pred_intensities.max() > 0:
        pred_intensities = pred_intensities / pred_intensities.max()

    # Find matches
    matched_obs = set()
    matched_pred = set()

    for i, obs_m in enumerate(valid_obs_masses):
        for j, pred_m in enumerate(pred_masses):
            if abs(obs_m - pred_m) < tolerance_da:
                matched_obs.add(i)
                matched_pred.add(j)

    # Plot observed (positive, blue/green for matched/unmatched)
    for i, (m, intens) in enumerate(zip(valid_obs_masses, valid_obs_intensities)):
        color = '#22C55E' if i in matched_obs else '#64748B'
        ax.vlines(m, 0, intens, colors=color, linewidth=1.5, alpha=0.8)

    # Plot predicted (negative, orange/gray for matched/unmatched)
    for j, (m, intens) in enumerate(zip(pred_masses, pred_intensities)):
        color = '#F97316' if j in matched_pred else '#94A3B8'
        ax.vlines(m, 0, -intens, colors=color, linewidth=1.5, alpha=0.8)

    # Add annotations for matched predicted peaks
    for j, (m, intens) in enumerate(zip(pred_masses, pred_intensities)):
        if j in matched_pred and m in pred_spectrum.ion_annotations:
            label = pred_spectrum.ion_annotations[m]
            ax.annotate(
                label, (m, -intens - 0.05),
                fontsize=6, ha='center', rotation=45, color='#F97316'
            )

    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('m/z (Da)', fontsize=10)
    ax.set_ylabel('Intensity', fontsize=10)
    ax.set_title('Spectrum Comparison: Observed (up) vs Predicted (down)', fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend
    obs_patch = mpatches.Patch(color='#22C55E', label=f'Observed matched ({len(matched_obs)})')
    obs_unmatch = mpatches.Patch(color='#64748B', label=f'Observed unmatched ({len(valid_obs_masses) - len(matched_obs)})')
    pred_patch = mpatches.Patch(color='#F97316', label=f'Predicted matched ({len(matched_pred)})')
    pred_unmatch = mpatches.Patch(color='#94A3B8', label=f'Predicted unmatched ({len(pred_masses) - len(matched_pred)})')
    ax.legend(handles=[obs_patch, obs_unmatch, pred_patch, pred_unmatch], loc='upper right', fontsize=8)

    # Compute and display coverage
    coverage = len(matched_obs) / len(valid_obs_masses) * 100 if len(valid_obs_masses) > 0 else 0
    precision = len(matched_pred) / len(pred_masses) * 100 if len(pred_masses) > 0 else 0
    ax.text(0.02, 0.98, f'Coverage: {coverage:.1f}%  Precision: {precision:.1f}%',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


def plot_prediction_evolution(
    ax: plt.Axes,
    target_sequence: str,
    step_predictions: List[str],
    step_tokens: List[List[int]],
    target_tokens: List[int],
):
    """Plot step-by-step prediction evolution with highlighting."""
    num_steps = len(step_predictions)
    max_len = max(len(target_sequence), max(len(p) for p in step_predictions) if step_predictions else 1)

    # Create color-coded visualization
    # Colors: correct (green), changed from previous (yellow), wrong (red)

    ax.set_xlim(-0.5, max_len + 0.5)
    ax.set_ylim(-0.5, num_steps + 1.5)

    # Plot target sequence at top
    y = num_steps + 0.5
    ax.text(-0.3, y, 'Target:', fontsize=10, ha='right', va='center', fontweight='bold')
    for i, aa in enumerate(target_sequence):
        ax.add_patch(plt.Rectangle((i - 0.4, y - 0.35), 0.8, 0.7,
                                    facecolor='#E0F2FE', edgecolor='#0284C7', linewidth=1))
        ax.text(i, y, aa, fontsize=11, ha='center', va='center', fontweight='bold', color='#0284C7')

    # Plot each step
    for step_idx in range(num_steps):
        y = num_steps - step_idx - 0.5
        pred = step_predictions[step_idx]
        prev_pred = step_predictions[step_idx - 1] if step_idx > 0 else ''

        ax.text(-0.3, y, f'Step {step_idx + 1}:', fontsize=9, ha='right', va='center')

        for i in range(max_len):
            if i < len(pred):
                aa = pred[i]
                target_aa = target_sequence[i] if i < len(target_sequence) else None
                prev_aa = prev_pred[i] if i < len(prev_pred) else None

                # Determine color
                if aa == target_aa:
                    facecolor = '#DCFCE7'  # Light green - correct
                    edgecolor = '#22C55E'
                    textcolor = '#166534'
                elif prev_aa is not None and aa != prev_aa:
                    facecolor = '#FEF3C7'  # Light yellow - changed
                    edgecolor = '#F59E0B'
                    textcolor = '#92400E'
                else:
                    facecolor = '#FEE2E2'  # Light red - wrong
                    edgecolor = '#EF4444'
                    textcolor = '#991B1B'

                ax.add_patch(plt.Rectangle((i - 0.4, y - 0.35), 0.8, 0.7,
                                            facecolor=facecolor, edgecolor=edgecolor, linewidth=1))
                ax.text(i, y, aa, fontsize=10, ha='center', va='center', color=textcolor)

    ax.set_title('Prediction Evolution by Refinement Step', fontsize=12, fontweight='bold')
    ax.axis('off')

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#DCFCE7', edgecolor='#22C55E', label='Correct'),
        mpatches.Patch(facecolor='#FEF3C7', edgecolor='#F59E0B', label='Changed'),
        mpatches.Patch(facecolor='#FEE2E2', edgecolor='#EF4444', label='Incorrect'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)


def plot_accuracy_progression(
    ax: plt.Axes,
    target_sequence: str,
    step_predictions: List[str],
):
    """Plot accuracy metrics over refinement steps."""
    accuracies = []
    for pred in step_predictions:
        if len(target_sequence) == 0:
            accuracies.append(0)
            continue
        matches = sum(1 for a, b in zip(pred, target_sequence) if a == b)
        acc = matches / len(target_sequence) * 100
        accuracies.append(acc)

    steps = range(1, len(accuracies) + 1)
    ax.plot(steps, accuracies, 'o-', color='#2E86AB', linewidth=2, markersize=8)
    ax.fill_between(steps, accuracies, alpha=0.2, color='#2E86AB')

    ax.set_xlabel('Refinement Step', fontsize=10)
    ax.set_ylabel('Token Accuracy (%)', fontsize=10)
    ax.set_title('Accuracy Progression', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.set_xticks(steps)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axhline(100, color='#22C55E', linestyle='--', alpha=0.5, label='Perfect')

    # Add final accuracy annotation
    if accuracies:
        ax.annotate(f'{accuracies[-1]:.1f}%',
                   (len(accuracies), accuracies[-1]),
                   textcoords='offset points', xytext=(5, 5),
                   fontsize=10, fontweight='bold', color='#2E86AB')


def plot_precursor_mass(
    ax: plt.Axes,
    target_mass: float,
    predicted_mass: float,
    charge: int,
):
    """Plot precursor mass comparison."""
    mass_error = predicted_mass - target_mass
    ppm_error = (mass_error / target_mass) * 1e6 if target_mass > 0 else 0

    # Bar chart
    bars = ax.bar(['Target', 'Predicted'], [target_mass, predicted_mass],
                  color=['#0284C7', '#F97316'], width=0.6, edgecolor='black')

    ax.set_ylabel('Mass (Da)', fontsize=10)
    ax.set_title(f'Precursor Mass (Charge: {charge}+)', fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add value labels
    for bar, val in zip(bars, [target_mass, predicted_mass]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{val:.2f}', ha='center', fontsize=9)

    # Error annotation
    error_color = '#22C55E' if abs(ppm_error) < 20 else '#EF4444'
    ax.text(0.5, 0.02, f'Error: {mass_error:+.2f} Da ({ppm_error:+.1f} ppm)',
            transform=ax.transAxes, ha='center', fontsize=10, fontweight='bold',
            color=error_color,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor=error_color, alpha=0.8))


def plot_sequence_info(
    ax: plt.Axes,
    data: VisualizationData,
):
    """Plot sequence information panel."""
    ax.axis('off')

    # Compute metrics
    target_len = len(data.target_sequence)
    pred_len = len(data.final_sequence)
    matches = sum(1 for a, b in zip(data.final_sequence, data.target_sequence) if a == b)
    accuracy = matches / target_len * 100 if target_len > 0 else 0

    mass_error = data.predicted_precursor_mass - data.precursor_mass
    ppm_error = (mass_error / data.precursor_mass) * 1e6 if data.precursor_mass > 0 else 0

    info_text = f"""
    Target:    {data.target_sequence}
    Predicted: {data.final_sequence}

    Length: {target_len} -> {pred_len}
    Accuracy: {matches}/{target_len} ({accuracy:.1f}%)
    Mass Error: {mass_error:+.2f} Da ({ppm_error:+.1f} ppm)
    Precursor Charge: {data.precursor_charge}+
    Data Source: {data.data_source}
    """

    ax.text(0.05, 0.95, info_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#F8FAFC', edgecolor='#E2E8F0'))
    ax.set_title('Sample Summary', fontsize=12, fontweight='bold', loc='left')


def plot_probability_heatmap(
    ax: plt.Axes,
    step_probs: np.ndarray,  # (T, seq_len, vocab_size)
    target_sequence: str,
    final_sequence: str,
):
    """
    Plot probability heatmap showing confidence evolution across refinement steps.

    Shows only the amino acid tokens (not special tokens) at each position for the final step,
    with entropy/confidence overlaid.
    """
    num_steps, seq_len, vocab_size = step_probs.shape

    # Get amino acid indices (skip special tokens 0-3)
    aa_start = 4
    aa_labels = VOCAB[aa_start:]  # 20 amino acids

    # Focus on positions that matter (based on target length)
    display_len = max(len(target_sequence), len(final_sequence)) + 2
    display_len = min(display_len, seq_len - 2)  # Exclude SOS/EOS positions

    # Get final step probabilities for amino acids only
    # Positions 1 to display_len+1 (skip SOS at position 0)
    final_probs = step_probs[-1, 1:display_len+1, aa_start:]  # (display_len, 20)

    # Create heatmap
    im = ax.imshow(final_probs.T, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)

    # Set ticks
    ax.set_xticks(range(display_len))
    ax.set_xticklabels([f'{i+1}' for i in range(display_len)], fontsize=8)
    ax.set_yticks(range(len(aa_labels)))
    ax.set_yticklabels(aa_labels, fontsize=8)

    ax.set_xlabel('Sequence Position', fontsize=10)
    ax.set_ylabel('Amino Acid', fontsize=10)
    ax.set_title('Final Step Probability Distribution', fontsize=12, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Probability', fontsize=9)

    # Highlight target and predicted tokens
    for pos in range(display_len):
        # Target token (blue border)
        if pos < len(target_sequence):
            target_aa = target_sequence[pos]
            if target_aa in aa_labels:
                target_idx = aa_labels.index(target_aa)
                ax.add_patch(plt.Rectangle((pos - 0.5, target_idx - 0.5), 1, 1,
                                            fill=False, edgecolor='#0284C7', linewidth=2))

        # Predicted token (green border if correct, red if wrong)
        if pos < len(final_sequence):
            pred_aa = final_sequence[pos]
            if pred_aa in aa_labels:
                pred_idx = aa_labels.index(pred_aa)
                target_aa = target_sequence[pos] if pos < len(target_sequence) else None
                color = '#22C55E' if pred_aa == target_aa else '#EF4444'
                ax.add_patch(plt.Rectangle((pos - 0.5, pred_idx - 0.5), 1, 1,
                                            fill=False, edgecolor=color, linewidth=2, linestyle='--'))

    # Legend
    target_patch = mpatches.Patch(fill=False, edgecolor='#0284C7', linewidth=2, label='Target')
    correct_patch = mpatches.Patch(fill=False, edgecolor='#22C55E', linewidth=2, linestyle='--', label='Pred (correct)')
    wrong_patch = mpatches.Patch(fill=False, edgecolor='#EF4444', linewidth=2, linestyle='--', label='Pred (wrong)')
    ax.legend(handles=[target_patch, correct_patch, wrong_patch], loc='upper right', fontsize=7)


def plot_confidence_evolution(
    ax: plt.Axes,
    step_probs: np.ndarray,  # (T, seq_len, vocab_size)
    target_sequence: str,
):
    """Plot how prediction confidence (max prob) evolves across refinement steps."""
    num_steps = step_probs.shape[0]
    display_len = len(target_sequence) + 2  # +2 for SOS/EOS

    # Compute max probability at each position for each step
    max_probs = step_probs[:, 1:display_len, :].max(axis=-1)  # (T, display_len-1)

    # Average confidence across positions
    avg_confidence = max_probs.mean(axis=1)

    # Also compute confidence for correct tokens
    correct_confidence = []
    for t in range(num_steps):
        probs = []
        for pos, aa in enumerate(target_sequence):
            if aa in AA_TO_IDX:
                token_idx = AA_TO_IDX[aa]
                probs.append(step_probs[t, pos + 1, token_idx])  # +1 for SOS offset
        correct_confidence.append(np.mean(probs) if probs else 0)

    steps = range(1, num_steps + 1)

    ax.plot(steps, avg_confidence * 100, 'o-', color='#2E86AB', linewidth=2,
            markersize=6, label='Max prob (any token)')
    ax.plot(steps, np.array(correct_confidence) * 100, 's-', color='#22C55E', linewidth=2,
            markersize=6, label='Prob of correct token')
    ax.fill_between(steps, np.array(correct_confidence) * 100, alpha=0.2, color='#22C55E')

    ax.set_xlabel('Refinement Step', fontsize=10)
    ax.set_ylabel('Probability (%)', fontsize=10)
    ax.set_title('Confidence Evolution', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.set_xticks(steps)
    ax.legend(loc='lower right', fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def create_visualization(
    data: VisualizationData,
    output_path: Optional[str] = None,
    show: bool = True,
):
    """Create the full multi-panel visualization."""

    fig = plt.figure(figsize=(20, 18))
    fig.suptitle('pepTRM Sample Analysis', fontsize=16, fontweight='bold', y=0.98)

    # Create grid layout - 5 rows now
    gs = GridSpec(5, 3, figure=fig, height_ratios=[0.9, 1.1, 1.0, 1.0, 0.7],
                  hspace=0.35, wspace=0.30)

    # Row 1: Input spectrum and summary info
    ax_spectrum = fig.add_subplot(gs[0, :2])
    ax_info = fig.add_subplot(gs[0, 2])

    # Row 2: Prediction evolution
    ax_evolution = fig.add_subplot(gs[1, :])

    # Row 3: Spectrum comparison and accuracy/confidence progression
    ax_comparison = fig.add_subplot(gs[2, :2])
    ax_accuracy = fig.add_subplot(gs[2, 2])

    # Row 4: Probability heatmap and confidence evolution
    ax_heatmap = fig.add_subplot(gs[3, :2])
    ax_confidence = fig.add_subplot(gs[3, 2])

    # Row 5: Precursor mass (centered)
    ax_precursor = fig.add_subplot(gs[4, 1])

    # Plot each panel
    plot_spectrum(
        ax_spectrum,
        data.spectrum_masses,
        data.spectrum_intensities,
        data.spectrum_mask,
        title='Input Mass Spectrum'
    )

    plot_sequence_info(ax_info, data)

    plot_prediction_evolution(
        ax_evolution,
        data.target_sequence,
        data.step_predictions,
        data.step_tokens,
        data.target_tokens,
    )

    plot_spectrum_comparison(
        ax_comparison,
        data.spectrum_masses,
        data.spectrum_intensities,
        data.spectrum_mask,
        data.final_theoretical_spectrum,
    )

    plot_accuracy_progression(
        ax_accuracy,
        data.target_sequence,
        data.step_predictions,
    )

    plot_probability_heatmap(
        ax_heatmap,
        data.step_probs,
        data.target_sequence,
        data.final_sequence,
    )

    plot_confidence_evolution(
        ax_confidence,
        data.step_probs,
        data.target_sequence,
    )

    plot_precursor_mass(
        ax_precursor,
        data.precursor_mass,
        data.predicted_precursor_mass,
        data.precursor_charge,
    )

    # Use constrained layout instead of tight_layout to handle all axes types
    try:
        fig.tight_layout(rect=[0, 0, 1, 0.96])
    except Exception:
        pass  # Ignore layout warnings

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved visualization to {output_path}")

    if show:
        plt.show()

    return fig


def generate_synthetic_sample(
    min_length: int = 8,
    max_length: int = 15,
    charge: int = 2,
) -> dict:
    """Generate a synthetic sample for testing."""
    dataset = SyntheticPeptideDataset(
        min_length=min_length,
        max_length=max_length,
        ion_types=['b', 'y'],
        clean_data_ratio=1.0,  # Pure synthetic
    )

    sample = next(iter(dataset))
    batch = collate_peptide_samples([sample])
    return batch


def load_real_sample(
    data_dir: str,
    sample_idx: int = 0,
    max_samples: int = 100,
) -> Tuple[dict, str]:
    """
    Load a real sample from Nine-Species benchmark.

    Returns batch and data source string.
    """
    from src.data.nine_species_dataset import NineSpeciesDataset

    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    print(f"Loading Nine-Species data from {data_path}...")

    dataset = NineSpeciesDataset(
        data_dir=data_path,
        split='val',
        max_samples=max_samples,
        use_balanced=True,  # Use smaller balanced version for quick loading
    )

    if len(dataset) == 0:
        raise ValueError("No samples found in dataset. Check data path and file format.")

    # Get sample
    idx = sample_idx % len(dataset)
    sample = dataset[idx]

    # Convert to batch format
    # Note: Nine-Species doesn't include SOS/EOS, so we need to add them
    sequence = sample.sequence.clone()
    sequence_mask = sample.sequence_mask.clone()

    # Shift sequence to add SOS at beginning
    new_sequence = torch.zeros_like(sequence)
    new_mask = torch.zeros_like(sequence_mask)

    new_sequence[0] = SOS_IDX
    new_mask[0] = True

    seq_len = sequence_mask.sum().item()
    new_sequence[1:seq_len+1] = sequence[:seq_len]
    new_mask[1:seq_len+1] = True

    # Add EOS
    if seq_len + 1 < len(new_sequence):
        new_sequence[seq_len + 1] = EOS_IDX
        new_mask[seq_len + 1] = True

    batch = {
        'spectrum_masses': sample.spectrum_masses.unsqueeze(0),
        'spectrum_intensities': sample.spectrum_intensities.unsqueeze(0),
        'spectrum_mask': sample.spectrum_mask.unsqueeze(0),
        'precursor_mass': sample.precursor_mass.unsqueeze(0),
        'precursor_charge': sample.precursor_charge.unsqueeze(0),
        'sequence': new_sequence.unsqueeze(0),
        'sequence_mask': new_mask.unsqueeze(0),
    }

    species = getattr(sample, 'species', 'unknown')
    return batch, f'nine_species ({species})'


def main():
    parser = argparse.ArgumentParser(
        description='Visualize pepTRM predictions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Synthetic data (quick test)
    python scripts/visualize_sample.py --checkpoint checkpoints/model.pt --synthetic

    # Real data from Nine-Species benchmark
    python scripts/visualize_sample.py --checkpoint checkpoints/model.pt --real-data data/nine-species

    # Multiple samples with output
    python scripts/visualize_sample.py --checkpoint checkpoints/model.pt --synthetic --num-samples 5 --output viz.png

    # Specific sample index from real data
    python scripts/visualize_sample.py --checkpoint checkpoints/model.pt --real-data data/nine-species --sample-idx 42
        """
    )
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--sample-idx', type=int, default=0, help='Sample index to visualize')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic data')
    parser.add_argument('--real-data', type=str, default=None, help='Path to real data directory (Nine-Species)')
    parser.add_argument('--output', type=str, default=None, help='Output path for figure (PNG)')
    parser.add_argument('--no-show', action='store_true', help='Do not display figure (useful for batch processing)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num-samples', type=int, default=1, help='Number of samples to visualize')
    parser.add_argument('--min-length', type=int, default=8, help='Min peptide length for synthetic data')
    parser.add_argument('--max-length', type=int, default=15, help='Max peptide length for synthetic data')

    args = parser.parse_args()

    # Validate arguments
    if not args.synthetic and not args.real_data:
        print("No data source specified. Use --synthetic or --real-data <path>")
        print("Defaulting to synthetic data...")
        args.synthetic = True

    print(f"Loading checkpoint from {args.checkpoint}...")
    model, checkpoint = load_checkpoint(args.checkpoint, args.device)
    print(f"Model loaded. Config: {model.config}")

    for sample_num in range(args.num_samples):
        print(f"\nGenerating visualization for sample {sample_num + 1}/{args.num_samples}...")

        # Generate or load sample
        if args.real_data:
            try:
                batch, data_source = load_real_sample(
                    args.real_data,
                    sample_idx=args.sample_idx + sample_num,
                )
            except Exception as e:
                print(f"Error loading real data: {e}")
                print("Falling back to synthetic data...")
                batch = generate_synthetic_sample(args.min_length, args.max_length)
                data_source = 'synthetic'
        else:
            batch = generate_synthetic_sample(args.min_length, args.max_length)
            data_source = 'synthetic'

        # Run inference
        print("Running inference...")
        vis_data = run_inference(model, batch, args.device, data_source)

        # Print summary
        matches = sum(1 for a, b in zip(vis_data.final_sequence, vis_data.target_sequence) if a == b)
        accuracy = matches / len(vis_data.target_sequence) * 100 if vis_data.target_sequence else 0

        print(f"Data Source: {data_source}")
        print(f"Target:      {vis_data.target_sequence}")
        print(f"Predicted:   {vis_data.final_sequence}")
        print(f"Accuracy:    {matches}/{len(vis_data.target_sequence)} ({accuracy:.1f}%)")

        # Determine output path
        if args.output:
            if args.num_samples > 1:
                base, ext = args.output.rsplit('.', 1) if '.' in args.output else (args.output, 'png')
                output_path = f"{base}_{sample_num + 1}.{ext}"
            else:
                output_path = args.output
        else:
            output_path = None

        # Create visualization
        create_visualization(
            vis_data,
            output_path=output_path,
            show=not args.no_show,
        )

    print("\nDone!")


if __name__ == '__main__':
    main()
