"""
Evaluation metrics for peptide sequencing.
"""

import torch
from torch import Tensor
from typing import Optional

from ..constants import PAD_IDX, EOS_IDX, IDX_TO_AA, AA_TO_IDX, AMINO_ACID_MASSES, WATER_MASS


def normalize_il_ambiguity(tokens: Tensor) -> Tensor:
    """
    Normalize I/L ambiguity by replacing all I with L.

    Isoleucine and Leucine have identical mass and are indistinguishable
    in mass spectrometry, so they should be treated as equivalent in metrics.
    """
    normalized = tokens.clone()
    i_idx = AA_TO_IDX['I']
    l_idx = AA_TO_IDX['L']
    normalized[tokens == i_idx] = l_idx
    return normalized


def token_accuracy(
    logits: Tensor,
    targets: Tensor,
    mask: Tensor,
) -> float:
    """
    Per-amino-acid accuracy (ignoring padding).

    Treats I/L as equivalent since they are indistinguishable by mass.

    Args:
        logits: (batch, seq_len, vocab)
        targets: (batch, seq_len)
        mask: (batch, seq_len)
    """
    predictions = logits.argmax(dim=-1)

    # Normalize I/L ambiguity
    predictions = normalize_il_ambiguity(predictions)
    targets = normalize_il_ambiguity(targets)

    correct = (predictions == targets) & mask
    return correct.sum().item() / mask.sum().item()


def sequence_accuracy(
    logits: Tensor,
    targets: Tensor,
    mask: Tensor,
) -> float:
    """
    Fraction of perfectly reconstructed sequences.

    Treats I/L as equivalent since they are indistinguishable by mass.
    """
    predictions = logits.argmax(dim=-1)

    # Normalize I/L ambiguity
    predictions = normalize_il_ambiguity(predictions)
    targets = normalize_il_ambiguity(targets)

    # For each sequence, check if all masked positions match
    correct = (predictions == targets) | ~mask
    all_correct = correct.all(dim=-1)

    return all_correct.float().mean().item()


def decode_sequence(indices: Tensor, mask: Optional[Tensor] = None) -> str:
    """Convert token indices to amino acid string."""
    seq = []
    for i, idx in enumerate(indices.tolist()):
        if mask is not None and not mask[i]:
            break
        if idx == EOS_IDX:
            break
        if idx >= len(IDX_TO_AA):
            continue
        aa = IDX_TO_AA[idx]
        if aa not in ['<PAD>', '<SOS>', '<EOS>', '<UNK>']:
            seq.append(aa)
    return ''.join(seq)


def compute_mass_error(
    predicted_sequence: str,
    target_mass: float,
) -> float:
    """
    Compute mass error between predicted sequence and target precursor mass.

    Returns error in Daltons.
    """
    if not predicted_sequence:
        return float('inf')

    predicted_mass = sum(
        AMINO_ACID_MASSES.get(aa, 0) for aa in predicted_sequence
    ) + WATER_MASS

    return abs(predicted_mass - target_mass)


def compute_mass_error_ppm(
    predicted_sequence: str,
    target_mass: float,
) -> float:
    """Compute mass error in parts per million."""
    error_da = compute_mass_error(predicted_sequence, target_mass)
    return (error_da / target_mass) * 1e6


def compute_metrics(
    logits: Tensor,
    targets: Tensor,
    mask: Tensor,
    precursor_masses: Optional[Tensor] = None,
) -> dict:
    """
    Compute all metrics for a batch.

    Returns:
        dict with token_acc, seq_acc, and optionally mass_error
    """
    metrics = {
        'token_accuracy': token_accuracy(logits, targets, mask),
        'sequence_accuracy': sequence_accuracy(logits, targets, mask),
    }

    if precursor_masses is not None:
        mass_errors = []
        predictions = logits.argmax(dim=-1)

        for i in range(len(predictions)):
            pred_seq = decode_sequence(predictions[i], mask[i])
            target_mass = precursor_masses[i].item()
            error = compute_mass_error_ppm(pred_seq, target_mass)
            if error < float('inf'):
                mass_errors.append(error)

        if mass_errors:
            metrics['mean_mass_error_ppm'] = sum(mass_errors) / len(mass_errors)

    return metrics


def compute_metrics_by_length(
    logits: Tensor,
    targets: Tensor,
    mask: Tensor,
) -> dict:
    """
    Compute token accuracy broken down by sequence length buckets.

    Returns:
        dict with accuracy for each length bucket:
        - 'length_7_12_acc': accuracy for lengths 7-12
        - 'length_13_17_acc': accuracy for lengths 13-17
        - 'length_18_25_acc': accuracy for lengths 18-25
        - 'avg_length': average sequence length in batch
    """
    predictions = logits.argmax(dim=-1)

    # Normalize I/L
    predictions = normalize_il_ambiguity(predictions)
    targets_norm = normalize_il_ambiguity(targets)

    # Get sequence lengths
    seq_lengths = mask.sum(dim=-1)  # (batch,)

    # Initialize accumulators for each bucket
    buckets = {
        'short': {'correct': 0, 'total': 0},   # 7-12
        'medium': {'correct': 0, 'total': 0},  # 13-17
        'long': {'correct': 0, 'total': 0},    # 18-25
    }

    for i in range(len(predictions)):
        seq_len = int(seq_lengths[i].item())
        correct = ((predictions[i] == targets_norm[i]) & mask[i]).sum().item()
        total = seq_len

        if seq_len <= 12:
            buckets['short']['correct'] += correct
            buckets['short']['total'] += total
        elif seq_len <= 17:
            buckets['medium']['correct'] += correct
            buckets['medium']['total'] += total
        else:
            buckets['long']['correct'] += correct
            buckets['long']['total'] += total

    metrics = {
        'avg_length': seq_lengths.float().mean().item(),
    }

    # Only add bucket metrics if there are samples in that bucket
    if buckets['short']['total'] > 0:
        metrics['length_7_12_acc'] = buckets['short']['correct'] / buckets['short']['total']
    if buckets['medium']['total'] > 0:
        metrics['length_13_17_acc'] = buckets['medium']['correct'] / buckets['medium']['total']
    if buckets['long']['total'] > 0:
        metrics['length_18_25_acc'] = buckets['long']['correct'] / buckets['long']['total']

    return metrics
