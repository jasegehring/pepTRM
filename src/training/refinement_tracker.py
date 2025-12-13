"""Track how sequences change across recursive refinement steps."""
import torch
from typing import Dict, List


def compute_refinement_metrics(
    all_logits: torch.Tensor,  # (T, B, S, V)
    targets: torch.Tensor,     # (B, S)
    target_mask: torch.Tensor, # (B, S)
    pad_id: int = 0,
) -> Dict[str, float]:
    """
    Compute how predictions change and improve across refinement steps.

    Returns metrics with 'recursion/' prefix for WandB organization:
        - recursion/accuracy_step_{t}: % of tokens correct at step t
        - recursion/edit_rate_step_{t}: % of tokens that changed from step t-1 to t
        - recursion/edit_quality_step_{t}: Net quality of edits (positive = helpful)

    Summary metrics:
        - recursion/accuracy_gain: accuracy_final - accuracy_step_0 (recursion benefit)
        - recursion/total_edits: Total edit rate across all steps
        - recursion/converged_by_step: First step where edit rate < 1%
    """
    num_steps = all_logits.shape[0]
    batch_size, seq_len = targets.shape

    metrics = {}

    # Get predictions at each step
    all_preds = []  # List of (B, S) tensors
    for t in range(num_steps):
        preds = all_logits[t].argmax(dim=-1)  # (B, S)
        all_preds.append(preds)

    accuracies = []
    edit_rates = []
    converged_step = num_steps  # Default: never converged

    # Track changes and improvements
    for t in range(num_steps):
        # Accuracy at this step
        correct = (all_preds[t] == targets) & target_mask.bool()
        accuracy = correct.float().sum() / target_mask.sum()
        accuracies.append(accuracy.item())
        metrics[f'recursion/accuracy_step_{t}'] = accuracy.item()

        if t > 0:
            # Edit rate: how many positions changed from previous step
            changed = (all_preds[t] != all_preds[t-1]) & target_mask.bool()
            edit_rate = changed.float().sum() / target_mask.sum()
            edit_rates.append(edit_rate.item())
            metrics[f'recursion/edit_rate_step_{t}'] = edit_rate.item()

            # Check for convergence (edit rate < 1%)
            if edit_rate.item() < 0.01 and converged_step == num_steps:
                converged_step = t

            # Edit quality: of the changes made, what's the net effect?
            # Positive = edits are helpful, Negative = edits are harmful
            if changed.any():
                # Was wrong at t-1, now correct at t
                was_wrong = (all_preds[t-1] != targets) & target_mask.bool()
                now_correct = (all_preds[t] == targets) & target_mask.bool()
                improved = changed & was_wrong & now_correct

                # Was correct at t-1, now wrong at t
                was_correct = (all_preds[t-1] == targets) & target_mask.bool()
                now_wrong = (all_preds[t] != targets) & target_mask.bool()
                regressed = changed & was_correct & now_wrong

                num_improved = improved.float().sum()
                num_regressed = regressed.float().sum()
                num_changed = changed.float().sum()

                # Net edit quality: (improved - regressed) / changed
                # Range: [-1, 1], where 1 = all edits helpful, -1 = all edits harmful
                edit_quality = (num_improved - num_regressed) / num_changed
                metrics[f'recursion/edit_quality_step_{t}'] = edit_quality.item()

    # Summary metrics - these tell the overall recursion story
    metrics['recursion/accuracy_gain'] = accuracies[-1] - accuracies[0]  # Key: recursion benefit
    metrics['recursion/total_edits'] = sum(edit_rates)  # Total editing activity
    metrics['recursion/converged_by_step'] = converged_step  # When predictions stabilized

    return metrics


def summarize_refinement(metrics: Dict[str, float], num_steps: int = 8) -> str:
    """
    Create human-readable summary of refinement process.

    Example output:
        Step 0: 85.2% acc
        Step 1: 92.3% acc | 15.2% edited | +85.3% quality
        Step 2: 93.1% acc |  3.1% edited | +62.1% quality
        ...
        Summary: +8.1% accuracy gain, converged at step 5
    """
    lines = []
    lines.append("Refinement Progress:")

    for t in range(num_steps):
        # Support both old and new key formats
        acc = metrics.get(f'recursion/accuracy_step_{t}', metrics.get(f'accuracy_step_{t}', 0.0)) * 100
        line = f"  Step {t}: {acc:5.1f}% acc"

        if t > 0:
            edit = metrics.get(f'recursion/edit_rate_step_{t}', metrics.get(f'edit_rate_step_{t}', 0.0)) * 100
            quality = metrics.get(f'recursion/edit_quality_step_{t}', metrics.get(f'improvement_step_{t}', 0.0)) * 100
            line += f" | {edit:5.1f}% edited | {quality:+6.1f}% quality"

        lines.append(line)

    # Add summary line
    acc_gain = metrics.get('recursion/accuracy_gain', 0.0) * 100
    converged = metrics.get('recursion/converged_by_step', num_steps)
    lines.append(f"  Summary: {acc_gain:+.1f}% accuracy gain, converged at step {converged}")

    return '\n'.join(lines)


def analyze_refinement_convergence(
    all_logits: torch.Tensor,  # (T, B, S, V)
    targets: torch.Tensor,     # (B, S)
    target_mask: torch.Tensor, # (B, S)
) -> Dict[str, float]:
    """
    Analyze when refinement converges (predictions stop changing).

    Returns:
        convergence_step: Average step at which predictions stabilize
        final_changes: % of tokens still changing in final step
    """
    num_steps = all_logits.shape[0]

    all_preds = []
    for t in range(num_steps):
        preds = all_logits[t].argmax(dim=-1)
        all_preds.append(preds)

    # For each sequence, find first step where it stops changing
    convergence_steps = []

    batch_size = targets.shape[0]
    for b in range(batch_size):
        mask = target_mask[b].bool()

        # Find first step where prediction doesn't change for rest of sequence
        converged_at = num_steps - 1  # Default: never converged

        for t in range(1, num_steps):
            # Check if prediction at step t matches final prediction
            matches_final = (all_preds[t][b] == all_preds[-1][b]) & mask

            if matches_final.all():
                converged_at = t
                break

        convergence_steps.append(converged_at)

    # Compute final change rate (step 7 vs step 6)
    if num_steps > 1:
        changed_final = (all_preds[-1] != all_preds[-2]) & target_mask.bool()
        final_change_rate = changed_final.float().sum() / target_mask.sum()
    else:
        final_change_rate = 0.0

    return {
        'convergence_step_mean': sum(convergence_steps) / len(convergence_steps),
        'convergence_step_median': sorted(convergence_steps)[len(convergence_steps)//2],
        'final_change_rate': final_change_rate.item(),
    }
