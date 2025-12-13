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

    Returns:
        metrics: Dict with:
            - edit_rate_step_{t}: % of tokens that changed from step t-1 to t
            - accuracy_step_{t}: % of tokens correct at step t
            - improvement_step_{t}: % of changed tokens that moved toward ground truth
    """
    num_steps = all_logits.shape[0]
    batch_size, seq_len = targets.shape

    metrics = {}

    # Get predictions at each step
    all_preds = []  # List of (B, S) tensors
    for t in range(num_steps):
        preds = all_logits[t].argmax(dim=-1)  # (B, S)
        all_preds.append(preds)

    # Track changes and improvements
    for t in range(num_steps):
        # Accuracy at this step
        correct = (all_preds[t] == targets) & target_mask.bool()
        accuracy = correct.float().sum() / target_mask.sum()
        metrics[f'accuracy_step_{t}'] = accuracy.item()

        if t > 0:
            # Edit rate: how many positions changed from previous step
            changed = (all_preds[t] != all_preds[t-1]) & target_mask.bool()
            edit_rate = changed.float().sum() / target_mask.sum()
            metrics[f'edit_rate_step_{t}'] = edit_rate.item()

            # Improvement rate: of the changed positions, how many got better?
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

                # Net improvement: (improved - regressed) / changed
                improvement_rate = (num_improved - num_regressed) / num_changed
                metrics[f'improvement_step_{t}'] = improvement_rate.item()

                # Also track separately
                metrics[f'improved_step_{t}'] = (num_improved / num_changed).item()
                metrics[f'regressed_step_{t}'] = (num_regressed / num_changed).item()

    return metrics


def summarize_refinement(metrics: Dict[str, float], num_steps: int = 8) -> str:
    """
    Create human-readable summary of refinement process.

    Example output:
        Step 0: 85.2% acc
        Step 1: 92.3% acc | 15.2% edited | +85.3% improved
        Step 2: 93.1% acc |  3.1% edited | +62.1% improved
        ...
    """
    lines = []
    lines.append("Refinement Progress:")

    for t in range(num_steps):
        acc = metrics.get(f'accuracy_step_{t}', 0.0) * 100
        line = f"  Step {t}: {acc:5.1f}% acc"

        if t > 0:
            edit = metrics.get(f'edit_rate_step_{t}', 0.0) * 100
            impr = metrics.get(f'improvement_step_{t}', 0.0) * 100
            line += f" | {edit:5.1f}% edited | {impr:+6.1f}% net improvement"

        lines.append(line)

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
