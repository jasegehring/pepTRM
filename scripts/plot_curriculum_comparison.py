"""
Plot and analyze curriculum comparison results.

Visualizes:
- Loss curves over time
- Validation accuracy progression
- Key milestones (when auxiliary losses are introduced)
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import re
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


def parse_log_file(log_path):
    """Parse training log file and extract metrics."""
    metrics = defaultdict(list)

    with open(log_path, 'r') as f:
        for line in f:
            # Parse step number
            step_match = re.search(r'Step (\d+)', line)
            if not step_match:
                continue

            step = int(step_match.group(1))

            # Parse loss
            loss_match = re.search(r'Loss: ([\d.]+)', line)
            if loss_match:
                metrics['step'].append(step)
                metrics['loss'].append(float(loss_match.group(1)))

            # Parse token accuracy
            acc_match = re.search(r'Token Acc: ([\d.]+)', line)
            if acc_match:
                metrics['token_acc'].append(float(acc_match.group(1)))

            # Parse validation metrics
            val_easy_match = re.search(r'Val \(Easy\).*Token Acc: ([\d.]+)', line)
            if val_easy_match:
                metrics['val_easy_step'].append(step)
                metrics['val_easy_acc'].append(float(val_easy_match.group(1)))

            val_hard_match = re.search(r'Val \(Hard\).*Token Acc: ([\d.]+)', line)
            if val_hard_match:
                metrics['val_hard_step'].append(step)
                metrics['val_hard_acc'].append(float(val_hard_match.group(1)))

    return metrics


def plot_comparison(early_metrics, phased_metrics, output_path):
    """Create comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Curriculum Comparison: Early vs Phased Introduction', fontsize=16, fontweight='bold')

    # Plot 1: Training Loss
    ax1 = axes[0, 0]
    ax1.plot(early_metrics['step'], early_metrics['loss'], label='Early Introduction', alpha=0.7, linewidth=1.5)
    ax1.plot(phased_metrics['step'], phased_metrics['loss'], label='Phased Introduction', alpha=0.7, linewidth=1.5)
    ax1.axvline(10000, color='red', linestyle='--', alpha=0.5, label='Early: Spectrum @ 10k')
    ax1.axvline(15000, color='green', linestyle='--', alpha=0.5, label='Phased: Spectrum @ 15k')
    ax1.axvline(20000, color='orange', linestyle='--', alpha=0.5, label='Early: Precursor @ 20k')
    ax1.axvline(30000, color='blue', linestyle='--', alpha=0.5, label='Phased: Precursor @ 30k')
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss Over Time')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Training Accuracy
    ax2 = axes[0, 1]
    ax2.plot(early_metrics['step'], early_metrics['token_acc'], label='Early Introduction', alpha=0.7, linewidth=1.5)
    ax2.plot(phased_metrics['step'], phased_metrics['token_acc'], label='Phased Introduction', alpha=0.7, linewidth=1.5)
    ax2.axvline(10000, color='red', linestyle='--', alpha=0.3)
    ax2.axvline(15000, color='green', linestyle='--', alpha=0.3)
    ax2.axvline(20000, color='orange', linestyle='--', alpha=0.3)
    ax2.axvline(30000, color='blue', linestyle='--', alpha=0.3)
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Token Accuracy')
    ax2.set_title('Training Accuracy Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Validation (Easy)
    ax3 = axes[1, 0]
    ax3.plot(early_metrics['val_easy_step'], early_metrics['val_easy_acc'],
             'o-', label='Early Introduction', alpha=0.7, markersize=4)
    ax3.plot(phased_metrics['val_easy_step'], phased_metrics['val_easy_acc'],
             's-', label='Phased Introduction', alpha=0.7, markersize=4)
    ax3.axvline(10000, color='red', linestyle='--', alpha=0.3)
    ax3.axvline(15000, color='green', linestyle='--', alpha=0.3)
    ax3.axvline(20000, color='orange', linestyle='--', alpha=0.3)
    ax3.axvline(30000, color='blue', linestyle='--', alpha=0.3)
    ax3.set_xlabel('Training Step')
    ax3.set_ylabel('Validation Accuracy (Easy)')
    ax3.set_title('Validation Performance (Easy Set)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Validation (Hard)
    ax4 = axes[1, 1]
    ax4.plot(early_metrics['val_hard_step'], early_metrics['val_hard_acc'],
             'o-', label='Early Introduction', alpha=0.7, markersize=4)
    ax4.plot(phased_metrics['val_hard_step'], phased_metrics['val_hard_acc'],
             's-', label='Phased Introduction', alpha=0.7, markersize=4)
    ax4.axvline(10000, color='red', linestyle='--', alpha=0.3)
    ax4.axvline(15000, color='green', linestyle='--', alpha=0.3)
    ax4.axvline(20000, color='orange', linestyle='--', alpha=0.3)
    ax4.axvline(30000, color='blue', linestyle='--', alpha=0.3)
    ax4.set_xlabel('Training Step')
    ax4.set_ylabel('Validation Accuracy (Hard)')
    ax4.set_title('Validation Performance (Hard Set)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved to: {output_path}")


def print_summary(early_metrics, phased_metrics):
    """Print numerical comparison."""
    print("=" * 80)
    print("Numerical Comparison")
    print("=" * 80)

    # Final metrics
    print("\nFinal Metrics (last values):")
    print(f"{'Metric':<30} | {'Early':<15} | {'Phased':<15} | {'Winner':<10}")
    print("-" * 80)

    if early_metrics['loss'] and phased_metrics['loss']:
        early_loss = early_metrics['loss'][-1]
        phased_loss = phased_metrics['loss'][-1]
        winner = 'Phased' if phased_loss < early_loss else 'Early'
        print(f"{'Training Loss':<30} | {early_loss:<15.4f} | {phased_loss:<15.4f} | {winner:<10}")

    if early_metrics['token_acc'] and phased_metrics['token_acc']:
        early_acc = early_metrics['token_acc'][-1]
        phased_acc = phased_metrics['token_acc'][-1]
        winner = 'Phased' if phased_acc > early_acc else 'Early'
        print(f"{'Training Accuracy':<30} | {early_acc:<15.4f} | {phased_acc:<15.4f} | {winner:<10}")

    if early_metrics['val_easy_acc'] and phased_metrics['val_easy_acc']:
        early_val = early_metrics['val_easy_acc'][-1]
        phased_val = phased_metrics['val_easy_acc'][-1]
        winner = 'Phased' if phased_val > early_val else 'Early'
        print(f"{'Validation (Easy)':<30} | {early_val:<15.4f} | {phased_val:<15.4f} | {winner:<10}")

    if early_metrics['val_hard_acc'] and phased_metrics['val_hard_acc']:
        early_val = early_metrics['val_hard_acc'][-1]
        phased_val = phased_metrics['val_hard_acc'][-1]
        winner = 'Phased' if phased_val > early_val else 'Early'
        print(f"{'Validation (Hard)':<30} | {early_val:<15.4f} | {phased_val:<15.4f} | {winner:<10}")

    # Check for NaN or instabilities
    print("\nStability Check:")
    early_has_nan = any(np.isnan(early_metrics['loss']))
    phased_has_nan = any(np.isnan(phased_metrics['loss']))

    print(f"  Early curriculum:  {'NaN detected ✗' if early_has_nan else 'Stable ✓'}")
    print(f"  Phased curriculum: {'NaN detected ✗' if phased_has_nan else 'Stable ✓'}")


def main():
    print("=" * 80)
    print("Curriculum Comparison Analysis")
    print("=" * 80)
    print()

    # Find log files
    early_log = project_root / "wandb" / "latest-run" / "files" / "output.log"
    phased_log = project_root / "checkpoints_phased" / "training.log"

    # Check if logs exist
    if not early_log.exists():
        print(f"⚠️  Early curriculum log not found: {early_log}")
        print("Looking for alternative log locations...")
        # Try to find most recent log
        wandb_runs = sorted((project_root / "wandb").glob("run-*/files/output.log"))
        if wandb_runs:
            early_log = wandb_runs[-1]
            print(f"Using: {early_log}")

    if not phased_log.exists():
        print(f"⚠️  Phased curriculum log not found: {phased_log}")
        print("Run phased training first: python scripts/train_phased_curriculum.py")
        return

    print(f"Early curriculum log:  {early_log}")
    print(f"Phased curriculum log: {phased_log}")
    print()

    # Parse logs
    print("Parsing logs...")
    early_metrics = parse_log_file(early_log)
    phased_metrics = parse_log_file(phased_log)

    print(f"✓ Early:  {len(early_metrics['step'])} training steps parsed")
    print(f"✓ Phased: {len(phased_metrics['step'])} training steps parsed")
    print()

    # Print summary
    print_summary(early_metrics, phased_metrics)

    # Create plot
    output_path = project_root / "curriculum_comparison.png"
    print()
    print("Creating comparison plot...")
    plot_comparison(early_metrics, phased_metrics, output_path)

    print()
    print("=" * 80)
    print("Analysis Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
