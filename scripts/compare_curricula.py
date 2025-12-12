"""
A/B comparison: Early vs Phased auxiliary loss introduction.

Runs both curricula and compares:
- Final validation accuracy
- Training stability
- Loss curves
- Convergence speed
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import subprocess
import json
import time
from datetime import datetime


def run_training(script_name, max_steps, checkpoint_dir, description):
    """Run a training script and capture results."""
    print("=" * 80)
    print(f"Running: {description}")
    print("=" * 80)
    print(f"Script: {script_name}")
    print(f"Max steps: {max_steps:,}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print()

    start_time = time.time()

    # Run training
    cmd = [
        "python", f"scripts/{script_name}",
        "--max_steps", str(max_steps),
    ]

    print(f"Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, capture_output=False, text=True)

    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)

    print()
    print(f"Completed in {hours}h {minutes}m")
    print()

    return {
        'script': script_name,
        'description': description,
        'max_steps': max_steps,
        'checkpoint_dir': checkpoint_dir,
        'elapsed_seconds': elapsed,
        'success': result.returncode == 0,
    }


def compare_checkpoints(early_dir, phased_dir, max_steps):
    """Load and compare final checkpoints."""
    print("=" * 80)
    print("Comparing Results")
    print("=" * 80)

    # TODO: Load checkpoint metrics and compare
    # For now, just print instructions

    print(f"\nCheckpoints saved to:")
    print(f"  Early introduction:  {early_dir}/")
    print(f"  Phased introduction: {phased_dir}/")
    print()
    print("To analyze results:")
    print("  1. Check validation accuracy in logs")
    print("  2. Plot loss curves using wandb or tensorboard")
    print("  3. Compare final model performance")
    print()
    print("Key metrics to compare:")
    print("  - Final validation accuracy (easy)")
    print("  - Final validation accuracy (hard)")
    print("  - Training stability (NaN, spikes)")
    print("  - Loss at step 15k (spectrum intro)")
    print("  - Loss at step 30k (precursor intro)")
    print(f"  - Final loss at step {max_steps}")


def main():
    parser = argparse.ArgumentParser(description="A/B test: Early vs Phased curriculum")
    parser.add_argument(
        "--max_steps",
        type=int,
        default=50000,
        help="Max training steps for each run (default: 50k)"
    )
    parser.add_argument(
        "--quick_test",
        action="store_true",
        help="Quick test with 10k steps"
    )
    parser.add_argument(
        "--skip_early",
        action="store_true",
        help="Skip early introduction run (already completed)"
    )
    parser.add_argument(
        "--skip_phased",
        action="store_true",
        help="Skip phased introduction run (already completed)"
    )
    args = parser.parse_args()

    if args.quick_test:
        max_steps = 10000
        print("Quick test mode: 10k steps per run")
    else:
        max_steps = args.max_steps

    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 80)
    print("A/B Test: Early vs Phased Auxiliary Loss Introduction")
    print("=" * 80)
    print()
    print("Testing two curricula:")
    print()
    print("A) EARLY INTRODUCTION (current)")
    print("   - Spectrum loss @ step 10k (40% acc, gradient=0.16)")
    print("   - Precursor loss @ step 20k (60% acc, gradient=0.47)")
    print()
    print("B) PHASED INTRODUCTION (new)")
    print("   - Spectrum loss @ step 15k (55% acc, gradient=0.25)")
    print("   - Precursor loss @ step 30k (70% acc, gradient=0.59)")
    print()
    print(f"Each run: {max_steps:,} steps")
    print()

    # Run A: Early introduction (current curriculum)
    if not args.skip_early:
        result_early = run_training(
            script_name="train_optimized.py",
            max_steps=max_steps,
            checkpoint_dir="checkpoints_optimized",
            description="A) EARLY Introduction (current curriculum)"
        )
        results.append(result_early)
    else:
        print("Skipping early introduction run (already completed)")
        print()

    # Run B: Phased introduction (new curriculum)
    if not args.skip_phased:
        result_phased = run_training(
            script_name="train_phased_curriculum.py",
            max_steps=max_steps,
            checkpoint_dir="checkpoints_phased",
            description="B) PHASED Introduction (delayed curriculum)"
        )
        results.append(result_phased)
    else:
        print("Skipping phased introduction run (already completed)")
        print()

    # Compare results
    compare_checkpoints(
        early_dir="checkpoints_optimized",
        phased_dir="checkpoints_phased",
        max_steps=max_steps
    )

    # Save results summary
    summary_file = project_root / f"ab_test_results_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'max_steps': max_steps,
            'results': results,
        }, f, indent=2)

    print()
    print(f"Results summary saved to: {summary_file}")
    print()
    print("=" * 80)
    print("A/B Test Complete")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  1. Review training logs in checkpoints_optimized/ and checkpoints_phased/")
    print("  2. Compare validation accuracy at key milestones:")
    print(f"     - Step 15k (spectrum intro)")
    print(f"     - Step 30k (precursor intro)")
    print(f"     - Step {max_steps} (final)")
    print("  3. Check for NaN or loss spikes")
    print("  4. Run: python scripts/plot_curriculum_comparison.py")
    print()
    print("Decision criteria:")
    print("  - Higher final accuracy → Winner")
    print("  - More stable training → Winner")
    print("  - If similar accuracy, phased is safer (stronger gradients)")


if __name__ == "__main__":
    main()
