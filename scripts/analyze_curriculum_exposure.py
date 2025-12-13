"""
Analyze how many samples at each length the model sees during training.

The curriculum progressively expands length range. But if most training
happens at shorter lengths, the model may not get enough exposure to
longer sequences.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.curriculum_aggressive_noise import AGGRESSIVE_NOISE_CURRICULUM
import numpy as np

def analyze_curriculum():
    print("=" * 70)
    print("CURRICULUM LENGTH EXPOSURE ANALYSIS")
    print("=" * 70)

    # Track exposure at each length
    total_steps = sum(stage.steps for stage in AGGRESSIVE_NOISE_CURRICULUM)
    length_exposure = {}  # length -> total steps exposed to this length

    print("\n--- CURRICULUM STAGES ---")
    print(f"{'Stage':<25} {'Steps':<12} {'Lengths':<15} {'Clean Ratio':<12}")
    print("-" * 65)

    for i, stage in enumerate(AGGRESSIVE_NOISE_CURRICULUM):
        print(f"{i+1}. {stage.name:<21} {stage.steps:<12,} {stage.min_length}-{stage.max_length:<11} {stage.clean_data_ratio:.0%}")

        # Each length in range gets equal probability
        lengths_in_stage = range(stage.min_length, stage.max_length + 1)
        num_lengths = len(lengths_in_stage)

        for length in lengths_in_stage:
            if length not in length_exposure:
                length_exposure[length] = 0
            # Each sample has 1/num_lengths probability of being this length
            # So expected exposure = steps / num_lengths
            length_exposure[length] += stage.steps / num_lengths

    print(f"\nTotal training steps: {total_steps:,}")

    print("\n--- LENGTH EXPOSURE ---")
    print(f"{'Length':<10} {'Exposure (steps)':<20} {'% of Training':<15}")
    print("-" * 45)

    for length in sorted(length_exposure.keys()):
        exposure = length_exposure[length]
        pct = exposure / total_steps * 100
        bar = "█" * int(pct / 2)
        print(f"{length:<10} {exposure:>15,.0f}     {pct:>5.1f}%  {bar}")

    print("\n--- KEY OBSERVATIONS ---")

    # Find cliff point
    prev_exposure = None
    cliff_length = None
    for length in sorted(length_exposure.keys()):
        exposure = length_exposure[length]
        if prev_exposure is not None and exposure < prev_exposure * 0.5:
            cliff_length = length
            print(f"⚠️  Exposure drops by >50% at length {cliff_length}")
        prev_exposure = exposure

    # How much exposure do lengths 13+ get vs 7-12?
    short_exposure = sum(length_exposure.get(l, 0) for l in range(7, 13))
    long_exposure = sum(length_exposure.get(l, 0) for l in range(13, 26))

    print(f"\nTotal exposure for lengths 7-12:  {short_exposure:>12,.0f} steps")
    print(f"Total exposure for lengths 13-25: {long_exposure:>12,.0f} steps")
    print(f"Ratio (short/long):               {short_exposure/long_exposure:.1f}x")

    # Average exposure per length
    short_avg = short_exposure / 6  # 6 lengths (7-12)
    long_avg = long_exposure / 13   # 13 lengths (13-25)
    print(f"\nAvg exposure per length (7-12):   {short_avg:>12,.0f} steps")
    print(f"Avg exposure per length (13-25):  {long_avg:>12,.0f} steps")
    print(f"Ratio (short/long):               {short_avg/long_avg:.1f}x")

    # When does each length first appear?
    print("\n--- FIRST APPEARANCE ---")
    print(f"{'Length':<10} {'First Stage':<25} {'Step':<10}")
    print("-" * 45)

    for length in sorted(length_exposure.keys()):
        cumulative = 0
        for i, stage in enumerate(AGGRESSIVE_NOISE_CURRICULUM):
            if stage.min_length <= length <= stage.max_length:
                print(f"{length:<10} {stage.name:<25} {cumulative:>,}")
                break
            cumulative += stage.steps

    # Analyze clean vs noisy exposure
    print("\n--- CLEAN DATA EXPOSURE ---")

    clean_exposure = {}
    for i, stage in enumerate(AGGRESSIVE_NOISE_CURRICULUM):
        lengths_in_stage = range(stage.min_length, stage.max_length + 1)
        num_lengths = len(lengths_in_stage)

        for length in lengths_in_stage:
            if length not in clean_exposure:
                clean_exposure[length] = 0
            # Clean samples = steps * clean_ratio / num_lengths
            clean_exposure[length] += stage.steps * stage.clean_data_ratio / num_lengths

    print(f"{'Length':<10} {'Clean Exposure':<20} {'% Clean':<15}")
    print("-" * 45)

    for length in sorted(clean_exposure.keys()):
        clean = clean_exposure[length]
        total = length_exposure[length]
        pct_clean = clean / total * 100 if total > 0 else 0
        print(f"{length:<10} {clean:>15,.0f}     {pct_clean:>5.1f}%")

    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)

    # Check if clean exposure is sufficient for longer lengths
    clean_12 = clean_exposure.get(12, 0)
    clean_14 = clean_exposure.get(14, 0)

    print(f"""
Lengths 7-12 get {short_avg/long_avg:.1f}x more average exposure than 13-25.

Clean data exposure:
- Length 12: {clean_12:,.0f} clean samples
- Length 14: {clean_14:,.0f} clean samples (only {clean_14/clean_12*100:.0f}% of length 12)

The curriculum introduces length 13-15 in Stage 2 (starting at step 10K)
with 60% clean, 40% noisy. By then, the model has already learned:
- Strong priors for lengths 7-12 from Stage 1 (80% clean)
- Length 13-15 appear WITH noise, making them harder to learn

RECOMMENDATION:
- Model needs MORE clean exposure to longer sequences BEFORE noise
- Current curriculum may be causing negative transfer:
  "Long sequence = noisy/hard" association
""")


if __name__ == '__main__':
    analyze_curriculum()
