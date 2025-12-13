#!/bin/bash
# Launch progressive length training
#
# Key differences from aggressive_noise training:
# - Phase 1 (0-50K): 100% clean data, progressively extend length range
# - Phase 2 (50K-100K): Introduce noise gradually while keeping all lengths
#
# Expected milestones:
# - Step 15K: >90% accuracy on length 14 (clean)
# - Step 40K: >90% accuracy on length 25 (clean)
# - Step 50K: All lengths mastered
# - Step 100K: Full noise robustness

set -e

echo "=================================================="
echo "PROGRESSIVE LENGTH TRAINING"
echo "=================================================="
echo ""
echo "This training run is designed to fix the length"
echo "generalization failure observed in aggressive noise"
echo "training, where accuracy dropped from 99% (len 12)"
echo "to 40% (len 14)."
echo ""
echo "Key changes:"
echo "  1. 100% clean data for first 50K steps"
echo "  2. Length extended gradually: 10 -> 12 -> 14 -> ... -> 25"
echo "  3. Noise introduced ONLY after all lengths are learned"
echo ""

# Activate conda environment if available
if command -v conda &> /dev/null; then
    echo "Activating conda environment..."
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate pepTRMenv 2>/dev/null || echo "Note: pepTRMenv not found, using current environment"
fi

# Create checkpoint directory
mkdir -p checkpoints_progressive_length

# Run training
echo ""
echo "Starting training..."
echo ""

python scripts/train_progressive_length.py "$@"
