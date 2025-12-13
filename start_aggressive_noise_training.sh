#!/bin/bash

# Aggressive Noise Training Launcher
# Tests if noise alone unlocks multi-step refinement

echo "======================================================================"
echo "AGGRESSIVE NOISE TRAINING - MIXED CLEAN/NOISY CURRICULUM"
echo "======================================================================"
echo ""
echo "Strategy:"
echo "  • Start with 80% clean data, gradually increase noise"
echo "  • Stage 1-2: Learn basics on mostly clean data"
echo "  • Stage 3-4: Transition to majority noisy data"
echo "  • Stage 5-7: Pure noisy data (100%)"
echo ""
echo "Configuration:"
echo "  • Exponential iteration weighting (force step 7 optimization)"
echo "  • Precursor loss enabled (0.05 → 0.3)"
echo "  • Spectrum loss DISABLED"
echo "  • Batch size 80 + torch.compile"
echo ""
echo "Checkpoint directory: checkpoints_aggressive_noise/"
echo ""
echo "======================================================================"
echo ""

# Clear CUDA cache
python -c "import torch; torch.cuda.empty_cache(); print('✓ CUDA cache cleared')"

# Run training
python scripts/train_aggressive_noise.py "$@"
