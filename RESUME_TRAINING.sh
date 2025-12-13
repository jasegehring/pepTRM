#!/bin/bash
#
# Resume training from step 15000 with FIXED spectrum loss
#
# This script will:
# 1. Load your checkpoint from step 15000 (71.9% token accuracy)
# 2. Continue training with the spectrum loss bug fix applied
# 3. Log to W&B for monitoring
#

echo "================================================================"
echo "RESUMING TRAINING WITH SPECTRUM LOSS FIX"
echo "================================================================"
echo ""
echo "Starting from checkpoint_step_15000.pt"
echo "Current state: 71.9% token accuracy, spectrum loss ~0.99"
echo ""
echo "Expected after fix:"
echo "  - Spectrum loss should start decreasing from 0.99"
echo "  - Token accuracy may dip slightly but should recover"
echo "  - Watch for spectrum loss < 0.90 (10% coverage)"
echo ""
echo "================================================================"
echo ""

python3 scripts/resume_training.py \
    --checkpoint checkpoints_optimized/checkpoint_step_15000.pt \
    --wandb-run-name gallant-water-62-FIXED

echo ""
echo "Training completed or interrupted."
