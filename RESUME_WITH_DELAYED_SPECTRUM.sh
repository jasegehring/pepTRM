#!/bin/bash
#
# Resume training with DELAYED spectrum loss curriculum
#
# Changes:
# - Delays spectrum loss introduction from step 15K → 30K
# - Gives model time to reach 65% accuracy before spectrum loss
# - Resumes wandb run ID: nbs1e6hk
#

echo "======================================================================"
echo "RESUMING TRAINING WITH DELAYED SPECTRUM CURRICULUM"
echo "======================================================================"
echo ""
echo "Checkpoint: checkpoints_optimized/checkpoint_step_15000.pt"
echo "W&B: Creates NEW run (for comparison to old run nbs1e6hk)"
echo ""
echo "Old run (nbs1e6hk):"
echo "  Steps 0-31K with BUGGY spectrum loss"
echo "  Spectrum loss introduced at step 15K"
echo "  Caused accuracy drop (48% → ?)"
echo ""
echo "New run (this one):"
echo "  Steps 15K-100K with FIXED spectrum loss"
echo "  Delayed curriculum: No spectrum until step 30K"
echo "  Should improve accuracy 48% → 65%+ by step 30K"
echo ""
echo "Comparison:"
echo "  You can compare old vs new in W&B to see the fix working!"
echo ""
echo "======================================================================"
echo ""

python3 scripts/resume_with_delayed_spectrum.py \
    --checkpoint checkpoints_optimized/checkpoint_step_15000.pt

echo ""
echo "Training completed or interrupted."
