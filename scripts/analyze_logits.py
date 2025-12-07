"""
Analyze raw logits to understand W bias.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn.functional as F
from src.model.trm import TRMConfig, create_model
from src.data.dataset import SyntheticPeptideDataset
from src.constants import IDX_TO_AA, VOCAB
from omegaconf import OmegaConf

# Load config and model
cfg = OmegaConf.load(project_root / 'configs' / 'default.yaml')
model_config = TRMConfig(**cfg.model)
model = create_model(model_config)

# Load checkpoint
checkpoint = torch.load(project_root / 'checkpoints' / 'checkpoint_step_10000.pt', map_location='cpu')
model.load_state_dict(checkpoint['ema_state_dict'])
model.eval()

# Create dataset
dataset = SyntheticPeptideDataset(
    min_length=7,
    max_length=10,
    max_peaks=model_config.max_peaks,
    max_seq_len=model_config.max_seq_len,
    ion_types=['b', 'y'],
)

# Get one sample
sample = next(iter(dataset))

# Run model
batch = {
    'spectrum_masses': sample.spectrum_masses.unsqueeze(0),
    'spectrum_intensities': sample.spectrum_intensities.unsqueeze(0),
    'spectrum_mask': sample.spectrum_mask.unsqueeze(0),
    'precursor_mass': sample.precursor_mass,
    'precursor_charge': sample.precursor_charge,
}

with torch.no_grad():
    all_logits, _ = model(**batch)

# Analyze final step logits
final_logits = all_logits[-1, 0]  # (seq_len, vocab)
final_probs = F.softmax(final_logits, dim=-1)

print("="*80)
print("LOGIT ANALYSIS - Single Sample")
print("="*80)

# For each position, show top 3 predictions
mask = sample.sequence_mask
target = sample.sequence

print("\nPosition-by-position Analysis:")
print("-"*80)
for pos in range(len(mask)):
    if not mask[pos]:
        break

    true_token = target[pos].item()
    true_aa = IDX_TO_AA[true_token]

    logits_at_pos = final_logits[pos]
    probs_at_pos = final_probs[pos]

    # Top 3 predictions
    top_k = torch.topk(probs_at_pos, k=3)

    print(f"\nPosition {pos}: True = {true_aa}")
    print(f"  Top 3 predictions:")
    for i, (prob, idx) in enumerate(zip(top_k.values, top_k.indices)):
        pred_aa = IDX_TO_AA[idx.item()]
        is_correct = "✓" if idx.item() == true_token else " "
        print(f"    {i+1}. {pred_aa:10s} {100*prob.item():5.1f}% {is_correct}")

# Overall statistics
print("\n" + "="*80)
print("AGGREGATE STATISTICS")
print("="*80)

# Average probability assigned to each token across all positions
avg_probs = final_probs[mask].mean(dim=0)  # (vocab,)

print("\nAverage probability by token (top 10):")
for idx in avg_probs.argsort(descending=True)[:10]:
    token = IDX_TO_AA[idx.item()]
    prob = avg_probs[idx].item()
    print(f"  {token:10s}: {100*prob:5.2f}%")

# Check W specifically
w_idx = VOCAB.index('W')
w_avg_prob = avg_probs[w_idx].item()
w_rank = (avg_probs >= w_avg_prob).sum().item()
print(f"\nW (Tryptophan):")
print(f"  Average probability: {100*w_avg_prob:.2f}%")
print(f"  Rank: {w_rank} / {len(VOCAB)}")

# Check entropy (uncertainty)
entropy = -(final_probs * torch.log(final_probs + 1e-10)).sum(dim=-1)
print(f"\nPer-position entropy (uncertainty):")
print(f"  Mean: {entropy[mask].mean().item():.2f} bits")
print(f"  Max: {entropy[mask].max().item():.2f} bits (position {entropy[mask].argmax().item()})")
print(f"  Min: {entropy[mask].min().item():.2f} bits (position {entropy[mask].argmin().item()})")
