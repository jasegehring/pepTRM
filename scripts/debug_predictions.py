"""
Quick diagnostic to understand what the model is predicting.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from src.model.trm import TRMConfig, create_model
from src.data.dataset import SyntheticPeptideDataset, collate_peptide_samples
from src.training.metrics import decode_sequence, compute_mass_error_ppm
from src.constants import IDX_TO_AA, AMINO_ACID_MASSES, WATER_MASS
from omegaconf import OmegaConf

# Load config
config_path = project_root / 'configs' / 'default.yaml'
cfg = OmegaConf.load(config_path)

# Create model
model_config = TRMConfig(**cfg.model)
model = create_model(model_config)

# Load EMA checkpoint
checkpoint_path = project_root / 'checkpoints' / 'checkpoint_step_10000.pt'
if checkpoint_path.exists():
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    # Load EMA weights if available
    if 'ema_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['ema_state_dict'])
        print(f"Loaded EMA checkpoint from step {checkpoint['global_step']}")
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model checkpoint from step {checkpoint['global_step']}")
else:
    print("No checkpoint found - using random initialization")

model.eval()

# Create dataset
dataset = SyntheticPeptideDataset(
    min_length=7,
    max_length=10,
    max_peaks=model_config.max_peaks,
    max_seq_len=model_config.max_seq_len,
    ion_types=['b', 'y'],
)

# Generate 10 samples
print("\n" + "="*80)
print("PREDICTION ANALYSIS - 10 Random Samples")
print("="*80)

total_tokens = 0
correct_tokens = 0
mass_errors = []
length_errors = []
token_distribution = {idx: 0 for idx in range(len(IDX_TO_AA))}

for sample_idx in range(10):
    # Get sample
    sample = next(iter(dataset))

    # Convert to batch format
    batch = {
        'spectrum_masses': sample.spectrum_masses.unsqueeze(0),
        'spectrum_intensities': sample.spectrum_intensities.unsqueeze(0),
        'spectrum_mask': sample.spectrum_mask.unsqueeze(0),
        'precursor_mass': sample.precursor_mass,  # Already has shape (1,)
        'precursor_charge': sample.precursor_charge,  # Already has shape (1,)
    }

    # Get ground truth
    true_seq = decode_sequence(sample.sequence, sample.sequence_mask)
    true_mass = sample.precursor_mass.item()

    # Predict
    with torch.no_grad():
        all_logits, _ = model(
            spectrum_masses=batch['spectrum_masses'],
            spectrum_intensities=batch['spectrum_intensities'],
            spectrum_mask=batch['spectrum_mask'],
            precursor_mass=batch['precursor_mass'],
            precursor_charge=batch['precursor_charge'],
        )

    # Get final prediction
    final_logits = all_logits[-1, 0]  # (seq_len, vocab)
    predictions = final_logits.argmax(dim=-1)

    # Count token distribution
    for idx in predictions.tolist():
        if idx < len(IDX_TO_AA):
            token_distribution[idx] += 1

    # Decode prediction
    pred_seq = decode_sequence(predictions, sample.sequence_mask)

    # Compute mass error
    if pred_seq:
        pred_mass = sum(AMINO_ACID_MASSES.get(aa, 0) for aa in pred_seq) + WATER_MASS
        mass_error_ppm = abs(pred_mass - true_mass) / true_mass * 1e6
        mass_errors.append(mass_error_ppm)
        length_errors.append(len(pred_seq) - len(true_seq))
    else:
        pred_mass = 0
        mass_error_ppm = float('inf')

    # Token accuracy for this sample
    mask = sample.sequence_mask
    target = sample.sequence
    correct = ((predictions == target) & mask).sum().item()
    total = mask.sum().item()
    correct_tokens += correct
    total_tokens += total

    print(f"\nSample {sample_idx + 1}:")
    print(f"  True:      {true_seq} ({len(true_seq)} AA, {true_mass:.1f} Da)")
    print(f"  Predicted: {pred_seq} ({len(pred_seq)} AA, {pred_mass:.1f} Da)")
    print(f"  Token Acc: {correct}/{total} ({100*correct/total:.1f}%)")
    print(f"  Mass Error: {mass_error_ppm:.0f} ppm")
    print(f"  Length Error: {len(pred_seq) - len(true_seq):+d}")

print("\n" + "="*80)
print("AGGREGATE STATISTICS")
print("="*80)
print(f"Overall Token Accuracy: {100*correct_tokens/total_tokens:.1f}%")
print(f"Mean Mass Error: {sum(mass_errors)/len(mass_errors) if mass_errors else float('inf'):.0f} ppm")
print(f"Mean Length Error: {sum(length_errors)/len(length_errors) if length_errors else 0:.1f} AA")

print("\n" + "="*80)
print("TOKEN DISTRIBUTION (Top 10 predicted tokens)")
print("="*80)
sorted_tokens = sorted(token_distribution.items(), key=lambda x: x[1], reverse=True)
for idx, count in sorted_tokens[:10]:
    token = IDX_TO_AA.get(idx, f"IDX_{idx}")
    pct = 100 * count / sum(token_distribution.values())
    print(f"  {token:10s}: {count:4d} ({pct:5.1f}%)")
