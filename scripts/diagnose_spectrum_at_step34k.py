"""Diagnose spectrum matching at current training step."""
import sys
sys.path.insert(0, '/home/jgehring/pepTRM')

import torch
from src.model.recursive_peptide_model import RecursivePeptideModel
from src.data.ms2pip_dataset import MS2PIPDataset
from src.data.ion_types import compute_theoretical_peaks
from omegaconf import OmegaConf

print("=" * 70)
print("SPECTRUM MATCHING DIAGNOSTIC - Step ~34K")
print("=" * 70)

# Load config
cfg = OmegaConf.load('configs/optimized_extended.yaml')

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RecursivePeptideModel(
    vocab_size=25,
    hidden_dim=cfg.model.hidden_dim,
    num_encoder_layers=cfg.model.num_encoder_layers,
    num_decoder_layers=cfg.model.num_decoder_layers,
    num_heads=cfg.model.num_heads,
    max_peaks=cfg.model.max_peaks,
    max_seq_len=cfg.model.max_seq_len,
    num_supervision_steps=cfg.model.num_supervision_steps,
    num_latent_steps=cfg.model.num_latent_steps,
    dropout=cfg.model.dropout
).to(device)

# Find latest checkpoint
import glob
checkpoints = sorted(glob.glob('checkpoints_optimized/checkpoint_step_*.pt'),
                      key=lambda x: int(x.split('_')[-1].split('.')[0]))
latest_ckpt = checkpoints[-1] if checkpoints else None

if latest_ckpt:
    print(f"\n✓ Loading checkpoint: {latest_ckpt}")
    checkpoint = torch.load(latest_ckpt, map_location=device)

    # Handle _orig_mod prefix if present
    model_state = checkpoint['model_state_dict']
    if any(k.startswith('_orig_mod.') for k in model_state.keys()):
        model_state = {k.replace('_orig_mod.', ''): v for k, v in model_state.items()}

    model.load_state_dict(model_state)
    step = checkpoint.get('step', 'unknown')
    print(f"   Step: {step}")
else:
    print("\n✗ No checkpoint found!")
    sys.exit(1)

model.eval()

# Create dataset (use curriculum stage 5 settings - step 30K+)
dataset = MS2PIPDataset(
    min_length=12,
    max_length=18,
    charge_distribution={2: 0.7, 3: 0.3},
    num_samples=100,
    ms2pip_model=cfg.data.ms2pip_model,
    noise_level=0.1,
    top_k_peaks=cfg.data.top_k_peaks
)

print(f"\n✓ Dataset created: {len(dataset)} samples (length 12-18)")

# Test on 5 samples
from src.data.vocabulary import Vocabulary
vocab = Vocabulary()

print("\n" + "=" * 70)
print("TESTING ON 5 SAMPLES:")
print("=" * 70)

coverage_scores = []

with torch.no_grad():
    for i in range(5):
        sample = dataset[i]

        # Prepare batch
        target_ids = sample['target_ids'].unsqueeze(0).to(device)
        observed_masses = sample['observed_masses'].unsqueeze(0).to(device)
        observed_intensities = sample['observed_intensities'].unsqueeze(0).to(device)
        peak_mask = sample['peak_mask'].unsqueeze(0).to(device)
        precursor_mass = sample['precursor_mass'].unsqueeze(0).to(device)
        precursor_charge = sample['precursor_charge'].unsqueeze(0).to(device)

        # Forward pass
        output = model(
            observed_masses=observed_masses,
            observed_intensities=observed_intensities,
            peak_mask=peak_mask,
            precursor_mass=precursor_mass,
            precursor_charge=precursor_charge,
            target_ids=target_ids[:, :-1]
        )

        # Get final predictions
        final_logits = output['logits'][:, -1, :, :]  # [B, S, V]
        final_probs = torch.softmax(final_logits, dim=-1)

        # Compute theoretical peaks
        target_mask = (target_ids[:, 1:] != vocab.pad_id).float()

        # Get AA masses
        from src.data.ion_types import AA_MASSES
        aa_mass_list = [AA_MASSES.get(vocab.idx_to_token[i], 0.0) for i in range(len(vocab))]
        aa_masses = torch.tensor(aa_mass_list, dtype=torch.float32, device=device)

        predicted_masses = compute_theoretical_peaks(
            sequence_probs=final_probs,
            aa_masses=aa_masses,
            ion_type_names=['b', 'y', 'b++', 'y++'],
            sequence_mask=target_mask
        )

        # Compute coverage (Gaussian matched filter)
        sigma = 10.0  # 10 Da tolerance
        distances = torch.abs(
            observed_masses.unsqueeze(2) - predicted_masses.unsqueeze(1)
        )
        similarities = torch.exp(-0.5 * (distances / sigma) ** 2)
        max_sim_per_obs_peak, _ = similarities.max(dim=2)

        # Weight by intensity and mask
        weighted_coverage = (max_sim_per_obs_peak * observed_intensities * peak_mask).sum(dim=1)
        total_intensity = (observed_intensities * peak_mask).sum(dim=1)
        coverage = (weighted_coverage / (total_intensity + 1e-8)).item()

        # Decode sequence
        pred_ids = final_probs.argmax(dim=-1)[0]
        pred_seq = ''.join([vocab.idx_to_token[idx.item()] for idx in pred_ids if idx.item() not in [vocab.sos_id, vocab.eos_id, vocab.pad_id]])
        true_seq = ''.join([vocab.idx_to_token[idx.item()] for idx in target_ids[0, 1:] if idx.item() not in [vocab.sos_id, vocab.eos_id, vocab.pad_id]])

        correct = (pred_ids == target_ids[0, 1:-1]).float().mean().item()

        print(f"\nSample {i+1}:")
        print(f"  True:      {true_seq}")
        print(f"  Predicted: {pred_seq}")
        print(f"  Token Acc: {correct*100:.1f}%")
        print(f"  Coverage:  {coverage*100:.1f}%")
        print(f"  # predicted peaks: {predicted_masses.shape[1]}")
        print(f"  # observed peaks:  {peak_mask.sum().item():.0f}")

        # Check for anomalies
        max_pred_mass = predicted_masses.max().item()
        print(f"  Max predicted mass: {max_pred_mass:.1f} Da")
        if max_pred_mass > 2000:
            print(f"  ⚠️  WARNING: Predicted mass > 2000 Da (PAD bug?)")

        coverage_scores.append(coverage * 100)

print("\n" + "=" * 70)
print(f"AVERAGE COVERAGE: {sum(coverage_scores)/len(coverage_scores):.1f}%")
print("=" * 70)

if sum(coverage_scores)/len(coverage_scores) < 10:
    print("\n❌ VERY LOW COVERAGE - Spectrum matching not working!")
    print("   Expected: 15-30% at 91% token accuracy")
    print("   Actual:   {:.1f}%".format(sum(coverage_scores)/len(coverage_scores)))
elif sum(coverage_scores)/len(coverage_scores) < 20:
    print("\n⚠️  LOW COVERAGE - Spectrum matching struggling")
    print("   Expected: 15-30% at 91% token accuracy")
    print("   Actual:   {:.1f}%".format(sum(coverage_scores)/len(coverage_scores)))
else:
    print("\n✓ Coverage looks reasonable for current accuracy")
