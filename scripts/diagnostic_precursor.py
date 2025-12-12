"""
Diagnostic Test 1: Wrong Precursor Mass

Tests whether the model relies on precursor mass constraint.
If accuracy collapses with wrong precursor, model is using mass correctly.
If accuracy stays high, model ignores precursor (potential issue).
"""

import sys
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model.trm import create_model, TRMConfig
from src.data.dataset import SyntheticPeptideDataset
from src.constants import WATER_MASS, PROTON_MASS
from src.training.metrics import token_accuracy, sequence_accuracy


def evaluate_with_precursor_offset(model, dataset, num_samples=500, offset_da=0.0, device='cuda'):
    """
    Evaluate model with precursor mass offset.

    Args:
        model: Trained TRM model
        dataset: Synthetic dataset
        num_samples: Number of samples to test
        offset_da: Precursor mass offset in Daltons (0 = correct, ±50 = wrong)
        device: Device to run on

    Returns:
        dict with token_acc, seq_acc, avg_loss
    """
    model.eval()
    model.to(device)

    total_token_acc = 0.0
    total_seq_acc = 0.0
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for i in tqdm(range(num_samples), desc=f"Offset={offset_da:+.0f} Da"):
            # Generate sample
            sample = dataset._generate_sample()
            batch = {
                'peak_masses': sample.spectrum_masses,
                'peak_intensities': sample.spectrum_intensities,
                'peak_mask': sample.spectrum_mask,
                'precursor_mass': sample.precursor_mass[0],
                'precursor_mz': (sample.precursor_mass[0] + sample.precursor_charge[0] * PROTON_MASS) / sample.precursor_charge[0],
                'precursor_charge': sample.precursor_charge[0],
                'sequence': sample.sequence,
                'sequence_mask': sample.sequence_mask,
            }

            # Apply precursor offset
            if offset_da != 0.0:
                # Offset the precursor mass
                precursor_mass = batch['precursor_mass'].item()
                precursor_charge = batch['precursor_charge'].item()

                # Add offset to m/z (detector sees m/z, not mass)
                original_mz = batch['precursor_mz'].item()
                offset_mass = precursor_mass + offset_da
                offset_mz = (offset_mass + precursor_charge * PROTON_MASS) / precursor_charge

                batch['precursor_mass'] = torch.tensor(offset_mass, dtype=torch.float32)
                batch['precursor_mz'] = torch.tensor(offset_mz, dtype=torch.float32)

            # Move to device
            masses = batch['peak_masses'].unsqueeze(0).to(device)
            intensities = batch['peak_intensities'].unsqueeze(0).to(device)
            peak_mask = batch['peak_mask'].unsqueeze(0).to(device)
            precursor_mass = batch['precursor_mass'].unsqueeze(0).to(device)
            precursor_charge = batch['precursor_charge'].unsqueeze(0).to(device)
            sequence = batch['sequence'].unsqueeze(0).to(device)
            sequence_mask = batch['sequence_mask'].unsqueeze(0).to(device)

            # Forward pass (model doesn't take sequence during inference)
            all_logits, _ = model(
                masses, intensities, peak_mask,
                precursor_mass, precursor_charge
            )

            # Use final step predictions
            final_logits = all_logits[-1]  # (batch=1, seq_len, vocab)

            # Calculate metrics
            token_acc = token_accuracy(final_logits, sequence, sequence_mask)
            seq_acc = sequence_accuracy(final_logits, sequence, sequence_mask)

            # Metrics already return floats, not tensors
            total_token_acc += token_acc if isinstance(token_acc, float) else token_acc.item()
            total_seq_acc += seq_acc if isinstance(seq_acc, float) else seq_acc.item()
            count += 1

    return {
        'token_acc': total_token_acc / count,
        'seq_acc': total_seq_acc / count,
        'num_samples': count
    }


def main():
    print("=" * 80)
    print("DIAGNOSTIC TEST 1: WRONG PRECURSOR MASS")
    print("=" * 80)
    print("\nPurpose: Test if model uses precursor mass constraint")
    print("Expected: Accuracy should collapse with wrong precursor mass\n")

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    # Load model checkpoint
    checkpoint_path = project_root / 'checkpoints_optimized' / 'best_model.pt'
    if not checkpoint_path.exists():
        checkpoint_path = project_root / 'checkpoints_optimized' / 'final_model.pt'

    print(f"Loading checkpoint: {checkpoint_path.name}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Create model
    model_config = TRMConfig(
        hidden_dim=256,
        num_encoder_layers=2,
        num_decoder_layers=2,
        num_heads=4,
        max_peaks=100,
        max_seq_len=25,
        num_supervision_steps=8,
        num_latent_steps=4,
        dropout=0.1,
    )
    model = create_model(model_config)

    # Handle torch.compile() wrapper (_orig_mod prefix)
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    print(f"Model loaded (step {checkpoint.get('step', 'unknown')})\n")

    # Create clean validation dataset (Stage 1 difficulty)
    print("Creating clean validation dataset...")
    dataset = SyntheticPeptideDataset(
        min_length=7,
        max_length=10,
        max_peaks=100,
        max_seq_len=25,
        ion_types=['b', 'y'],
        include_neutral_losses=False,
        noise_peaks=0,
        peak_dropout=0.0,
        mass_error_ppm=0.0,
        charge_distribution={2: 0.7, 3: 0.3},
    )
    print(f"Dataset created (clean, length 7-10aa)\n")

    # Test configurations
    num_samples = 500
    offsets = [0.0, 10.0, 25.0, 50.0, 100.0, -50.0]  # Daltons

    print(f"Testing with {num_samples} samples per offset...")
    print("-" * 80)

    results = {}
    for offset in offsets:
        result = evaluate_with_precursor_offset(
            model, dataset, num_samples, offset_da=offset, device=device
        )
        results[offset] = result

        print(f"\nOffset: {offset:+6.1f} Da")
        print(f"  Token Accuracy:    {result['token_acc']*100:5.2f}%")
        print(f"  Sequence Accuracy: {result['seq_acc']*100:5.2f}%")

    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    baseline_token = results[0.0]['token_acc']
    baseline_seq = results[0.0]['seq_acc']

    print(f"\nBaseline (correct precursor):")
    print(f"  Token Accuracy:    {baseline_token*100:.2f}%")
    print(f"  Sequence Accuracy: {baseline_seq*100:.2f}%")

    print(f"\nAccuracy degradation with offset:")
    for offset in offsets:
        if offset == 0.0:
            continue
        token_drop = (baseline_token - results[offset]['token_acc']) * 100
        seq_drop = (baseline_seq - results[offset]['seq_acc']) * 100

        print(f"  {offset:+6.1f} Da: Token Δ={token_drop:+5.2f}%, Seq Δ={seq_drop:+5.2f}%")

    # Interpretation
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    # Check if accuracy drops significantly
    worst_offset = max([o for o in offsets if o != 0], key=lambda o: abs(o))
    worst_token = results[worst_offset]['token_acc']
    token_drop_pct = ((baseline_token - worst_token) / baseline_token) * 100

    print(f"\nWith {worst_offset:+.0f} Da offset:")
    print(f"  Token accuracy drops {token_drop_pct:.1f}% (relative)")

    if token_drop_pct > 30:
        print("\n✅ PASS: Model relies on precursor mass constraint")
        print("   Large accuracy drop indicates model uses precursor information")
    elif token_drop_pct > 10:
        print("\n⚠️  PARTIAL: Model uses precursor mass, but not strongly")
        print("   Moderate drop suggests partial reliance on precursor")
    else:
        print("\n❌ FAIL: Model ignores precursor mass")
        print("   Small accuracy drop suggests model doesn't use precursor")
        print("   Recommendation: Add explicit precursor mass loss term")

    # Save results
    results_file = project_root / 'DIAGNOSTIC_PRECURSOR_RESULTS.txt'
    with open(results_file, 'w') as f:
        f.write("DIAGNOSTIC TEST 1: WRONG PRECURSOR MASS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Checkpoint: {checkpoint_path.name}\n")
        f.write(f"Samples per offset: {num_samples}\n\n")
        f.write("RESULTS:\n")
        for offset in offsets:
            f.write(f"  {offset:+6.1f} Da: Token={results[offset]['token_acc']*100:.2f}%, ")
            f.write(f"Seq={results[offset]['seq_acc']*100:.2f}%\n")
        f.write(f"\nToken accuracy drop with {worst_offset:+.0f} Da: {token_drop_pct:.1f}%\n")

    print(f"\nResults saved to: {results_file}")
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
