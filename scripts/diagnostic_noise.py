"""
Diagnostic Test 3: Noise Decomposition & Charge State Analysis

Tests which noise component hurts performance most:
- Peak dropout (missing fragments)
- Noise peaks (contaminants)
- Mass error (instrument precision)

Also tests charge state (+2 vs +3) performance.
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
from src.training.metrics import token_accuracy, sequence_accuracy


def evaluate_noise_configuration(model, noise_config, num_samples=500, device='cuda'):
    """
    Evaluate model with specific noise configuration.

    Args:
        model: Trained TRM model
        noise_config: Dict with noise parameters
        num_samples: Number of samples
        device: Device to run on

    Returns:
        dict with token_acc, seq_acc
    """
    model.eval()
    model.to(device)

    # Create dataset with specified noise
    dataset = SyntheticPeptideDataset(
        min_length=7,
        max_length=15,  # Medium range
        max_peaks=100,
        max_seq_len=25,
        ion_types=['b', 'y'],
        include_neutral_losses=False,
        charge_distribution=noise_config.get('charge_distribution', {2: 0.7, 3: 0.3}),
        noise_peaks=noise_config.get('noise_peaks', 0),
        peak_dropout=noise_config.get('peak_dropout', 0.0),
        mass_error_ppm=noise_config.get('mass_error_ppm', 0.0),
        intensity_variation=noise_config.get('intensity_variation', 0.0),
    )

    total_token_acc = 0.0
    total_seq_acc = 0.0
    count = 0

    with torch.no_grad():
        for i in tqdm(range(num_samples), desc=noise_config.get('name', 'Testing'), leave=False):
            # Generate sample
            sample = dataset._generate_sample()
            batch = {
                'peak_masses': sample.spectrum_masses,
                'peak_intensities': sample.spectrum_intensities,
                'peak_mask': sample.spectrum_mask,
                'precursor_mass': sample.precursor_mass[0],
                'precursor_mz': (sample.precursor_mass[0] + sample.precursor_charge[0] * 1.007276) / sample.precursor_charge[0],
                'precursor_charge': sample.precursor_charge[0],
                'sequence': sample.sequence,
                'sequence_mask': sample.sequence_mask,
            }

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
            final_logits = all_logits[-1]

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
    print("DIAGNOSTIC TEST 3: NOISE DECOMPOSITION & CHARGE STATE")
    print("=" * 80)
    print("\nPurpose: Identify which noise type hurts performance most")
    print("Expected: Some noise types worse than others\n")

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    # Load model
    checkpoint_path = project_root / 'checkpoints_optimized' / 'best_model.pt'
    if not checkpoint_path.exists():
        checkpoint_path = project_root / 'checkpoints_optimized' / 'final_model.pt'

    print(f"Loading checkpoint: {checkpoint_path.name}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

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

    # Test configurations
    num_samples = 500

    # Part 1: Noise Decomposition
    print("=" * 80)
    print("PART 1: NOISE DECOMPOSITION")
    print("=" * 80)

    noise_configs = [
        {
            'name': 'Clean (baseline)',
            'noise_peaks': 0,
            'peak_dropout': 0.0,
            'mass_error_ppm': 0.0,
        },
        {
            'name': 'Dropout only (15%)',
            'noise_peaks': 0,
            'peak_dropout': 0.15,
            'mass_error_ppm': 0.0,
        },
        {
            'name': 'Noise peaks only (10)',
            'noise_peaks': 10,
            'peak_dropout': 0.0,
            'mass_error_ppm': 0.0,
        },
        {
            'name': 'Mass error only (15ppm)',
            'noise_peaks': 0,
            'peak_dropout': 0.0,
            'mass_error_ppm': 15.0,
        },
        {
            'name': 'Dropout + Noise peaks',
            'noise_peaks': 10,
            'peak_dropout': 0.15,
            'mass_error_ppm': 0.0,
        },
        {
            'name': 'Dropout + Mass error',
            'noise_peaks': 0,
            'peak_dropout': 0.15,
            'mass_error_ppm': 15.0,
        },
        {
            'name': 'Noise peaks + Mass error',
            'noise_peaks': 10,
            'peak_dropout': 0.0,
            'mass_error_ppm': 15.0,
        },
        {
            'name': 'ALL (realistic)',
            'noise_peaks': 10,
            'peak_dropout': 0.15,
            'mass_error_ppm': 15.0,
        },
    ]

    noise_results = {}
    for config in noise_configs:
        print(f"\nTesting: {config['name']}")
        result = evaluate_noise_configuration(model, config, num_samples, device)
        noise_results[config['name']] = result

        print(f"  Token Accuracy:    {result['token_acc']*100:5.2f}%")
        print(f"  Sequence Accuracy: {result['seq_acc']*100:5.2f}%")

    # Part 2: Charge State Analysis
    print("\n" + "=" * 80)
    print("PART 2: CHARGE STATE ANALYSIS")
    print("=" * 80)

    charge_configs = [
        {
            'name': 'Charge +2 only (clean)',
            'charge_distribution': {2: 1.0},
            'noise_peaks': 0,
            'peak_dropout': 0.0,
            'mass_error_ppm': 0.0,
        },
        {
            'name': 'Charge +3 only (clean)',
            'charge_distribution': {3: 1.0},
            'noise_peaks': 0,
            'peak_dropout': 0.0,
            'mass_error_ppm': 0.0,
        },
        {
            'name': 'Charge +2 only (realistic)',
            'charge_distribution': {2: 1.0},
            'noise_peaks': 10,
            'peak_dropout': 0.15,
            'mass_error_ppm': 15.0,
        },
        {
            'name': 'Charge +3 only (realistic)',
            'charge_distribution': {3: 1.0},
            'noise_peaks': 10,
            'peak_dropout': 0.15,
            'mass_error_ppm': 15.0,
        },
    ]

    charge_results = {}
    for config in charge_configs:
        print(f"\nTesting: {config['name']}")
        result = evaluate_noise_configuration(model, config, num_samples, device)
        charge_results[config['name']] = result

        print(f"  Token Accuracy:    {result['token_acc']*100:5.2f}%")
        print(f"  Sequence Accuracy: {result['seq_acc']*100:5.2f}%")

    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    baseline = noise_results['Clean (baseline)']
    baseline_token = baseline['token_acc'] * 100

    print(f"\nBaseline (clean): {baseline_token:.2f}% token accuracy\n")
    print("Impact of individual noise types:")

    # Calculate individual impacts
    dropout_impact = baseline_token - noise_results['Dropout only (15%)']['token_acc'] * 100
    noise_peaks_impact = baseline_token - noise_results['Noise peaks only (10)']['token_acc'] * 100
    mass_error_impact = baseline_token - noise_results['Mass error only (15ppm)']['token_acc'] * 100
    combined_impact = baseline_token - noise_results['ALL (realistic)']['token_acc'] * 100

    print(f"  Dropout (15%):       Δ={dropout_impact:+5.2f}%")
    print(f"  Noise peaks (10):    Δ={noise_peaks_impact:+5.2f}%")
    print(f"  Mass error (15ppm):  Δ={mass_error_impact:+5.2f}%")
    print(f"  ALL combined:        Δ={combined_impact:+5.2f}%")

    # Find worst offender
    impacts = {
        'Dropout': dropout_impact,
        'Noise peaks': noise_peaks_impact,
        'Mass error': mass_error_impact,
    }
    worst_noise = max(impacts.items(), key=lambda x: x[1])

    print(f"\n🔴 Worst offender: {worst_noise[0]} (Δ={worst_noise[1]:.2f}%)")
    print(f"   This noise type has the largest negative impact on accuracy")

    # Charge state analysis
    print("\nCharge state comparison:")
    charge2_clean = charge_results['Charge +2 only (clean)']['token_acc'] * 100
    charge3_clean = charge_results['Charge +3 only (clean)']['token_acc'] * 100
    charge_diff = charge2_clean - charge3_clean

    print(f"  +2 (clean): {charge2_clean:.2f}%")
    print(f"  +3 (clean): {charge3_clean:.2f}%")
    print(f"  Difference: {charge_diff:+.2f}%")

    if abs(charge_diff) < 3:
        print("  ✅ Model handles both charge states equally")
    elif charge_diff > 0:
        print("  ⚠️  Model prefers +2 charge state")
    else:
        print("  ⚠️  Model prefers +3 charge state")

    # Interpretation
    print("\n" + "=" * 80)
    print("INTERPRETATION & RECOMMENDATIONS")
    print("=" * 80)

    if worst_noise[0] == 'Dropout' and worst_noise[1] > 10:
        print("\n1. DROPOUT is the main challenge")
        print("   Recommendation:")
        print("   - Increase dropout during training (train harder)")
        print("   - Consider adding attention to precursor mass (global constraint)")
        print("   - Longer peptides suffer more from dropout (fewer peaks)")

    if worst_noise[0] == 'Noise peaks' and worst_noise[1] > 10:
        print("\n2. NOISE PEAKS are the main challenge")
        print("   Recommendation:")
        print("   - Model may be treating all peaks as signal")
        print("   - Consider attention weights analysis (which peaks attended to)")
        print("   - May need better peak filtering/ranking")

    if worst_noise[0] == 'Mass error' and worst_noise[1] > 10:
        print("\n3. MASS ERROR is the main challenge")
        print("   Recommendation:")
        print("   - Increase mass_tolerance in spectrum matching loss")
        print("   - Current tolerance may be too strict")
        print("   - Review mass embedding resolution")

    # Check additive vs synergistic effects
    expected_combined = dropout_impact + noise_peaks_impact + mass_error_impact
    actual_combined = combined_impact
    synergy = actual_combined - expected_combined

    print(f"\nNoise interaction effect:")
    print(f"  Expected (additive): {expected_combined:.2f}%")
    print(f"  Actual (combined):   {actual_combined:.2f}%")
    print(f"  Synergy:             {synergy:+.2f}%")

    if synergy > 5:
        print("  ⚠️  SYNERGISTIC: Noise types amplify each other")
        print("     Multiple noise sources make problem much harder")
    elif synergy < -5:
        print("  ✅ SUBADDITIVE: Some noise types overlap")
        print("     Model partially robust to combined noise")
    else:
        print("  ✅ ADDITIVE: Noise effects are independent")

    # Save results
    results_file = project_root / 'DIAGNOSTIC_NOISE_RESULTS.txt'
    with open(results_file, 'w') as f:
        f.write("DIAGNOSTIC TEST 3: NOISE DECOMPOSITION & CHARGE STATE\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Checkpoint: {checkpoint_path.name}\n")
        f.write(f"Samples per config: {num_samples}\n\n")

        f.write("NOISE DECOMPOSITION:\n")
        for name, result in noise_results.items():
            f.write(f"  {name:30s}: Token={result['token_acc']*100:5.2f}%, ")
            f.write(f"Seq={result['seq_acc']*100:5.2f}%\n")

        f.write("\nCHARGE STATE ANALYSIS:\n")
        for name, result in charge_results.items():
            f.write(f"  {name:30s}: Token={result['token_acc']*100:5.2f}%, ")
            f.write(f"Seq={result['seq_acc']*100:5.2f}%\n")

        f.write(f"\nWorst noise type: {worst_noise[0]} (Δ={worst_noise[1]:.2f}%)\n")
        f.write(f"Noise synergy: {synergy:+.2f}%\n")

    print(f"\nResults saved to: {results_file}")
    print("=" * 80)


if __name__ == '__main__':
    main()
