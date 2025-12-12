"""
Diagnostic Test 2: Performance vs Peptide Length

Analyzes how model accuracy degrades with peptide length.
Helps identify if model has length-dependent performance issues.
"""

import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model.trm import create_model, TRMConfig
from src.data.dataset import SyntheticPeptideDataset
from src.training.metrics import token_accuracy, sequence_accuracy


def evaluate_by_length(model, lengths, samples_per_length=200, noise_level='clean', device='cuda'):
    """
    Evaluate model on different peptide lengths.

    Args:
        model: Trained TRM model
        lengths: List of peptide lengths to test
        samples_per_length: Number of samples per length
        noise_level: 'clean', 'moderate', or 'realistic'
        device: Device to run on

    Returns:
        dict mapping length -> {token_acc, seq_acc, samples}
    """
    model.eval()
    model.to(device)

    # Noise configurations
    noise_configs = {
        'clean': {
            'noise_peaks': 0,
            'peak_dropout': 0.0,
            'mass_error_ppm': 0.0,
        },
        'moderate': {
            'noise_peaks': 5,
            'peak_dropout': 0.10,
            'mass_error_ppm': 10.0,
        },
        'realistic': {
            'noise_peaks': 10,
            'peak_dropout': 0.15,
            'mass_error_ppm': 15.0,
        }
    }

    results = {}

    for length in lengths:
        print(f"\nTesting length {length}...")

        # Create dataset for this length
        dataset = SyntheticPeptideDataset(
            min_length=length,
            max_length=length,  # Fixed length
            max_peaks=100,
            max_seq_len=25,
            ion_types=['b', 'y'],
            include_neutral_losses=False,
            charge_distribution={2: 0.7, 3: 0.3},
            **noise_configs[noise_level]
        )

        total_token_acc = 0.0
        total_seq_acc = 0.0
        count = 0

        with torch.no_grad():
            for i in tqdm(range(samples_per_length), desc=f"Length {length}", leave=False):
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

        results[length] = {
            'token_acc': total_token_acc / count,
            'seq_acc': total_seq_acc / count,
            'samples': count
        }

        print(f"  Token Accuracy:    {results[length]['token_acc']*100:.2f}%")
        print(f"  Sequence Accuracy: {results[length]['seq_acc']*100:.2f}%")

    return results


def plot_length_analysis(results, noise_level, output_path):
    """Create visualization of length vs accuracy."""
    lengths = sorted(results.keys())
    token_accs = [results[l]['token_acc'] * 100 for l in lengths]
    seq_accs = [results[l]['seq_acc'] * 100 for l in lengths]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Token accuracy plot
    ax1.plot(lengths, token_accs, 'o-', linewidth=2, markersize=8, color='#2E86AB')
    ax1.set_xlabel('Peptide Length (amino acids)', fontsize=12)
    ax1.set_ylabel('Token Accuracy (%)', fontsize=12)
    ax1.set_title(f'Token Accuracy vs Length ({noise_level})', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 100])

    # Sequence accuracy plot
    ax2.plot(lengths, seq_accs, 'o-', linewidth=2, markersize=8, color='#A23B72')
    ax2.set_xlabel('Peptide Length (amino acids)', fontsize=12)
    ax2.set_ylabel('Sequence Accuracy (%)', fontsize=12)
    ax2.set_title(f'Sequence Accuracy vs Length ({noise_level})', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 100])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")


def main():
    print("=" * 80)
    print("DIAGNOSTIC TEST 2: PERFORMANCE VS PEPTIDE LENGTH")
    print("=" * 80)
    print("\nPurpose: Analyze accuracy degradation with peptide length")
    print("Expected: Gradual degradation (not cliff at specific length)\n")

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
    lengths = [7, 9, 11, 13, 15, 17, 18]
    samples_per_length = 200
    noise_levels = ['clean', 'realistic']

    all_results = {}

    for noise_level in noise_levels:
        print(f"\n{'=' * 80}")
        print(f"TESTING: {noise_level.upper()} DATA")
        print("=" * 80)

        results = evaluate_by_length(
            model, lengths, samples_per_length, noise_level, device
        )
        all_results[noise_level] = results

        # Create plot
        plot_path = project_root / f'diagnostic_length_{noise_level}.png'
        plot_length_analysis(results, noise_level, plot_path)

    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    for noise_level in noise_levels:
        print(f"\n{noise_level.upper()} DATA:")
        results = all_results[noise_level]

        lengths_sorted = sorted(results.keys())
        token_accs = [results[l]['token_acc'] * 100 for l in lengths_sorted]
        seq_accs = [results[l]['seq_acc'] * 100 for l in lengths_sorted]

        # Calculate degradation rate
        token_drop = token_accs[0] - token_accs[-1]
        seq_drop = seq_accs[0] - seq_accs[-1]
        length_range = lengths_sorted[-1] - lengths_sorted[0]

        print(f"  Length {lengths_sorted[0]}aa → {lengths_sorted[-1]}aa:")
        print(f"    Token accuracy:    {token_accs[0]:.2f}% → {token_accs[-1]:.2f}% (Δ={token_drop:.2f}%)")
        print(f"    Sequence accuracy: {seq_accs[0]:.2f}% → {seq_accs[-1]:.2f}% (Δ={seq_drop:.2f}%)")
        print(f"    Degradation rate:  {token_drop/length_range:.2f}% per amino acid")

        # Check for cliffs
        for i in range(len(lengths_sorted) - 1):
            l1, l2 = lengths_sorted[i], lengths_sorted[i+1]
            acc1, acc2 = token_accs[i], token_accs[i+1]
            drop = acc1 - acc2

            if drop > 15:  # >15% drop between consecutive lengths
                print(f"    ⚠️  CLIFF DETECTED: {l1}aa → {l2}aa (Δ={drop:.2f}%)")

    # Interpretation
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    clean_results = all_results['clean']
    clean_degradation = (clean_results[7]['token_acc'] - clean_results[18]['token_acc']) * 100 / 11

    print(f"\nClean data degradation rate: {clean_degradation:.2f}% per amino acid")

    if clean_degradation < 1.0:
        print("\n✅ EXCELLENT: Model handles length well")
        print("   Minimal degradation with length")
    elif clean_degradation < 2.0:
        print("\n✅ GOOD: Gradual performance degradation")
        print("   Expected behavior for recursive models")
    elif clean_degradation < 4.0:
        print("\n⚠️  MODERATE: Noticeable length-dependent degradation")
        print("   Consider: Increase max_seq_len or model capacity")
    else:
        print("\n❌ POOR: Severe length-dependent degradation")
        print("   Recommendation: Increase model capacity or use beam search")

    # Check realistic data
    realistic_results = all_results['realistic']
    realistic_short = realistic_results[7]['token_acc'] * 100
    realistic_long = realistic_results[18]['token_acc'] * 100

    print(f"\nRealistic data (7aa vs 18aa):")
    print(f"  7aa:  {realistic_short:.2f}%")
    print(f"  18aa: {realistic_long:.2f}%")

    if realistic_long < 50:
        print("\n⚠️  Long peptides on realistic noise are challenging")
        print("   This is expected but limits real-world applicability")

    # Save results
    results_file = project_root / 'DIAGNOSTIC_LENGTH_RESULTS.txt'
    with open(results_file, 'w') as f:
        f.write("DIAGNOSTIC TEST 2: PERFORMANCE VS PEPTIDE LENGTH\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Checkpoint: {checkpoint_path.name}\n")
        f.write(f"Samples per length: {samples_per_length}\n\n")

        for noise_level in noise_levels:
            f.write(f"\n{noise_level.upper()} DATA:\n")
            results = all_results[noise_level]
            for length in sorted(results.keys()):
                f.write(f"  {length:2d}aa: Token={results[length]['token_acc']*100:5.2f}%, ")
                f.write(f"Seq={results[length]['seq_acc']*100:5.2f}%\n")

    print(f"\nResults saved to: {results_file}")
    print("=" * 80)


if __name__ == '__main__':
    main()
