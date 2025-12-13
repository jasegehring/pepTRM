"""
Comprehensive diagnostic script for analyzing training run quality.

This script applies a falsification mindset to identify potential issues:
1. Recursion effectiveness - is refinement actually helping?
2. Training accuracy skepticism - could the model be cheating?
3. Validation alignment - does val difficulty match training?
4. Real data readiness - how far are we from real MS/MS?

Usage:
    python scripts/diagnose_training_run.py --checkpoint path/to/checkpoint.pt
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader

from src.model.trm import TRMConfig, RecursivePeptideModel
from src.data.ms2pip_dataset import MS2PIPSyntheticDataset
from src.data.proteometools_dataset import ProteomeToolsDataset
from src.data.nine_species_dataset import NineSpeciesDataset
from src.training.refinement_tracker import compute_refinement_metrics, summarize_refinement
from src.training.metrics import compute_metrics
from src.constants import AMINO_ACID_MASSES, WATER_MASS, PROTON_MASS, IDX_TO_AA


def collate_fn(batch):
    return {
        'spectrum_masses': torch.stack([s.spectrum_masses for s in batch]),
        'spectrum_intensities': torch.stack([s.spectrum_intensities for s in batch]),
        'spectrum_mask': torch.stack([s.spectrum_mask for s in batch]),
        'precursor_mass': torch.stack([s.precursor_mass for s in batch]),
        'precursor_charge': torch.stack([s.precursor_charge for s in batch]),
        'sequence': torch.stack([s.sequence for s in batch]),
        'sequence_mask': torch.stack([s.sequence_mask for s in batch]),
    }


def load_model(checkpoint_path: str, device: str = 'cuda'):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config_dict = checkpoint['config']

    # Get model config from checkpoint or use defaults
    model_config = TRMConfig(
        hidden_dim=config_dict.get('hidden_dim', 384),
        num_encoder_layers=3,
        num_decoder_layers=3,
        num_heads=6,
        max_peaks=100,
        max_seq_len=35,
        num_supervision_steps=8,
        num_latent_steps=6,
        dropout=0.1,
    )

    model = RecursivePeptideModel(model_config).to(device)

    # Handle compiled model prefix
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()

    return model, model_config


@torch.no_grad()
def evaluate_dataset(model, dataloader, device, num_batches=50, description=""):
    """Evaluate model on a dataset and return detailed metrics."""
    all_metrics = defaultdict(list)

    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break

        batch = {k: v.to(device) for k, v in batch.items()}

        # Calculate precursor m/z for model
        precursor_mass = batch['precursor_mass']
        precursor_charge = batch['precursor_charge']
        precursor_mz = (precursor_mass + precursor_charge * PROTON_MASS) / precursor_charge

        all_logits, _ = model(
            spectrum_masses=batch['spectrum_masses'],
            spectrum_intensities=batch['spectrum_intensities'],
            spectrum_mask=batch['spectrum_mask'],
            precursor_mass=precursor_mz,
            precursor_charge=precursor_charge,
        )

        # Basic metrics
        final_logits = all_logits[-1]
        metrics = compute_metrics(
            logits=final_logits,
            targets=batch['sequence'],
            mask=batch['sequence_mask'],
        )

        # Refinement metrics
        ref_metrics = compute_refinement_metrics(
            all_logits=all_logits,
            targets=batch['sequence'],
            target_mask=batch['sequence_mask'],
        )
        metrics.update(ref_metrics)

        for k, v in metrics.items():
            all_metrics[k].append(v)

    # Average
    avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
    return avg_metrics


def analyze_recursion_effectiveness(metrics: dict, num_steps: int = 8):
    """Analyze whether recursion is providing real benefit."""
    print("\n" + "=" * 70)
    print("RECURSION EFFECTIVENESS ANALYSIS")
    print("=" * 70)

    # Get accuracy progression
    accuracies = [metrics.get(f'recursion/accuracy_step_{t}', 0) for t in range(num_steps)]
    edit_rates = [metrics.get(f'recursion/edit_rate_step_{t}', 0) for t in range(1, num_steps)]

    print("\nAccuracy progression across steps:")
    for t, acc in enumerate(accuracies):
        if t == 0:
            print(f"  Step {t}: {acc*100:.1f}%")
        else:
            delta = accuracies[t] - accuracies[t-1]
            edit = edit_rates[t-1]
            print(f"  Step {t}: {acc*100:.1f}% ({delta*100:+.2f}%) | {edit*100:.2f}% edits")

    # Key metrics
    acc_gain = accuracies[-1] - accuracies[0]
    total_edits = sum(edit_rates)

    print(f"\nSUMMARY:")
    print(f"  Total accuracy gain from recursion: {acc_gain*100:+.2f}%")
    print(f"  Total editing activity: {total_edits*100:.2f}%")

    # Diagnosis
    if acc_gain < 0.01:
        print("\n  ⚠️  WARNING: Recursion provides < 1% accuracy gain")
        print("     Possible causes:")
        print("     - Task too easy (model solves in step 0)")
        print("     - Recursion not being utilized effectively")
        print("     - Need harder data or incentive for late-step refinement")
    elif acc_gain < 0.05:
        print("\n  ℹ️  Recursion provides modest benefit (1-5%)")
        print("     Consider increasing task difficulty")
    else:
        print(f"\n  ✓ Recursion providing {acc_gain*100:.1f}% accuracy improvement!")

    return acc_gain


def test_precursor_mass_cheating(model, device, num_samples=200):
    """
    FALSIFICATION TEST 1: Can model predict sequence length from precursor mass?

    If precursor mass gives away length, model can "cheat" by counting residues.
    Test: Generate peptides of same length but different masses and vice versa.
    """
    print("\n" + "=" * 70)
    print("FALSIFICATION TEST 1: Precursor Mass Information Leakage")
    print("=" * 70)

    # Create test dataset: fixed length, varying mass
    dataset = MS2PIPSyntheticDataset(
        min_length=15,  # Fixed length
        max_length=15,
        noise_peaks=0,
        peak_dropout=0.0,
        mass_error_ppm=0.0,
        ms2pip_model='HCDch2',
    )

    loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)

    # Collect samples with similar masses but same length
    masses = []
    lengths = []

    for i, batch in enumerate(loader):
        if i >= num_samples // 32:
            break

        batch = {k: v.to(device) for k, v in batch.items()}
        precursor_mass = batch['precursor_mass'].cpu().numpy()
        seq_len = batch['sequence_mask'].sum(dim=1).cpu().numpy()

        masses.extend(precursor_mass.flatten())
        lengths.extend(seq_len)

    masses = np.array(masses)
    lengths = np.array(lengths)

    # Check correlation between mass and length
    mass_per_aa = masses / lengths
    mass_std = np.std(mass_per_aa)

    print(f"\nFixed-length peptides (length=15):")
    print(f"  Mass range: {masses.min():.1f} - {masses.max():.1f} Da")
    print(f"  Mass per AA: {mass_per_aa.mean():.1f} ± {mass_std:.1f} Da")

    # For real peptides, average AA mass is ~110 Da with std ~20-30 Da
    # If precursor mass is informative, model could estimate length as mass/110
    length_estimate_error = np.abs(masses / 110 - lengths).mean()
    print(f"  Length estimation error from mass/110: {length_estimate_error:.1f} residues")

    if length_estimate_error < 1.5:
        print("\n  ⚠️  WARNING: Precursor mass strongly predicts sequence length!")
        print("     Model may be using mass to estimate length.")
        print("     Consider: (1) Not using precursor mass, or (2) varying length more")
    else:
        print("\n  ✓ Precursor mass does not strongly predict length")

    return length_estimate_error


def test_ms2pip_pattern_memorization(model, device, num_batches=20):
    """
    FALSIFICATION TEST 2: Is model memorizing MS2PIP intensity patterns?

    MS2PIP generates systematic intensity patterns. Test if model performance
    degrades when we scramble intensities (keeping masses).
    """
    print("\n" + "=" * 70)
    print("FALSIFICATION TEST 2: MS2PIP Pattern Dependence")
    print("=" * 70)

    # Create test dataset
    dataset = MS2PIPSyntheticDataset(
        min_length=10,
        max_length=20,
        noise_peaks=0,
        peak_dropout=0.0,
        mass_error_ppm=0.0,
        ms2pip_model='HCDch2',
    )

    loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)

    acc_normal = []
    acc_scrambled = []

    for i, batch in enumerate(loader):
        if i >= num_batches:
            break

        batch = {k: v.to(device) for k, v in batch.items()}

        precursor_mass = batch['precursor_mass']
        precursor_charge = batch['precursor_charge']
        precursor_mz = (precursor_mass + precursor_charge * PROTON_MASS) / precursor_charge

        # Test 1: Normal intensities
        all_logits, _ = model(
            spectrum_masses=batch['spectrum_masses'],
            spectrum_intensities=batch['spectrum_intensities'],
            spectrum_mask=batch['spectrum_mask'],
            precursor_mass=precursor_mz,
            precursor_charge=precursor_charge,
        )

        preds = all_logits[-1].argmax(dim=-1)
        correct = (preds == batch['sequence']) & batch['sequence_mask']
        acc_normal.append(correct.float().sum().item() / batch['sequence_mask'].sum().item())

        # Test 2: Scrambled intensities (but keep masses intact)
        scrambled_intensities = batch['spectrum_intensities'].clone()
        for b in range(scrambled_intensities.shape[0]):
            mask = batch['spectrum_mask'][b]
            valid_idx = mask.nonzero().squeeze(-1)
            perm = torch.randperm(len(valid_idx), device=device)
            scrambled_intensities[b, valid_idx] = scrambled_intensities[b, valid_idx[perm]]

        all_logits_scrambled, _ = model(
            spectrum_masses=batch['spectrum_masses'],
            spectrum_intensities=scrambled_intensities,
            spectrum_mask=batch['spectrum_mask'],
            precursor_mass=precursor_mz,
            precursor_charge=precursor_charge,
        )

        preds_scrambled = all_logits_scrambled[-1].argmax(dim=-1)
        correct_scrambled = (preds_scrambled == batch['sequence']) & batch['sequence_mask']
        acc_scrambled.append(correct_scrambled.float().sum().item() / batch['sequence_mask'].sum().item())

    acc_normal = np.mean(acc_normal)
    acc_scrambled = np.mean(acc_scrambled)

    print(f"\nResults:")
    print(f"  Normal MS2PIP intensities: {acc_normal*100:.1f}%")
    print(f"  Scrambled intensities:     {acc_scrambled*100:.1f}%")
    print(f"  Drop from scrambling:      {(acc_normal - acc_scrambled)*100:.1f}%")

    if acc_scrambled > acc_normal - 0.05:
        print("\n  ℹ️  Model is robust to intensity scrambling")
        print("     This is GOOD - means model uses mass positions, not intensity patterns")
    else:
        print("\n  ⚠️  Significant accuracy drop from scrambling")
        print("     Model may be relying heavily on MS2PIP intensity patterns")
        print("     This could hurt generalization to real data")

    return acc_normal, acc_scrambled


def test_noise_level_sensitivity(model, device):
    """
    FALSIFICATION TEST 3: How does model handle varying noise levels?

    If model is trained on easy data, it should fail on hard noise.
    """
    print("\n" + "=" * 70)
    print("FALSIFICATION TEST 3: Noise Level Sensitivity")
    print("=" * 70)

    noise_levels = [
        ("Clean", 0, 0.0, 0.0),
        ("Light noise (5 peaks, 10% dropout)", 5, 0.1, 5.0),
        ("Medium noise (15 peaks, 25% dropout)", 15, 0.25, 10.0),
        ("Heavy noise (30 peaks, 40% dropout)", 30, 0.4, 15.0),
        ("Extreme noise (50 peaks, 50% dropout)", 50, 0.5, 20.0),
    ]

    results = []

    for name, noise_peaks, peak_dropout, mass_error_ppm in noise_levels:
        dataset = MS2PIPSyntheticDataset(
            min_length=10,
            max_length=20,
            noise_peaks=noise_peaks,
            peak_dropout=peak_dropout,
            mass_error_ppm=mass_error_ppm,
            clean_data_ratio=0.0,  # Force all noisy
            ms2pip_model='HCDch2',
        )

        loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)

        accs = []
        for i, batch in enumerate(loader):
            if i >= 10:
                break

            batch = {k: v.to(device) for k, v in batch.items()}

            precursor_mass = batch['precursor_mass']
            precursor_charge = batch['precursor_charge']
            precursor_mz = (precursor_mass + precursor_charge * PROTON_MASS) / precursor_charge

            all_logits, _ = model(
                spectrum_masses=batch['spectrum_masses'],
                spectrum_intensities=batch['spectrum_intensities'],
                spectrum_mask=batch['spectrum_mask'],
                precursor_mass=precursor_mz,
                precursor_charge=precursor_charge,
            )

            preds = all_logits[-1].argmax(dim=-1)
            correct = (preds == batch['sequence']) & batch['sequence_mask']
            accs.append(correct.float().sum().item() / batch['sequence_mask'].sum().item())

        acc = np.mean(accs)
        results.append((name, acc))
        print(f"  {name}: {acc*100:.1f}%")

    # Check degradation pattern
    clean_acc = results[0][1]
    extreme_acc = results[-1][1]
    degradation = clean_acc - extreme_acc

    print(f"\nDegradation clean → extreme: {degradation*100:.1f}%")

    if degradation < 0.1:
        print("  ⚠️  WARNING: < 10% degradation suggests task may be too easy")
        print("     Model might be overfitting to simple patterns")
    elif degradation > 0.5:
        print("  ⚠️  WARNING: > 50% degradation suggests model not robust to noise")
    else:
        print("  ✓ Reasonable noise sensitivity")

    return results


def analyze_validation_alignment(model, device, num_batches=30):
    """
    Check if validation sets are properly aligned with training difficulty.
    """
    print("\n" + "=" * 70)
    print("VALIDATION SET ALIGNMENT ANALYSIS")
    print("=" * 70)

    # Create datasets matching training script
    val_easy = MS2PIPSyntheticDataset(
        min_length=7,
        max_length=10,
        noise_peaks=0,
        peak_dropout=0.0,
        mass_error_ppm=0.0,
        ms2pip_model='HCDch2',
    )

    val_hard = MS2PIPSyntheticDataset(
        min_length=8,
        max_length=25,
        noise_peaks=20,
        peak_dropout=0.25,
        mass_error_ppm=15.0,
        intensity_variation=0.3,
        clean_data_ratio=0.0,  # Force noisy
        ms2pip_model='HCDch2',
    )

    # Create curriculum-final-stage validation (what training sees at end)
    val_curriculum_end = MS2PIPSyntheticDataset(
        min_length=12,
        max_length=25,
        noise_peaks=30,
        peak_dropout=0.45,
        mass_error_ppm=20.0,
        intensity_variation=0.7,
        clean_data_ratio=0.0,  # Force noisy
        ms2pip_model='HCDch2',
    )

    datasets = [
        ("Val Easy (clean, short)", val_easy),
        ("Val Hard (as configured)", val_hard),
        ("Curriculum Final Stage", val_curriculum_end),
    ]

    results = {}
    for name, dataset in datasets:
        loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
        metrics = evaluate_dataset(model, loader, device, num_batches=num_batches)
        results[name] = metrics

        print(f"\n{name}:")
        print(f"  Token Accuracy: {metrics['token_accuracy']*100:.1f}%")
        print(f"  Sequence Accuracy: {metrics['sequence_accuracy']*100:.1f}%")
        print(f"  Accuracy Gain (recursion): {metrics.get('recursion/accuracy_gain', 0)*100:+.2f}%")

    # Check alignment issue
    val_hard_acc = results["Val Hard (as configured)"]['token_accuracy']
    curriculum_end_acc = results["Curriculum Final Stage"]['token_accuracy']

    print(f"\n⚠️  ALIGNMENT CHECK:")
    if val_hard_acc > curriculum_end_acc + 0.1:
        print(f"  Val Hard ({val_hard_acc*100:.1f}%) is EASIER than curriculum end ({curriculum_end_acc*100:.1f}%)")
        print("  This explains why val accuracy caps out - it's not challenging enough!")
        print("  RECOMMENDATION: Update val_hard to match curriculum final stage parameters")
    elif curriculum_end_acc > val_hard_acc + 0.1:
        print(f"  Val Hard ({val_hard_acc*100:.1f}%) is HARDER than curriculum end ({curriculum_end_acc*100:.1f}%)")
        print("  Validation may be out-of-distribution from training")
    else:
        print(f"  Val Hard and curriculum end are well-aligned")

    return results


def analyze_real_data(model, device, data_dir: Path):
    """Analyze real data datasets to understand the sim-to-real gap."""
    print("\n" + "=" * 70)
    print("REAL DATA ANALYSIS")
    print("=" * 70)

    # ProteomeTools
    pt_dir = data_dir / 'proteometools'
    if pt_dir.exists():
        print("\n--- ProteomeTools ---")
        try:
            pt_dataset = ProteomeToolsDataset(
                data_dir=pt_dir,
                split='val',
                max_peaks=100,
                max_seq_len=35,
                max_samples=1000,
            )
            print(f"  Loaded {len(pt_dataset)} samples")

            # Sample analysis
            sample = pt_dataset[0]
            print(f"  Sample spectrum: {sample.spectrum_mask.sum().item()} peaks")
            print(f"  Sample precursor mass: {sample.precursor_mass.item():.1f} Da")
            print(f"  Sample charge: {sample.precursor_charge.item()}")
            print(f"  Sample sequence length: {sample.sequence_mask.sum().item()}")

            # Evaluate
            loader = DataLoader(pt_dataset, batch_size=32, collate_fn=collate_fn)
            metrics = evaluate_dataset(model, loader, device, num_batches=20)
            print(f"  Token Accuracy: {metrics['token_accuracy']*100:.1f}%")
            print(f"  Sequence Accuracy: {metrics['sequence_accuracy']*100:.1f}%")

        except Exception as e:
            print(f"  Error loading: {e}")

    # Nine Species
    ns_dir = data_dir / 'nine_species'
    if ns_dir.exists():
        print("\n--- Nine Species ---")
        try:
            ns_dataset = NineSpeciesDataset(
                data_dir=ns_dir,
                split='val',
                max_peaks=100,
                max_seq_len=35,
                use_balanced=True,
                max_samples=1000,
            )
            print(f"  Loaded {len(ns_dataset)} samples")

            # Sample analysis
            sample = ns_dataset[0]
            print(f"  Sample spectrum: {sample.spectrum_mask.sum().item()} peaks")
            print(f"  Sample precursor mass: {sample.precursor_mass.item():.1f} Da")
            print(f"  Sample charge: {sample.precursor_charge.item()}")
            print(f"  Sample sequence length: {sample.sequence_mask.sum().item()}")
            print(f"  Sample species: {sample.species}")

            # Evaluate
            loader = DataLoader(ns_dataset, batch_size=32, collate_fn=collate_fn)
            metrics = evaluate_dataset(model, loader, device, num_batches=20)
            print(f"  Token Accuracy: {metrics['token_accuracy']*100:.1f}%")
            print(f"  Sequence Accuracy: {metrics['sequence_accuracy']*100:.1f}%")

        except Exception as e:
            print(f"  Error loading: {e}")


def main():
    parser = argparse.ArgumentParser(description="Diagnose training run")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--device", type=str, default='cuda', help="Device to use")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load model
    print(f"\nLoading checkpoint: {args.checkpoint}")
    model, config = load_model(args.checkpoint, device)
    print(f"Model loaded successfully")

    # Run diagnostics
    print("\n" + "=" * 70)
    print("RUNNING COMPREHENSIVE DIAGNOSTICS")
    print("=" * 70)

    # 1. Recursion analysis on training-like data
    print("\n>>> Evaluating on training-like data...")
    dataset = MS2PIPSyntheticDataset(
        min_length=12,
        max_length=25,
        noise_peaks=30,
        peak_dropout=0.45,
        mass_error_ppm=20.0,
        intensity_variation=0.7,
        clean_data_ratio=0.0,
        ms2pip_model='HCDch2',
    )
    loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
    metrics = evaluate_dataset(model, loader, device, num_batches=50)

    print(f"\nTraining-like data evaluation:")
    print(f"  Token Accuracy: {metrics['token_accuracy']*100:.1f}%")
    print(f"  Sequence Accuracy: {metrics['sequence_accuracy']*100:.1f}%")

    acc_gain = analyze_recursion_effectiveness(metrics)

    # 2. Falsification tests
    test_precursor_mass_cheating(model, device)
    test_ms2pip_pattern_memorization(model, device)
    test_noise_level_sensitivity(model, device)

    # 3. Validation alignment
    analyze_validation_alignment(model, device)

    # 4. Real data analysis
    analyze_real_data(model, device, project_root / 'data')

    print("\n" + "=" * 70)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
