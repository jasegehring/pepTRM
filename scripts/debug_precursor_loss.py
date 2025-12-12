"""
Debug precursor mass loss to find the root cause of spiking.

This script performs a detailed trace of:
1. Dataset precursor mass values
2. What the model receives (m/z vs neutral)
3. What the loss function receives
4. What the loss function predicts
5. The actual error values
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn.functional as F
from src.data.ms2pip_dataset import MS2PIPSyntheticDataset
from src.model.trm import TRMConfig, create_model
from src.training.losses import CombinedLoss
from src.constants import AMINO_ACID_MASSES, WATER_MASS, PROTON_MASS, VOCAB, AA_TO_IDX

def debug_single_sample():
    """Debug a single sample through the entire pipeline."""
    print("=" * 80)
    print("PRECURSOR MASS LOSS DEBUG")
    print("=" * 80)

    # Create dataset
    dataset = MS2PIPSyntheticDataset(
        min_length=7,
        max_length=10,
        ms2pip_model='HCDch2',
    )

    # Get one sample
    sample = next(iter(dataset))

    # Decode sequence
    sequence = sample.sequence
    sequence_mask = sample.sequence_mask
    valid_tokens = sequence[sequence_mask]
    decoded_aas = [dataset.idx_to_token[idx.item()] for idx in valid_tokens]
    peptide = ''.join(decoded_aas)

    print(f"\n1. DATASET OUTPUT")
    print(f"   Peptide: {peptide}")
    print(f"   Precursor mass (dataset): {sample.precursor_mass.item():.4f} Da")
    print(f"   Precursor charge: {sample.precursor_charge.item()}")
    print(f"   Sequence tokens: {valid_tokens.tolist()}")

    # Verify dataset mass calculation
    expected_neutral_mass = sum(AMINO_ACID_MASSES[aa] for aa in peptide) + WATER_MASS
    print(f"   Expected neutral mass: {expected_neutral_mass:.4f} Da")
    print(f"   Dataset mass error: {abs(sample.precursor_mass.item() - expected_neutral_mass):.6f} Da")

    # Simulate what trainer does
    precursor_neutral_mass = sample.precursor_mass.unsqueeze(0)  # (1,) for batch
    precursor_charge = sample.precursor_charge.unsqueeze(0)  # (1,) for batch
    precursor_mz = (precursor_neutral_mass + precursor_charge * PROTON_MASS) / precursor_charge

    print(f"\n2. TRAINER PROCESSING")
    print(f"   Neutral mass: {precursor_neutral_mass.item():.4f} Da")
    print(f"   Charge: {precursor_charge.item()}")
    print(f"   Calculated m/z: {precursor_mz.item():.4f}")
    print(f"   Model receives: m/z = {precursor_mz.item():.4f}")
    print(f"   Loss receives: neutral mass = {precursor_neutral_mass.item():.4f} Da")

    # Create model
    config = TRMConfig(
        hidden_dim=128,  # Smaller for speed
        num_encoder_layers=2,
        num_decoder_layers=2,
        num_heads=4,
        num_supervision_steps=4,
        num_latent_steps=2,
    )
    model = create_model(config)
    model.eval()

    # Run forward pass
    with torch.no_grad():
        all_logits, _ = model(
            spectrum_masses=sample.spectrum_masses.unsqueeze(0),
            spectrum_intensities=sample.spectrum_intensities.unsqueeze(0),
            spectrum_mask=sample.spectrum_mask.unsqueeze(0),
            precursor_mass=precursor_mz,
            precursor_charge=precursor_charge,
        )

    print(f"\n3. MODEL OUTPUT")
    print(f"   Logits shape: {all_logits.shape}")
    print(f"   Final logits shape: {all_logits[-1].shape}")

    # Get probabilities
    final_probs = F.softmax(all_logits[-1], dim=-1)  # (1, seq_len, vocab)

    print(f"\n4. LOSS FUNCTION - PRECURSOR MASS CALCULATION")

    # Create aa_masses like in loss function
    aa_masses = torch.tensor([
        AMINO_ACID_MASSES.get(aa, 0.0) for aa in VOCAB
    ])

    print(f"   AA masses tensor shape: {aa_masses.shape}")
    print(f"   First 10 AA masses: {aa_masses[:10].tolist()}")
    print(f"   - Index 0 (<PAD>): {aa_masses[0]:.4f} Da")
    print(f"   - Index 1 (<SOS>): {aa_masses[1]:.4f} Da")
    print(f"   - Index 4 (A): {aa_masses[4]:.4f} Da")

    # Calculate expected masses like in PrecursorMassLoss
    expected_masses = torch.einsum('bsv,v->bs', final_probs, aa_masses)
    print(f"   Expected masses per position: {expected_masses[0, :10].tolist()}")

    # Apply mask
    masked_masses = expected_masses * sequence_mask.unsqueeze(0).float()
    predicted_peptide_mass = masked_masses.sum(dim=1)
    predicted_precursor_mass = predicted_peptide_mass + WATER_MASS

    print(f"   Predicted peptide mass: {predicted_peptide_mass.item():.4f} Da")
    print(f"   Predicted precursor mass: {predicted_precursor_mass.item():.4f} Da")
    print(f"   Target precursor mass: {precursor_neutral_mass.item():.4f} Da")

    # Calculate error
    mass_error = abs(predicted_precursor_mass.item() - precursor_neutral_mass.item())
    ppm_error = (mass_error / precursor_neutral_mass.item()) * 1e6

    print(f"\n5. ERROR CALCULATION")
    print(f"   Absolute error: {mass_error:.4f} Da")
    print(f"   PPM error: {ppm_error:.2f} ppm")
    print(f"   Clamp threshold: 100 ppm")
    print(f"   Would be clamped: {ppm_error > 100}")

    # Check if vocabulary is consistent
    print(f"\n6. VOCABULARY CONSISTENCY CHECK")
    print(f"   Dataset token_to_idx == AA_TO_IDX: {dataset.token_to_idx == AA_TO_IDX}")

    # Check a few amino acids
    for aa in ['A', 'C', 'D', 'E']:
        dataset_idx = dataset.token_to_idx[aa]
        expected_idx = AA_TO_IDX[aa]
        aa_mass_at_idx = aa_masses[dataset_idx].item()
        expected_mass = AMINO_ACID_MASSES[aa]
        print(f"   {aa}: dataset_idx={dataset_idx}, expected_idx={expected_idx}, "
              f"mass_at_idx={aa_mass_at_idx:.4f}, expected_mass={expected_mass:.4f}")

    # Now check what happens with actual sequence tokens
    print(f"\n7. TOKEN MASS LOOKUP CHECK")
    for i, (token_id, valid) in enumerate(zip(sequence[:len(peptide)], sequence_mask[:len(peptide)])):
        if valid:
            aa = dataset.idx_to_token[token_id.item()]
            mass_from_lookup = aa_masses[token_id].item()
            expected_mass = AMINO_ACID_MASSES[aa]
            print(f"   Position {i}: token_id={token_id.item()}, aa={aa}, "
                  f"mass_from_lookup={mass_from_lookup:.4f}, expected={expected_mass:.4f}")

def test_with_trained_model():
    """Test with a trained model where precursor loss has been active."""
    print("\n" + "=" * 80)
    print("TESTING WITH REALISTIC MODEL PREDICTIONS")
    print("=" * 80)

    # Check if there's a checkpoint
    from pathlib import Path
    checkpoint_path = Path("checkpoints_optimized/best_model.pt")

    if not checkpoint_path.exists():
        print("\nNo trained checkpoint found. Skipping this test.")
        return

    print(f"\nLoading checkpoint: {checkpoint_path}")

    # Load config
    from omegaconf import OmegaConf
    cfg = OmegaConf.load('configs/optimized_extended.yaml')

    # Create model
    config = TRMConfig(**cfg.model)
    model = create_model(config)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded checkpoint from step {checkpoint['step']}")

    # Create dataset
    dataset = MS2PIPSyntheticDataset(
        min_length=7,
        max_length=10,
        ms2pip_model='HCDch2',
    )

    # Test on multiple samples
    total_errors = []
    total_ppm_errors = []

    for i in range(10):
        sample = next(iter(dataset))

        # Prepare inputs
        precursor_neutral_mass = sample.precursor_mass.unsqueeze(0)
        precursor_charge = sample.precursor_charge.unsqueeze(0)
        precursor_mz = (precursor_neutral_mass + precursor_charge * PROTON_MASS) / precursor_charge

        with torch.no_grad():
            all_logits, _ = model(
                spectrum_masses=sample.spectrum_masses.unsqueeze(0),
                spectrum_intensities=sample.spectrum_intensities.unsqueeze(0),
                spectrum_mask=sample.spectrum_mask.unsqueeze(0),
                precursor_mass=precursor_mz,
                precursor_charge=precursor_charge,
            )

        # Calculate predicted mass
        final_probs = F.softmax(all_logits[-1], dim=-1)
        aa_masses = torch.tensor([AMINO_ACID_MASSES.get(aa, 0.0) for aa in VOCAB])
        expected_masses = torch.einsum('bsv,v->bs', final_probs, aa_masses)
        masked_masses = expected_masses * sample.sequence_mask.unsqueeze(0).float()
        predicted_peptide_mass = masked_masses.sum(dim=1)
        predicted_precursor_mass = predicted_peptide_mass + WATER_MASS

        error = abs(predicted_precursor_mass.item() - precursor_neutral_mass.item())
        ppm_error = (error / precursor_neutral_mass.item()) * 1e6

        total_errors.append(error)
        total_ppm_errors.append(ppm_error)

    print(f"\nResults over 10 samples:")
    print(f"  Mean absolute error: {sum(total_errors)/len(total_errors):.4f} Da")
    print(f"  Mean PPM error: {sum(total_ppm_errors)/len(total_ppm_errors):.2f} ppm")
    print(f"  Max PPM error: {max(total_ppm_errors):.2f} ppm")
    print(f"  Errors > 100 ppm: {sum(1 for e in total_ppm_errors if e > 100)}/10")

if __name__ == '__main__':
    debug_single_sample()
    test_with_trained_model()
