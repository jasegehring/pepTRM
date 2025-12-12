"""Test torch.compile with the model to diagnose the shape mismatch issue."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from omegaconf import OmegaConf

from src.model.trm import TRMConfig, create_model

def test_compile():
    """Test different torch.compile configurations."""

    # Load config
    config_path = project_root / 'configs' / 'optimized_extended.yaml'
    cfg = OmegaConf.load(config_path)

    model_config = TRMConfig(**cfg.model)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 60)
    print("TORCH.COMPILE DIAGNOSIS")
    print("=" * 60)

    # Create test inputs
    batch_size = 4
    spectrum_masses = torch.randn(batch_size, model_config.max_peaks).abs().to(device) * 1000
    spectrum_intensities = torch.randn(batch_size, model_config.max_peaks).abs().to(device)
    spectrum_mask = torch.ones(batch_size, model_config.max_peaks, dtype=torch.bool).to(device)
    precursor_mass = torch.randn(batch_size).abs().to(device) * 1500
    precursor_charge = torch.randint(2, 4, (batch_size,)).to(device)

    # Test different compile configurations
    configs_to_test = [
        ("No compile (baseline)", {}),
        ("fullgraph=False, dynamic=True", {"fullgraph": False, "dynamic": True}),
        ("fullgraph=False, mode='default'", {"fullgraph": False, "mode": "default"}),
        ("fullgraph=False, mode='reduce-overhead'", {"fullgraph": False, "mode": "reduce-overhead"}),
        ("fullgraph=True, mode='default'", {"fullgraph": True, "mode": "default"}),
    ]

    for name, compile_kwargs in configs_to_test:
        print(f"\n{'-'*60}")
        print(f"Testing: {name}")
        print(f"-"*60)

        try:
            # Create fresh model
            model = create_model(model_config).to(device)

            if compile_kwargs:
                print(f"Compiling with: {compile_kwargs}")
                model = torch.compile(model, **compile_kwargs)

            # Try forward pass
            model.eval()
            with torch.no_grad():
                all_logits, final_z = model(
                    spectrum_masses=spectrum_masses,
                    spectrum_intensities=spectrum_intensities,
                    spectrum_mask=spectrum_mask,
                    precursor_mass=precursor_mass,
                    precursor_charge=precursor_charge,
                )

            print(f"✓ SUCCESS!")
            print(f"  Output shape: {all_logits.shape}")

        except Exception as e:
            print(f"✗ FAILED: {type(e).__name__}")
            print(f"  Error: {str(e)[:200]}")

        # Clean up
        del model
        torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("DIAGNOSIS COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    test_compile()
