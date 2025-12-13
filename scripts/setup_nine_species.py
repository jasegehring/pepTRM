"""
Setup script for downloading and preparing the Nine-Species benchmark dataset.

This script provides utilities to download the dataset from Zenodo and
verify its structure.

Dataset DOI: 10.5281/zenodo.13685813
Paper DOI: 10.1038/s41597-024-04068-4
"""

import argparse
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def print_download_instructions(data_dir: Path):
    """Print instructions for downloading the dataset."""
    print("=" * 80)
    print("Nine-Species Benchmark Dataset Setup")
    print("=" * 80)
    print()
    print("The Nine-Species benchmark is a multi-species proteomics dataset with")
    print("2.8M high-confidence peptide-spectrum matches from 9 species.")
    print()
    print("Dataset Information:")
    print("  - DOI: 10.5281/zenodo.13685813")
    print("  - Paper: https://www.nature.com/articles/s41597-024-04068-4")
    print("  - Size: ~50 GB (main) or ~15 GB (balanced)")
    print()
    print("Download Instructions:")
    print("-" * 80)
    print()
    print("Option 1: Manual Download")
    print("  1. Visit: https://zenodo.org/records/13685813")
    print("  2. Download either:")
    print("     - main.tar.gz (2.8M PSMs, full dataset)")
    print("     - balanced.tar.gz (780K PSMs, balanced across species)")
    print("  3. Extract to:", data_dir.absolute())
    print()
    print("Option 2: Command Line Download (requires zenodo_get)")
    print("  pip install zenodo-get")
    print(f"  cd {data_dir.absolute()}")
    print("  zenodo_get 10.5281/zenodo.13685813")
    print()
    print("Expected directory structure:")
    print(f"  {data_dir}/")
    print("    ├── main/")
    print("    │   └── mgf/")
    print("    │       ├── homo_sapiens_*.mgf")
    print("    │       ├── mus_musculus_*.mgf")
    print("    │       └── ...")
    print("    └── balanced/")
    print("        └── mgf/")
    print("            └── ...")
    print()
    print("=" * 80)


def verify_dataset(data_dir: Path, use_balanced: bool = False):
    """Verify that the dataset is properly downloaded and structured."""
    print("\nVerifying dataset structure...")

    version = 'balanced' if use_balanced else 'main'
    mgf_dir = data_dir / version / 'mgf'

    if not mgf_dir.exists():
        print(f"❌ MGF directory not found: {mgf_dir}")
        print("\nPlease download the dataset first (see instructions above).")
        return False

    # Check for MGF files
    mgf_files = list(mgf_dir.glob("*.mgf"))
    if not mgf_files:
        print(f"❌ No MGF files found in {mgf_dir}")
        return False

    print(f"✓ Found {len(mgf_files)} MGF files")

    # Expected species
    species_list = [
        'vigna_mungo',
        'mus_musculus',
        'methanosarcina_mazei',
        'bacillus_subtilis',
        'candidatus_endoloripes',
        'solanum_lycopersicum',
        'saccharomyces_cerevisiae',
        'apis_mellifera',
        'homo_sapiens',
    ]

    # Check for each species
    found_species = []
    for species in species_list:
        species_files = [f for f in mgf_files if species in f.name.lower()]
        if species_files:
            found_species.append(species)
            print(f"  ✓ {species}: {len(species_files)} file(s)")
        else:
            print(f"  ⚠ {species}: not found")

    if len(found_species) < 9:
        print(f"\n⚠ Warning: Only found {len(found_species)}/9 species")
    else:
        print(f"\n✓ All 9 species found!")

    print("\n✓ Dataset verification complete!")
    return True


def test_dataloader(data_dir: Path, use_balanced: bool = False):
    """Test loading a small batch from the dataset."""
    print("\nTesting data loader...")

    from src.data.nine_species_dataset import create_nine_species_dataloader

    try:
        # Create a small test loader
        dataloader = create_nine_species_dataloader(
            data_dir=data_dir,
            batch_size=4,
            split='train',
            test_species='homo_sapiens',  # Use human as test species
            use_balanced=use_balanced,
            num_workers=0,  # Single process for testing
        )

        # Load one batch
        batch = next(iter(dataloader))

        print("✓ Successfully loaded a batch!")
        print(f"  - Batch size: {batch['sequence'].shape[0]}")
        print(f"  - Spectrum shape: {batch['peak_masses'].shape}")
        print(f"  - Sequence shape: {batch['sequence'].shape}")
        print(f"  - Sample sequence length: {batch['sequence_mask'][0].sum().item()} tokens")
        print(f"  - Sample peak count: {batch['peak_mask'][0].sum().item()} peaks")

        return True

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Setup and verify Nine-Species benchmark dataset"
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/nine_species',
        help='Directory to store/find the dataset'
    )
    parser.add_argument(
        '--balanced',
        action='store_true',
        help='Use balanced version (780K PSMs) instead of main (2.8M PSMs)'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify dataset structure'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test data loader by loading a batch'
    )

    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    # Always print instructions
    print_download_instructions(data_dir)

    # Verify if requested
    if args.verify or args.test:
        verified = verify_dataset(data_dir, args.balanced)
        if not verified:
            sys.exit(1)

    # Test loader if requested
    if args.test:
        success = test_dataloader(data_dir, args.balanced)
        if not success:
            sys.exit(1)

    print("\n✓ Setup complete!")


if __name__ == '__main__':
    main()
