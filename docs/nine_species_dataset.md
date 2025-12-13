# Nine-Species Benchmark Dataset

## Overview

The Nine-Species benchmark is a standardized proteomics dataset for training and evaluating de novo sequencing models. It contains 2.8M high-confidence peptide-spectrum matches from 9 diverse species.

**Paper**: [A multi-species benchmark for training and validating mass spectrometry proteomics machine learning models](https://www.nature.com/articles/s41597-024-04068-4) (Scientific Data, 2024)

**Dataset**: [Zenodo DOI 10.5281/zenodo.13685813](https://zenodo.org/records/13685813)

## Dataset Statistics

| Version | PSMs | Distinct Peptides | Raw Files | Size |
|---------|------|-------------------|-----------|------|
| Main | 2.8M | 168K | 343 | ~50 GB |
| Balanced | 780K | 133K | 343 | ~15 GB |

**Species Included**:
- *Homo sapiens* (human)
- *Mus musculus* (mouse)
- *Saccharomyces cerevisiae* (yeast)
- *Bacillus subtilis* (bacterium)
- *Candidatus endoloripes* (bacterium)
- *Methanosarcina mazei* (archaeon)
- *Apis mellifera* (honeybee)
- *Solanum lycopersicum* (tomato)
- *Vigna mungo* (plant)

## Setup

### 1. Download Dataset

```bash
# Create data directory
mkdir -p data/nine_species

# Option A: Download via browser
# Visit: https://zenodo.org/records/13685813
# Download: main.tar.gz or balanced.tar.gz
# Extract to: data/nine_species/

# Option B: Download via command line (requires zenodo-get)
pip install zenodo-get
cd data/nine_species
zenodo_get 10.5281/zenodo.13685813
```

### 2. Verify Setup

```bash
# Run setup script to verify dataset structure
python scripts/setup_nine_species.py --verify

# Test data loader
python scripts/setup_nine_species.py --test
```

Expected directory structure:
```
data/nine_species/
├── main/
│   └── mgf/
│       ├── homo_sapiens_*.mgf
│       ├── mus_musculus_*.mgf
│       └── ...
└── balanced/
    └── mgf/
        └── ...
```

## Usage

### Basic Usage

```python
from pathlib import Path
from src.data.nine_species_dataset import create_nine_species_dataloader

# Create data loader
dataloader = create_nine_species_dataloader(
    data_dir=Path('data/nine_species'),
    batch_size=64,
    split='train',
    test_species='homo_sapiens',  # Hold out human for testing
    use_balanced=False,  # Use full dataset
)

# Iterate over batches
for batch in dataloader:
    # batch contains same fields as MS2PIP synthetic data:
    # - peak_masses: (batch, max_peaks)
    # - peak_intensities: (batch, max_peaks)
    # - peak_mask: (batch, max_peaks)
    # - precursor_mass: (batch,)
    # - precursor_charge: (batch,)
    # - sequence: (batch, max_seq_len)
    # - sequence_mask: (batch, max_seq_len)
    pass
```

### 9-Fold Cross-Validation

The recommended evaluation protocol is 9-fold cross-validation, training on 8 species and testing on the held-out 9th species:

```python
species_list = [
    'homo_sapiens',
    'mus_musculus',
    'saccharomyces_cerevisiae',
    'bacillus_subtilis',
    'candidatus_endoloripes',
    'methanosarcina_mazei',
    'apis_mellifera',
    'solanum_lycopersicum',
    'vigna_mungo',
]

for test_species in species_list:
    print(f"Training with {test_species} held out...")

    train_loader = create_nine_species_dataloader(
        data_dir=Path('data/nine_species'),
        batch_size=64,
        split='train',
        test_species=test_species,
    )

    val_loader = create_nine_species_dataloader(
        data_dir=Path('data/nine_species'),
        batch_size=64,
        split='val',
        test_species=test_species,
    )

    test_loader = create_nine_species_dataloader(
        data_dir=Path('data/nine_species'),
        batch_size=64,
        split='test',
        test_species=test_species,
    )

    # Train and evaluate...
```

### Custom Parameters

```python
dataloader = create_nine_species_dataloader(
    data_dir=Path('data/nine_species'),
    batch_size=64,
    split='train',
    test_species='homo_sapiens',
    use_balanced=False,  # Use full or balanced dataset
    max_peaks=100,       # Maximum peaks per spectrum
    max_seq_len=35,      # Maximum sequence length
    min_peaks=10,        # Filter spectra with fewer peaks
    min_length=7,        # Minimum peptide length
    max_length=30,       # Maximum peptide length
    normalize_intensities=True,  # Normalize to [0, 1]
    num_workers=4,       # DataLoader workers
)
```

## Data Format

Each sample contains:

- **spectrum_masses**: Peak m/z values (max_peaks,)
- **spectrum_intensities**: Peak intensities normalized to [0, 1] (max_peaks,)
- **spectrum_mask**: Valid peak mask (max_peaks,)
- **precursor_mass**: Precursor mass in Da (scalar)
- **precursor_charge**: Precursor charge state (scalar)
- **sequence**: Tokenized peptide sequence (max_seq_len,)
- **sequence_mask**: Valid sequence position mask (max_seq_len,)

This format is **identical** to the MS2PIP synthetic dataset, allowing seamless switching between synthetic and real data.

## Curriculum Learning Compatibility

The Nine-Species dataset supports the `set_difficulty()` interface for curriculum learning compatibility:

```python
dataset.set_difficulty(
    min_length=7,
    max_length=15,
)
```

**Note**: Unlike synthetic data, real spectra cannot have noise/dropout dynamically applied. The `set_difficulty()` method only supports length filtering, which requires reloading the dataset.

## Integration with Training Scripts

To use Nine-Species data with existing training scripts, modify the dataset creation:

```python
# Old (MS2PIP synthetic):
from src.data.ms2pip_dataset import create_ms2pip_dataloader
train_loader = create_ms2pip_dataloader(batch_size=64, ...)

# New (Nine-Species real):
from src.data.nine_species_dataset import create_nine_species_dataloader
train_loader = create_nine_species_dataloader(
    data_dir=Path('data/nine_species'),
    batch_size=64,
    split='train',
    test_species='homo_sapiens',
)
```

The batch format is identical, so no other changes are needed!

## Tips for Training on Real Data

1. **Start with balanced version**: The balanced dataset (780K PSMs) is faster to load and train on initially.

2. **Use cross-validation**: The recommended protocol is 9-fold CV to evaluate generalization across species.

3. **Check MS2PIP results first**: Ensure your model works on synthetic data before moving to real data.

4. **Monitor metrics per species**: Real data quality varies by species - track per-species metrics.

5. **Expect lower accuracy initially**: Real spectra have noise, missing peaks, and PTMs that synthetic data doesn't capture.

6. **Consider data augmentation**: For curriculum learning on real data, you may want to add augmentation (future enhancement).

## Citation

If you use this dataset, please cite:

```bibtex
@article{eloff2024multispecies,
  title={A multi-species benchmark for training and validating mass spectrometry proteomics machine learning models},
  author={Eloff, Ralf and others},
  journal={Scientific Data},
  volume={11},
  year={2024},
  publisher={Nature Publishing Group},
  doi={10.1038/s41597-024-04068-4}
}
```
