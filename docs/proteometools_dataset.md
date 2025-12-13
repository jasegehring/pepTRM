

# ProteomeTools Dataset

## Overview

ProteomeTools is a comprehensive synthetic peptide library covering the human proteome with >1M tryptic peptides and 21M high-quality MS/MS spectra. This dataset was used to train Prosit, one of the most accurate spectrum prediction models.

**Paper**: [Prosit: proteome-wide prediction of peptide tandem mass spectra by deep learning](https://www.nature.com/articles/s41592-019-0426-7) (Nature Methods, 2019)

**Dataset**: [Zenodo DOI 10.5281/zenodo.15705607](https://zenodo.org/records/15705607)

## Dataset Characteristics

### Key Differences from Nine-Species:
- **Synthetic peptides**: Chemically synthesized, not from biological samples
- **Comprehensive coverage**: Systematic coverage of human tryptic peptides
- **High quality**: Consistent, high-quality spectra
- **Multiple collision energies**: HCD spectra at 20, 23, 25, 28, 30, 35 NCE
- **Large scale**: 21M spectra vs Nine-Species' 2.8M PSMs

### Use Cases:
- Training spectrum prediction models
- Large-scale pre-training before fine-tuning on real data
- Benchmarking against Prosit
- Data augmentation for real datasets

## Dataset Statistics

| File | NCE | Size | Peptides |
|------|-----|------|----------|
| FTMS_HCD_20 | 20 | 158.6 MB | ~100K |
| FTMS_HCD_23 | 23 | 191.0 MB | ~100K |
| FTMS_HCD_25 | 25 | 197.9 MB | ~100K |
| FTMS_HCD_28 | 28 | 253.2 MB | ~100K |
| FTMS_HCD_30 | 30 | 209.7 MB | ~100K |
| FTMS_HCD_35 | 35 | 192.2 MB | ~100K |
| ITMS_CID_35 | 35 (CID) | 559.6 MB | ~100K |
| ITMS_HCD_28 | 28 (ITMS) | 514.8 MB | ~100K |

**Total**: 2.3 GB

## Setup

### 1. Download Dataset

```bash
# Create data directory
mkdir -p data/proteometools

# Download HCD 28 NCE (recommended, most common)
cd data/proteometools
wget -O FTMS_HCD_28_annotated.zip "https://zenodo.org/records/15705607/files/FTMS_HCD_28_annotated_2019-11-12.zip?download=1"

# Extract
unzip FTMS_HCD_28_annotated.zip
```

### 2. Download Additional Collision Energies (Optional)

```bash
# Download other NCE values for multi-condition training
wget -O FTMS_HCD_25_annotated.zip "https://zenodo.org/records/15705607/files/FTMS_HCD_25_annotated_2019-11-12.zip?download=1"
wget -O FTMS_HCD_30_annotated.zip "https://zenodo.org/records/15705607/files/FTMS_HCD_30_annotated_2019-11-12.zip?download=1"

# Extract all
unzip "*.zip"
```

Expected directory structure:
```
data/proteometools/
├── FTMS_HCD_28_annotated_2019-11-12.msp
├── FTMS_HCD_25_annotated_2019-11-12.msp  (optional)
└── FTMS_HCD_30_annotated_2019-11-12.msp  (optional)
```

## Usage

### Basic Usage

```python
from pathlib import Path
from src.data.proteometools_dataset import create_proteometools_dataloader

# Create data loader
dataloader = create_proteometools_dataloader(
    data_dir=Path('data/proteometools'),
    batch_size=64,
    split='train',
    collision_energy=28,  # Optional: filter by NCE
)

# Iterate over batches
for batch in dataloader:
    # Same interface as MS2PIP and Nine-Species:
    # - peak_masses, peak_intensities, peak_mask
    # - precursor_mass, precursor_charge
    # - sequence, sequence_mask
    pass
```

### Custom Parameters

```python
dataloader = create_proteometools_dataloader(
    data_dir=Path('data/proteometools'),
    batch_size=64,
    split='train',
    max_peaks=100,              # Maximum peaks per spectrum
    max_seq_len=35,             # Maximum sequence length
    min_peaks=10,               # Filter spectra with fewer peaks
    min_length=7,               # Minimum peptide length
    max_length=30,              # Maximum peptide length
    normalize_intensities=True, # Normalize to [0, 1]
    collision_energy=28,        # Filter by NCE (optional)
    val_fraction=0.1,           # Validation split
    random_seed=42,             # Reproducible splits
    num_workers=4,              # DataLoader workers
)
```

## Training Strategies

### 1. Pre-training + Fine-tuning (Recommended)

```python
# Stage 1: Pre-train on ProteomeTools (synthetic)
proteometools_loader = create_proteometools_dataloader(
    data_dir=Path('data/proteometools'),
    batch_size=96,
    split='train',
)

# Train for 50K steps on synthetic data
trainer.train(proteometools_loader, max_steps=50000)

# Stage 2: Fine-tune on Nine-Species (real)
ninespecies_loader = create_nine_species_dataloader(
    data_dir=Path('data/nine_species'),
    batch_size=64,
    split='train',
    test_species='homo_sapiens',
)

# Fine-tune for 10K steps on real data
trainer.train(ninespecies_loader, max_steps=10000)
```

### 2. Mixed Training

```python
# Combine synthetic and real data in single training run
from torch.utils.data import ConcatDataset

pt_dataset = ProteomeToolsDataset(data_dir=Path('data/proteometools'))
ns_dataset = NineSpeciesDataset(data_dir=Path('data/nine_species'))

combined_dataset = ConcatDataset([pt_dataset, ns_dataset])
loader = DataLoader(combined_dataset, batch_size=64, shuffle=True)
```

### 3. Progressive Training

```python
# Start with ProteomeTools, gradually introduce real data
weights = {
    'step_0-20K': {'proteometools': 1.0, 'ninespecies': 0.0},
    'step_20K-40K': {'proteometools': 0.7, 'ninespecies': 0.3},
    'step_40K-60K': {'proteometools': 0.3, 'ninespecies': 0.7},
    'step_60K+': {'proteometools': 0.0, 'ninespecies': 1.0},
}
```

## Data Format

Each sample contains (identical to other datasets):

- **spectrum_masses**: Peak m/z values (max_peaks,)
- **spectrum_intensities**: Peak intensities [0, 1] (max_peaks,)
- **spectrum_mask**: Valid peak mask (max_peaks,)
- **precursor_mass**: Precursor mass in Da (scalar)
- **precursor_charge**: Precursor charge state (scalar)
- **sequence**: Tokenized peptide sequence (max_seq_len,)
- **sequence_mask**: Valid sequence position mask (max_seq_len,)

## Advantages & Limitations

### Advantages:
✓ Comprehensive coverage of peptide space
✓ High-quality, consistent spectra
✓ Large scale (21M spectra)
✓ Multiple collision energies available
✓ Good for pre-training and data augmentation
✓ Same interface as real data - easy to switch

### Limitations:
✗ Synthetic peptides differ from biological samples
✗ No post-translational modifications
✗ May not capture all real-world complexities
✗ Human proteome only (but can generalize)
✗ Smaller file format (.msp) has less metadata than raw files

## Comparison: ProteomeTools vs Nine-Species

| Aspect | ProteomeTools | Nine-Species |
|--------|---------------|--------------|
| **Type** | Synthetic peptides | Real biological data |
| **Size** | 21M spectra | 2.8M PSMs |
| **Coverage** | Comprehensive (human) | 9 diverse species |
| **Quality** | Very high, consistent | Variable, realistic |
| **Use case** | Pre-training | Real-world validation |
| **Download** | 2.3 GB (.msp) | 50 GB (MGF) |
| **Modifications** | None | Native PTMs present |

**Recommendation**: Use ProteomeTools for initial training/pre-training, then validate and fine-tune on Nine-Species.

## Integration with Training Scripts

ProteomeTools uses the same interface as all other datasets:

```python
# Simply swap the data loader - no other changes needed!

# Option 1: MS2PIP synthetic
from src.data.ms2pip_dataset import create_ms2pip_dataloader
train_loader = create_ms2pip_dataloader(...)

# Option 2: ProteomeTools synthetic
from src.data.proteometools_dataset import create_proteometools_dataloader
train_loader = create_proteometools_dataloader(
    data_dir=Path('data/proteometools'), ...
)

# Option 3: Nine-Species real
from src.data.nine_species_dataset import create_nine_species_dataloader
train_loader = create_nine_species_dataloader(
    data_dir=Path('data/nine_species'), ...
)

# Trainer works identically with all three!
trainer.train(train_loader)
```

## Citation

If you use ProteomeTools, please cite:

```bibtex
@article{gessulat2019prosit,
  title={Prosit: proteome-wide prediction of peptide tandem mass spectra by deep learning},
  author={Gessulat, Siegfried and Schmidt, Tobias and others},
  journal={Nature Methods},
  volume={16},
  pages={509--518},
  year={2019},
  doi={10.1038/s41592-019-0426-7}
}

@article{zolg2017building,
  title={Building ProteomeTools based on a complete synthetic human proteome},
  author={Zolg, Daniel P and others},
  journal={Nature Methods},
  volume={14},
  pages={259--262},
  year={2017}
}
```
