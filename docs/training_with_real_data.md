# Training with Real Data: Complete Guide

## Overview

This guide covers training strategies for transitioning from synthetic MS2PIP data to real-world proteomics datasets. We support three data sources with identical interfaces:

1. **MS2PIP** - Synthetic spectra (for initial development)
2. **ProteomeTools** - High-quality synthetic peptide library (21M spectra)
3. **Nine-Species** - Real biological MS/MS data (2.8M PSMs from 9 species)

## Quick Start

All three datasets use the **exact same interface**, allowing seamless switching:

```python
from pathlib import Path

# Option 1: MS2PIP (infinite synthetic data)
from src.data.ms2pip_dataset import create_ms2pip_dataloader
loader = create_ms2pip_dataloader(batch_size=96)

# Option 2: ProteomeTools (finite, high-quality synthetic)
from src.data.proteometools_dataset import create_proteometools_dataloader
loader = create_proteometools_dataloader(
    data_dir=Path('data/proteometools'),
    batch_size=96,
)

# Option 3: Nine-Species (real biological data)
from src.data.nine_species_dataset import create_nine_species_dataloader
loader = create_nine_species_dataloader(
    data_dir=Path('data/nine_species'),
    batch_size=96,
    test_species='homo_sapiens',
)

# Use any loader with your existing trainer - no code changes!
trainer.train(loader)
```

## Dataset Comparison

| Feature | MS2PIP | ProteomeTools | Nine-Species |
|---------|---------|---------------|--------------|
| **Type** | Synthetic (on-the-fly) | Synthetic (pre-generated) | Real biological |
| **Size** | Infinite | 21M spectra | 2.8M PSMs |
| **Coverage** | Random peptides | Human proteome | 9 species |
| **Quality** | Consistent | Very high | Variable |
| **Noise** | Configurable | Minimal | Realistic |
| **PTMs** | None | None | Native |
| **Download** | None (uses MS2PIP) | 2.3 GB | 5-50 GB |
| **Best for** | Early development | Pre-training | Validation & production |

## Recommended Training Strategy

### Stage 1: Development (MS2PIP)
**Goal**: Verify model architecture and training pipeline

```python
# Quick iteration with infinite synthetic data
loader = create_ms2pip_dataloader(
    batch_size=96,
    min_length=7,
    max_length=15,
)

trainer.train(loader, max_steps=10000)
```

**Signs of success**:
- ✓ Loss converges steadily
- ✓ Validation sequence accuracy >80% on easy examples
- ✓ Curriculum progresses smoothly through stages
- ✓ No crashes or NaN losses

---

### Stage 2: Pre-training (ProteomeTools)
**Goal**: Learn from comprehensive, high-quality synthetic data

```python
# Pre-train on large-scale synthetic library
loader = create_proteometools_dataloader(
    data_dir=Path('data/proteometools'),
    batch_size=96,
    split='train',
    collision_energy=28,  # Match HCD conditions
)

trainer.train(loader, max_steps=50000)
```

**Advantages**:
- More diverse peptides than random MS2PIP
- Matches real instrument conditions
- Fixed dataset allows reproducible comparisons
- Good initialization for real data

---

### Stage 3: Real Data Training (Nine-Species)
**Goal**: Learn from real biological complexity

```python
# Train or fine-tune on real data
loader = create_nine_species_dataloader(
    data_dir=Path('data/nine_species'),
    batch_size=64,  # May need smaller batch due to complexity
    split='train',
    test_species='homo_sapiens',  # Hold out for testing
    use_balanced=True,  # Start with balanced version
)

trainer.train(loader, max_steps=30000)
```

**Expected challenges**:
- Lower accuracy initially (real data is harder)
- More variable spectra quality
- Missing peaks and noise
- May need hyperparameter tuning

---

### Stage 4: Cross-Validation (Nine-Species)
**Goal**: Evaluate generalization across species

```python
species_list = [
    'homo_sapiens', 'mus_musculus', 'saccharomyces_cerevisiae',
    'bacillus_subtilis', 'candidatus_endoloripes', 'methanosarcina_mazei',
    'apis_mellifera', 'solanum_lycopersicum', 'vigna_mungo',
]

results = {}
for test_species in species_list:
    # Train on 8 species
    train_loader = create_nine_species_dataloader(
        data_dir=Path('data/nine_species'),
        batch_size=64,
        split='train',
        test_species=test_species,
    )

    # Test on held-out species
    test_loader = create_nine_species_dataloader(
        data_dir=Path('data/nine_species'),
        batch_size=64,
        split='test',
        test_species=test_species,
    )

    model = train_model(train_loader)
    results[test_species] = evaluate_model(model, test_loader)

# Average across all folds
avg_accuracy = np.mean([r['accuracy'] for r in results.values()])
```

## Advanced Training Strategies

### 1. Progressive Real Data Introduction

Gradually transition from synthetic to real data:

```python
schedule = [
    (0, 10000, 'ms2pip'),           # Steps 0-10K: MS2PIP
    (10000, 30000, 'proteometools'), # Steps 10-30K: ProteomeTools
    (30000, 60000, 'ninespecies'),   # Steps 30-60K: Nine-Species
]

for start, end, dataset_type in schedule:
    if dataset_type == 'ms2pip':
        loader = create_ms2pip_dataloader(batch_size=96)
    elif dataset_type == 'proteometools':
        loader = create_proteometools_dataloader(...)
    else:
        loader = create_nine_species_dataloader(...)

    trainer.train(loader, max_steps=end-start)
```

### 2. Mixed Data Training

Combine datasets in a single training run:

```python
from torch.utils.data import ConcatDataset, DataLoader

# Load both datasets
pt_dataset = ProteomeToolsDataset(data_dir=Path('data/proteometools'))
ns_dataset = NineSpeciesDataset(data_dir=Path('data/nine_species'))

# Combine with weighting
# Repeat Nine-Species 3x to balance with larger ProteomeTools
combined = ConcatDataset([pt_dataset] + [ns_dataset] * 3)

loader = DataLoader(combined, batch_size=64, shuffle=True)
trainer.train(loader)
```

### 3. Pre-train + Fine-tune (Recommended)

Best approach for real-world performance:

```python
# Phase 1: Pre-train on ProteomeTools
print("Phase 1: Pre-training on ProteomeTools...")
pt_loader = create_proteometools_dataloader(
    data_dir=Path('data/proteometools'),
    batch_size=96,
)
trainer.train(pt_loader, max_steps=50000)
trainer.save_checkpoint('checkpoints/pretrained.pt')

# Phase 2: Fine-tune on Nine-Species with lower LR
print("Phase 2: Fine-tuning on Nine-Species...")
trainer.learning_rate = 5e-5  # Lower LR for fine-tuning
ns_loader = create_nine_species_dataloader(
    data_dir=Path('data/nine_species'),
    batch_size=64,
    test_species='homo_sapiens',
)
trainer.train(ns_loader, max_steps=20000)
trainer.save_checkpoint('checkpoints/finetuned.pt')
```

## Curriculum Learning with Real Data

The curriculum scheduler works with all datasets:

```python
from src.training.curriculum import CurriculumScheduler, DEFAULT_CURRICULUM

# Create curriculum
curriculum = CurriculumScheduler(
    stages=DEFAULT_CURRICULUM,
    dataset=real_dataset,  # Works with any dataset!
)

# Training loop
for step in range(max_steps):
    # Curriculum automatically adjusts dataset difficulty
    curriculum.step(step)

    batch = next(dataloader)
    loss = trainer.train_step(batch)
```

**Note**: Real datasets (ProteomeTools, Nine-Species) don't support dynamic noise like MS2PIP, but curriculum can still control:
- Peptide length filtering
- Peak filtering
- Data mixing ratios (if combining datasets)

## Monitoring and Debugging

### Key Metrics to Track

```python
metrics_to_log = {
    # Core metrics (all datasets)
    'loss': train_loss,
    'token_accuracy': token_acc,
    'sequence_accuracy': seq_acc,

    # Real data specific
    'avg_peaks_per_spectrum': avg_peaks,
    'avg_sequence_length': avg_len,
    'data_source': dataset_name,  # Track which dataset

    # Per-species (Nine-Species)
    'species_accuracy': {species: acc for species, acc in species_metrics},
}
```

### Expected Performance

| Dataset | Token Accuracy | Sequence Accuracy |
|---------|----------------|-------------------|
| MS2PIP (easy) | 90-95% | 80-90% |
| MS2PIP (hard) | 70-80% | 30-50% |
| ProteomeTools | 85-90% | 70-80% |
| Nine-Species | 60-75% | 20-40% |

**Note**: Real data will always have lower accuracy due to noise, missing peaks, modifications, etc. This is expected!

### Debugging Checklist

If performance is poor on real data:

- [ ] Verify MS2PIP training worked first
- [ ] Check data loading (visualize a few spectra)
- [ ] Reduce batch size (real data may need more memory)
- [ ] Lower learning rate for fine-tuning
- [ ] Increase training steps
- [ ] Check for data quality issues (filter low-quality spectra)
- [ ] Verify peptide length distribution matches training range

## Data Pipeline Summary

```
Development:
  MS2PIP (infinite) → Quick iteration, verify pipeline

Pre-training:
  ProteomeTools (21M) → Learn general fragmentation patterns

Real Data Training:
  Nine-Species (2.8M) → Learn biological complexity

Cross-Validation:
  9-fold CV on Nine-Species → Robust evaluation

Production:
  Deploy best model from CV → Real-world inference
```

## File Organization

Recommended directory structure:

```
pepTRM/
├── data/
│   ├── proteometools/
│   │   ├── FTMS_HCD_28_annotated.msp
│   │   └── ...
│   └── nine_species/
│       ├── balanced/
│       │   └── mgf/
│       │       ├── homo_sapiens_*.mgf
│       │       └── ...
│       └── main/ (optional, 50GB)
├── checkpoints/
│   ├── ms2pip_baseline.pt
│   ├── proteometools_pretrained.pt
│   └── ninespecies_finetuned.pt
├── configs/
│   ├── optimized_extended.yaml  # MS2PIP config
│   ├── proteometools.yaml        # ProteomeTools config
│   └── ninespecies.yaml          # Nine-Species config
└── docs/
    ├── nine_species_dataset.md
    ├── proteometools_dataset.md
    └── training_with_real_data.md (this file)
```

## Next Steps

1. ✓ Verify MS2PIP training works (wait for current run)
2. ⏳ Download both real datasets (in progress)
3. ⏸️ Test data loaders (when GPU available)
4. ⏸️ Run small ProteomeTools pre-training test
5. ⏸️ Run small Nine-Species fine-tuning test
6. ⏸️ Full training pipeline
7. ⏸️ 9-fold cross-validation

## Getting Help

- **Data issues**: Check respective dataset docs (nine_species_dataset.md, proteometools_dataset.md)
- **Training issues**: Review configs/optimized_extended.yaml
- **Integration issues**: All datasets use identical interface - if one works, all should work

---

**You're now ready to train on real data!** Start with ProteomeTools pre-training once your MS2PIP baseline shows good results.
