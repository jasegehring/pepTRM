# Phase 2: Advanced Synthetic → Real Data Transition

**Created**: December 9, 2025
**Status**: 🟢 ACTIVE
**Current Stage**: Stage 0 (Diagnostics)

## Overview

This document tracks the execution plan for transitioning from basic synthetic data to real MS/MS data. We've validated the core TRM architecture (95.9% token acc on clean synthetic) and achieved stable curriculum learning. Now we're addressing the performance gap on realistic noise and preparing for real-world deployment.

**Starting Point**: 100K training run complete (66.7% token / 17.3% seq acc on realistic synthetic)
**End Goal**: Model validated on Nine-Species benchmark with competitive performance

---

## Stage 0: Diagnostic Analysis ⏳ IN PROGRESS

**Goal**: Identify current model's limitations before investing in new data pipeline

**Duration**: 1-2 hours
**Checkpoint**: `checkpoints_optimized/best_model.pt` (step 20K) or `final_model.pt` (step 100K)

### Tasks

- [ ] **0.1: Wrong Precursor Mass Test**
  - Evaluate model with precursor mass offset by ±50 Da
  - **Expected**: Accuracy should collapse if model uses mass constraint
  - **If fails**: Add explicit precursor mass loss term
  - **Script**: Create `scripts/diagnostic_precursor.py`

- [ ] **0.2: Performance vs Peptide Length**
  - Analyze token accuracy for 7aa, 10aa, 13aa, 15aa, 18aa separately
  - **Expected**: Gradual degradation with length
  - **Red flag**: Cliff at specific length (memorization)
  - **Script**: Create `scripts/diagnostic_length.py`

- [ ] **0.3: Noise Decomposition**
  - Test model on:
    - Clean (baseline)
    - Dropout only (no noise peaks)
    - Noise peaks only (no dropout)
    - Mass error only
  - **Goal**: Identify which noise type hurts most
  - **Script**: Create `scripts/diagnostic_noise.py`

- [ ] **0.4: Charge State Analysis**
  - Compare accuracy on +2 vs +3 precursors
  - **Goal**: Understand if model handles higher charge
  - **Script**: Add to `diagnostic_noise.py`

- [ ] **0.5: Training/Validation Gap Analysis**
  - Measure gap at each curriculum stage
  - Check if gap widens with harder stages
  - **Current gap**: 83% train vs 67% val (16 points)
  - **Script**: Parse existing logs

### Success Criteria

✅ All diagnostic scripts complete and results analyzed
✅ Bottleneck identified (dropout, noise, mass error, length, or architecture)
✅ Decision made on what to prioritize in physics review

### Deliverables

- `scripts/diagnostic_*.py` - Diagnostic test suite
- `DIAGNOSTIC_RESULTS.md` - Analysis summary with recommendations

---

## Stage 1: Advanced Synthetic Data + Physics

**Goal**: Improve synthetic data quality and add missing physics (doubly-charged ions)

**Duration**: 5-7 days
**Dependencies**: Stage 0 complete

### Stage 1A: Physics Review & Model Updates (2-3 days)

- [ ] **1A.1: Review MS/MS Fragmentation Chemistry**
  - Document b, y, a, x, c, z ion formation
  - Understand doubly-charged ion prevalence (charge +2, +3)
  - Review neutral losses (H2O: -18 Da, NH3: -17 Da)
  - Literature: Mann & Kelleher reviews, Steen & Mann 2004
  - **Deliverable**: `Documentation/MS_MS_PHYSICS.md`

- [ ] **1A.2: Update Synthetic Data Generator**
  - **File**: `src/data/synthetic.py`
  - Add doubly-charged b++ and y++ ions
  - Add neutral losses (b-H2O, y-NH3, etc.)
  - Make charge state dependent (b++ more common for charge +3)
  - **Test**: `tests/test_doubly_charged.py`

- [ ] **1A.3: Update Spectrum Matching Loss**
  - **File**: `src/training/losses.py`
  - Extend `compute_theoretical_peaks()` for b++, y++
  - Add neutral loss peaks to matching
  - Weight by expected prevalence
  - **Test**: Verify loss decreases with b++/y++ in synthetic data

- [ ] **1A.4: Validation Test**
  - Generate synthetic spectrum with b++, y++, neutral losses
  - Verify all peaks accounted for in loss
  - **Script**: `scripts/test_physics_v2.py`

### Stage 1B: Prosit Integration (2-3 days)

**Decision**: Use Prosit for intensity prediction (more realistic than current uniform distribution)

- [ ] **1B.1: Prosit Setup**
  - Install Prosit dependencies: `pip install prosit`
  - Download pretrained models (HCD, charge +2/+3)
  - **Test**: Generate spectrum for "PEPTIDE" and verify

- [ ] **1B.2: Create PrositDataset Class**
  - **File**: `src/data/prosit_dataset.py`
  - Wrapper that calls Prosit for intensity prediction
  - Keep current mass calculation (we control physics)
  - Use Prosit for realistic intensity distribution
  - Add noise/dropout on top of Prosit output

- [ ] **1B.3: Prosit vs Current Comparison**
  - Generate 1000 spectra with current pipeline
  - Generate same 1000 with Prosit
  - Compare intensity distributions, peak prevalence
  - **Deliverable**: `PROSIT_COMPARISON.md`

- [ ] **1B.4: Fallback: MS2PIP** (if Prosit fails)
  - Only implement if Prosit integration blocked
  - Similar wrapper approach
  - **File**: `src/data/ms2pip_dataset.py`

### Stage 1C: Training Runs with Improved Synthetic (2 days)

Run 3 training experiments in parallel:

- [ ] **1C.1: Baseline + Doubly Charged**
  - Current synthetic pipeline + b++/y++ ions
  - Config: `configs/baseline_plus_doubly_charged.yaml`
  - 50K steps, evaluate on clean and realistic
  - **Target**: >70% token acc on realistic

- [ ] **1C.2: Prosit-Generated Spectra**
  - Use `PrositDataset` with all improvements
  - Config: `configs/prosit_synthetic.yaml`
  - 50K steps
  - **Target**: >75% token acc (better intensity → better learning)

- [ ] **1C.3: Ablation: Spectrum Loss On/Off**
  - Same as 1C.2 but `spectrum_loss_weight=0.0`
  - Test if spectrum loss helps or hurts with realistic intensities
  - **Goal**: Verify spectrum loss still valuable

### Success Criteria

✅ Doubly-charged ions integrated and tested
✅ Prosit integration working OR MS2PIP fallback implemented
✅ Training runs show improvement over 66.7% baseline
✅ Best model checkpoint saved for Stage 2

### Deliverables

- `Documentation/MS_MS_PHYSICS.md` - Physics reference
- `src/data/prosit_dataset.py` - Advanced synthetic data
- `configs/prosit_synthetic.yaml` - Training config
- `STAGE_1_RESULTS.md` - Training comparison & best checkpoint

---

## Stage 2: Real Data Planning (Parallel with Stage 1)

**Goal**: Design real data pipeline and transfer learning strategy

**Duration**: 3-5 days (while Stage 1 trains)
**Dependencies**: None (can run in parallel)

### Stage 2A: Literature Review (1 day)

- [ ] **2A.1: Survey Recent De Novo Papers**
  - Casanovo (2023) - Current SOTA
  - DeepNovo-DIA (2022)
  - PointNovo (2023)
  - Focus on: preprocessing, data augmentation, transfer learning

- [ ] **2A.2: Nine-Species Benchmark Analysis**
  - Download from PRIDE: MSV000081382
  - Analyze: charge distribution, length distribution, PTMs
  - Identify challenges: incomplete fragmentation, noise
  - **Deliverable**: `Documentation/NINE_SPECIES_ANALYSIS.md`

- [ ] **2A.3: Preprocessing Best Practices**
  - Peak filtering (top-K, intensity threshold)
  - Normalization (base peak, TIC, sqrt)
  - Binning vs continuous m/z
  - Charge deconvolution

### Stage 2B: Real Data Pipeline Design (2 days)

- [ ] **2B.1: RealPeptideDataset Specification**
  - Input: .mgf or .mzML files
  - Preprocessing pipeline
  - Output format (same as synthetic for compatibility)
  - Handle missing annotations (for unlabeled data)
  - **Deliverable**: `Documentation/REAL_DATA_PIPELINE.md`

- [ ] **2B.2: Preprocessing Implementation**
  - **File**: `src/data/preprocessing.py`
  - Functions: normalize, filter_peaks, bin_spectrum
  - Make configurable (different strategies for experiments)

- [ ] **2B.3: RealPeptideDataset Class**
  - **File**: `src/data/real_dataset.py`
  - Use `pyteomics` for file parsing
  - Apply preprocessing pipeline
  - Integrate with PyTorch DataLoader
  - **Test**: Load 10 spectra from Nine-Species

### Stage 2C: Transfer Learning Strategy (1 day)

- [ ] **2C.1: Design Synthetic→Real Transition**
  - **Option A**: Fine-tune all weights (simple)
  - **Option B**: Freeze encoder, train decoder (conservative)
  - **Option C**: Layer-wise unfreezing (gradual)
  - **Decision**: Based on literature review

- [ ] **2C.2: Create Fine-Tuning Script**
  - **File**: `scripts/finetune.py`
  - Load pretrained checkpoint
  - Optionally freeze layers
  - Lower learning rate (1e-5 typical)
  - Mixed synthetic/real batches (optional)

- [ ] **2C.3: Evaluation Metrics for Real Data**
  - Peptide-spectrum match (PSM) rate
  - Accuracy at AA-level vs spectrum-level
  - Comparison to Casanovo baseline
  - **File**: `src/training/real_metrics.py`

### Success Criteria

✅ Nine-Species benchmark downloaded and analyzed
✅ RealPeptideDataset can load and preprocess real spectra
✅ Fine-tuning script ready to use
✅ Evaluation metrics defined

### Deliverables

- `Documentation/NINE_SPECIES_ANALYSIS.md`
- `Documentation/REAL_DATA_PIPELINE.md`
- `src/data/real_dataset.py`
- `scripts/finetune.py`
- `TRANSFER_LEARNING_STRATEGY.md`

---

## Stage 3: Real Data Integration & Evaluation

**Goal**: Train on real MS/MS data and benchmark performance

**Duration**: 7-10 days
**Dependencies**: Stage 1 & 2 complete

### Stage 3A: Baseline Evaluation (1 day)

- [ ] **3A.1: Zero-Shot Evaluation**
  - Load best Stage 1 checkpoint
  - Evaluate on Nine-Species (no fine-tuning)
  - **Metrics**: Token acc, seq acc, PSM rate
  - **Goal**: Measure synthetic→real transfer gap

- [ ] **3A.2: Compare to Casanovo**
  - Run Casanovo on same Nine-Species subset
  - Direct comparison on identical data
  - **Deliverable**: `BASELINE_COMPARISON.md`

### Stage 3B: Fine-Tuning Experiments (3-5 days)

- [ ] **3B.1: Full Fine-Tuning**
  - Strategy A: Fine-tune all weights
  - 20K steps, learning rate 1e-5
  - **Config**: `configs/finetune_full.yaml`

- [ ] **3B.2: Frozen Encoder**
  - Strategy B: Freeze encoder, train decoder
  - 20K steps
  - **Config**: `configs/finetune_frozen.yaml`

- [ ] **3B.3: Mixed Synthetic/Real Batches**
  - 50% synthetic (Prosit) + 50% real
  - Prevents catastrophic forgetting
  - 30K steps
  - **Config**: `configs/finetune_mixed.yaml`

- [ ] **3B.4: Select Best Strategy**
  - Compare all three on validation set
  - Choose best for final training

### Stage 3C: Final Training & Evaluation (2-3 days)

- [ ] **3C.1: Full Training Run**
  - Best fine-tuning strategy
  - 50K steps on full Nine-Species training set
  - Save checkpoints every 5K steps

- [ ] **3C.2: Comprehensive Evaluation**
  - Test on Nine-Species test set (unseen)
  - Metrics: AA acc, PSM rate, sequence acc
  - Compare to Casanovo, DeepNovo, PointNovo
  - **Deliverable**: `FINAL_RESULTS.md`

- [ ] **3C.3: Error Analysis**
  - Analyze failure cases
  - Length-dependent performance
  - Charge state effects
  - PTM handling (if present)
  - **Deliverable**: `ERROR_ANALYSIS.md`

### Stage 3D: Ablation Studies (Optional)

If time permits and results are promising:

- [ ] **3D.1: Recursion Ablation (T=1 vs T=8)**
  - Prove recursive refinement helps on real data
  - Train with `num_supervision_steps=1`
  - Compare to `num_supervision_steps=8`

- [ ] **3D.2: Spectrum Loss Ablation**
  - Train with `spectrum_loss_weight=0.0` (CE only)
  - Compare to `spectrum_loss_weight=0.15`
  - Measure impact on real data

### Success Criteria

✅ Model fine-tuned on real Nine-Species data
✅ Competitive performance vs Casanovo (within 10% token acc)
✅ Clear understanding of strengths/weaknesses
✅ Paper-ready results with ablations

### Deliverables

- `BASELINE_COMPARISON.md` - Zero-shot vs Casanovo
- `FINAL_RESULTS.md` - Complete evaluation on Nine-Species
- `ERROR_ANALYSIS.md` - Failure mode analysis
- Trained checkpoints for deployment

---

## Decision Points & Contingencies

### Decision Point 1: After Stage 0 Diagnostics

**If diagnostics show**:
- ✅ Length is main issue → Increase `max_seq_len` before Stage 1
- ✅ Dropout is main issue → Increase dropout in curriculum
- ✅ Model not using precursor → Add precursor mass loss
- ✅ Charge state problems → Prioritize doubly-charged ions

### Decision Point 2: After Stage 1 Training

**If Prosit-trained model shows**:
- ✅ Worse than baseline → Stick with current synthetic, skip Prosit
- ✅ Marginal improvement (<5%) → Optional, not critical
- ✅ Significant improvement (>10%) → Use for all future training

### Decision Point 3: After Stage 3A Zero-Shot Eval

**If zero-shot performance is**:
- ✅ Competitive (>50% token acc) → Fine-tuning will likely succeed
- ⚠️ Poor (<30% token acc) → Synthetic→Real gap too large, revisit Stage 1
- 🔴 Catastrophic (<10%) → Fundamental issue, need architecture changes

---

## Resources & References

### Key Papers
- Tran et al. (2023) - Casanovo: De novo peptide sequencing
- Steen & Mann (2004) - The ABC's of PTMs
- Tran et al. (2017) - De novo peptide sequencing by deep learning

### Datasets
- Nine-Species Benchmark: PRIDE MSV000081382
- MassIVE: MSV000080679 (alternative)

### Tools
- Prosit: https://www.proteomicsdb.org/prosit/
- MS2PIP: https://github.com/compomics/ms2pip
- Pyteomics: https://pyteomics.readthedocs.io/
- Casanovo: https://github.com/Noble-Lab/casanovo

---

## Progress Tracking

### Current Status: Stage 0 - Diagnostics ⏳

**Last Updated**: December 9, 2025
**Days Elapsed**: 0
**Estimated Completion**: December 9, 2025 (end of day)

### Timeline

| Stage | Start Date | End Date | Status |
|-------|------------|----------|--------|
| Stage 0: Diagnostics | Dec 9 | Dec 9 | ⏳ In Progress |
| Stage 1: Advanced Synthetic | Dec 10 | Dec 16 | ⚪ Planned |
| Stage 2: Real Data Planning | Dec 10 | Dec 14 | ⚪ Planned |
| Stage 3: Real Data Integration | Dec 17 | Dec 27 | ⚪ Planned |

**Total Estimated Duration**: 18 days

---

## Notes & Insights

*This section will be updated as we learn during execution*

### Stage 0 Insights
- TBD after diagnostics complete

### Stage 1 Insights
- TBD

### Stage 2 Insights
- TBD

### Stage 3 Insights
- TBD

---

## Maintenance

This document will be updated after each major milestone:
- ✅ Checkboxes updated as tasks complete
- 📝 Insights added to Notes section
- 🔄 Decision points evaluated and paths chosen
- 📊 Results summarized in Deliverables

**Maintainers**: User + Claude Code
**Review Frequency**: After each stage completion
