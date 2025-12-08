# PepTRM Development Roadmap: From Synthetic Success to Real-World Application

### Introduction

The PepTRM model has achieved high accuracy on a robust synthetic dataset, demonstrating that the core recursive architecture is working and capable of learning the fundamental physics of peptide fragmentation. This document outlines a strategic roadmap for the next phases of development, designed to bridge the gap between this initial success and a state-of-the-art, real-world-ready de novo sequencing tool.

This is a living document meant to guide our future iteration sessions. Priorities and implementation details can be adjusted as we gather more data.

---

### Phase 0: Infrastructure & Diagnostics Setup

**Goal**: Prepare the technical infrastructure and diagnostic tools needed for efficient experimentation on GPU hardware.

**Priority 0.1: Migration to NVIDIA GPU Training**
*   **Why**: MacBook MPS training (1.1 it/s) is ~10x slower than NVIDIA GPU training. For 50K+ step runs, this means hours vs days. Remote GPU will enable faster iteration.
*   **Hardware Context**:
    - Current: M3 Pro MacBook (14 GPU cores, MPS backend, ~1.1 it/s)
    - Target: Remote NVIDIA GPU (CUDA backend, expect ~5-15 it/s depending on GPU)
*   **Implementation Checklist**:
    1.  **Environment Setup on Remote Machine**:
        ```bash
        # Install CUDA-compatible PyTorch
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

        # Verify CUDA availability
        python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

        # Install other dependencies
        pip install tqdm omegaconf wandb
        ```
    2.  **Code Changes (Minimal)**:
        - **`configs/default.yaml`**: Change `device: 'mps'` → `device: 'cuda'`
        - **`src/training/trainer.py`**: The existing device handling should work automatically:
          ```python
          self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
          ```
        - **Verify**: No hardcoded MPS references in codebase
    3.  **Data Transfer**:
        - Use `rsync` or `scp` to copy codebase to remote machine
        - WandB will sync results automatically (already configured)
    4.  **Remote Training Workflow**:
        ```bash
        # SSH into remote machine
        ssh user@remote-gpu-machine

        # Navigate to project
        cd ~/pepTRM

        # Run training in tmux/screen (survives disconnect)
        tmux new -s training
        python3 scripts/train.py
        # Ctrl+B then D to detach

        # Monitor via WandB dashboard from local machine
        ```
    5.  **Validation Test**:
        - Run short 1K-step test on GPU to verify:
          - CUDA tensors work
          - Speed improvement achieved (expect 5-10x faster)
          - Checkpoints save correctly
          - WandB logging works

**Priority 0.2: Add Validation Accuracy Tracking** ✅ **BLOCKING for Priority 1.2**
*   **Why**: Currently only tracking training metrics. Need val_sequence_accuracy to know if model is truly learning vs overfitting.
*   **Implementation**:
    1.  **`scripts/train.py`**: Create fixed validation dataset (e.g., 1000 samples, seed=42)
    2.  **`src/training/trainer.py`**: Already has `evaluate()` method - ensure it's called every `eval_interval`
    3.  **Log to WandB**: `wandb.log({'val_token_accuracy': ..., 'val_sequence_accuracy': ...})`
    4.  **Critical**: This will let us see the 58-59% val_seq_acc you mentioned at steps 17K-20K!

**Priority 0.3: Implement "Cheating" Diagnostic Tests**
*   **Why**: To validate that 93% token accuracy / 58% sequence accuracy is genuine learning, not exploitation of shortcuts.
*   **Tests to Implement**:
    1.  **Wrong Precursor Mass Test**:
        - Evaluate model with precursor mass deliberately offset by ±50 Da
        - **Expected if learning is real**: Accuracy should collapse
        - **Expected if cheating**: Accuracy stays high (model ignores mass)
    2.  **Accuracy vs Length Analysis**:
        - Plot token accuracy for 7-aa, 8-aa, 9-aa, 10-aa peptides separately
        - **Expected**: Smooth degradation with length
        - **Red flag**: Cliff at specific length (would indicate memorization)
    3.  **Amino Acid Frequency Bias**:
        - Generate validation set with uniform AA distribution (force rare AAs)
        - Compare accuracy vs normal validation
        - **Expected**: Similar performance (model understands physics, not just statistics)

---

### Phase 1: The Robustness Regimen

**Goal**: Solve the "perfectionist model" problem by redesigning the training process to teach robustness against noise and mass error from an earlier stage. This is the top priority before moving to more complex physics or real data.

**Priority 1.1: Tune Loss Function Hyperparameters** ⚠️ **RECONSIDER - May Not Be Needed**
*   **Why**: Original hypothesis was to make matching more flexible for noisy data. However, results show spectrum loss actually HELPED on clean data (19K→6.7K ppm mass error).
*   **Counter-Evidence from Results**:
    - Current `mass_tolerance=0.5` worked well on clean data
    - The problem isn't the loss function - it's the curriculum cliff
    - Narrowing tolerance might make it WORSE on noisy data
*   **RECOMMENDATION**: **Skip this priority for now**. Address curriculum first (Priority 1.2). Revisit only if smoother curriculum still shows issues.
*   **Original Implementation Proposal** (for reference):
    1.  **`configs/default.yaml`**: Decrease `mass_tolerance` from `0.5` to **`0.1`**
    2.  **`configs/default.yaml`**: Add `temperature` parameter to **`0.2`**
*   **Alternative**: If needed after curriculum fix, try INCREASING tolerance (0.5→1.0) to be more forgiving of noisy peaks.

**Priority 1.2: Redesign the Curriculum for Gradual Difficulty** ⚠️ **CRITICAL - TOP PRIORITY**
*   **Why**: To prevent the model from overfitting to perfect data by introducing imperfections early and gradually. **Current curriculum causes catastrophic forgetting at steps 20K→21K and 35K→36K (93%→49% and 71%→45% drops).**
*   **Root Cause Analysis**: Going from ZERO noise peaks to 5+ noise peaks is a distribution shift the model cannot handle. It learned "every peak is signal" on clean data, then completely fails when noise appears.
*   **Implementation Proposal**:
    1.  **`src/training/curriculum.py`**: Replace the `DEFAULT_CURRICULUM` with a 6-stage schedule that introduces noise in tiny increments.
    2.  **REVISED New Schedule** (Based on 50K Training Results):
        *   **Stage 1 (0-8K steps)**: Pure CE on clean data (length 7-10, no noise, no dropout, no mass error)
            - `spectrum_loss_weight=0.0` (learn basic sequence patterns first)
            - Expected: 80-85% token accuracy by step 8K
        *   **Stage 2 (8K-16K steps)**: Add spectrum loss, KEEP data clean
            - `spectrum_loss_weight=0.1`, still length 7-10, **zero noise**
            - **Critical**: This proved spectrum loss helps (93% achieved, mass error 6.7K ppm)
            - Expected: 90-93% token accuracy by step 16K
        *   **Stage 2.5 (16K-22K steps)**: First exposure to imperfection ⭐ **NEW STAGE**
            - `noise_peaks=1`, `peak_dropout=0.02`, `mass_error_ppm=2.0`
            - `spectrum_loss_weight=0.1`, length 7-12
            - **Goal**: Gentle introduction - model sees "mostly good" spectra
            - Expected: 85-88% token accuracy (slight drop is OK)
        *   **Stage 2.75 (22K-28K steps)**: Gradual noise increase ⭐ **NEW STAGE**
            - `noise_peaks=2`, `peak_dropout=0.05`, `mass_error_ppm=5.0`
            - `spectrum_loss_weight=0.12`, length 7-12
            - Expected: 80-85% token accuracy (continued adaptation)
        *   **Stage 3 (28K-38K steps)**: Moderate difficulty
            - `noise_peaks=5`, `peak_dropout=0.10`, `mass_error_ppm=10.0`
            - `spectrum_loss_weight=0.15`, length 7-15
            - Expected: 70-75% token accuracy (realistic for noisy data)
        *   **Stage 4 (38K-50K steps)**: Realistic conditions
            - `noise_peaks=8`, `peak_dropout=0.15`, `mass_error_ppm=15.0`
            - `spectrum_loss_weight=0.15`, length 7-18, intensity_variation=0.2
            - Expected: 60-70% token accuracy (challenging but learnable)
    3.  **Success Criteria**: NO catastrophic drops >20% between stages. Gradual degradation curve instead of cliffs.
    4.  **Validation**: Run for 50K steps, monitor token accuracy at every stage boundary (steps 8K, 16K, 22K, 28K, 38K, 50K)

---

### Phase 2: Enhancing Data Realism & Deepening Analysis

**Goal**: Make the synthetic data more challenging and realistic to build a more robust model, while also gaining a deeper understanding of the current model's performance.

**Priority 2.1: Add Doubly-Charged Fragment Ions**
*   **Why**: In spectra from precursors with a charge of +2 or higher, doubly-charged `b` and `y` ions are common. Our current model has never seen these and would misinterpret them as noise. This is a critical piece of missing physics.
*   **Implementation Proposal**:
    1.  **`src/data/synthetic.py`**: Update `generate_theoretical_spectrum` to optionally generate `b++` and `y++` ions. Their m/z is calculated as `(neutral_mass + 2 * PROTON_MASS) / 2`. Generation should be more frequent for precursors with a charge > 2.
    2.  **`src/training/losses.py`**: Update `SpectrumMatchingLoss` to also calculate the expected masses for these doubly-charged ions, ensuring the loss function is consistent with the new data.

**Priority 2.2: Implement Dual Validation Sets**
*   **Why**: To provide better diagnostics. Using a fixed `val_easy` (clean data) and `val_hard` (realistic data) set allows us to track if the model is retaining its knowledge of simple cases while measuring its progress on the difficult final task.
*   **Implementation Proposal**:
    1.  **`scripts/train.py`**: Create two separate validation `SyntheticPeptideDataset` instances, `val_easy` and `val_hard`, with the appropriate difficulty settings.
    2.  **`src/training/trainer.py`**: Modify the `Trainer` to accept a dictionary of validation dataloaders. Update the `evaluate` method to loop through them, logging metrics with descriptive names (e.g., `val_easy_seq_acc`, `val_hard_seq_acc`).

**Priority 2.3: Analyze Performance vs. Peptide Length**
*   **Why**: To understand the current model's limitations. High average accuracy can sometimes mask weaker performance on longer, more difficult peptides.

---

### Phase 3: Expanding the Chemical Space

**Goal**: Teach the model to handle the most common chemical modifications that occur in real biological samples.

**Priority 3.1: Incorporate Common Post-Translational Modifications (PTMs)**
*   **Why**: To have any success on real-world data, the model must be able to identify peptides with common modifications, such as the oxidation of Methionine.
*   **Implementation Proposal**:
    1.  **`src/constants.py`**: Add new entries to `AMINO_ACID_MASSES` for modified residues (e.g., `"M(ox)": 147.0354`).
    2.  **Update Vocabulary**: The global `VOCAB` and `AA_TO_IDX` mappings will need to be expanded. This will change the size of the model's embedding and output layers, requiring re-initialization of those weights.
    3.  **`src/data/synthetic.py`**: Update `generate_random_peptide` to have a chance of including these modified residues.
    4.  The loss function and model architecture should adapt automatically, provided they use the shared constants.

---

### Phase 4: Transition to Real Data

**Goal**: Validate the model on public benchmarks and adapt it to the nuances of real experimental data. This phase builds on the robustness developed in Phases 1, 2, and 3.

**Priority 4.1: Implement a Real Data Pipeline**
*   **Why**: The model needs to be able to ingest and process data from standard mass spectrometry file formats.
*   **Implementation Proposal**:
    1.  **Create `RealPeptideDataset`**: Build a new `torch.utils.data.Dataset` class.
    2.  **Use `pyteomics`**: Integrate the `pyteomics` library to parse `.mgf` or `.mzML` files, extracting peak lists and precursor information.
    3.  **Add Preprocessing**: Implement crucial preprocessing steps for real data, most importantly **intensity normalization** (e.g., scaling all peak intensities in a spectrum relative to the most intense "base peak").

**Priority 4.2: Execute Pre-training & Fine-tuning Strategy**
*   **Why**: This is the state-of-the-art strategy for bridging the synthetic-to-real gap. It leverages the vastness of synthetic data while adapting the model to the specific characteristics of real instruments.
*   **Implementation Proposal**:
    1.  **Stage 1 (Pre-training)**: Train a model for a large number of steps using the enhanced synthetic data from Phases 1 and 2.
    2.  **Stage 2 (Fine-tuning)**: Load the weights from the pre-trained model and continue training on a real, annotated dataset (e.g., from the Nine-Species benchmark) using a smaller learning rate. This adapts the model to real noise and intensity patterns without "forgetting" the fundamental physics learned during pre-training.

---

### Summary: Immediate Next Steps

Based on the 50K training run results (93% token accuracy, 58% sequence accuracy on clean data, but catastrophic drops when noise introduced), here's the prioritized action plan:

**Week 1: Infrastructure & Validation** (Phase 0)
1. ✅ **Set up GPU training** (Priority 0.1) - Migrate to remote NVIDIA GPU for 5-10x speedup
2. ✅ **Verify validation tracking** (Priority 0.2) - Ensure val_sequence_accuracy is logged (seems already working based on your 58-59% observation)
3. ⚪ **Run diagnostic tests** (Priority 0.3) - Validate learning is genuine (wrong precursor mass test, accuracy vs length)

**Week 2-3: Critical Curriculum Fix** (Phase 1)
4. ✅ **Implement 6-stage curriculum** (Priority 1.2) - Replace catastrophic 4-stage with smooth 6-stage
5. ✅ **Run 50K training on GPU** - Validate no cliffs at stage boundaries
6. ✅ **Compare results** - Expect gradual degradation (90%→85%→75%→65%) instead of cliffs (93%→49%)

**Week 4: Realism Enhancements** (Phase 2)
7. ⚪ **Add doubly-charged ions** (Priority 2.1) - Critical missing physics
8. ⚪ **Dual validation sets** (Priority 2.2) - Track clean vs noisy performance separately

**Week 5+: Real Data Pipeline** (Phase 4)
9. ⚪ **Build RealPeptideDataset** (Priority 4.1) - pyteomics integration
10. ⚪ **Pre-train + fine-tune** (Priority 4.2) - Synthetic→Real transfer

**Phase 3 (PTMs) can wait** - Not needed until real data shows need

---

### Key Insights from Initial 50K Run

**✅ What Worked:**
- TRM architecture fundamentally sound (93% token acc, 58% seq acc on clean data)
- Spectrum matching loss HELPS (mass error 19K→6.7K ppm in Stage 2)
- Model capacity sufficient (3.7M params adequate for clean data)
- Physics-based loss + iterative refinement = winning combination

**❌ What Failed:**
- Curriculum too aggressive (0→5 noise peaks = distribution shift)
- Model has NO robustness to noise (93%→49% cliff)
- "Every peak is signal" assumption breaks catastrophically

**🎯 Root Cause:**
Not an architecture problem. Not a loss function problem. It's a **curriculum/data distribution problem**. Model needs to see "mostly good + a little noise" before "very noisy".

**📊 Success Metrics for Next Run:**
- ✅ Token accuracy: 90% @ 16K (clean), 85% @ 22K (1 noise), 75% @ 28K (2 noise), 65% @ 50K (realistic)
- ✅ NO drops >15% between ANY consecutive checkpoints
- ✅ Val sequence accuracy: >50% on clean, >20% on moderate noise
- ✅ Mass error: <10K ppm throughout (spectrum loss keeps working)

---

### Long-Term Vision (Beyond This Roadmap)

Once Phases 0-2 are complete and model handles noisy synthetic data gracefully:
1. **Benchmark against Casanovo** on Nine-Species dataset
2. **Publish results** if competitive (TRM for peptide sequencing is novel)
3. **Scale up** - Try 10M param model, 100K training steps
4. **Real-world deployment** - Build inference API, integrate with proteomics pipelines

The 93%/58% clean-data result validates the approach. Now it's about robustness.

