# Training Execution Analysis for `scripts/train_optimized.py`

This document details the scripts involved in a training run initiated by `scripts/train_optimized.py`, their purposes, and potential sources of instability, particularly concerning the `precursor_loss`.

## 1. Entry Point: `scripts/train_optimized.py`

*   **Purpose:** The main script to launch a training run with optimizations for high-performance GPUs.
*   **Key Actions:**
    *   Loads the configuration from `configs/optimized_extended.yaml`.
    *   Initializes the `MS2PIPSyntheticDataset` for generating training data.
    *   Initializes the `CurriculumScheduler` with `DEFAULT_CURRICULUM` from `src/training/curriculum.py`. **This is a major inconsistency**, as the optimized trainer is likely intended to be used with the `EXTENDED_CURRICULUM`.
    *   Instantiates the `OptimizedTrainer`.
    *   Starts the training loop.

## 2. Core Components

### 2.1. Data Generation

*   **`src/data/ms2pip_dataset.py`**:
    *   **Purpose:** Generates synthetic peptide spectra using the `ms2pip` library. This provides more realistic training data than the simple theoretical model.
    *   **Key Features:** It generates peptides and their corresponding spectra on the fly. It calculates the **neutral precursor mass**.
*   **`src/data/synthetic.py`**:
    *   **Purpose:** A simpler synthetic data generator. Not directly used by `train_optimized.py`, but it's a dependency of `src/data/dataset.py`.
*   **`src/data/dataset.py`**:
    *   **Purpose:** Defines the `SyntheticPeptideDataset`, which is a simpler iterable dataset. It is not directly used by `train_optimized.py`, but is a dependency of `src/training/curriculum.py`.
*   **`src/data/encoding.py`**:
    *   **Purpose:** Implements sinusoidal embeddings for continuous mass values. This is crucial for the model to understand mass relationships. It provides `PeakEncoder` and `PrecursorEncoder`.
*   **`src/data/ion_types.py`**:
    *   **Purpose:** Provides a flexible system for defining and computing different fragment ion types (b, y, a, with neutral losses, etc.). This is used for spectrum matching loss.

### 2.2. Model Architecture

*   **`src/model/trm.py`**:
    *   **Purpose:** Defines the main `RecursivePeptideModel` (TRM), which is the core of the sequencing model. It orchestrates the encoder and the recursive decoder.
*   **`src/model/encoder.py`**:
    *   **Purpose:** The `SpectrumEncoder` transforms the input MS/MS spectrum (peaks and precursor info) into a set of contextual embeddings. It uses a Transformer encoder architecture.
*   **`src/model/decoder.py`**:
    *   **Purpose:** The `RecursiveCore` is the heart of the TRM model. It iteratively refines the predicted peptide sequence through a series of "latent" and "answer" steps.
*   **`src/model/layers.py`**:
    *   **Purpose:** Contains building blocks for the Transformer architecture, such as `MultiHeadAttention`, `FeedForward` networks, and `TransformerEncoderLayer`/`TransformerDecoderLayer`.

### 2.3. Training Process

*   **`src/training/trainer_optimized.py`**:
    *   **Purpose:** Implements the main training loop with optimizations like mixed-precision (AMP) and `torch.compile`.
    *   **Key Actions:**
        *   Handles the training step, including the forward and backward passes.
        *   Calculates the precursor m/z from the neutral mass and passes it to the model. This is the location of the recent "fix".
        *   Calls the `CombinedLoss` function with the appropriate inputs.
        *   Updates the curriculum.
*   **`src/training/losses.py`**:
    *   **Purpose:** Defines all the loss functions used in training.
    *   **Components:**
        *   `DeepSupervisionLoss`: The primary cross-entropy loss applied at each step of the recursive decoder.
        *   `SpectrumMatchingLoss`: An auxiliary loss that encourages the predicted peptide's theoretical spectrum to match the observed one.
        *   `PrecursorMassLoss`: The loss that is causing issues. It penalizes predictions where the summed mass of the amino acids does not match the precursor mass.
        *   `CombinedLoss`: A wrapper that combines all the above losses with their respective weights.
*   **`src/training/metrics.py`**:
    *   **Purpose:** Defines evaluation metrics like token accuracy and sequence accuracy.
*   **`src/training/curriculum.py`**:
    *   **Purpose:** Defines the `DEFAULT_CURRICULUM`. This curriculum is being used by `scripts/train_optimized.py`.
    *   **Key Observation:** The `precursor_loss_weight` is introduced at step 8000 and ramps up aggressively. This is the likely cause of the training instability.
*   **`src/training/curriculum_extended.py`**:
    *   **Purpose:** Defines the `EXTENDED_CURRICULUM`. This curriculum appears to be intended for the optimized trainer, but it is **not being used**. It has a more gradual introduction of the `precursor_loss_weight`.

## 3. Key Findings and Potential Issues

1.  **Curriculum Mismatch:** The primary issue is that `scripts/train_optimized.py` is using the `DEFAULT_CURRICULUM` from `src/training/curriculum.py` instead of the `EXTENDED_CURRICULUM` from `src/training/curriculum_extended.py`. The `DEFAULT_CURRICULUM` is much more aggressive in introducing the `precursor_loss`, which is the likely cause of the training instability and NaN values.
2.  **Aggressive `precursor_loss` Introduction:** The `DEFAULT_CURRICULUM` introduces the `precursor_loss` at 8k steps with a weight of 0.2, and quickly ramps it up to 0.5 and 0.7 in the 16k-28k step range. This is likely too fast for the model to adapt, leading to the catastrophic failure.

## 4. Recommendations

1.  **Fix the Curriculum:** The most critical fix is to make `scripts/train_optimized.py` use the `EXTENDED_CURRICULUM`. This can be done by changing the import in `scripts/train_optimized.py`.
2.  **Further Refine the Curriculum:** Even with the `EXTENDED_CURRICULUM`, the introduction of the precursor loss might still be too aggressive. The curriculum I proposed in the previous turn, with a more gradual ramp-up, should be considered.

This analysis provides a clear path to resolving the training instability. The core issue seems to be a configuration problem (the wrong curriculum being used) rather than a deep bug in the model or loss function logic.
