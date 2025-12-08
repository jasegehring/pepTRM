### **PepTRM Project: Action Plan & Roadmap**

This document outlines a prioritized plan to address critical bugs, enhance model performance, and prepare the PepTRM project for training on real-world mass spectrometry data. The core architecture is sound; the following steps are designed to unlock its full potential.

---

### **Part 1: Immediate Priorities (Critical Bug Fixes)**

These three issues directly impact the model's ability to learn the underlying physics of mass spectrometry. They should be addressed before any further training or experimentation to ensure the results are meaningful.

**1. Fix Incomplete Physics in the Auxiliary Loss**
*   **Priority**: **Critical**
*   **Component**: `src/training/losses.py` (in `SpectrumMatchingLoss.compute_theoretical_peaks`)
*   **Issue**: The auxiliary loss only calculates theoretical peaks for **b- and y-ions**. It is blind to the **a-ions** and **neutral loss** peaks that are generated in the synthetic data.
*   **Impact**: The model receives conflicting training signals. It is punished for correctly matching peaks that the loss function doesn't know about, severely hindering learning.
*   **Actionable Fix**: Modify the `compute_theoretical_peaks` method to calculate expected masses for all ion types that the `synthetic.py` generator can produce. This involves adding logic for a-ions and neutral losses, mirroring the data generation process.
    ```python
    # In SpectrumMatchingLoss.compute_theoretical_peaks:
    # ...
    # After calculating b_ions and y_ions
    
    # Expected a-ion masses (a = b - CO)
    # Note: must be consistent with how b-ions are calculated (i.e. with PROTON_MASS)
    a_ions = b_ions - CO_MASS 

    # Combine all ion types
    theoretical_peaks = torch.cat([b_ions, y_ions, a_ions], dim=1)
    
    # Add logic for neutral losses based on expected residue probabilities if desired
    # ...
    ```

**2. Fix Unused `mass_tolerance` Parameter in Auxiliary Loss**
*   **Priority**: **Critical**
*   **Component**: `src/training/losses.py` (in `SpectrumMatchingLoss.forward`)
*   **Issue**: The `mass_tolerance` parameter is initialized but never used. The loss currently considers matches between theoretical peaks and *all* observed peaks, no matter how far apart they are.
*   **Impact**: The loss signal is noisy and diluted by irrelevant peaks, making training less efficient and less physically meaningful.
*   **Actionable Fix**: Use the `mass_tolerance` to create a window. Only calculate the loss for observed peaks that fall within `+/- mass_tolerance` of a given theoretical peak. This creates a sparse, focused, and more stable loss.
    ```python
    # In SpectrumMatchingLoss.forward, before calculating scores:
    # ...
    distances = torch.abs(theo_expanded - obs_expanded)
    
    # Create a mask for peaks within the tolerance window
    in_window = (distances < self.mass_tolerance)
    
    # Combine with the original peak padding mask
    combined_mask = in_window & peak_mask.unsqueeze(1)
    
    # Apply the combined mask to the scores
    scores = scores.masked_fill(~combined_mask, float('-inf'))
    # ...
    ```

**3. Fix Inconsistent a-ion Mass Calculation in Data Generation**
*   **Priority**: **Critical**
*   **Component**: `src/data/synthetic.py` (in `generate_theoretical_spectrum`)
*   **Issue**: The m/z for `b-` and `y-ions` is calculated as `neutral_mass + PROTON_MASS`, but the `a-ion` calculation is missing the `+ PROTON_MASS`.
*   **Impact**: The training data is physically inconsistent. The model receives flawed data where `a-ions` are systematically misplaced.
*   **Actionable Fix**: Add `PROTON_MASS` to the `a-ion` mass calculation to make it consistent with the other ion types.
    ```python
    # In generate_theoretical_spectrum:
    # ...
    # Inside the 'a' in ion_types block:
    mass = cumulative - CO_MASS + PROTON_MASS # Add PROTON_MASS
    # ...
    ```

---

### **Part 2: Next Steps (Performance Enhancements)**

Once the critical bugs are fixed, these features will most likely provide the largest performance boosts.

**1. Implement Beam Search Decoding**
*   **Priority**: **High**
*   **Rationale**: The current greedy `argmax` decoding is suboptimal for sequence generation. Beam search explores multiple high-probability sequences at each step, significantly increasing the chances of finding the globally optimal peptide sequence.
*   **Implementation Sketch**: Create a new inference function or class that, instead of taking the `argmax` at each of the `T` refinement steps, maintains a "beam" of `k` candidate sequences and their corresponding scores. The recursive loop would be applied to all `k` candidates.

**2. Incorporate Post-Translational Modifications (PTMs)**
*   **Priority**: **Medium**
*   **Rationale**: To move towards real-world utility, the model must be able to handle common PTMs. This is a crucial step for benchmarking on real datasets.
*   **Implementation Sketch**:
    1.  Add PTMs (e.g., `M(ox)`, `S(ph)`) to the `VOCAB` and `AMINO_ACID_MASSES` in `src/constants.py`.
    2.  Update `src/data/synthetic.py` to randomly include these modified amino acids when generating peptides.
    3.  The model architecture itself requires no changes, but the vocabulary size will increase.

---

### **Part 3: Future Work (Transition to Real Data)**

This section outlines the strategy for moving beyond synthetic data to tackle real-world challenges.

**1. Adopt a Pre-training and Fine-tuning Strategy**
*   **Priority**: **Long-term**
*   **Rationale**: This is the proven, state-of-the-art method. Train the model extensively on millions of synthetic spectra to learn the fundamental physics of fragmentation, then fine-tune it on a smaller, high-quality real dataset to adapt it to real instrument noise and characteristics.

**2. Implement a Real Data Pipeline**
*   **Priority**: **Long-term**
*   **Rationale**: The model needs to be able to ingest data from standard mass spectrometry file formats.
*   **Implementation Sketch**:
    1.  **Create a New Data Loader**: Build a `Dataset` class that uses a library like `pyteomics` to read `.mgf` or `.mzML` files. It will need to parse peak lists (m/z, intensity) and precursor information.
    2.  **Add Data Preprocessing**: Implement an intensity normalization step (e.g., normalize to the base peak intensity) within the data loader.
    3.  **Handle Target Sequences**: The pipeline must be able to read peptide sequence annotations and associate them with the correct spectra for training and validation.
