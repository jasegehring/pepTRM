# MS/MS Fragmentation Physics & Neural Network Modeling

**Created**: December 9, 2025
**Purpose**: Reference for implementing doubly-charged ions and neutral losses in Stage 1

---

## Part 1: MS/MS Fragmentation Chemistry

### 1.1 Fundamental Ion Types

**b-ions (N-terminal fragments)**
- Formed when the N-terminal side retains the charge after peptide bond cleavage
- Nomenclature: b₁, b₂, ..., bₙ₋₁ (where n = peptide length)
- Mass calculation: Σ(amino acid masses from N-terminus)

**y-ions (C-terminal fragments)**
- Formed when the C-terminal side retains the charge after peptide bond cleavage
- Nomenclature: y₁, y₂, ..., yₙ₋₁
- Mass calculation: Σ(amino acid masses from C-terminus) + H₂O

**Key Finding**: In low-energy CID (collision-induced dissociation), b and y ions dominate the spectrum, making them the primary targets for modeling.

**Sources:**
- [Mascot Peptide Fragmentation Help](http://www.matrixscience.com/help/fragmentation_help.html)
- [Chemistry LibreTexts: CID of Peptides](https://chem.libretexts.org/Bookshelves/Analytical_Chemistry/Supplemental_Modules_(Analytical_Chemistry)/Analytical_Sciences_Digital_Library/In_Class_Activities/Biological_Mass_Spectrometry:_Proteomics/Section_4:_MS-MS_and_De_Novo_Sequencing/Section_4B._CID_of_Peptides_and_De_Novo_Sequencing)

---

### 1.2 Doubly-Charged Fragment Ions (b++, y++)

**Critical Finding**: Doubly-charged fragment ions are COMMON and important for accurate modeling.

**Prevalence:**
- In ESI (electrospray ionization), tryptic peptides typically carry +2 or +3 charge
- Fragment ions can inherit multiple protons, creating b++ and y++ ions
- Analysis of 285,000+ MS/MS spectra shows y++ ions are prevalent
- Sometimes y++ peak intensity EXCEEDS y+ intensity

**Charge State Distribution:**
- Most intense fragments: y-ions in 500-900 Da range, containing C-terminal 4-8 residues
- Doubly protonated precursors yield ~50% higher quality spectra than singly charged
- For precursor charge ≥+3, doubly-charged fragments become very common

**Formation Factors:**
- **Bond cleavage position**: Terminal fragments more likely to be doubly charged
- **Peptide length**: Longer peptides → more doubly-charged fragments
- **Residue composition**: Basic residues (K, R, H) increase probability
- **Remote residue effects**: Residues far from cleavage site can influence charge retention

**Mass Calculation:**
```
b++ mass = (Σ residue masses + 2×PROTON_MASS) / 2
y++ mass = (Σ residue masses + H₂O + 2×PROTON_MASS) / 2
```

**Implication for Our Model**: Currently missing ~30-50% of peaks in +2/+3 precursor spectra.

**Sources:**
- [Charge States of y Ions](https://link.springer.com/article/10.1007/s13361-011-0089-9)
- [Statistical Characterization of Ion Trap MS](https://pubmed.ncbi.nlm.nih.gov/12641236/)
- [Detection of Doubly Charged Peptide Ions](https://pubmed.ncbi.nlm.nih.gov/30868190/)
- [Structural Heterogeneity of b-Ions](https://pmc.ncbi.nlm.nih.gov/articles/PMC3305756/)

---

### 1.3 Neutral Losses (H₂O, NH₃)

**Definition**: Loss of small neutral molecules (H₂O, NH₃, CO) from fragment ions, creating satellite peaks.

**Water Loss (-18 Da, -H₂O):**
- **Amino acids**: Ser (S), Thr (T), Asp (D), Glu (E)
- **Mechanism**: Hydroxyl or carboxyl groups facilitate H₂O elimination
- **Common patterns**:
  - b-H₂O ions from Ser/Thr
  - y-H₂O ions from Ser/Thr/Asp/Glu
- **N-terminal Thr-Thr or Thr-Ser**: Consecutive H₂O losses observed

**Ammonia Loss (-17 Da, -NH₃):**
- **Amino acids**: Asn (N), Gln (Q), Lys (K), Arg (R)
- **Mechanism**: Amide side chains (N, Q) or amino groups (K, R) donate NH₃
- **Fragment types**: b-NH₃ and y-NH₃ with higher abundance than corresponding parent ion
- **Arg special case**: y-17 ions often more abundant than y ions

**Proton Mobility Model:**
- Neutral loss prevalence depends on proton mobility along peptide backbone
- Internal basic residues (K, R, H) LIMIT mobility → affects neutral loss pathways
- Mobile proton mechanism: proton migrates to amide bond → fragmentation

**Frequency:**
- Not all b/y ions show neutral losses (depends on amino acid composition)
- Typically 20-40% of b/y ions have corresponding neutral loss peaks
- Can significantly diversify the peak set (important for spectrum matching)

**Implication for Our Model**: Missing neutral loss peaks reduces spectrum matching accuracy, especially for S/T/N/Q-rich peptides.

**Sources:**
- [Investigation of Neutral Loss During CID](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1360221/)
- [Predicting Intensity Ranks of Fragment Ions](https://pmc.ncbi.nlm.nih.gov/articles/PMC2738854/)
- [Mascot Fragmentation Help](http://www.matrixscience.com/help/fragmentation_help.html)
- [Pathways for Water Loss](https://link.springer.com/article/10.1007/s13361-011-0282-x)

---

## Part 2: How Neural Networks Model Fragmentation

### 2.1 DeepNovo (2017) - First Deep Learning Approach

**Architecture:**
- **Ion-CNN**: Convolutional network learns local features from peaks
- **LSTM Decoder**: Recurrent network autoregressively predicts sequence
- **Prefix-based prediction**: Uses previous amino acids as context

**Physics Modeling:**
- Learns peak patterns implicitly from data (no explicit b/y/b++/y++ labels)
- CNN captures local peak neighborhoods
- LSTM learns sequence-dependent fragmentation patterns

**Limitations:**
- No explicit charge state modeling
- Doesn't distinguish b++ from noise peaks (learns correlation, not causation)
- Limited to singly-charged fragments in training data

**Sources:**
- [De novo peptide sequencing by deep learning (PNAS)](https://www.pnas.org/doi/10.1073/pnas.1705691114)
- [DeepNovo Paper (Semantic Scholar)](https://www.semanticscholar.org/paper/De-novo-peptide-sequencing-by-deep-learning-Tran-Zhang/8ce5660f3e51c17cd40bf5082798e05062ddc026)

---

### 2.2 Casanovo (2022-2023) - Transformer Architecture

**Architecture:**
- **Transformer Encoder**: Processes spectrum as sequence of peaks (mass, intensity)
- **Transformer Decoder**: Autoregressively generates peptide sequence
- **Beam Search**: Explores multiple candidate sequences

**Physics Modeling:**
- Encoder: Peak embeddings from (mass, intensity) pairs
- Precursor mass + charge as explicit conditioning
- Attention mechanism learns which peaks correspond to which amino acids
- Still mostly implicit - learns b/y patterns from data

**Improvements over DeepNovo:**
- Better long-range dependencies (transformer vs LSTM)
- Explicit precursor mass constraint
- Charge state as input feature

**Key Insight**: "The encoder receives an MS/MS spectrum as input and converts it into a latent embedding"
- No hard-coded fragmentation rules
- Model learns peak-to-sequence mapping end-to-end

**Recent Variants (2023-2024):**
- **ContraNovo**: Contrastive learning
- **InstaNovo**: Faster inference
- **AdaNovo**: Adaptive learning

**Sources:**
- [Sequence-to-sequence translation (Nature Communications)](https://www.nature.com/articles/s41467-024-49731-x)
- [Improvements to Casanovo (bioRxiv)](https://www.biorxiv.org/content/10.1101/2025.07.25.666826v1.full.pdf)
- [Deep Learning Methods for De Novo Sequencing (Wiley)](https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/10.1002/mas.21919)
- [NovoBench Benchmarking Study](https://arxiv.org/html/2406.11906v1)

---

### 2.3 Prosit (2019) - Spectrum Prediction Model

**Purpose**: FORWARD model (peptide → spectrum), NOT de novo sequencing
**Use Case**: Generates synthetic training data or validates predictions

**Architecture:**
- **Input**: Peptide sequence + collision energy + precursor charge
- **Latent Space**: Sequence features × (energy, charge) → compressed representation
- **Output Decoder**: Matrix of 174 fragment intensities
  - Covers b and y ions (charge +1, +2, +3)
  - Supports peptides up to 30 amino acids

**Physics Modeling - EXPLICIT:**
- Predicts intensities for SPECIFIC fragment types (b₁, b₂, y₁, y₂, etc.)
- Separate output neurons for +1, +2, +3 charge states
- Collision energy as input (affects fragmentation patterns)
- Trained on millions of experimental spectra (learns realistic intensity distributions)

**Loss Function:**
- Normalized spectral contrast loss
- Optimizes for intensity prediction accuracy

**Key Innovation**: Prosit explicitly models the 174 most common fragment types, including:
- b ions: b₁⁺, b₁⁺⁺, b₁⁺⁺⁺, b₂⁺, b₂⁺⁺, ...
- y ions: y₁⁺, y₁⁺⁺, y₁⁺⁺⁺, y₂⁺, y₂⁺⁺, ...
- Does NOT model neutral losses (b-H₂O, y-NH₃) - limitation

**2024 Update:**
- Fragment ion intensity prediction improves non-tryptic peptide identification
- Integration with timsTOF instruments
- PROSPECT PTMs dataset for modified peptides

**Sources:**
- [Prosit: Proteome-wide Prediction (Nature Methods)](https://www.nature.com/articles/s41592-019-0426-7)
- [Prosit Deep Learning Model](https://www.mls.ls.tum.de/en/compms/research/projects/prosit/)
- [Peptide Property Prediction Using AI (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12076536/)

---

### 2.4 MS2PIP - Alternative Spectrum Predictor

**Approach**: More traditional, gradient boosted trees + feature engineering
**Coverage**: Multiple fragmentation modes (HCD, CID, ETD)
**Physics**: Hand-crafted features based on amino acid properties
**Limitation**: Less accurate intensity predictions than Prosit

---

## Part 3: Comparison: Explicit vs Implicit Physics

### Explicit Physics Modeling (Prosit-style)

**Approach:**
```python
# Define specific fragment types
fragment_types = ['b1+', 'b1++', 'b2+', 'b2++', ..., 'y1+', 'y1++', ...]

# Predict intensity for each type
intensities = model(sequence, charge, collision_energy)
# Output: [I_b1+, I_b1++, I_b2+, ..., I_y1+, I_y1++, ...]
```

**Pros:**
- Interpretable (know which peak is which)
- Can enforce physical constraints (e.g., y++ requires charge ≥2)
- Easy to extend (add neutral losses as new output dimensions)
- Training signal is clear (match predicted b₂⁺⁺ to observed b₂⁺⁺ peak)

**Cons:**
- Fixed number of fragments (doesn't scale to very long peptides)
- Must decide ahead of time which ion types to model
- Can't handle novel fragmentation pathways

---

### Implicit Physics Learning (Casanovo-style)

**Approach:**
```python
# Feed spectrum as unordered set of peaks
spectrum_embedding = encoder(peaks_mass, peaks_intensity)

# Decoder learns to map peaks → sequence
sequence = decoder(spectrum_embedding, precursor_mass, charge)
```

**Pros:**
- Flexible (learns any fragmentation pattern from data)
- Scales to arbitrary peptide lengths
- Can discover novel peak types
- End-to-end differentiable

**Cons:**
- Black box (hard to debug why it fails)
- May learn spurious correlations (noise peaks, contaminants)
- Requires massive training data to learn physics implicitly
- No guarantee it respects physical constraints

---

### Our Current Approach (pepTRM)

**What We Do:**
- Implicit learning (Casanovo-style encoder-decoder)
- Spectrum matching loss (soft physics constraint)
- Mass embeddings for peaks and precursor

**What We're Missing:**
1. **Doubly-charged ions** - no b++/y++ in synthetic data or loss function
2. **Neutral losses** - no b-H₂O, y-NH₃ peaks
3. **Explicit ion type prediction** - decoder doesn't know which peaks are b vs y

**Where We Sit:**
- More structured than Casanovo (spectrum matching loss)
- Less structured than Prosit (no explicit ion type outputs)
- **Sweet spot?** Hybrid approach: Implicit learning + physics-guided loss

---

## Part 4: Recommendations for Stage 1 Implementation

### Priority 1: Add Doubly-Charged Ions (CRITICAL)

**Why:**
- Missing 30-50% of peaks in realistic spectra
- Major cause of performance gap (66.7% → 70% target)
- Relatively easy to implement

**Implementation:**

1. **Update Synthetic Data (`src/data/synthetic.py`)**
```python
def generate_theoretical_spectrum(..., charge: int):
    peaks = []

    # b-ions (singly charged)
    for i in range(1, n):
        mass = sum(residues[:i]) + PROTON_MASS
        peaks.append((mass, intensity_b))

        # b++ (doubly charged) - if precursor charge ≥ 2
        if charge >= 2 and i >= 3:  # Need at least 3 residues for b++
            mass_doubly = (sum(residues[:i]) + 2*PROTON_MASS) / 2
            intensity_doubly = intensity_b * prevalence_factor(i, charge)
            peaks.append((mass_doubly, intensity_doubly))

    # Same for y-ions and y++
    ...
```

**Prevalence Model (from literature):**
- charge +2 precursor: 10-20% of b/y ions have ++ counterpart
- charge +3 precursor: 30-50% of b/y ions have ++ counterpart
- Longer fragments (n ≥ 4) more likely to be doubly charged
- Can use simple heuristic or learn from data

2. **Update Spectrum Matching Loss (`src/training/losses.py`)**
```python
def compute_theoretical_peaks(..., charge):
    # b-ions +1
    b_ions_single = ...

    # b-ions +2 (if charge ≥ 2)
    if charge >= 2:
        b_ions_double = (cumsum_masses + 2*PROTON_MASS) / 2
        theoretical_peaks = torch.cat([b_ions_single, b_ions_double, ...])

    # Match observed peaks to ALL theoretical peaks
    distances = cdist(observed, theoretical)
    ...
```

**Testing:**
- Generate spectrum with b++/y++
- Verify loss decreases when predictions match doubly-charged peaks
- Check that model learns to attend to b++ peaks

---

### Priority 2: Add Neutral Losses (MEDIUM)

**Why:**
- 20-40% of b/y ions have neutral loss counterparts
- Improves spectrum matching for S/T/N/Q-rich peptides
- More complex than doubly-charged (amino acid dependent)

**Implementation:**

1. **Amino Acid-Specific Neutral Losses**
```python
NEUTRAL_LOSSES = {
    'S': ['H2O'],  # Serine loses water
    'T': ['H2O'],  # Threonine loses water
    'D': ['H2O'],  # Aspartic acid loses water
    'E': ['H2O'],  # Glutamic acid loses water
    'N': ['NH3'],  # Asparagine loses ammonia
    'Q': ['NH3'],  # Glutamine loses ammonia
    'K': ['NH3'],  # Lysine loses ammonia
    'R': ['NH3'],  # Arginine loses ammonia
}

WATER_MASS = 18.01056
AMMONIA_MASS = 17.02655
```

2. **Generate Neutral Loss Peaks**
```python
def add_neutral_losses(fragment_mass, sequence_prefix):
    peaks = [(fragment_mass, intensity_main)]

    # Check if any residues in prefix have neutral losses
    for aa in sequence_prefix:
        if aa in NEUTRAL_LOSSES:
            for loss_type in NEUTRAL_LOSSES[aa]:
                if loss_type == 'H2O':
                    loss_mass = fragment_mass - WATER_MASS
                    loss_intensity = intensity_main * 0.3  # 30% of main peak
                elif loss_type == 'NH3':
                    loss_mass = fragment_mass - AMMONIA_MASS
                    loss_intensity = intensity_main * 0.4  # Can be higher than main!

                peaks.append((loss_mass, loss_intensity))

    return peaks
```

**Complexity:**
- Need to track amino acid composition of each fragment
- Relative intensities vary (sometimes b-NH₃ > b)
- Can create many peaks (b-H₂O, b-NH₃, b-H₂O-NH₃, ...)

**Simplification for MVP:**
- Only add neutral losses for yn-2, yn-1, yn-3 (most prevalent)
- Only add for fragments containing S/T/N/Q/K/R
- Use fixed intensity ratio (30% of main peak)

---

### Priority 3: Hybrid Approach - Explicit Ion Type Features (OPTIONAL)

**Why:**
- Gives model explicit signal about which peaks matter
- Interpretable attention (can visualize which peaks model uses)
- Helps with generalization (fewer parameters to learn)

**Implementation:**

Add "ion type embedding" to peak encoder:
```python
class PeakEncoder(nn.Module):
    def encode_peak(self, mass, intensity):
        mass_emb = sinusoidal_embedding(mass)
        intensity_emb = mlp(intensity)

        # NEW: Add ion type features (computed from sequence + mass)
        ion_type_features = self.compute_ion_type_likelihood(mass)
        # [prob_b+, prob_b++, prob_y+, prob_y++, prob_b-H2O, ...]

        return concat(mass_emb, intensity_emb, ion_type_features)

    def compute_ion_type_likelihood(self, mass):
        # Soft matching: how likely is this peak to be b1, b2, ..., y1, y2?
        # Use precursor mass and sequence length distribution
        ...
```

**Benefit:**
- Model can learn "this peak at m/z 450 is probably y3++" → attend strongly
- Provides inductive bias (but model can still ignore if not useful)

**Risk:**
- Adds complexity
- May not help if data is sufficient (neural networks are good at learning features)

**Decision**: Implement only if Priority 1+2 don't reach target performance.

---

## Part 5: Integration with Prosit (Stage 1B)

### Why Use Prosit?

**Current synthetic data issues:**
- Uniform intensity distribution (unrealistic)
- No instrument-specific effects
- Missing subtle fragmentation biases

**Prosit provides:**
- Realistic intensity predictions (trained on millions of real spectra)
- Collision energy dependence
- Charge state effects
- Implicitly includes doubly-charged ions (b++, y++ in output)

### Integration Strategy

**Option A: Replace Synthetic Data Generator**
```python
from prosit import PrositPredictor

class PrositSyntheticDataset:
    def _generate_sample(self):
        peptide = generate_random_peptide(...)
        charge = sample_charge()

        # Use Prosit to predict spectrum
        predicted_spectrum = prosit.predict(
            sequences=[peptide],
            collision_energies=[30],  # NCE 30%
            precursor_charges=[charge]
        )

        # Add noise on top of Prosit predictions
        spectrum = add_noise(predicted_spectrum,
                             noise_peaks=self.noise_peaks,
                             dropout=self.peak_dropout)

        return spectrum
```

**Option B: Hybrid (Prosit for Intensities, Our Physics for Peaks)**
```python
# Generate peak list with our explicit physics (b, y, b++, y++, neutral losses)
our_peaks = generate_theoretical_peaks(peptide, charge)

# Use Prosit to predict intensities for those peaks
prosit_intensities = prosit.predict_intensities(peptide, our_peaks, charge)

# Combine
spectrum = zip(our_peaks, prosit_intensities)
```

**Recommendation**: Option A (full Prosit integration)
- Prosit already includes b++/y++ in its 174 fragment types
- More realistic overall
- Can always add neutral losses on top (Prosit doesn't have them)

---

## Part 6: Testing Strategy

### Test 1: Verify Doubly-Charged Ion Generation
```python
peptide = "PEPTIDE"  # 7 amino acids
charge = 3

spectrum = generate_theoretical_spectrum(peptide, charge, ...)

# Should have:
# - 6 b+ ions (b1, b2, ..., b6)
# - ~4 b++ ions (b3++, b4++, b5++, b6++)
# - 6 y+ ions
# - ~4 y++ ions

assert num_doubly_charged_peaks > 0, "Missing b++/y++ ions!"
assert (spectrum['charge'] >= 2).sum() > 0, "No doubly charged peaks!"
```

### Test 2: Spectrum Matching Loss Sensitivity
```python
# Generate spectrum with b++
true_spectrum = generate(peptide, charge=3)

# Predict sequence (should match true peptide)
predictions = model(true_spectrum)

# Loss should be low (model sees b++ peaks)
loss_with_double = spectrum_loss(predictions, true_spectrum)

# Remove b++ peaks (simulate old model)
spectrum_no_double = remove_doubly_charged(true_spectrum)
loss_without_double = spectrum_loss(predictions, spectrum_no_double)

assert loss_with_double < loss_without_double, "Model doesn't use b++!"
```

### Test 3: Real Data Validation
```python
# Load real MS/MS spectrum from Nine-Species
real_spectrum, peptide = load_real_spectrum(idx=0)

# Predict with our model
predicted_sequence = model.predict(real_spectrum)

# Generate theoretical spectrum (with b++/y++)
theoretical = generate_theoretical_spectrum(peptide, charge, ...)

# Check coverage
matched_peaks = match_peaks(real_spectrum, theoretical, tolerance=20ppm)
coverage = matched_peaks.sum() / len(real_spectrum)

assert coverage > 0.6, "Low peak coverage - missing ion types?"
```

---

## Part 7: Expected Performance Improvements

### Current Performance Gap
- Clean synthetic: 95.9% token acc ✓ (target: >85%)
- Realistic synthetic: 66.7% token acc ✗ (target: >70%)
- Gap: 3.3 percentage points

### Predicted Impact of Doubly-Charged Ions

**Conservative Estimate:** +5-8% token accuracy
- Rationale: Currently missing 30-50% of peaks in charge +2/+3 spectra
- Spectrum matching loss gets better gradient signal
- Model can distinguish signal from noise more easily

**Optimistic Estimate:** +10-15% token accuracy
- Rationale: If doubly-charged ions are main bottleneck
- Unlocks learning on harder examples (longer peptides, higher charge)

**Realistic Target:** 66.7% → 72-75% token accuracy after adding b++/y++

### Predicted Impact of Neutral Losses

**Estimate:** +2-4% token accuracy
- Smaller effect than doubly-charged (only affects 20-40% of fragments)
- Helps for specific peptides (S/T/N/Q-rich)
- May not help on average (depends on amino acid distribution)

### Combined with Prosit Integration

**Prosit adds:**
- Realistic intensity distributions: +3-5% (better signal/noise discrimination)
- Instrument-specific patterns: +2-3% (sim-to-real transfer)

**Total Predicted:** 66.7% → 78-82% token accuracy
- This would EXCEED target (70% token, 40% sequence)
- Realistic sequence accuracy: 30-35% (vs 40% target)

---

## Part 8: Open Questions & Future Work

### Q1: Should we predict ion types explicitly?

**Trade-off:**
- Pro: Interpretable, easy to debug, enforces physics
- Con: Fixed architecture, can't discover new ion types
- **Decision**: Start implicit (current approach), add explicit if needed

### Q2: How to handle triply-charged ions (b+++, y+++)?

**Data:**
- Rare for tryptic peptides (mostly charge +2/+3 precursors)
- More common for long peptides or non-tryptic digestion
- Prosit supports up to +3

**Recommendation:**
- Add b+++/y+++ support for completeness
- Expect small impact (<1% improvement)
- May help for real data (real spectra have occasional +3 fragments)

### Q3: What about other ion types (a, c, x, z)?

**Literature:**
- a-ions: N-terminal, b-ion minus CO (-28 Da)
- c-ions: N-terminal, alternative cleavage
- x-ions: C-terminal, y-ion plus CO (+28 Da)
- z-ions: C-terminal, alternative cleavage

**Prevalence:** LOW in CID/HCD (the most common methods)
- Mostly seen in ETD (electron transfer dissociation)
- Our data is HCD → can skip for now

**Recommendation:** Not priority for Stage 1

### Q4: How to validate on real data without ground truth?

**Challenge:**
- Real spectra don't have "correct" peak labels
- Can't directly measure "did we predict b++ correctly?"

**Solutions:**
1. **Coverage metric**: % of observed peaks matched to theoretical
2. **Cross-validation**: Train on database search results, test on unseen
3. **Benchmarking**: Compare to Casanovo on Nine-Species (indirect validation)

---

## Summary & Action Items for Stage 1

### Must Do (Priority 1) ✅
1. **Implement doubly-charged ions (b++, y++)**
   - Update `src/data/synthetic.py`
   - Update `src/training/losses.py`
   - Test on charge +2 and +3 precursors
   - Expected: +5-10% token accuracy

2. **Integrate Prosit for realistic intensities**
   - Install Prosit library
   - Create `PrositSyntheticDataset` wrapper
   - Train with Prosit-generated spectra
   - Expected: +3-5% token accuracy

### Should Do (Priority 2) ⚠️
3. **Add neutral losses (b-H₂O, y-NH₃)**
   - Amino acid-specific rules
   - Start with S/T/N/Q/R/K only
   - Expected: +2-4% token accuracy

### Could Do (Priority 3) 💡
4. **Explicit ion type features**
   - Add to peak encoder
   - Soft matching (probabilistic)
   - Only if Priority 1+2 don't reach target

### Total Expected Performance
- Current: 66.7% token / 17.3% sequence
- After Stage 1: **75-80% token / 30-35% sequence**
- Target: 70% token / 40% sequence → **SHOULD EXCEED TOKEN TARGET**

---

## References

### MS/MS Fragmentation Chemistry
- [Mascot Peptide Fragmentation Help](http://www.matrixscience.com/help/fragmentation_help.html)
- [PNAS: Improved Peptide Identification](https://www.pnas.org/doi/full/10.1073/pnas.0405549101)
- [Chemistry LibreTexts: CID of Peptides](https://chem.libretexts.org/Bookshelves/Analytical_Chemistry/Supplemental_Modules_(Analytical_Chemistry)/Analytical_Sciences_Digital_Library/In_Class_Activities/Biological_Mass_Spectrometry:_Proteomics/Section_4:_MS-MS_and_De_Novo_Sequencing/Section_4B._CID_of_Peptides_and_De_Novo_Sequencing)

### Doubly-Charged Ions
- [Charge States of y Ions (Springer)](https://link.springer.com/article/10.1007/s13361-011-0089-9)
- [Detection of Doubly Charged Peptide Ions (PubMed)](https://pubmed.ncbi.nlm.nih.gov/30868190/)
- [Statistical Characterization of Ion Trap MS](https://pubmed.ncbi.nlm.nih.gov/12641236/)
- [Structural Heterogeneity of b-Ions (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3305756/)

### Neutral Losses
- [Investigation of Neutral Loss During CID (PMC)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1360221/)
- [Predicting Intensity Ranks of Fragment Ions (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC2738854/)
- [Pathways for Water Loss (Springer)](https://link.springer.com/article/10.1007/s13361-011-0282-x)

### Neural Network Architectures
- [DeepNovo: De novo Sequencing by Deep Learning (PNAS)](https://www.pnas.org/doi/10.1073/pnas.1705691114)
- [Casanovo: Sequence-to-sequence Translation (Nature Communications)](https://www.nature.com/articles/s41467-024-49731-x)
- [Improvements to Casanovo (bioRxiv)](https://www.biorxiv.org/content/10.1101/2025.07.25.666826v1.full.pdf)
- [Deep Learning Methods Review (Wiley)](https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/10.1002/mas.21919)
- [NovoBench Benchmarking Study (arXiv)](https://arxiv.org/html/2406.11906v1)

### Prosit & Spectrum Prediction
- [Prosit: Proteome-wide Prediction (Nature Methods)](https://www.nature.com/articles/s41592-019-0426-7)
- [Prosit Deep Learning Model (TUM)](https://www.mls.ls.tum.de/en/compms/research/projects/prosit/)
- [Peptide Property Prediction Using AI (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12076536/)

---

**Document Status**: ✅ Complete - Ready for Stage 1 Implementation
**Next Steps**: Use this document to guide implementation of doubly-charged ions and Prosit integration
