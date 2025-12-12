# Advanced Features Roadmap: Residual Spectrum Embeddings & Mass Gap Tokens

**Document Purpose**: Comprehensive implementation guide for physics-guided enhancements to the peptide TRM model.

**Status**: Planning phase - implement after baseline model is working on real data

**Last Updated**: 2025-12-11

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Feature 1: Residual Spectrum Embeddings](#feature-1-residual-spectrum-embeddings)
3. [Feature 2: Mass Gap Token](#feature-2-mass-gap-token)
4. [Implementation Phases](#implementation-phases)
5. [Technical Specifications](#technical-specifications)
6. [Success Metrics](#success-metrics)
7. [Risk Mitigation](#risk-mitigation)
8. [References & Resources](#references--resources)

---

## Executive Summary

### The Problem
Current recursive architecture has a fundamental limitation: **the model never sees what spectrum its current prediction would produce**. It cross-attends to the observed spectrum (static) but lacks explicit feedback about what peaks remain unexplained.

### The Solution
Two complementary enhancements:

**1. Residual Spectrum Embeddings** (Core Feature)
- At each recursive step, compute: `Residual = Observed_Spectrum - Predicted_Spectrum`
- Embed the residual and inject into recursive loop
- Provides explicit "what remains unexplained" signal
- **Impact**: Better convergence, reduced stagnation, clearer gradient signal

**2. Mass Gap Token** (Advanced Feature)
- Add `[GAP]` token representing unknown mass
- Dual prediction: token identity (classification) + mass delta (regression)
- Enables discovery of post-translational modifications (PTMs)
- **Impact**: Handles modified peptides, enables novel PTM discovery

### Why This Matters

**Scientific Impact**:
- Current de novo tools ignore modifications → [GAP] token solves this
- Database search fails on unknown PTMs → residuals guide model to find them
- Potential for **Nature Methods** publication if successful

**Technical Innovation**:
- First physics-guided recursive architecture for proteomics
- Novel combination of discrete (tokens) + continuous (masses) prediction
- Explicit error feedback loop (residuals) unique in peptide sequencing

### Timeline Estimate
- **Phase 1** (Residual Embeddings): 2-3 weeks implementation + 2-4 weeks validation
- **Phase 2** (Mass Gap Token): 3-4 weeks implementation + 4-8 weeks validation
- **Total**: 3-6 months for full deployment and evaluation

---

## Feature 1: Residual Spectrum Embeddings

### Conceptual Overview

**The Core Idea**: Transform recursive refinement from "implicit" to "explicit" error correction.

**Current Architecture**:
```
Step t: Model predicts sequence → Cross-attend to observed spectrum → Update prediction
        ↑ No explicit feedback about what's explained vs. unexplained
```

**With Residual Embeddings**:
```
Step t: Model predicts sequence → Compute predicted spectrum →
        Residual = Observed - Predicted → Embed residual →
        Inject into next step → Model sees exactly what's missing
```

**Analogy**: Like showing a student their test with wrong answers highlighted, rather than just saying "you got 60%".

### Why Current Cross-Attention Isn't Enough

**The "Stateless Attention" Problem**:
- Cross-attention sees the entire observed spectrum at every step
- Must implicitly learn "ignore peak at 500 Da, I already used it"
- No explicit memory of what's been explained
- Residuals provide **explicit memory** by removing matched peaks

**Evidence This Helps**:
1. Gradient Boosting (XGBoost) uses residuals → state-of-the-art performance
2. Iterative refinement works better with explicit error signals
3. Your model has latent state `z` but it's limited capacity (384-dim)

### Technical Architecture

#### Component 1: Fast Linear Interpolation Renderer

**Purpose**: Convert masses → spectrum bins (differentiable, fast)

**Key Design Decisions**:
- **Binning**: 0.5 Da resolution (4000 bins for 0-2000 Da)
  - Tradeoff: 0.1 Da (20k bins) too expensive, 1.0 Da (2k bins) too coarse
- **Method**: Linear interpolation (2× scatter_add per peak)
  - Fully differentiable (gradients flow)
  - 50× faster than Gaussian rendering
  - 99% of Gaussian's gradient information

**Implementation**:
```python
class FastLinearBinRenderer(nn.Module):
    """
    Fast differentiable spectrum rendering using linear interpolation.

    Computational Cost:
    - Memory: (batch × num_peaks × 2) = ~1 MB (negligible)
    - Time: 2× scatter_add operations = ~0.1ms per batch
    - Total overhead: ~5% of training time
    """

    def __init__(self, num_bins: int = 4000, max_mz: float = 2000.0):
        super().__init__()
        self.num_bins = num_bins
        self.max_mz = max_mz
        self.bin_size = max_mz / num_bins  # 0.5 Da

    def forward(self, masses: Tensor, intensities: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            masses: (batch, num_peaks) - peak m/z values
            intensities: (batch, num_peaks) - optional weights (default: 1.0)
        Returns:
            spectrum: (batch, num_bins) - rendered spectrum
        """
        batch_size = masses.shape[0]

        # Compute continuous bin positions
        bin_pos = (masses / self.max_mz * self.num_bins).clamp(0, self.num_bins - 1)

        # Lower and upper bin indices
        bin_lower = bin_pos.floor().long()
        bin_upper = (bin_lower + 1).clamp(max=self.num_bins - 1)

        # Interpolation weights (this is where gradient flows!)
        # If mass = 100.4 Da: bin_lower=100 gets 0.6, bin_upper=101 gets 0.4
        # If mass shifts to 100.5: weights become 0.5/0.5 → gradient!
        weight_upper = bin_pos - bin_lower.float()
        weight_lower = 1.0 - weight_upper

        # Apply intensities if provided
        if intensities is not None:
            weight_lower = weight_lower * intensities
            weight_upper = weight_upper * intensities

        # Scatter into bins (only 2 operations!)
        spectrum = torch.zeros(batch_size, self.num_bins, device=masses.device)
        spectrum.scatter_add_(1, bin_lower, weight_lower)
        spectrum.scatter_add_(1, bin_upper, weight_upper)

        return spectrum
```

**Why Linear Interpolation Wins**:
| Method | Gradient Flow | Speed | Bins Affected | Use Case |
|--------|--------------|-------|---------------|----------|
| Hard binning | ❌ None | Fastest (1×) | 1 | Read-only embeddings |
| **Linear** | ✅ **Smooth** | **Fast (2×)** | **2** | **General purpose** |
| Gaussian | ✅ Very smooth | Slow (20,000×) | All | Loss functions only |

#### Component 2: Residual Embedding Module

**Purpose**: Convert residual spectrum → compact embedding for injection

**Architecture**:
```python
class ResidualSpectrumEmbedding(nn.Module):
    """
    Computes residual (observed - predicted) and embeds it for injection
    into the recursive loop.

    Design Philosophy:
    - 1D-CNN extracts shift patterns (critical for PTM detection)
    - Projects to hidden_dim for seamless integration
    - Computes residuals only at supervision steps (8×) not latent steps (48×)
    """

    def __init__(self, hidden_dim: int = 384, num_bins: int = 4000):
        super().__init__()

        # Fast renderer (linear interpolation)
        self.renderer = FastLinearBinRenderer(num_bins=num_bins, max_mz=2000.0)

        # 1D-CNN to extract patterns from residual
        # Why CNN? Detects "shift patterns" critical for PTM identification
        # Kernel size 5 = ~2.5 Da receptive field (good for isotope patterns)
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
        )

        # Project to hidden_dim for injection
        # Output size after 3× stride-2 convs: num_bins // 8 = 500
        self.projector = nn.Linear(256 * (num_bins // 8), hidden_dim)

        # Cache observed spectrum (computed once per forward pass)
        self.register_buffer('obs_spectrum_cached', None, persistent=False)

    def cache_observed_spectrum(
        self,
        masses: Tensor,
        intensities: Tensor,
        mask: Tensor
    ):
        """Call once at start of forward pass to cache observed spectrum."""
        # Mask out padding peaks
        masked_masses = masses * mask.float()
        masked_intens = intensities * mask.float()

        # Render and cache
        self.obs_spectrum_cached = self.renderer(masked_masses, masked_intens)

    def forward(
        self,
        sequence_probs: Tensor,  # (batch, seq_len, vocab)
        aa_masses: Tensor,       # (vocab,) - amino acid masses
        ion_types: List[str],    # e.g., ['b', 'y', 'b++', 'y++']
    ) -> Tensor:
        """
        Compute residual embedding from current prediction.

        Returns:
            residual_embed: (batch, hidden_dim) - compressed residual info
        """
        # 1. Compute predicted theoretical spectrum
        pred_masses = compute_theoretical_peaks(
            sequence_probs, aa_masses, ion_types
        )  # (batch, num_theoretical_peaks)

        pred_spectrum = self.renderer(pred_masses)  # (batch, num_bins)

        # 2. Compute residual (what remains unexplained)
        # Positive values: "You're missing peaks here" (recall errors)
        # Negative values: "You predicted peaks that don't exist" (precision errors)
        residual = self.obs_spectrum_cached - pred_spectrum  # (batch, num_bins)

        # 3. Extract shift patterns with CNN
        residual_features = self.cnn(residual.unsqueeze(1))  # (batch, 256, 500)
        residual_features = residual_features.flatten(1)

        # 4. Project to hidden_dim
        residual_embed = self.projector(residual_features)  # (batch, hidden_dim)

        return residual_embed
```

**Why 1D-CNN Instead of Linear Projection?**
- **Shift Pattern Detection**: CNNs naturally detect "peak at position X shifted by Δ"
- **Translation Invariance**: Same PTM at different sequence positions → same pattern
- **Local Context**: Kernel captures isotope envelopes, doubly-charged ions
- **Compression**: 4000 bins → 256 features efficiently

#### Component 3: Integration into Recursive Loop

**Where to Inject**: Into cross-attention context (not into sequence embedding)

**Why Context Injection?**
```python
# Option A: Inject into sequence embedding (what Gemini suggested initially)
x = y_embed + latent_z + pos_embed + residual_embed  # ❌ Mixing semantics

# Option B: Inject into cross-attention context (better)
augmented_context = encoded_spectrum + residual_embed.unsqueeze(1)  # ✅ Physics info
```

**Reasoning**:
- `encoded_spectrum` = "what you see" (observed peaks)
- `residual_embed` = "what you're missing" (unexplained peaks)
- Both are physics signals, not sequence semantics
- Augmenting context keeps semantics clean

**Modified Decoder Implementation**:
```python
# In src/model/decoder.py, RecursiveCore class:

def __init__(self, ...):
    # ... existing code ...

    # NEW: Residual embedding module
    self.residual_embedder = ResidualSpectrumEmbedding(
        hidden_dim=hidden_dim,
        num_bins=4000,
    )

def forward(
    self,
    encoded_spectrum: Tensor,
    spectrum_mask: Tensor,
    observed_masses: Tensor,      # NEW: for residual computation
    observed_intensities: Tensor, # NEW: for residual computation
    peak_mask: Tensor,            # NEW: for residual computation
    aa_masses: Tensor,            # NEW: for theoretical spectrum
    ion_types: List[str],         # NEW: for theoretical spectrum
    num_supervision_steps: int = 8,
    return_all_steps: bool = True,
) -> tuple[Tensor, Tensor]:

    batch_size = encoded_spectrum.shape[0]
    device = encoded_spectrum.device

    # Cache observed spectrum (compute once, reuse 8 times)
    self.residual_embedder.cache_observed_spectrum(
        observed_masses, observed_intensities, peak_mask
    )

    # Initialize
    y_logits, z = self.decoder.get_initial_state(batch_size, device)
    all_logits = []

    for t in range(num_supervision_steps):
        # Convert logits to probabilities
        y_probs = torch.softmax(y_logits, dim=-1)

        # NEW: Compute residual embedding at each supervision step
        residual_embed = self.residual_embedder(
            y_probs, aa_masses, ion_types
        )  # (batch, hidden_dim)

        # NEW: Augment context with residual information
        # Broadcast residual to all sequence positions
        augmented_context = encoded_spectrum + residual_embed.unsqueeze(1)

        # Latent reasoning with augmented context
        for _ in range(self.num_latent_steps):
            z = self.decoder.latent_step(
                augmented_context,  # ← Using augmented context!
                spectrum_mask,
                y_probs,
                z
            )

        # Answer update
        y_logits = self.decoder.answer_step(y_probs, z)

        if return_all_steps:
            all_logits.append(y_logits)

    if return_all_steps:
        return torch.stack(all_logits, dim=0), z
    else:
        return y_logits, z
```

### Computational Cost Analysis

**Baseline (current architecture)**:
- Spectrum encoding: 1× (once per forward pass)
- Recursive steps: 8 supervision × 6 latent = 48 transformer calls
- Loss computation: 1× Gaussian rendering for final step

**With Residual Embeddings**:
- Spectrum encoding: 1× (unchanged)
- Observed spectrum caching: 1× linear binning (~0.1ms)
- Recursive steps: 48 transformer calls (unchanged)
- Residual computation: 8× per supervision step
  - Theoretical spectrum: 8× `compute_theoretical_peaks` (~0.5ms each = 4ms)
  - Linear binning: 8× renderer (~0.1ms each = 0.8ms)
  - CNN projection: 8× forward pass (~0.2ms each = 1.6ms)
  - Total: ~6.4ms per batch
- Loss computation: 1× (unchanged)

**Total overhead**: ~6.4ms / ~120ms per batch = **~5% training time increase**

**Memory overhead**: ~10 MB per batch (4000 bins × 8 steps × 4 bytes)

**Verdict**: ✅ Negligible cost for potentially major improvement

### Expected Benefits

**1. Reduced Stagnation**
- **Problem**: Model reaches "good enough" prediction, stops improving
- **Solution**: Residual shows exactly what's still wrong
- **Metric**: Track per-step accuracy improvement (should be smoother)

**2. Better Gradient Signal**
- **Problem**: Gradients diluted through 48 recursive steps
- **Solution**: Direct feedback at each of 8 supervision steps
- **Metric**: Gradient norm analysis (should be more stable)

**3. Faster Convergence**
- **Problem**: Model wastes capacity re-learning "what's been explained"
- **Solution**: Explicit residual frees up capacity for actual prediction
- **Metric**: Steps to reach target accuracy (should decrease 20-30%)

**4. PTM Readiness**
- **Problem**: Modified peptides have unexplained mass shifts
- **Solution**: Residual highlights shift patterns
- **Metric**: Foundation for [GAP] token (Phase 2)

### Implementation Checklist

- [ ] Implement `FastLinearBinRenderer` class
- [ ] Test gradient flow (verify `mass.requires_grad = True` → spectrum gradient exists)
- [ ] Implement `ResidualSpectrumEmbedding` module
- [ ] Test on dummy data (verify residual decreases over steps)
- [ ] Modify `RecursiveCore.forward()` to accept new arguments
- [ ] Integrate context augmentation in latent step
- [ ] Update training loop to pass observed spectrum data
- [ ] Add residual visualization to logging (plot residual per step)
- [ ] Ablation study: Train with/without residuals (same compute budget)
- [ ] Analyze failure modes: When do residuals help most?

---

## Feature 2: Mass Gap Token

### Conceptual Overview

**The Problem**: Database search fails when peptides have unknown modifications.

**Current De Novo Sequencing**:
```
Observed: PEPTK+80 Da IDE (phosphorylated lysine)
Model predicts: PEPTIDE (wrong - ignores +80 Da)
```

**With [GAP] Token**:
```
Observed: PEPTK+80 Da IDE
Model predicts: PEPTK[GAP=80]IDE (correct - explicitly models unknown mass)
```

**Dual Prediction Architecture**:
1. **Token Classification**: What is it? (A, K, [GAP], etc.)
2. **Mass Regression**: How much does it weigh? (continuous scalar)

**For Standard AAs**: Mass = lookup_table(token) + delta (small, for PTMs)
**For [GAP] Token**: Mass = 0 (no base mass) + delta (the entire mass)

### Why This is Revolutionary

**Current State of Field**:
| Method | Speed | Handles Mods | Discovery | Limitation |
|--------|-------|--------------|-----------|------------|
| Database Search | Fast | ❌ No (or pre-specified) | ❌ | Requires knowing sequence |
| De Novo | Medium | ❌ Ignores | ❌ | Assumes standard AAs |
| Open Search | Slow | ⚠️ Some (known mods) | ⚠️ Limited | Combinatorial explosion |

**Your Model with [GAP]**:
| Metric | Performance | Why |
|--------|-------------|-----|
| Speed | Fast (single forward pass) | No database, no search |
| Handles Mods | ✅ Yes (any mass) | [GAP] is flexible wildcard |
| Discovery | ✅ Novel PTMs | Not limited to known mods |
| Scalability | ✅ Linear | No combinatorial search |

**Impact**: First truly "open" de novo sequencer that can discover novel modifications.

### Technical Architecture

#### Component 1: Extended Vocabulary

**Current Vocabulary**:
```python
VOCAB = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
         '<PAD>', '<SOS>', '<EOS>']  # 23 tokens
```

**Extended Vocabulary**:
```python
VOCAB = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
         '<PAD>', '<SOS>', '<EOS>', '<GAP>']  # 24 tokens
```

**Mass Lookup Table**:
```python
AMINO_ACID_MASSES = {
    'A': 71.037, 'C': 103.009, 'D': 115.027, ...,
    '<PAD>': 0.0, '<SOS>': 0.0, '<EOS>': 0.0,
    '<GAP>': 0.0,  # NEW: Base mass is 0, relies entirely on delta
}
```

#### Component 2: Mass Delta Prediction Head

**Architecture**:
```python
class DualPredictionDecoder(nn.Module):
    """
    Decoder that predicts both token identity and mass delta.

    Key Design Decisions:
    - Separate heads (not shared) for better specialization
    - Tanh activation on mass → bounded predictions
    - Delta applied to ALL tokens (enables PTM on standard AAs too)
    """

    def __init__(self, hidden_dim: int = 384, vocab_size: int = 24):
        super().__init__()

        # Token classification head (existing)
        self.token_head = nn.Linear(hidden_dim, vocab_size)

        # NEW: Mass delta regression head
        # Predicts mass shift for each position
        # Architecture: hidden → 128 → 1 (adds nonlinearity for complex mappings)
        self.mass_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh(),  # Output in [-1, 1]
        )

        # Mass range: [-500, +500] Da
        # Covers: Most PTMs (-18 Da dehydration to +305 Da sumoylation)
        self.mass_scale = 500.0

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            x: (batch, seq_len, hidden_dim) - decoder hidden states
        Returns:
            logits: (batch, seq_len, vocab_size) - token predictions
            deltas: (batch, seq_len) - mass deltas in Da
        """
        logits = self.token_head(x)  # (batch, seq_len, vocab_size)

        # Mass delta: tanh output [-1, 1] scaled to [-500, 500] Da
        delta_raw = self.mass_head(x)  # (batch, seq_len, 1)
        deltas = delta_raw.squeeze(-1) * self.mass_scale  # (batch, seq_len)

        return logits, deltas
```

**Why Bounded Regression (Tanh)?**
- **Stability**: Unbounded regression can diverge
- **Prior knowledge**: Most PTMs are < 500 Da
- **Gradient flow**: Tanh has smooth gradients in [-1, 1], explodes outside
- **Interpretability**: Model can't predict absurd masses (e.g., 5000 Da)

**Alternative**: Softplus for [0, ∞) if you want unbounded but non-negative.

#### Component 3: Physics Engine with [GAP] Support

**Purpose**: Compute theoretical fragment masses from mixed token + delta predictions

```python
def compute_theoretical_masses_with_gap(
    sequence_ids: Tensor,      # (batch, seq_len) - token indices
    mass_deltas: Tensor,       # (batch, seq_len) - predicted mass shifts
    aa_mass_tensor: Tensor,    # (vocab,) - base mass lookup
    gap_idx: int,              # Index of <GAP> token
    ion_types: List[str] = ['b', 'y'],
) -> Tensor:
    """
    Compute theoretical fragment masses with [GAP] support.

    Logic:
    - Standard AA: final_mass = lookup_mass(AA) + delta
      Example: K (128 Da) + delta (42 Da) = 170 Da (acetylated K)

    - GAP token: final_mass = 0 + delta
      Example: [GAP] (0 Da) + delta (163 Da) = 163 Da (unknown mod)

    This unified formulation handles both cases seamlessly.
    """
    # 1. Get base masses from lookup table
    # For <GAP>, this returns 0.0
    base_masses = F.embedding(sequence_ids, aa_mass_tensor)  # (batch, seq_len)

    # 2. Add predicted deltas
    # For standard AA: small correction for PTMs
    # For <GAP>: entire mass comes from delta
    final_masses = base_masses + mass_deltas  # (batch, seq_len)

    # 3. Mask out padding
    mask = (sequence_ids != PAD_IDX).float()
    masked_masses = final_masses * mask

    # 4. Compute ion series (cumulative sums)
    theoretical_peaks = []

    if 'b' in ion_types:
        # b-ions: Sum from N-terminus (left to right)
        b_ions = torch.cumsum(masked_masses, dim=1)
        theoretical_peaks.append(b_ions)

    if 'y' in ion_types:
        # y-ions: Sum from C-terminus (right to left)
        # Flip, cumsum, flip back
        y_ions = torch.flip(
            torch.cumsum(torch.flip(masked_masses, dims=[1]), dim=1),
            dims=[1]
        )
        theoretical_peaks.append(y_ions)

    # Handle doubly-charged ions if needed
    if 'b++' in ion_types:
        b_ions_2 = b_ions / 2.0  # Divide by charge
        theoretical_peaks.append(b_ions_2)

    if 'y++' in ion_types:
        y_ions_2 = y_ions / 2.0
        theoretical_peaks.append(y_ions_2)

    # 5. Concatenate all ion types
    all_masses = torch.cat(theoretical_peaks, dim=1)  # (batch, num_ions * seq_len)

    return all_masses
```

**Key Insight**: The same function handles both standard AAs and [GAP] tokens - no special casing needed!

#### Component 4: Integrated Model Architecture

**Full Forward Pass**:
```python
class TRMWithMassGap(RecursivePeptideModel):
    """
    Extended TRM with [GAP] token and mass delta prediction.

    Inherits from base RecursivePeptideModel but adds:
    - Dual prediction heads (token + mass)
    - Residual spectrum embedding with [GAP] support
    - Physics feedback loop
    """

    def __init__(self, config: TRMConfig):
        super().__init__(config)

        # Replace standard decoder with dual-prediction decoder
        self.recursive_core.decoder.output_head = DualPredictionDecoder(
            hidden_dim=config.hidden_dim,
            vocab_size=config.vocab_size,  # Now includes <GAP>
        )

        # Residual spectrum embedder (from Feature 1)
        self.residual_embedder = ResidualSpectrumEmbedding(
            hidden_dim=config.hidden_dim,
            num_bins=4000,
        )

        # Mass lookup tensor (includes <GAP> = 0.0)
        aa_masses = torch.tensor([
            AMINO_ACID_MASSES.get(aa, 0.0) for aa in VOCAB
        ])
        self.register_buffer('aa_masses', aa_masses)

        # Gap token index
        self.gap_idx = VOCAB.index('<GAP>')

    def forward(
        self,
        spectrum_masses: Tensor,
        spectrum_intensities: Tensor,
        spectrum_mask: Tensor,
        precursor_mass: Tensor,
        precursor_charge: Tensor,
        num_supervision_steps: int = 8,
    ) -> tuple[Tensor, Tensor]:
        """
        Forward pass with [GAP] token support.

        Returns:
            all_logits: (T, batch, seq_len, vocab) - token predictions
            all_deltas: (T, batch, seq_len) - mass delta predictions
        """
        batch_size = spectrum_masses.shape[0]
        device = spectrum_masses.device

        # 1. Encode spectrum (unchanged)
        encoded_spectrum, full_mask = self.encoder(
            spectrum_masses,
            spectrum_intensities,
            spectrum_mask,
            precursor_mass,
            precursor_charge,
        )

        # 2. Cache observed spectrum for residual computation
        self.residual_embedder.cache_observed_spectrum(
            spectrum_masses, spectrum_intensities, spectrum_mask
        )

        # 3. Initialize predictions
        y_logits, z = self.recursive_core.decoder.get_initial_state(
            batch_size, device
        )
        # Initialize mass deltas to zero
        current_deltas = torch.zeros(
            batch_size, self.config.max_seq_len, device=device
        )

        all_logits = []
        all_deltas = []

        # 4. Recursive refinement loop
        for t in range(num_supervision_steps):
            # 4a. Convert logits to probabilities
            y_probs = torch.softmax(y_logits, dim=-1)

            # 4b. Compute current prediction's theoretical spectrum
            # Use soft probabilities for differentiability
            current_tokens = y_logits.argmax(dim=-1)  # Hard for physics
            theo_masses = compute_theoretical_masses_with_gap(
                current_tokens,
                current_deltas.detach(),  # Detach for stability
                self.aa_masses,
                self.gap_idx,
                ion_types=['b', 'y', 'b++', 'y++'],
            )

            # 4c. Compute residual embedding (what's unexplained)
            residual_embed = self.residual_embedder(
                y_probs,
                self.aa_masses,
                ion_types=['b', 'y', 'b++', 'y++'],
            )

            # 4d. Augment context with residual
            augmented_context = encoded_spectrum + residual_embed.unsqueeze(1)

            # 4e. Latent reasoning steps
            for _ in range(self.config.num_latent_steps):
                z = self.recursive_core.decoder.latent_step(
                    augmented_context,
                    full_mask,
                    y_probs,
                    z
                )

            # 4f. Dual prediction
            y_logits, mass_deltas = self.recursive_core.decoder.output_head.forward(z)

            # 4g. Store predictions
            all_logits.append(y_logits)
            all_deltas.append(mass_deltas)

            # 4h. Update current state for next iteration
            current_deltas = mass_deltas.detach()  # Detach for stability

        return torch.stack(all_logits, dim=0), torch.stack(all_deltas, dim=0)
```

**Critical Design Decisions**:

1. **Detached Deltas in Physics** (line 120):
   - `current_deltas.detach()` prevents gradient explosion
   - Still allows gradient flow through current step's mass head
   - Tradeoff: Limits cross-step optimization but ensures stability

2. **Residual Computed Every Step**:
   - Shows model exactly what remains unexplained
   - Guides it toward using [GAP] when appropriate
   - Without this, model has no signal for when to insert [GAP]

3. **Hard Tokens for Physics, Soft for Loss**:
   - Physics engine uses `argmax` (hard tokens) for interpretability
   - Loss uses soft probabilities for differentiability
   - Standard practice in discrete-continuous hybrid models

### Loss Functions

**Triple Loss Design**:
```python
class GapAwareLoss(nn.Module):
    """
    Combined loss for [GAP] token training.

    Three components:
    1. Token Classification (CE) - What is it?
    2. Mass Regression (L1) - How much does it weigh?
    3. Gap Sparsity (Penalty) - Don't overuse [GAP]
    """

    def __init__(
        self,
        ce_weight: float = 1.0,
        mass_weight: float = 0.5,
        sparsity_weight: float = 0.1,
        gap_idx: int = 23,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.mass_weight = mass_weight
        self.sparsity_weight = sparsity_weight
        self.gap_idx = gap_idx

    def forward(
        self,
        all_logits: Tensor,        # (T, batch, seq_len, vocab)
        all_deltas: Tensor,        # (T, batch, seq_len)
        target_tokens: Tensor,     # (batch, seq_len) - ground truth tokens
        target_masses: Tensor,     # (batch, seq_len) - ground truth masses
        sequence_mask: Tensor,     # (batch, seq_len)
    ) -> tuple[Tensor, dict]:

        T = all_logits.shape[0]

        # 1. Token Classification Loss (standard CE)
        ce_loss = 0.0
        for t in range(T):
            logits_t = all_logits[t].view(-1, all_logits.shape[-1])
            targets_t = target_tokens.view(-1)
            mask_t = sequence_mask.view(-1).float()

            ce_t = F.cross_entropy(
                logits_t, targets_t,
                ignore_index=PAD_IDX,
                reduction='none'
            )
            ce_loss += (ce_t * mask_t).sum() / mask_t.sum().clamp(min=1)

        ce_loss = ce_loss / T

        # 2. Mass Regression Loss (only for final step)
        # L1 loss on predicted vs. true mass delta
        final_deltas = all_deltas[-1]  # (batch, seq_len)

        mass_error = torch.abs(final_deltas - target_masses)  # (batch, seq_len)
        masked_error = mass_error * sequence_mask.float()
        mass_loss = masked_error.sum() / sequence_mask.float().sum().clamp(min=1)

        # 3. Gap Sparsity Penalty
        # Encourage model to use [GAP] only when necessary
        final_probs = torch.softmax(all_logits[-1], dim=-1)
        gap_probs = final_probs[:, :, self.gap_idx]  # (batch, seq_len)

        # Penalize high gap probability (L1 regularization)
        gap_usage = (gap_probs * sequence_mask.float()).sum() / \
                    sequence_mask.float().sum().clamp(min=1)
        sparsity_loss = gap_usage

        # Combine losses
        total_loss = (
            self.ce_weight * ce_loss +
            self.mass_weight * mass_loss +
            self.sparsity_weight * sparsity_loss
        )

        # Metrics for logging
        metrics = {
            'ce_loss': ce_loss.item(),
            'mass_loss': mass_loss.item(),
            'mass_error_da': masked_error.mean().item(),
            'gap_usage': gap_usage.item(),
            'total_loss': total_loss.item(),
        }

        return total_loss, metrics
```

**Why Sparsity Penalty?**
- Without it, model might spam [GAP] to avoid committing
- Penalty encourages: "Use standard AAs when possible, [GAP] only when necessary"
- Weight 0.1 is gentle nudge, not hard constraint
- Tunable hyperparameter based on validation performance

### Training Data Generation

**Challenge**: How to create ground truth for [GAP] tokens?

**Strategy 1: Known Modifications as [GAP]** (Recommended for Phase 1)

```python
def create_gap_training_data(peptide, spectrum, known_ptms):
    """
    Convert known PTMs to [GAP] tokens for training.

    Example:
        Input: "PEPTK(+42.01)IDE" (acetylated lysine)
        Output tokens: ['P', 'E', 'P', 'T', 'K', '<GAP>', 'I', 'D', 'E']
        Output masses: [0, 0, 0, 0, 0, 42.01, 0, 0, 0]

    Or alternatively:
        Output tokens: ['P', 'E', 'P', 'T', 'K', 'I', 'D', 'E']
        Output masses: [0, 0, 0, 0, 42.01, 0, 0, 0]
        (K gets delta=42.01, representing K+acetyl)
    """
    sequence = []
    masses = []

    for i, (aa, ptm_mass) in enumerate(zip(peptide, known_ptms)):
        if ptm_mass > 0:
            # Option A: Insert explicit <GAP> token
            sequence.append(aa)
            masses.append(0.0)
            sequence.append('<GAP>')
            masses.append(ptm_mass)

            # Option B: Add delta to existing AA (cleaner)
            # sequence.append(aa)
            # masses.append(ptm_mass)
        else:
            sequence.append(aa)
            masses.append(0.0)

    return sequence, masses
```

**Common PTMs to Use**:
- Phosphorylation: +79.966 Da (S, T, Y)
- Acetylation: +42.011 Da (K, N-term)
- Oxidation: +15.995 Da (M, W)
- Deamidation: +0.984 Da (N, Q)
- Carbamidomethylation: +57.021 Da (C) - often fixed, not variable

**Data Sources**:
1. **UniMod Database**: 2000+ known modifications
2. **Existing Proteomics Datasets**: PRIDE, PeptideAtlas with PTM annotations
3. **Synthetic with MS2PIP**: Generate modified peptide spectra

**Strategy 2: Curriculum Learning**

```
Stage 1 (0-30k steps):   Train on standard peptides only (no [GAP])
                         → Learn basic sequencing

Stage 2 (30k-60k steps): Introduce [GAP] on known PTMs (10% of data)
                         → Learn when to insert [GAP]

Stage 3 (60k-100k):      Mix of standard + [GAP] (30% modified)
                         → Robust to modifications

Stage 4 (100k+):         Add random [GAP] for robustness
                         → Generalize to novel modifications
```

**Why Curriculum?**
- Prevents [GAP] spam in early training
- Gives model strong foundation before introducing complexity
- Gradual difficulty increase improves convergence

### Evaluation Metrics

**Challenge**: How do you evaluate [GAP] predictions?

**Metric 1: Known PTM Recovery** (Validation Set)
- Hold out peptides with known modifications
- Measure: Can model predict [GAP] at correct position with correct mass?

```python
def evaluate_gap_recovery(predictions, ground_truth):
    """
    Metrics:
    - Position Accuracy: Is [GAP] at the right position?
    - Mass Accuracy: Is predicted mass within ±10 Da of true mass?
    - Combined Accuracy: Both position and mass correct?
    """
    position_correct = (pred_positions == true_positions).float().mean()

    mass_errors = abs(pred_masses - true_masses)
    mass_within_10da = (mass_errors < 10.0).float().mean()

    combined = (position_correct * mass_within_10da).mean()

    return {
        'position_acc': position_correct,
        'mass_acc_10da': mass_within_10da,
        'combined_acc': combined,
        'mass_error_mean': mass_errors.mean(),
        'mass_error_median': mass_errors.median(),
    }
```

**Metric 2: Precursor Mass Error** (Sanity Check)
- Even with [GAP], total predicted mass must match precursor
- Should be no worse than baseline model

**Metric 3: Comparison vs. Baseline** (Standard Peptides)
- Does adding [GAP] hurt performance on unmodified peptides?
- Should maintain ≥95% of baseline accuracy

**Metric 4: Novel PTM Discovery** (Real Data, Qualitative)
- Run on open search datasets
- Manually validate: Do predicted [GAP] masses correspond to real modifications?
- Literature search: Are discovered masses consistent with known biology?

### Expected Challenges & Mitigations

**Challenge 1: Model Spams [GAP] Tokens**
- **Symptom**: >50% of predicted tokens are [GAP]
- **Cause**: [GAP] is easier than committing to specific AA
- **Fix**: Increase sparsity penalty weight (0.1 → 0.3)
- **Fix**: Curriculum learning (delay [GAP] introduction)

**Challenge 2: Mass Predictions Are Random**
- **Symptom**: Mass errors >100 Da, no correlation with truth
- **Cause**: Regression task is too hard without guidance
- **Fix**: Add auxiliary loss: "Predicted spectrum must match observed"
- **Fix**: Warm-start: Train token classification first, add mass head later

**Challenge 3: Ambiguity Between F and [GAP]=147**
- **Symptom**: Model alternates between F and [GAP] for 147 Da masses
- **Cause**: Both explanations are physics-valid
- **Fix**: Sparsity penalty biases toward F (prefer standard AAs)
- **Fix**: Add prior: [GAP] mass distribution should be bimodal (small PTMs or large mods, not standard AA masses)

**Challenge 4: Poor Generalization to Novel PTMs**
- **Symptom**: Works on known PTMs (validation) but fails on novel ones (test)
- **Cause**: Overfitting to training PTM distribution
- **Fix**: Data augmentation: Random mass shifts during training
- **Fix**: Regularization: Dropout on mass head (0.2)

**Challenge 5: Training Data Scarcity**
- **Symptom**: Not enough PTM-labeled examples
- **Cause**: Proteomics databases have sparse modification annotations
- **Fix**: Synthetic data generation with MS2PIP
- **Fix**: Semi-supervised learning: Use predictions on unlabeled data as pseudo-labels

---

## Implementation Phases

### Prerequisites (Before Starting)
- ✅ Baseline model achieves >70% accuracy on standard peptides
- ✅ Model trains stably for 100k steps
- ✅ Real data evaluation pipeline established
- ✅ Compute resources: 1× GPU (RTX 4090 or better)

### Phase 1: Residual Spectrum Embeddings (Weeks 1-6)

**Week 1-2: Implementation**
- [ ] Implement `FastLinearBinRenderer`
  - Unit tests: Gradient flow, bin assignment correctness
  - Benchmark: Speed vs. Gaussian rendering
- [ ] Implement `ResidualSpectrumEmbedding`
  - Verify CNN output shapes
  - Test on dummy data: Residual should decrease over steps
- [ ] Modify `RecursiveCore` for context augmentation
  - Update forward signature
  - Add residual computation at supervision steps
- [ ] Update training loop
  - Pass observed spectrum to model
  - Add residual visualization to TensorBoard

**Week 3-4: Validation**
- [ ] Ablation study: Train with/without residuals (same budget)
  - Metrics: Final accuracy, convergence speed, gradient norms
- [ ] Analyze residual patterns
  - Plot: Residual magnitude vs. training step (should decrease)
  - Visualize: Which peaks are explained first? (large peaks prioritized?)
- [ ] Failure mode analysis
  - When do residuals help most? (longer peptides? noisy spectra?)
  - When do they not help? (very short peptides? clean data?)

**Week 5-6: Tuning & Integration**
- [ ] Hyperparameter sweep
  - CNN architecture (kernel size, channels)
  - Bin resolution (0.5 Da vs. 1.0 Da)
  - Injection method (context vs. input)
- [ ] Integrate into main training pipeline
  - Update default configs
  - Document usage in README
- [ ] Write technical report
  - Document improvements
  - Include visualizations

**Success Criteria**:
- ✅ Training time overhead <10%
- ✅ Convergence speed improves ≥15%
- ✅ Final accuracy improves ≥3% OR
- ✅ Gradient stability improves (lower variance)

### Phase 2: Mass Gap Token - Preparation (Weeks 7-10)

**Week 7-8: Data Preparation**
- [ ] Create PTM-labeled dataset
  - Source: PRIDE, PeptideAtlas, or MS2PIP synthetic
  - Target: 10k peptides with known modifications
  - Format: (sequence, modification_sites, modification_masses)
- [ ] Implement data generation pipeline
  - Convert known PTMs → [GAP] tokens
  - Validate: Mass balance (total predicted mass = precursor)
- [ ] Split data: Train 70% / Val 15% / Test 15%
  - Stratify by modification type
  - Ensure novel PTMs in test set

**Week 9-10: Curriculum Design**
- [ ] Define curriculum stages
  - Stage 1: Standard peptides (30k steps)
  - Stage 2: +10% PTMs (20k steps)
  - Stage 3: +30% PTMs (30k steps)
  - Stage 4: Random augmentation (20k steps)
- [ ] Implement curriculum scheduler
  - Dynamically adjust PTM frequency
  - Log stage transitions
- [ ] Dry-run: Test curriculum on small dataset
  - Verify smooth transitions
  - Check data loading pipeline

**Success Criteria**:
- ✅ 10k+ PTM-labeled examples
- ✅ Data pipeline tested and validated
- ✅ Curriculum schedule defined

### Phase 3: Mass Gap Token - Implementation (Weeks 11-16)

**Week 11-12: Core Architecture**
- [ ] Extend vocabulary: Add `<GAP>` token
- [ ] Implement `DualPredictionDecoder`
  - Token classification head
  - Mass regression head
  - Test gradient flow through both heads
- [ ] Implement `compute_theoretical_masses_with_gap`
  - Unit tests: Standard AAs, [GAP] tokens, mixed
  - Verify ion type calculations (b, y, b++, y++)
- [ ] Update model forward pass
  - Integrate dual prediction
  - Return both logits and deltas

**Week 13-14: Loss Functions**
- [ ] Implement `GapAwareLoss`
  - Token CE loss
  - Mass regression loss
  - Sparsity penalty
- [ ] Hyperparameter tuning
  - Loss weights (ce=1.0, mass=?, sparsity=?)
  - Mass error tolerance (±10 Da? ±20 Da?)
- [ ] Test on synthetic data
  - Overfit to 10 examples (sanity check)
  - Verify mass predictions converge

**Week 15-16: Integration & Training**
- [ ] Full training run (100k steps with curriculum)
- [ ] Monitor metrics
  - Gap usage over time (should increase gradually)
  - Mass error over time (should decrease)
  - Position accuracy on validation set
- [ ] Debug issues
  - Gap spam → Increase sparsity penalty
  - Poor mass predictions → Check loss weights
  - No [GAP] used → Decrease sparsity penalty

**Success Criteria**:
- ✅ Model completes training without crashes
- ✅ [GAP] usage stabilizes (not 0%, not 100%)
- ✅ Mass predictions converge (error <50 Da on validation)

### Phase 4: Evaluation & Refinement (Weeks 17-20)

**Week 17-18: Known PTM Evaluation**
- [ ] Evaluate on held-out PTM validation set
  - Position accuracy: % correct [GAP] positions
  - Mass accuracy: % within ±10 Da of true mass
  - Combined accuracy: Both correct
- [ ] Error analysis
  - Which modifications are hardest? (small vs. large mass?)
  - Which amino acids cause confusion? (F vs. [GAP]=147?)
- [ ] Comparison vs. baseline
  - Does [GAP] hurt performance on standard peptides?
  - Is the tradeoff worth it?

**Week 19-20: Novel PTM Discovery**
- [ ] Run on open search datasets
  - Source: Literature datasets with unexplained mass shifts
- [ ] Manual validation
  - Do predicted [GAP] masses make sense?
  - Literature search: Known biology?
- [ ] Sensitivity analysis
  - Vary mass error tolerance: ±5, ±10, ±20 Da
  - Vary sparsity penalty: 0.05, 0.1, 0.2
  - Find optimal hyperparameters

**Success Criteria**:
- ✅ Known PTM recovery: >60% position accuracy, >70% mass within ±10 Da
- ✅ Standard peptide performance: ≥95% of baseline accuracy
- ✅ Novel PTM discovery: At least 5 validated novel modifications

### Phase 5: Publication & Deployment (Weeks 21-24)

**Week 21-22: Benchmarking**
- [ ] Compare against existing tools
  - PEAKS, Novor, DeepNovo, Casanovo
  - Metrics: Accuracy, speed, PTM discovery rate
- [ ] Ablation studies
  - Residuals only vs. [GAP] only vs. both
  - Quantify contribution of each component
- [ ] Performance optimization
  - Profile code: Identify bottlenecks
  - Optimize inference speed (important for deployment)

**Week 23-24: Documentation & Release**
- [ ] Write manuscript
  - Methods: Architecture, training, evaluation
  - Results: Benchmarks, novel discoveries
  - Discussion: Limitations, future work
- [ ] Prepare code release
  - Clean up code, add docstrings
  - Create examples/tutorials
  - Package for PyPI or Conda
- [ ] Create web demo (optional)
  - Upload spectrum → Get sequence + modifications
  - Visualization of residuals and [GAP] predictions

**Success Criteria**:
- ✅ Manuscript submitted to top-tier venue (Nature Methods, PNAS, Bioinformatics)
- ✅ Code released on GitHub with documentation
- ✅ At least 1 novel biological discovery validated

---

## Technical Specifications

### System Requirements

**Hardware**:
- GPU: RTX 4090 (24GB VRAM) or A100 (40GB)
- RAM: 64GB+ (for large dataset loading)
- Storage: 500GB+ SSD (for datasets and checkpoints)

**Software**:
- Python 3.10+
- PyTorch 2.0+ (for `torch.compile` support)
- CUDA 11.8+

**Dependencies**:
```txt
torch>=2.0.0
numpy>=1.24.0
ms2pip>=4.0.0
omegaconf>=2.3.0
wandb>=0.15.0  # For experiment tracking
tensorboard>=2.13.0
pytest>=7.4.0  # For testing
```

### Model Configuration

**Baseline + Residual Embeddings**:
```yaml
model:
  hidden_dim: 384
  num_encoder_layers: 3
  num_decoder_layers: 3
  num_heads: 6
  max_peaks: 100
  max_seq_len: 35
  vocab_size: 23  # Standard AAs + special tokens

residual_embedding:
  num_bins: 4000
  max_mz: 2000.0
  bin_size: 0.5  # Automatically computed
  cnn_channels: [64, 128, 256]
  cnn_kernel_size: 5
  cnn_stride: 2

training:
  batch_size: 96
  learning_rate: 1.5e-4
  max_steps: 100000
  use_amp: true
  use_compile: false  # Enable after validation
```

**With Mass Gap Token**:
```yaml
model:
  vocab_size: 24  # +1 for <GAP>
  gap_idx: 23

mass_prediction:
  mass_range: 500.0  # [-500, +500] Da
  hidden_dim: 128  # For mass head

loss:
  ce_weight: 1.0
  mass_weight: 0.5  # Tune based on validation
  sparsity_weight: 0.1  # Tune based on gap usage

curriculum:
  stage_1_steps: 30000  # Standard peptides
  stage_2_steps: 20000  # +10% PTMs
  stage_3_steps: 30000  # +30% PTMs
  stage_4_steps: 20000  # +random augmentation
```

### File Structure

```
pepTRM/
├── src/
│   ├── model/
│   │   ├── trm.py                    # Base model
│   │   ├── encoder.py                # Spectrum encoder
│   │   ├── decoder.py                # Recursive decoder
│   │   ├── residual_embedding.py     # NEW: Residual spectrum module
│   │   └── dual_prediction.py        # NEW: Token + mass heads
│   ├── training/
│   │   ├── losses.py                 # Base losses
│   │   ├── gap_aware_loss.py         # NEW: Loss for [GAP] training
│   │   ├── curriculum_gap.py         # NEW: PTM curriculum
│   │   └── trainer_optimized.py      # Updated trainer
│   ├── data/
│   │   ├── dataset.py                # Base dataset
│   │   ├── ms2pip_dataset.py         # MS2PIP synthetic data
│   │   └── ptm_dataset.py            # NEW: PTM-labeled data
│   └── evaluation/
│       ├── metrics.py                # Base metrics
│       └── gap_metrics.py            # NEW: PTM evaluation metrics
├── configs/
│   ├── baseline.yaml
│   ├── residual_embedding.yaml       # NEW
│   └── mass_gap.yaml                 # NEW
├── scripts/
│   ├── train_residual.py             # NEW: Train with residuals
│   ├── train_gap.py                  # NEW: Train with [GAP]
│   ├── evaluate_gap.py               # NEW: Evaluate PTM recovery
│   └── create_ptm_dataset.py         # NEW: Data preparation
└── tests/
    ├── test_residual_embedding.py    # NEW
    ├── test_dual_prediction.py       # NEW
    └── test_gap_physics.py           # NEW
```

---

## Success Metrics

### Feature 1: Residual Spectrum Embeddings

**Primary Metrics**:
| Metric | Baseline | Target | Measurement |
|--------|----------|--------|-------------|
| Token Accuracy (Easy) | 75% | 78%+ | Validation set (clean, short) |
| Token Accuracy (Hard) | 60% | 63%+ | Validation set (noisy, long) |
| Convergence Speed | 50k steps | 40k steps | Steps to reach 70% accuracy |
| Training Time Overhead | 0% | <10% | Wall-clock time increase |

**Secondary Metrics**:
| Metric | Target | Measurement |
|--------|--------|-------------|
| Gradient Norm Stability | Lower variance | Std dev of gradient norms |
| Residual Magnitude | Decreases over steps | Mean residual per supervision step |
| Peak Coverage | Improves | % of observed peaks explained |

**Qualitative Analysis**:
- Visualize residuals: Do they highlight unexplained peaks?
- Attention patterns: Does model attend more to high-residual regions?
- Failure modes: When do residuals NOT help?

### Feature 2: Mass Gap Token

**Primary Metrics (Known PTMs)**:
| Metric | Target | Measurement |
|--------|--------|-------------|
| Position Accuracy | >60% | [GAP] at correct sequence position |
| Mass Accuracy (±10 Da) | >70% | Predicted mass within ±10 Da of truth |
| Combined Accuracy | >50% | Both position and mass correct |
| Standard Peptide Acc. | ≥95% baseline | Performance on unmodified peptides |

**PTM-Specific Metrics**:
| PTM Type | Target Accuracy | Common Examples |
|----------|----------------|-----------------|
| Phosphorylation (+80 Da) | >70% | Ser, Thr, Tyr |
| Acetylation (+42 Da) | >70% | Lys, N-term |
| Oxidation (+16 Da) | >60% | Met, Trp |
| Large mods (>100 Da) | >50% | Glycosylation, ubiquitination |

**Discovery Metrics (Novel PTMs)**:
| Metric | Target | Measurement |
|--------|--------|-------------|
| Novel PTM candidates | >20 | Unique [GAP] masses found in test set |
| Validation rate | >25% | % confirmed by literature/manual validation |
| False positive rate | <50% | % of predictions that are artifacts |

**Comparison vs. Existing Tools**:
| Tool | Accuracy | Speed | PTM Discovery | Notes |
|------|----------|-------|---------------|-------|
| PEAKS | ~65% | Slow | Known only | Commercial |
| Novor | ~60% | Medium | None | Free |
| DeepNovo | ~55% | Fast | None | Deep learning |
| **Ours (Target)** | **>70%** | **Fast** | **Yes** | Open source |

---

## Risk Mitigation

### Technical Risks

**Risk 1: Residual Embeddings Don't Help**
- **Probability**: 30%
- **Impact**: Medium (wasted 4-6 weeks)
- **Mitigation**:
  - Early validation: Test on small dataset first
  - Ablation studies: Isolate contribution of residuals
  - Fallback: Use residuals for visualization only, not training
- **Decision Point**: If no improvement after 20k steps, revert to baseline

**Risk 2: [GAP] Token Training Unstable**
- **Probability**: 40%
- **Impact**: High (could derail entire Phase 3)
- **Mitigation**:
  - Curriculum learning (delay [GAP] introduction)
  - Warm-start from baseline model
  - Gradient clipping (max_norm=1.0)
  - Lower learning rate for mass head
- **Decision Point**: If training diverges 3× in a row, simplify architecture

**Risk 3: Model Spams [GAP] Tokens**
- **Probability**: 50%
- **Impact**: Medium (predictions are useless)
- **Mitigation**:
  - Sparsity penalty (start at 0.1, increase to 0.3 if needed)
  - Hard constraint: Max 20% of tokens can be [GAP]
  - Curriculum: Only introduce [GAP] after model is confident with standard AAs
- **Decision Point**: If gap usage >50% at convergence, increase sparsity penalty

**Risk 4: Poor Generalization to Novel PTMs**
- **Probability**: 60%
- **Impact**: High (limits scientific impact)
- **Mitigation**:
  - Data augmentation: Random mass shifts during training
  - Regularization: Dropout, weight decay
  - Diverse training data: Cover wide range of PTM masses
  - Ensemble: Train multiple models, average predictions
- **Decision Point**: If test accuracy <40%, focus on known PTMs only

### Resource Risks

**Risk 5: Insufficient PTM-Labeled Data**
- **Probability**: 40%
- **Impact**: High (can't train [GAP] model)
- **Mitigation**:
  - Synthetic data generation with MS2PIP
  - Data augmentation: Artificial PTM insertion
  - Transfer learning: Pre-train on standard peptides
  - Collaboration: Partner with proteomics labs for data
- **Decision Point**: If <5k PTM examples, delay Phase 2

**Risk 6: Computational Resources Insufficient**
- **Probability**: 20%
- **Impact**: Medium (slower iteration)
- **Mitigation**:
  - Cloud compute: AWS/GCP with A100 GPUs
  - Mixed precision (already enabled)
  - Gradient checkpointing for memory
  - Smaller batch sizes if needed
- **Decision Point**: If training takes >1 week, request more compute

### Scientific Risks

**Risk 7: Reviewers Reject Novelty**
- **Probability**: 30%
- **Impact**: Medium (publication delayed)
- **Mitigation**:
  - Emphasize novel architecture (physics-guided recursion)
  - Strong baselines: Compare against PEAKS, Novor
  - Real-world validation: At least 1 novel PTM discovery
  - Multiple submission targets: Nature Methods → Bioinformatics → PLOS Comp Bio
- **Decision Point**: After 2 rejections, pivot to ML conference (ICML, NeurIPS)

**Risk 8: Biological Validation Fails**
- **Probability**: 40%
- **Impact**: High (limits impact)
- **Mitigation**:
  - Collaborate with experimental biologists
  - Focus on well-studied PTMs first (phosphorylation, acetylation)
  - Computational validation: Cross-reference with databases (UniMod, PhosphoSitePlus)
  - Transparent reporting: Report both successes and failures
- **Decision Point**: If <3 discoveries validate, focus on computational metrics

---

## References & Resources

### Original Concepts
- `spectrum_embedding_recursion.md` - Core residual embedding idea
- `gemini_residual_embedding.md` - Engineering optimizations
- `mass_gap_implementation_example_gemini.txt` - [GAP] token architecture

### Key Techniques
- **Linear Interpolation Binning**: Fast differentiable rendering
- **1D-CNN for Shift Patterns**: PTM detection via convolution
- **Curriculum Learning**: Gradual complexity increase
- **Sparsity Regularization**: Prevent [GAP] spam

### Related Work

**Peptide Sequencing**:
- PEAKS: Combinatorial search + ML scoring
- Novor: Ensemble of DNNs
- DeepNovo: Seq2seq with attention
- Casanovo: Transformer-based

**PTM Discovery**:
- Open search: MSFragger, Metamorpheus
- Blind search: PIPI, MODa
- Database-dependent, slow, limited to known mods

**Your Innovation**:
- First de novo + PTM discovery in single model
- Physics-guided via residual embeddings
- Flexible wildcard ([GAP]) for any modification

### Datasets

**For Training**:
- MS2PIP synthetic data (unlimited, customizable)
- NIST Peptide Libraries (high-quality reference)
- PRIDE Archive (public proteomics data)
- PeptideAtlas (curated, PTM-annotated)

**For Evaluation**:
- MassIVE (community datasets)
- ProteomeXchange (multi-lab datasets)
- Published PTM studies (ground truth validations)

### Tools & Libraries

**Proteomics**:
- `ms2pip` - Fragment intensity prediction
- `pyteomics` - Mass spec data parsing
- `matchms` - Spectrum similarity
- `unimod` - PTM database

**Machine Learning**:
- `torch.compile` - Model optimization
- `wandb` - Experiment tracking
- `tensorboard` - Visualization
- `pytest` - Testing

**Visualization**:
- `matplotlib` - Plotting
- `seaborn` - Statistical plots
- `plotly` - Interactive visualizations

---

## Appendix: Quick Start Commands

### Phase 1: Residual Embeddings

```bash
# Install dependencies
pip install -r requirements_advanced.txt

# Test residual embedding module
pytest tests/test_residual_embedding.py

# Train with residuals (small test run)
python scripts/train_residual.py --max_steps 1000 --use_wandb

# Full training run
python scripts/train_residual.py --max_steps 100000 --use_wandb

# Evaluate against baseline
python scripts/compare_models.py \
  --baseline checkpoints/baseline_100k.pt \
  --residual checkpoints/residual_100k.pt
```

### Phase 2: Mass Gap Token

```bash
# Create PTM dataset
python scripts/create_ptm_dataset.py \
  --source data/pride_ptms.csv \
  --output data/gap_training_data.pkl

# Train with [GAP] token
python scripts/train_gap.py \
  --config configs/mass_gap.yaml \
  --use_wandb

# Evaluate PTM recovery
python scripts/evaluate_gap.py \
  --checkpoint checkpoints/gap_100k.pt \
  --test_data data/gap_test_data.pkl

# Novel PTM discovery
python scripts/discover_ptms.py \
  --checkpoint checkpoints/gap_100k.pt \
  --spectra data/open_search_spectra.mgf \
  --output results/novel_ptms.csv
```

---

## Version History

- **v1.0** (2025-12-11): Initial roadmap created
  - Defined residual embedding architecture
  - Defined mass gap token architecture
  - Established implementation phases
  - Identified risks and mitigations

---

## Contact & Collaboration

**For Questions**:
- Review this document first
- Check existing issues on GitHub
- Create new issue with `[Roadmap]` tag

**For Contributions**:
- Follow implementation phases in order
- Write tests for new components
- Document architectural decisions
- Submit PRs with clear descriptions

**For Collaborations**:
- Proteomics data providers: Contact for PTM datasets
- Experimental validation: Contact for wet-lab validation
- Tool developers: Contact for benchmark comparisons

---

**End of Roadmap**

This document will be updated as the project progresses. Next review: After Phase 1 completion.
