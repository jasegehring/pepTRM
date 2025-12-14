# PepTRM Master Development Roadmap
**Date**: December 12, 2024
**Status**: Aggressive noise training in progress (testing recursion hypothesis)
**Version**: 2.0 - Consolidated from all previous roadmaps and proposals

---

## 📊 Current Status & Recent Progress

### ✅ Achievements
- **Model Architecture**: TRM working well (91.3% token accuracy on clean data)
- **Infrastructure**: torch.compile working with batch_size=80 (2.1x speedup)
- **Training Pipeline**: Optimized trainer with gradient accumulation, EMA, mixed precision
- **Data Pipeline**: MS2PIP synthetic data working, ProteomeTools and Nine-Species datasets ready

### 🔄 In Progress
- **Aggressive Noise Training**: Testing if noise alone unlocks multi-step refinement
  - Exponential iteration weighting (force step 7 optimization)
  - Clean/noisy mixing curriculum (80% clean → 0% clean over 45K steps)
  - Precursor loss ramping (0.05 → 0.3)
  - Spectrum loss disabled (sigma too narrow)

### ❌ Known Issues
1. **Recursion Plateau**: Model refines step 0→1 but plateaus steps 2-7 (all ~0.82 loss)
2. **Spectrum Loss Challenge**: Stuck at 94-96% with sigma=0.2 Da (only 4-6% coverage)
3. **Sim-to-Real Gap**: Unknown until we test on real data

---

## 🎯 Master Priority List

### **Phase 1: Unlock Multi-Step Recursion** (Weeks 1-2)
**Goal**: Prove model can use all 8 refinement steps effectively

#### Priority 1.1: Monitor Current Aggressive Noise Run ⏳ **IN PROGRESS**
- **What**: Training with exponential weighting + aggressive noise curriculum
- **Expected**: See refinement across ALL 8 steps (not just step 0→1)
- **Success Criteria**:
  - `ce_step_7` < `ce_step_0` (final step better than initial)
  - Edit rate >5% at steps 2-7 (model still making changes)
  - No catastrophic drops between curriculum stages
- **If Fails**: Move to Priority 1.2 (architectural improvements)

#### Priority 1.2: Implement Step Embeddings ✅ **COMPLETED 2024-12-13**
**Why**: Tell model which refinement step it's on (rough draft vs polishing)

**Implementation**: Added `step_embedding = nn.Embedding(8, hidden_dim)` to RecursiveDecoder.
Step embedding is injected into both `latent_step` and `answer_step` via:
```python
x = y_embed + latent_z + pos_embed + step_emb
```

**Precedent**: Diffusion models, DDPM, iterative refinement literature

#### Priority 1.3: Fix Residual Format for Answer Step ✅ **COMPLETED 2024-12-13**
**Why**: Current architecture only partially uses residuals

**Solution Implemented**: GRU-style gated residual (safer than direct delta):
```python
# In answer_step:
candidate_logits = self.output_head(x)           # New hypothesis
gate = torch.sigmoid(self.gate_head(x))          # Confidence [0,1]
new_logits = (1 - gate) * prev_logits + gate * candidate_logits
```

- Gate bias initialized to -2.0 (conservative start, gate ≈ 0.12)
- Forces model to "earn" the right to make changes
- Bounded updates (no exploding deltas)

#### Priority 1.4: Implement Refinement Tracker Integration 📊
**Why**: Monitor how sequences change across steps (already created, need to integrate)

**Integration Points**:
```python
# In trainer_optimized.py, during evaluation:
from src.training.refinement_tracker import compute_refinement_metrics, summarize_refinement

metrics = compute_refinement_metrics(
    all_logits=all_logits,  # (T, B, S, V)
    targets=targets,
    target_mask=target_mask,
)

# Log to W&B
wandb.log({
    **metrics,  # edit_rate_step_{t}, accuracy_step_{t}, improvement_step_{t}
    'refinement_summary': summarize_refinement(metrics),
})
```

**Effort**: Easy (file exists, just needs integration)
**Impact**: High visibility into recursion behavior

---

### **Phase 2: Advanced Architecture Features** (Weeks 3-4)
**Goal**: Physics-aware improvements for modified peptides and complex data

#### Priority 2.1: Mass Gap Token & Delta Mass Prediction ⭐⭐ **ADVANCED**
**Why**: Handle unknown modifications, PTMs, non-standard amino acids

**Concept**:
- Add `[GAP]` token to vocabulary (represents "unknown mass shift")
- Add `delta_mass_head` to predict mass shift per position
- Compute theoretical masses as: `base_mass + predicted_delta`

**Use Cases**:
- Oxidation (+15.99 Da on M)
- Phosphorylation (+79.97 Da on S/T/Y)
- Unknown modifications in open search
- Unusual cleavage sites

**Implementation Sketch**:
```python
# 1. Update constants.py
VOCAB = [...existing..., '[GAP]']
GAP_IDX = len(VOCAB) - 1

# 2. Add to decoder
self.delta_mass_head = nn.Linear(hidden_dim, 1)  # Predict mass shift

# 3. In forward loop
deltas = torch.tanh(self.delta_mass_head(x)).squeeze(-1) * 500.0  # ±500 Da range
predicted_masses = compute_theoretical_masses_with_gap(
    sequence_ids=sequence_ids,
    predicted_deltas=deltas,
    aa_masses=self.aa_masses,
    gap_idx=GAP_IDX,
)
```

**Effort**: High (requires physics engine updates, loss changes)
**Impact**: Medium-High (critical for real data with PTMs)
**Status**: **Defer until Phase 3** (after recursion works and real data tested)

#### Priority 2.2: Residual Spectrum Embedding ⭐⭐ **ADVANCED**
**Why**: Explicit physics feedback - show model what it missed

**Concept**:
1. Render observed spectrum (reality)
2. Render predicted spectrum (expectation)
3. Compute residual: `residual = observed - predicted`
4. Feed residual through CNN to detect "shift patterns"
5. Inject CNN output into latent state

**Key Insights from Analysis**:
- Use **cheap rendering**: Hard binning at 0.5 Da (not Gaussian)
- Inject only at **supervision steps** (not latent steps) → sparse feedback
- Detach gradients through renderer (no backprop through "eye")
- 3-layer CNN: 5→2→1 Da receptive field to detect mass shifts

**Implementation** (Low-Cost Version):
```python
class ResidualSpectrumEncoder(nn.Module):
    def __init__(self, max_mz=2000, bin_size=0.5, embed_dim=256):
        self.num_bins = int(max_mz / bin_size)  # 4000 bins at 0.5 Da

        # Lightweight CNN (only ~50K params)
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32), nn.LeakyReLU(0.1),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64), nn.LeakyReLU(0.1),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.Flatten(),
        )
        self.projector = nn.Linear(cnn_out_dim, embed_dim)

    def forward(self, observed_masses, observed_intensities, theoretical_masses):
        # Fast hard binning (detached - no gradients)
        with torch.no_grad():
            obs_spec = self._render(observed_masses, observed_intensities)
            theo_spec = self._render(theoretical_masses)
            residual = obs_spec - theo_spec  # What did we miss?

        # Encode shift pattern
        features = self.cnn(residual.unsqueeze(1))
        embedding = self.projector(features)
        return embedding
```

**Inject at supervision steps**:
```python
# In recursive loop (only at supervision steps, not latent steps):
if is_supervision_step:
    residual_emb = self.residual_encoder(observed, intensities, predicted)
    z = z + residual_emb.unsqueeze(1)  # Inject physics feedback
```

**Computational Cost**: +5-10% training time (negligible)
**Effort**: High (requires architectural changes)
**Impact**: High (for modified peptides, shift detection)
**Status**: **Defer to Phase 3** (after simpler methods tested)

---

### **Phase 3: Spectrum Loss Improvements** (Week 5)
**Goal**: Fix or replace spectrum matching loss

#### Priority 3.1: Implement Matched Filter Loss (Recall-Only) ⭐⭐
**Why**: Current symmetric loss struggles with dropout

**Philosophy**:
- **Trust observed peaks** (they're real signal)
- **Don't trust absence** (peaks may have dropped out)
- Focus on recall: "Did we predict peaks where we observed them?"
- Let precursor loss handle precision (prevent hallucinations)

**Implementation** (Already Proposed):
```python
class MatchedFilterSpectrumLoss(nn.Module):
    """Robust recall-only loss that ignores missing peaks."""

    def forward(self, sequence_probs, observed_masses, observed_intensities, peak_mask):
        # 1. Get predicted masses
        predicted_masses = compute_theoretical_peaks(sequence_probs, self.aa_masses, ...)

        # 2. Gaussian alignment scores (batch, num_obs, num_pred)
        diff = observed_masses.unsqueeze(-1) - predicted_masses.unsqueeze(1)
        alignment_scores = torch.exp(-0.5 * (diff / self.sigma) ** 2)

        # 3. Best match per OBSERVED peak (recall)
        peak_match_score = alignment_scores.max(dim=-1)[0]  # (batch, num_obs)

        # 4. Weight by observed intensity (important peaks matter more)
        masked_intensities = observed_intensities * peak_mask.float()
        weights = masked_intensities / (masked_intensities.sum(dim=1, keepdim=True) + 1e-8)

        # 5. Recall score: Did we explain the observed peaks?
        recall_score = (peak_match_score * weights).sum(dim=1)

        # Loss = 1 - Recall
        return 1.0 - recall_score.mean()
```

**Advantages**:
- Robust to peak dropout (common in real data)
- Simpler gradient signal
- Precursor loss prevents hallucinations (no precision term needed)

**Effort**: Easy (code already written, just needs integration)
**Impact**: Medium (may improve real data performance)

#### Priority 3.2: Adaptive Sigma Curriculum 📊
**Why**: Current sigma=0.2 too narrow for early training

**Schedule**:
```python
SIGMA_CURRICULUM = [
    (0, 10000, 10.0),     # Stage 1: Very forgiving (learn rough patterns)
    (10000, 20000, 5.0),  # Stage 2: Tighten (90% accuracy)
    (20000, 40000, 2.0),  # Stage 3: Realistic (high accuracy)
    (40000, None, 0.5),   # Stage 4+: Tight matching
]
```

**Integration**:
```python
# In curriculum_scheduler.step():
current_sigma = get_sigma_for_step(global_step)
self.loss_fn.sigma = current_sigma
```

**Effort**: Easy
**Impact**: Medium (may unblock spectrum loss)
**Status**: **Test after matched filter loss**

---

### **Phase 4: Real Data Transition** (Weeks 6-8)
**Goal**: Bridge synthetic→real gap, achieve SOTA on benchmarks

#### Priority 4.1: Implement Mixed Sim/Real Curriculum ⭐⭐⭐ **RECOMMENDED**
**Why**: Proven technique for domain adaptation

**Strategy**: Same as clean/noisy mixing but with synthetic/real data

**Proposed Schedule**:
```python
MIXED_DATA_CURRICULUM = [
    # Stage 1 (0-20K): Pure synthetic - learn basics
    {'steps': 20000, 'synthetic_ratio': 1.0, 'real_ratio': 0.0},

    # Stage 2 (20K-40K): Introduce real data
    {'steps': 20000, 'synthetic_ratio': 0.8, 'real_ratio': 0.2},

    # Stage 3 (40K-60K): Balance
    {'steps': 20000, 'synthetic_ratio': 0.5, 'real_ratio': 0.5},

    # Stage 4 (60K-80K): Mostly real
    {'steps': 20000, 'synthetic_ratio': 0.2, 'real_ratio': 0.8},

    # Stage 5 (80K+): Pure real
    {'steps': 20000, 'synthetic_ratio': 0.0, 'real_ratio': 1.0},
]
```

**Implementation**:
```python
# In dataset._generate_sample():
if np.random.rand() < self.real_data_ratio:
    return self._load_real_sample()  # From ProteomeTools or Nine-Species
else:
    return self._generate_synthetic_sample()  # From MS2PIP
```

**Effort**: Medium (dataset integration)
**Impact**: High (proven for domain adaptation)

#### Priority 4.2: Evaluate on ProteomeTools ⭐⭐
**Why**: High-quality synthetic benchmark (21M spectra)

**Steps**:
1. Download ProteomeTools FTMS_HCD_28 (253 MB)
2. Implement data loader (already designed)
3. Pre-train on ProteomeTools for 50K steps
4. Evaluate vs MS2PIP baseline

**Expected Performance**: 85-90% token accuracy (between MS2PIP clean and real data)

**Effort**: Easy (data loader already designed)
**Impact**: Medium (good intermediate benchmark)

#### Priority 4.3: Evaluate on Nine-Species Benchmark ⭐⭐⭐
**Why**: Standard real-data benchmark for de novo sequencing

**Steps**:
1. Download Nine-Species balanced dataset (15 GB)
2. Implement 9-fold cross-validation protocol
3. Train on 8 species, test on held-out 9th
4. Compare against Casanovo baseline

**Expected Performance**: 60-75% token accuracy on real data

**Effort**: Medium (requires CV infrastructure)
**Impact**: High (publication-ready benchmark)

#### Priority 4.4: Alternative Sim-to-Real Strategies

**Option A: Two-Stage Pre-training + Fine-tuning** ⭐⭐
- Stage 1: Train on MS2PIP/ProteomeTools to convergence
- Stage 2: Fine-tune on Nine-Species with lower LR (5e-5)
- **Pros**: Simple, proven
- **Cons**: Risk of catastrophic forgetting

**Option B: Noise Model Learning** ⭐
- Train VAE on residuals: `Real Spectrum - MS2PIP Prediction`
- Use learned noise to augment synthetic data
- **Pros**: Improves synthetic realism
- **Cons**: Requires real data upfront, complex

**Option C: Adversarial Domain Adaptation** ⭐
- Add discriminator: "Is this synthetic or real?"
- Train encoder to fool discriminator (domain-invariant features)
- **Pros**: Proven in computer vision
- **Cons**: Complex training, GAN instability

**Recommendation**: Start with **Priority 4.1 (Mixed Curriculum)**, fall back to **Option A** if needed

---

### **Phase 5: Missing Physics & Advanced Features** (Weeks 9-12)
**Goal**: Complete feature set for production deployment

#### Priority 5.1: Doubly-Charged Ions (b++, y++) ⭐⭐⭐ **CRITICAL MISSING PHYSICS**
**Why**: Common in charge 2+ precursors, currently treated as noise

**Implementation**:
```python
# Update compute_theoretical_peaks to include:
ion_types = ['b', 'y', 'b++', 'y++']

# Compute doubly-charged masses:
b_doubly = (b_masses + 2 * PROTON_MASS) / 2
y_doubly = (y_masses + 2 * PROTON_MASS) / 2
```

**Curriculum**: Introduce gradually
- Stages 1-2: Only singly-charged (learn basics)
- Stage 3+: Add doubly-charged ions

**Effort**: Easy (just mass calculation updates)
**Impact**: High (critical for real data)

#### Priority 5.2: Common PTM Support ⭐⭐
**Why**: Real biological samples have modifications

**Common PTMs**:
- Oxidation: M(+15.99)
- Phosphorylation: S/T/Y(+79.97)
- Acetylation: N-term(+42.01)
- Deamidation: N/Q(+0.98)

**Implementation Options**:
1. **Explicit tokens**: Add `M(ox)`, `pS`, etc. to vocabulary
   - Pros: Simple, interpretable
   - Cons: Vocabulary explosion (20 AA × 10 mods = 200 tokens)

2. **Mass gap token** (Priority 2.1)
   - Pros: Handles any modification
   - Cons: Less interpretable, harder to train

**Recommendation**: Start with **explicit tokens for top 3-5 PTMs**, then add mass gap for open search

**Effort**: Medium (vocabulary changes)
**Impact**: High (essential for real data)

#### Priority 5.3: Uncertainty Quantification 📊
**Why**: Know when model is confident vs guessing

**Approaches**:
1. **Entropy-based**: High entropy = uncertain
   ```python
   probs = torch.softmax(logits, dim=-1)
   entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
   confidence = 1.0 - (entropy / math.log(vocab_size))
   ```

2. **Ensemble**: Train 3-5 models, measure disagreement
   - High agreement = confident
   - High disagreement = uncertain

3. **Dropout-based** (MC Dropout): Run inference with dropout on
   - Variance across runs = uncertainty

**Use Cases**:
- Filter low-confidence predictions
- Prioritize manual validation
- Detect out-of-distribution samples

**Effort**: Medium
**Impact**: Medium (important for production)

---

### **Phase 6: Production & Deployment** (Months 3-4)
**Goal**: Optimize for real-world use

#### Priority 6.1: Inference Optimization
- **Quantization**: FP32 → INT8 (4x smaller, 2-3x faster)
- **Pruning**: Remove low-magnitude weights
- **Distillation**: Train smaller student model
- **TorchScript**: Compile for C++ deployment

**Target**: <100ms per spectrum on CPU

#### Priority 6.2: Multi-Instrument Generalization
- Train on data from different instruments (Orbitrap, Q-TOF, TOF-TOF)
- Instrument-specific fine-tuning
- Meta-learning for fast adaptation

#### Priority 6.3: Integration with Proteomics Pipelines
- MGF/mzML file readers
- Integration with MaxQuant, Proteome Discoverer
- REST API for cloud deployment
- Batch processing scripts

---

## 📐 Implementation Timeline

### **Sprint 1-2 (Weeks 1-2): Recursion First**
- ✅ Monitor aggressive noise training
- 🔲 Implement step embeddings (if needed)
- 🔲 Fix residual format for answer_step (if needed)
- 🔲 Integrate refinement tracker
- **Goal**: Achieve multi-step refinement (ce_step_7 < ce_step_0)

### **Sprint 3-4 (Weeks 3-4): Real Data Prep**
- 🔲 Download ProteomeTools dataset
- 🔲 Download Nine-Species dataset
- 🔲 Implement matched filter spectrum loss
- 🔲 Add doubly-charged ions
- **Goal**: Infrastructure ready for real data

### **Sprint 5-6 (Weeks 5-6): First Real Data Training**
- 🔲 Pre-train on ProteomeTools
- 🔲 Implement mixed sim/real curriculum
- 🔲 First Nine-Species evaluation
- **Goal**: Measure sim-to-real gap

### **Sprint 7-8 (Weeks 7-8): Optimization & Benchmarking**
- 🔲 9-fold cross-validation on Nine-Species
- 🔲 Hyperparameter tuning for real data
- 🔲 Compare against Casanovo baseline
- **Goal**: Competitive performance on benchmark

### **Sprint 9-12 (Weeks 9-12): Advanced Features**
- 🔲 PTM support (explicit tokens)
- 🔲 Uncertainty quantification
- 🔲 Mass gap token (if needed)
- 🔲 Residual spectrum embedding (if needed)
- **Goal**: Publication-ready feature set

---

## 🎯 Success Criteria

### **Recursion (Phase 1)**
- ✅ Clear improvement across all 8 steps (not just step 0→1)
- ✅ Edit rate >5% at steps 2-7
- ✅ ce_step_7 consistently lower than ce_step_0
- ✅ Refinement tracker shows positive improvements

### **Synthetic Data (Phases 1-2)**
- ✅ >90% token accuracy on clean MS2PIP
- ✅ >75% token accuracy on noisy MS2PIP (45% dropout, 30 noise peaks)
- ✅ >85% token accuracy on ProteomeTools

### **Real Data (Phases 4-5)**
- ✅ >70% token accuracy on Nine-Species (averaged across 9 folds)
- ✅ >60% sequence accuracy on Nine-Species easy subset
- ✅ Competitive with Casanovo on standard benchmarks
- ✅ Handles doubly-charged ions correctly

### **Production (Phase 6)**
- ✅ <100ms inference time per spectrum (CPU)
- ✅ Uncertainty calibration (high confidence → high accuracy)
- ✅ Works across multiple instruments
- ✅ Integration with standard proteomics tools

---

## 💡 Key Technical Insights

### **On Recursion**
- **Linear weighting insufficient**: Step 7 only gets 22% of loss
- **Exponential weighting**: Step 7 gets 50% - forces optimization
- **Noise is essential**: Model needs challenge to use recursion
- **Step embeddings**: Tell model "which iteration am I on?"
- **Residual format**: Learn deltas, not states (prevents fixed points)

### **On Spectrum Loss**
- **Sigma curriculum**: Start wide (10 Da), narrow over time (0.5 Da)
- **Matched filter**: Recall-only (trust observed, not absence)
- **Precursor loss balances**: Prevents hallucinations (no need for precision term)
- **Clean data first**: Spectrum loss helps even on clean data (19K→6.7K ppm mass error)

### **On Sim-to-Real**
- **Mixed curriculum best**: 100% synthetic → 100% real over 60K steps
- **ProteomeTools good middle ground**: Better than MS2PIP, easier than Nine-Species
- **Pre-train + fine-tune**: Simpler fallback if mixing fails
- **Noise model learning**: Advanced technique if simple methods insufficient

### **On Missing Physics**
- **Doubly-charged ions critical**: Common in charge 2+ precursors
- **PTMs essential for real data**: Top 5 mods cover 80% of real spectra
- **Mass gap advanced**: For open search, unknown modifications
- **Residual embedding powerful**: Explicit physics feedback for shift detection

---

## 📚 References & Prior Art

### **Recursive Models**
- AlphaFold (recycling iterations)
- Diffusion models (DDPM, step embeddings)
- Neural ODEs (residual updates as continuous dynamics)
- DETR (iterative refinement for object detection)

### **Peptide Sequencing**
- **Casanovo**: Current SOTA, transformer-based
- **DeepNovo**: Original deep learning approach, LSTM
- **PointNovo**: Graph-based, handles PTMs
- **Prosit**: Spectrum prediction (not sequencing)

### **Domain Adaptation**
- DANN (Domain-Adversarial Neural Networks)
- CycleGAN (image translation)
- Pre-training + fine-tuning (BERT, GPT paradigm)
- Progressive domain adaptation

### **Mass Spectrometry Datasets**
- **MS2PIP**: Synthetic prediction tool
- **ProteomeTools**: 21M synthetic spectra
- **Nine-Species**: 2.8M real PSMs benchmark
- **MassIVE-KB**: Large-scale community data

---

## 🔧 Deferred/Rejected Ideas

### **Rejected**
- ❌ **Narrower sigma without curriculum**: Hurts learning
- ❌ **Reduce batch size below 40**: Quality degradation
- ❌ **Skip real data benchmarking**: Essential for publication
- ❌ **Train on all PTMs simultaneously**: Vocabulary explosion

### **Deferred to Later**
- ⏸️ **Gradient checkpointing**: Not needed for model this size
- ⏸️ **Multi-task learning**: Focus on sequencing first
- ⏸️ **Cross-linking peptides**: Rare, complex
- ⏸️ **Glycosylation**: Very complex modifications
- ⏸️ **De novo assembly**: Protein-level inference

### **Queued for Testing (Post Step Embeddings)**

#### Priority A: Multi-Point Step Embedding Injection ⭐⭐
**Why**: In diffusion models, timestep embeddings are injected at every layer, not just the input.
Currently we add step_emb once at input fusion. Better approach: inject into each transformer layer.

**Implementation**:
```python
# In TransformerDecoderLayer:
def forward(self, x, context, step_emb=None):
    if step_emb is not None:
        x = x + step_emb  # Inject at each layer
    # ... rest of layer
```

**Effort**: Medium (modify transformer layers)
**Expected Impact**: Stronger step-awareness, better differentiation between early/late steps

#### Priority B: pLDDT-Style Confidence Head ⭐⭐⭐
**Why**: Know which positions are confident vs uncertain (like AlphaFold's pLDDT).

**Implementation**:
```python
# Add confidence head to decoder
self.confidence_head = nn.Sequential(
    nn.Linear(hidden_dim, hidden_dim // 2),
    nn.ReLU(),
    nn.Linear(hidden_dim // 2, 1),
    nn.Sigmoid()
)

# During training: target is per-position correctness (1 if correct, 0 if wrong)
confidence_target = (predictions == targets).float()
confidence_loss = F.binary_cross_entropy(confidence_pred, confidence_target)

# During inference: color-code output
# "PEPTIDE" where P is red (uncertain, confidence<0.5) and EPTIDE is blue (confident)
```

**Use Cases**:
- Filter uncertain positions during inference
- Prioritize manual validation of low-confidence regions
- Quality control for downstream analysis

**Effort**: Easy-Medium
**Expected Impact**: High for production use, interpretability

#### Priority C: Noise/Curriculum Conditioning ⭐⭐
**Why**: Tell the model about data quality ("Greedy Mode" vs "Careful Mode").

**Implementation**:
```python
# Add noise level embedding (indexed by curriculum stage or estimated SNR)
self.noise_embedding = nn.Embedding(num_curriculum_stages, hidden_dim)

# In forward loop, inject alongside step embedding:
noise_emb = self.noise_embedding(curriculum_stage_idx)
x = y_embed + latent_z + pos_embed + step_emb + noise_emb
```

**Alternative**: Estimate SNR from spectrum statistics (peak count, intensity distribution)
and use continuous embedding.

**Use Cases**:
- Model adapts processing based on data quality
- Cleaner data → more confident, noisier data → more cautious
- Could improve sim-to-real transfer

**Effort**: Easy
**Expected Impact**: Medium (especially for curriculum training)

---

## 📊 Metrics Dashboard

### **Training Metrics**
- Token accuracy (overall, per-step)
- Sequence accuracy (full match)
- Edit rate per step (refinement tracker)
- Improvement rate per step
- Mass error (ppm, Daltons)
- Loss breakdown (CE, spectrum, precursor)

### **Validation Metrics**
- Token/sequence accuracy on val_easy (clean)
- Token/sequence accuracy on val_hard (noisy)
- Per-species accuracy (Nine-Species)
- Coverage (% of observed peaks explained)
- Precision/Recall on peak matching

### **Production Metrics**
- Inference time (ms/spectrum)
- Model size (MB)
- Confidence calibration (ECE)
- Out-of-distribution detection (AUROC)

---

## 🎓 Open Research Questions

1. **Is recursion necessary for noisy data?** Testing now with aggressive noise
2. **What's the optimal number of refinement steps?** Currently 8, could be 4-16
3. **Can mass gap generalize to any PTM?** Unknown - needs testing
4. **Does residual embedding help on real data?** Theoretical benefit, needs validation
5. **What's the sim-to-real performance ceiling?** Unknown until tested
6. **Can one model handle all instruments?** Or need instrument-specific fine-tuning?
7. **What's the right balance of synthetic/real data?** Testing 50/50, could vary

---

**Last Updated**: 2024-12-13
**Next Review**: After step embeddings + gated residuals training run
**Maintainer**: Development team

---

## 📝 Change Log

**v2.1 (2024-12-13)**:
- ✅ **IMPLEMENTED: Step Embeddings** - Model now knows which refinement iteration it's on (0-7)
- ✅ **IMPLEMENTED: Gated Residuals** - GRU-style update in answer_step: `new = (1-gate)*prev + gate*candidate`
  - Gate initialized to -2.0 (conservative start, ~12% initial gate activation)
  - Forces model to "earn" the right to make changes
- Added local metrics logging to trainer (logs/metrics_*.jsonl)
- Added analyze_runs.py tool for parsing training metrics
- Added future ideas to roadmap: multi-point injection, pLDDT confidence, noise conditioning
- Parameter count: 12,482,816 → 12,495,128 (+12K for step_emb + gate_head)
- **NOTE**: Old checkpoints are NOT compatible with new model architecture

**v2.0 (2024-12-12)**:
- Consolidated all previous roadmaps and proposals
- Added mass gap token and residual embedding (from Gemini docs)
- Integrated recursion improvements (step embeddings, residual format)
- Added matched filter spectrum loss
- Updated sim-to-real strategies with mixed curriculum
- Reorganized by priority and timeline
- Added success criteria and metrics

**v1.0 (Earlier)**:
- Initial roadmap from 50K training run analysis
- Focus on curriculum improvements
- Basic real data pipeline
