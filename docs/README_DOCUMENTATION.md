# Documentation Index

**Last Updated**: 2024-12-12

## 📋 Master Documents

### **MASTER_ROADMAP_2024-12-12.md** ⭐ **PRIMARY REFERENCE**
**Comprehensive development roadmap consolidating ALL previous documents.**

**What's inside**:
- ✅ Complete priority list (Phases 1-6)
- ✅ Recursion improvements (step embeddings, residual format)
- ✅ Advanced features (mass gap token, residual spectrum embedding)
- ✅ Spectrum loss fixes (matched filter, sigma curriculum)
- ✅ Sim-to-real strategies (mixed curriculum, pre-train+fine-tune)
- ✅ Real data pipelines (ProteomeTools, Nine-Species)
- ✅ Timeline, success criteria, open questions

**Consolidated from**:
- `pepTRM_development_roadmap.md`
- `mass_gap_implementation_example_gemini.txt`
- `gemini_residual_embedding.md`
- `precursor_mass_loss_proposal.txt`
- `precision_recall_spectrum_loss.txt`
- `robust_spectral_match_recall.txt`
- `recursion_update.txt`

---

## 📚 Dataset Documentation

### **training_with_real_data.md**
Complete guide for transitioning from synthetic to real data.

**Contents**:
- Dataset comparison (MS2PIP, ProteomeTools, Nine-Species)
- Training strategies (progressive, mixed, pre-train+fine-tune)
- Quick start examples with identical interfaces
- Expected performance benchmarks

### **proteometools_dataset.md**
ProteomeTools setup and usage guide.

**Contents**:
- Dataset overview (21M spectra, synthetic peptides)
- Download instructions (2.3 GB)
- Data loader usage
- Integration with training pipeline

### **nine_species_dataset.md**
Nine-Species benchmark setup and usage.

**Contents**:
- Dataset overview (2.8M PSMs, 9 species)
- Download instructions (15-50 GB)
- 9-fold cross-validation protocol
- Expected performance on real data

---

## 🔧 Technical Documentation

### **memory_management.md**
Memory optimization strategies for RTX 4090 (24GB VRAM).

**Contents**:
- Memory budget breakdown
- Gradient accumulation strategy
- Batch size trade-offs
- torch.compile considerations
- OOM troubleshooting

### **recursion_update.txt**
Step embeddings and residual format for improving recursion.

**Contents**:
- Step embedding implementation
- Residual update format
- Code examples
- Rationale from diffusion models

---

## 🗂️ Deprecated/Historical Documents

These documents have been **consolidated into MASTER_ROADMAP** but are kept for reference:

### **pepTRM_development_roadmap.md**
- Original roadmap from 50K training run
- Curriculum learning insights
- Phase 0-4 breakdown
- **Status**: Superseded by MASTER_ROADMAP

### **mass_gap_implementation_example_gemini.txt**
- Mass gap token implementation
- Delta mass prediction head
- Physics-aware solver approach
- **Status**: Consolidated into MASTER_ROADMAP Phase 2.1

### **gemini_residual_embedding.md**
- Residual spectrum embedding analysis
- Computational cost discussion
- Low-cost implementation strategy
- **Status**: Consolidated into MASTER_ROADMAP Phase 2.2

### **precursor_mass_loss_proposal.txt**
- Normalize mass error by AA mass (110 Da)
- Smooth L1 loss
- **Status**: Already implemented in codebase

### **precision_recall_spectrum_loss.txt**
- Symmetric loss (recall + precision)
- F1-style combination
- **Status**: Alternative to matched filter in MASTER_ROADMAP Phase 3.1

### **robust_spectral_match_recall.txt**
- Matched filter approach (recall-only)
- Trust observed peaks, not absence
- **Status**: Recommended approach in MASTER_ROADMAP Phase 3.1

---

## 📖 How to Use This Documentation

### **Starting a new feature?**
1. Check **MASTER_ROADMAP** for priority and phase
2. Review technical details in referenced section
3. Check dataset docs if using real data

### **Need dataset info?**
1. **Synthetic data**: See training_with_real_data.md overview
2. **ProteomeTools**: See proteometools_dataset.md
3. **Nine-Species**: See nine_species_dataset.md

### **Hitting memory issues?**
1. See memory_management.md
2. Check batch size recommendations
3. Review gradient accumulation setup

### **Want to improve recursion?**
1. See MASTER_ROADMAP Phase 1
2. Check recursion_update.txt for implementation details
3. Review step embedding and residual format code

### **Planning sim-to-real transition?**
1. See MASTER_ROADMAP Phase 4
2. Read training_with_real_data.md for strategies
3. Download datasets per proteometools_dataset.md and nine_species_dataset.md

---

## 🎯 Quick Reference

| Task | Document |
|------|----------|
| **Overall strategy & priorities** | MASTER_ROADMAP_2024-12-12.md |
| **Improve recursion** | MASTER_ROADMAP Phase 1 + recursion_update.txt |
| **Fix spectrum loss** | MASTER_ROADMAP Phase 3 |
| **Add advanced features** | MASTER_ROADMAP Phases 2 & 5 |
| **Train on real data** | training_with_real_data.md |
| **Memory optimization** | memory_management.md |
| **Dataset setup** | proteometools_dataset.md, nine_species_dataset.md |

---

## 🔄 Maintenance

**When to update MASTER_ROADMAP**:
- Major architectural changes
- New experimental results
- Completed phases
- Revised priorities

**How to update**:
1. Edit MASTER_ROADMAP_2024-12-12.md
2. Update version number in change log
3. Update this README if new docs added

---

## 📊 Current Status (2024-12-12)

**Active Training**: Aggressive noise curriculum (testing recursion hypothesis)

**Next Steps**:
1. Monitor current run for multi-step refinement
2. Implement step embeddings if needed (Phase 1.2)
3. Fix residual format if needed (Phase 1.3)
4. Prepare for real data evaluation (Phase 4)

**See MASTER_ROADMAP for detailed timeline and success criteria.**
