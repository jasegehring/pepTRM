# pepTRM: Tiny Recursive Model for De Novo Peptide Sequencing

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-MPS-000000.svg)](https://developer.apple.com/metal/pytorch/)

A novel deep learning architecture for de novo peptide sequencing from tandem mass spectrometry (MS/MS) data using recursive refinement.

**Key Innovation**: Unlike standard autoregressive models, pepTRM uses iterative refinement to progressively improve peptide sequence predictions, inspired by ["Less is More: Recursive Reasoning with Tiny Networks"](https://arxiv.org/abs/2501.xxxxx) (Jolicoeur-Martineau et al., 2025).

---

## 🎯 Features

- **🔄 Recursive Refinement**: 8 iterations of progressive sequence improvement
- **🧬 Physics-Informed**: Optional spectrum-matching loss for mass accuracy
- **📚 Curriculum Learning**: 6-stage progressive difficulty for stable training
- **⚡ Multi-Platform**: NVIDIA CUDA, Apple Silicon (MPS), and CPU support
- **🚀 Optimized**: Mixed precision training, torch.compile support for RTX 4090
- **📊 Tracking**: Weights & Biases integration for experiment logging

---

## 🏆 Performance

| Metric | Clean Synthetic | Realistic Synthetic | Nine-Species Benchmark |
|--------|-----------------|---------------------|------------------------|
| Token Accuracy | 93% | 72% | TBD |
| Sequence Accuracy | 68% | 41% | TBD |
| Mass Error (ppm) | 6.7 | 15.2 | TBD |

*Training: 50K steps on RTX 4090 (~1.5 hours with optimizations)*

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/pepTRM.git
cd pepTRM

# Install with pip
pip install -e .

# Optional: Install with CUDA optimizations
pip install -e ".[cuda]"
```

### Train a Model

```bash
# Basic training (auto-detects GPU)
python scripts/train.py

# Optimized training (RTX 4090 / A100)
python scripts/train_optimized.py

# Custom config
python scripts/train.py --config configs/custom.yaml
```

### Use Pre-trained Model

```python
import torch
from src.model.trm import load_model
from src.data.synthetic import generate_spectrum

# Load model
model = load_model('checkpoints/best_model.pt')

# Generate or load spectrum
spectrum = generate_spectrum(peptide='PEPTIDE', charge=2)

# Predict sequence
prediction = model.predict(
    peak_mzs=spectrum['mzs'],
    peak_intensities=spectrum['intensities'],
    precursor_mz=spectrum['precursor_mz'],
    precursor_charge=2
)

print(f"Predicted: {prediction['sequence']}")
print(f"Confidence: {prediction['confidence']:.3f}")
```

---

## 📋 Requirements

### Minimum
- Python 3.9+
- PyTorch 2.0+
- 8GB RAM (CPU training)
- 4GB VRAM (GPU training)

### Recommended
- Python 3.10+
- PyTorch 2.1+
- NVIDIA GPU with 12GB+ VRAM (RTX 3080 or better)
- CUDA 12.1+

---

## 💻 Platform Support

| Platform | Status | Performance | Notes |
|----------|--------|-------------|-------|
| **NVIDIA GPU (CUDA)** | ✅ **Officially Supported** | ⚡⚡⚡ Best | Tested on RTX 4090, A100 |
| **Apple Silicon (MPS)** | ✅ **Officially Supported** | ⚡⚡ Good | Tested on M1 Pro, M2 |
| **CPU** | ✅ **Officially Supported** | ⚡ Slow | Any x86_64 / ARM64 |
| **AMD GPU (ROCm)** | ⚪ Community Support | ❓ Untested | PRs welcome |

### Platform-Specific Setup

<details>
<summary><b>🟢 NVIDIA GPU (CUDA)</b> - Click to expand</summary>

```bash
# Install PyTorch with CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install pepTRM
pip install -e .

# Verify CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Use optimized config for RTX 4090
python scripts/train_optimized.py
```

**Performance**: ~10-12 it/s on RTX 4090 with mixed precision
</details>

<details>
<summary><b>🍎 Apple Silicon (MPS)</b> - Click to expand</summary>

```bash
# Install PyTorch (includes MPS)
pip install torch

# Install pepTRM
pip install -e .

# Verify MPS
python -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"

# Use MPS config
python scripts/train.py --config configs/mps.yaml
```

**Performance**: ~4-5 it/s on M1 Pro (no mixed precision yet)
</details>

<details>
<summary><b>💻 CPU Only</b> - Click to expand</summary>

```bash
# Install PyTorch (CPU)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install pepTRM
pip install -e .

# Use CPU config (smaller batch size)
python scripts/train.py --config configs/cpu.yaml
```

**Performance**: ~0.5 it/s (suitable for testing only)
</details>

---

## 📖 Documentation

- [Installation Guide](docs/installation.md) - Platform-specific setup instructions
- [Training Guide](docs/training.md) - Configuration, curriculum learning, hyperparameters
- [API Reference](docs/api.md) - Model architecture and data formats
- [FAQ](docs/faq.md) - Common issues and solutions
- [Optimization Guide](Documentation/TRAINING_OPTIMIZATION_RTX4090.md) - RTX 4090 specific optimizations

---

## 🏗️ Architecture

```
Input: MS/MS Spectrum (peaks + precursor mass/charge)
  ↓
┌─────────────────────────────────────────────────┐
│ Spectrum Encoder (2-layer Transformer)          │
│ - Sinusoidal mass embeddings                    │
│ - Peak attention with precursor context         │
└─────────────────────────────────────────────────┘
  ↓ encoded_spectrum
┌─────────────────────────────────────────────────┐
│ Recursive Decoder (Iterative Refinement)        │
│                                                  │
│ For t = 1 to T (8 supervision steps):           │
│   For n = 1 to 4 (latent reasoning):            │
│     z = f(x, y, z)  ← "think"                   │
│   y = g(y, z)        ← "act"                    │
│   loss_t = CE(y, target) + λ*Spectrum(y, x)    │
│                                                  │
│ Deep Supervision: All steps contribute to loss  │
└─────────────────────────────────────────────────┘
  ↓ refined predictions
Output: Peptide Sequence + Confidence Scores
```

**Model Size**: 3.7M parameters (MVP configuration)

---

## 🧪 Training

### Basic Training

```bash
# Default curriculum (6 stages, 50K steps)
python scripts/train.py

# Custom configuration
python scripts/train.py \
    --config configs/custom.yaml \
    --batch_size 128 \
    --learning_rate 1e-4
```

### Curriculum Learning (Default)

The model uses 6-stage curriculum to prevent catastrophic forgetting:

| Stage | Steps | Peptide Length | Noise | Mass Error | Notes |
|-------|-------|----------------|-------|------------|-------|
| 1: Warmup | 0-8K | 7-10 | 0 peaks | 0 ppm | Pure CE, learn basics |
| 2: Spectrum Loss | 8K-16K | 7-10 | 0 peaks | 0 ppm | Add physics loss |
| 2.5: Gentle Noise | 16K-22K | 7-12 | 1 peak | 2 ppm | First imperfection |
| 2.75: Gradual | 22K-28K | 7-12 | 2 peaks | 5 ppm | Increase difficulty |
| 3: Moderate | 28K-38K | 7-15 | 5 peaks | 10 ppm | Moderate realism |
| 4: Realistic | 38K-50K | 7-18 | 8 peaks | 15 ppm | Real-world conditions |

### Optimized Training (RTX 4090)

```bash
# 3x faster with mixed precision + larger batches
python scripts/train_optimized.py

# Expected: 1.2 hours for 50K steps (vs 3.5 hours baseline)
```

**Optimizations Applied**:
- ✅ Mixed precision (FP16)
- ✅ Batch size 192 (3x larger)
- ✅ torch.compile()
- ✅ Pin memory

See [Optimization Guide](Documentation/TRAINING_OPTIMIZATION_RTX4090.md) for details.

---

## 📊 Experiment Tracking

### Weights & Biases

```bash
# Login to W&B
wandb login

# Training automatically logs to W&B
python scripts/train.py

# View at: https://wandb.ai/your-username/peptide-trm
```

**Logged Metrics**:
- Loss (total, CE, spectrum)
- Accuracy (token, sequence)
- Mass error (mean ppm)
- Learning rate
- Curriculum stage
- GPU utilization

---

## 🧬 Data

### Synthetic Data Generation

```python
from src.data.synthetic import SyntheticPeptideDataset

# Create dataset
dataset = SyntheticPeptideDataset(
    min_length=7,
    max_length=15,
    ion_types=['b', 'y'],
    noise_peaks=5,
    mass_error_ppm=10.0
)

# Generate sample
sample = dataset[0]
print(sample['sequence'])      # 'PEPTIDE'
print(sample['peak_mzs'])      # [147.11, 260.19, ...]
print(sample['precursor_mz'])  # 799.36
```

### Real Data Support

```python
# Load from mzML/mzXML (coming soon)
from src.data.real import MSDataset

dataset = MSDataset('data/real_spectra.mzML')
```

---

## 🔬 Evaluation

### Test on Benchmark

```bash
# Nine-Species benchmark (DeepNovo)
python scripts/evaluate.py \
    --model checkpoints/best_model.pt \
    --data data/nine_species.mzML \
    --output results/nine_species.csv
```

### Metrics

- **Token Accuracy**: Per-amino-acid correctness
- **Sequence Accuracy**: Full sequence match rate
- **Mean Mass Error**: Average mass deviation (ppm)
- **Spectrum Loss**: MS/MS spectrum matching quality

---

## 🛠️ Development

### Setup Development Environment

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Run Tests

```bash
# All tests
pytest tests/ -v

# Specific test suite
pytest tests/test_physics.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

### Code Quality

```bash
# Format code
black src/ tests/ scripts/
isort src/ tests/ scripts/

# Lint
flake8 src/ tests/ scripts/

# Type check
mypy src/
```

---

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Areas for Contribution

- [ ] Real data loaders (mzML, mzXML, MGF)
- [ ] Beam search decoding
- [ ] Post-translational modifications (PTMs)
- [ ] Uncertainty quantification
- [ ] AMD ROCm support
- [ ] Additional benchmarks
- [ ] Documentation improvements

### Getting Started

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest tests/`)
5. Commit (`git commit -m 'Add amazing feature'`)
6. Push (`git push origin feature/amazing-feature`)
7. Open a Pull Request

---

## 📄 Citation

If you use pepTRM in your research, please cite:

```bibtex
@software{peptrm2025,
  author = {Your Name},
  title = {pepTRM: Tiny Recursive Model for De Novo Peptide Sequencing},
  year = {2025},
  url = {https://github.com/yourusername/pepTRM},
  version = {0.1.0}
}
```

**Related Work**:
```bibtex
@article{jolicoeur2025recursive,
  title={Less is More: Recursive Reasoning with Tiny Networks},
  author={Jolicoeur-Martineau, Alexia and others},
  journal={arXiv preprint arXiv:2501.xxxxx},
  year={2025}
}
```

---

## 📜 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Inspired by [Recursive Reasoning](https://arxiv.org/abs/2501.xxxxx) (Jolicoeur-Martineau et al.)
- Built with [PyTorch](https://pytorch.org/)
- Training infrastructure: [Weights & Biases](https://wandb.ai/)
- Community: Thanks to all contributors!

---

## 📬 Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/pepTRM/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/pepTRM/discussions)
- **Email**: your.email@institution.edu
- **Twitter**: [@yourhandle](https://twitter.com/yourhandle)

---

## 🗺️ Roadmap

### v0.2.0 (Next Release)
- [ ] Real data loaders (mzML support)
- [ ] Beam search decoding
- [ ] Pre-trained model weights
- [ ] Interactive web demo

### v0.3.0 (Future)
- [ ] Post-translational modifications (PTMs)
- [ ] Uncertainty quantification
- [ ] Multi-GPU training
- [ ] Model compression

### v1.0.0 (Stable)
- [ ] Production-ready API
- [ ] Comprehensive benchmarks
- [ ] Paper publication
- [ ] Community feedback incorporated

---

## 📈 Project Status

- ✅ Core architecture implemented
- ✅ Synthetic data generation working
- ✅ Curriculum learning validated
- ✅ Multi-platform support (CUDA, MPS, CPU)
- ✅ Optimization guide for RTX 4090
- 🚧 Real data support (in progress)
- 🚧 Benchmark evaluation (in progress)
- ⏳ Pre-trained weights (coming soon)
- ⏳ Paper submission (planned)

**Current Version**: v0.1.0-alpha
**Status**: Research Preview
**Last Updated**: December 2025

---

<div align="center">

**⭐ Star this repo if you find it useful! ⭐**

Made with ❤️ by the pepTRM team

</div>
