# Repository Setup Guide for Sharing pepTRM

## Overview

This guide covers best practices for sharing the pepTRM codebase publicly, including multi-platform support, documentation, and reproducibility.

---

## Recommended Repository Structure

```
pepTRM/
├── .github/
│   ├── workflows/
│   │   ├── tests.yml              # CI/CD testing
│   │   └── publish.yml            # Package publishing
│   ├── ISSUE_TEMPLATE/
│   └── PULL_REQUEST_TEMPLATE.md
├── configs/
│   ├── default.yaml               # Generic config
│   ├── cuda_optimized.yaml        # RTX 4090 / A100
│   ├── mps.yaml                   # Apple Silicon M1/M2/M3
│   └── cpu.yaml                   # CPU-only training
├── docs/
│   ├── installation.md            # Platform-specific setup
│   ├── quickstart.md              # Getting started
│   ├── training.md                # Training guide
│   ├── api.md                     # API documentation
│   └── faq.md                     # Common issues
├── src/
│   ├── __init__.py
│   ├── constants.py
│   ├── data/
│   ├── model/
│   ├── training/
│   ├── inference/
│   └── utils/
├── scripts/
│   ├── train.py                   # Basic training
│   ├── train_optimized.py         # Optimized for CUDA
│   ├── evaluate.py                # Model evaluation
│   └── demo.py                    # Interactive demo
├── tests/
│   ├── test_physics.py
│   ├── test_model.py
│   ├── test_synthetic.py
│   └── test_training.py
├── notebooks/
│   ├── 01_introduction.ipynb      # Overview & examples
│   ├── 02_data_generation.ipynb   # Data pipeline
│   └── 03_inference.ipynb         # Using trained models
├── .gitignore
├── .pre-commit-config.yaml        # Code quality automation
├── LICENSE                        # Choose appropriate license
├── README.md                      # Main documentation
├── CITATION.cff                   # Citation metadata
├── CONTRIBUTING.md                # Contribution guidelines
├── pyproject.toml                 # Modern Python packaging
├── requirements.txt               # Pip requirements (fallback)
└── environment.yml                # Conda environment (optional)
```

---

## Key Files to Add

### 1. Comprehensive README.md

See example in next section.

### 2. LICENSE

**Recommendation**: MIT or Apache 2.0 for research code

**MIT License** (most permissive):
- ✅ Commercial use allowed
- ✅ Modification allowed
- ✅ Distribution allowed
- ⚠️ No warranty/liability

**Apache 2.0** (patent protection):
- ✅ Everything MIT has
- ✅ Patent grant included
- ⚠️ More complex

**GPL-3.0** (copyleft):
- ✅ Strong open-source enforcement
- ⚠️ Derivative works must be GPL
- ⚠️ Can limit commercial adoption

For academic research: **MIT is most common**

### 3. CITATION.cff

```yaml
# CITATION.cff
cff-version: 1.2.0
message: "If you use this software, please cite it as below."
authors:
  - family-names: "Your Last Name"
    given-names: "Your First Name"
    orcid: "https://orcid.org/0000-0000-0000-0000"
title: "pepTRM: Tiny Recursive Model for De Novo Peptide Sequencing"
version: 0.1.0
date-released: 2025-01-01
url: "https://github.com/yourusername/pepTRM"
repository-code: "https://github.com/yourusername/pepTRM"
keywords:
  - mass spectrometry
  - peptide sequencing
  - deep learning
  - transformer
  - recursive reasoning
license: MIT
```

### 4. pyproject.toml (Modern Python Packaging)

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "peptrm"
version = "0.1.0"
description = "Tiny Recursive Model for De Novo Peptide Sequencing"
readme = "README.md"
authors = [
    {name = "Your Name", email = "your.email@institution.edu"}
]
license = {text = "MIT"}
keywords = ["mass spectrometry", "peptide", "deep learning", "transformer"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Topic :: Scientific/Engineering :: Bio-Informatics",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
requires-python = ">=3.9"
dependencies = [
    "torch>=2.0.0",
    "numpy>=1.24.0",
    "omegaconf>=2.3.0",
    "tqdm>=4.65.0",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
# CUDA-specific optimizations
cuda = [
    # Add CUDA-specific packages if needed
]

# Development dependencies
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "black>=23.0.0",
    "isort>=5.12.0",
    "flake8>=6.0.0",
    "mypy>=1.0.0",
]

# Logging and experiment tracking
logging = [
    "wandb>=0.15.0",
    "tensorboard>=2.13.0",
]

# Jupyter notebooks
notebooks = [
    "jupyter>=1.0.0",
    "ipykernel>=6.0.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
]

# All extras
all = [
    "peptrm[cuda,dev,logging,notebooks]",
]

[project.urls]
Homepage = "https://github.com/yourusername/pepTRM"
Documentation = "https://peptrm.readthedocs.io"
Repository = "https://github.com/yourusername/pepTRM"
"Bug Tracker" = "https://github.com/yourusername/pepTRM/issues"

[tool.setuptools.packages.find]
where = ["."]
include = ["src*"]

[tool.black]
line-length = 100
target-version = ['py39', 'py310', 'py311']

[tool.isort]
profile = "black"
line_length = 100

[tool.mypy]
python_version = "3.9"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = false

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_functions = "test_*"
addopts = "-v --cov=src --cov-report=html --cov-report=term"
```

### 5. requirements.txt (Fallback for Pip)

```txt
# Core dependencies
torch>=2.0.0
numpy>=1.24.0
omegaconf>=2.3.0
tqdm>=4.65.0
pyyaml>=6.0

# Optional: experiment tracking
wandb>=0.15.0

# Optional: development
pytest>=7.0.0
black>=23.0.0
```

### 6. environment.yml (For Conda Users)

```yaml
name: peptrm
channels:
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pytorch>=2.0.0
  - pytorch-cuda=12.1  # For CUDA 12.1
  - numpy>=1.24.0
  - pip
  - pip:
    - omegaconf>=2.3.0
    - tqdm>=4.65.0
    - wandb>=0.15.0
    - pytest>=7.0.0
```

### 7. .gitignore

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# PyTorch
*.pth
*.pt
*.ckpt

# Jupyter
.ipynb_checkpoints
*.ipynb_checkpoints/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# Project specific
checkpoints/
checkpoints_optimized/
wandb/
*.log
*.csv

# Data (if large)
data/
*.mzML
*.mzXML

# OS
.DS_Store
Thumbs.db

# Environment
.env
.venv
env/
venv/
ENV/
```

### 8. GitHub Actions CI/CD

```yaml
# .github/workflows/tests.yml
name: Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest]
        python-version: ['3.9', '3.10', '3.11']

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"

    - name: Run tests
      run: |
        pytest tests/ -v --cov=src

    - name: Check code formatting
      run: |
        black --check src/ tests/
        isort --check src/ tests/

  # GPU tests (if you have self-hosted runners)
  test-gpu:
    runs-on: self-hosted  # Requires self-hosted runner with GPU
    if: github.event_name == 'push'

    steps:
    - uses: actions/checkout@v3

    - name: Run GPU tests
      run: |
        python -m pytest tests/ -v -m "gpu"
```

---

## Platform-Specific Installation Documentation

### docs/installation.md

```markdown
# Installation Guide

## Quick Start (Any Platform)

### Option 1: Pip (Recommended)
\`\`\`bash
git clone https://github.com/yourusername/pepTRM.git
cd pepTRM
pip install -e .
\`\`\`

### Option 2: Conda
\`\`\`bash
git clone https://github.com/yourusername/pepTRM.git
cd pepTRM
conda env create -f environment.yml
conda activate peptrm
\`\`\`

---

## Platform-Specific Setup

### 🟢 NVIDIA GPU (CUDA)

**Requirements:**
- CUDA 11.8+ or 12.1+
- NVIDIA Driver 525+
- PyTorch with CUDA support

**Installation:**
\`\`\`bash
# Check CUDA version
nvidia-smi

# Install PyTorch with CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install pepTRM
pip install -e .

# Optional: Install CUDA-specific optimizations
pip install -e ".[cuda]"
\`\`\`

**Verify Installation:**
\`\`\`python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
\`\`\`

**Recommended Configuration:**
- For RTX 4090 / A100: Use `configs/cuda_optimized.yaml`
- Enables mixed precision (AMP) and larger batch sizes

---

### 🍎 Apple Silicon (M1/M2/M3)

**Requirements:**
- macOS 12.3+
- Apple Silicon Mac (M1/M2/M3)
- PyTorch 2.0+ with MPS support

**Installation:**
\`\`\`bash
# Install PyTorch (includes MPS support)
pip install torch torchvision

# Install pepTRM
pip install -e .
\`\`\`

**Verify Installation:**
\`\`\`python
import torch
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")
\`\`\`

**Recommended Configuration:**
- Use `configs/mps.yaml`
- Batch size: 32-64 (depending on RAM)
- No mixed precision (MPS doesn't support AMP yet)

**Known Limitations:**
- MPS doesn't support all PyTorch operations (fallback to CPU automatic)
- No AMP support (uses FP32 only)
- Performance: ~50% of equivalent NVIDIA GPU

---

### 💻 CPU Only

**Requirements:**
- Any modern CPU (x86_64 or ARM64)
- 16GB+ RAM recommended

**Installation:**
\`\`\`bash
# Install PyTorch (CPU version)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install pepTRM
pip install -e .
\`\`\`

**Recommended Configuration:**
- Use `configs/cpu.yaml`
- Batch size: 8-16 (limited by RAM)
- Reduce model size for faster training

**Performance:**
- ~10-20x slower than GPU training
- Suitable for: testing, debugging, small-scale experiments

---

## Docker (Advanced)

### NVIDIA GPU
\`\`\`dockerfile
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace
COPY . .

RUN pip install -e .

ENTRYPOINT ["python", "scripts/train.py"]
\`\`\`

\`\`\`bash
docker build -t peptrm:cuda .
docker run --gpus all -v $(pwd)/checkpoints:/workspace/checkpoints peptrm:cuda
\`\`\`

---

## Troubleshooting

### CUDA Out of Memory
\`\`\`bash
# Reduce batch size
python scripts/train.py --batch_size 32
\`\`\`

### MPS Backend Error
\`\`\`bash
# Fallback to CPU for unsupported ops (automatic)
# Or disable MPS entirely:
export PYTORCH_ENABLE_MPS_FALLBACK=1
\`\`\`

### Import Errors
\`\`\`bash
# Make sure you're in the pepTRM directory
pip install -e .
\`\`\`
```

---

## Multi-Platform Testing Strategy

### Automated Testing (GitHub Actions)

1. **Matrix Testing**: Test on Ubuntu (CUDA) + macOS (MPS) + Windows (CPU)
2. **Python Versions**: Test 3.9, 3.10, 3.11
3. **Hardware**: CPU tests in CI, GPU tests on self-hosted runners

### Manual Testing Checklist

Before release, validate on each platform:

- [ ] **NVIDIA GPU (CUDA)**
  - [ ] Training runs without errors
  - [ ] Mixed precision works
  - [ ] Checkpointing works
  - [ ] Multi-GPU (if available)

- [ ] **Apple Silicon (MPS)**
  - [ ] Model loads and runs
  - [ ] Training completes
  - [ ] No MPS-specific errors
  - [ ] Performance acceptable

- [ ] **CPU**
  - [ ] Can complete small training run
  - [ ] Tests pass
  - [ ] Memory usage reasonable

---

## Documentation Best Practices

### What to Include

1. **README.md**
   - Project overview
   - Key features
   - Quick start
   - Citation
   - License

2. **docs/installation.md**
   - Platform-specific setup
   - Troubleshooting
   - Docker instructions

3. **docs/quickstart.md**
   - Simple training example
   - Using pre-trained models
   - Common use cases

4. **docs/training.md**
   - Configuration options
   - Curriculum learning
   - Hyperparameter tuning
   - Multi-GPU training

5. **docs/api.md**
   - Model architecture
   - Data formats
   - API reference

6. **CONTRIBUTING.md**
   - How to contribute
   - Code style
   - Testing requirements
   - PR process

---

## Release Checklist

Before making repository public:

- [ ] Add LICENSE file
- [ ] Write comprehensive README
- [ ] Add CITATION.cff
- [ ] Set up .gitignore
- [ ] Remove sensitive data/tokens
- [ ] Add installation docs
- [ ] Add example notebooks
- [ ] Set up CI/CD
- [ ] Tag initial release (v0.1.0)
- [ ] Create DOI (Zenodo)
- [ ] Announce on Twitter/X

---

## Maintenance Plan

### Versioning (Semantic Versioning)

```
v0.1.0 - Initial release
v0.2.0 - Added feature X
v0.2.1 - Bug fix
v1.0.0 - First stable release
```

### Support Policy

**Recommended**: State clearly what you officially support

Example:
```markdown
## Platform Support

| Platform | Status | Notes |
|----------|--------|-------|
| NVIDIA GPU (CUDA) | ✅ Officially Supported | Tested on RTX 4090, A100 |
| Apple Silicon (MPS) | ✅ Officially Supported | Tested on M1 Pro, M2 |
| CPU | ✅ Officially Supported | Any platform |
| AMD GPU (ROCm) | ⚪ Community Support | Not tested, PRs welcome |
| TPU | ❌ Not Supported | No plans currently |
```

---

## Community Building

### Communication Channels

1. **GitHub Issues**: Bug reports, feature requests
2. **GitHub Discussions**: Q&A, ideas, show-and-tell
3. **Discord/Slack** (optional): Real-time community chat
4. **Twitter/X**: Announcements, papers, results

### Engagement

- Respond to issues within 1-2 days
- Label issues (bug, enhancement, question, good first issue)
- Welcome contributions with clear guidelines
- Acknowledge contributors in README
- Publish roadmap for transparency

---

## Example Projects to Model After

### Excellent Multi-Platform Research Repos

1. **Hugging Face Transformers**
   - URL: github.com/huggingface/transformers
   - Platform support: CUDA, MPS, CPU, TPU
   - Documentation: Excellent
   - Community: Very active

2. **PyTorch Lightning**
   - URL: github.com/Lightning-AI/lightning
   - Hardware abstraction: Best-in-class
   - Testing: Comprehensive CI/CD

3. **Fairseq** (Meta)
   - URL: github.com/facebookresearch/fairseq
   - Research code quality: High
   - Documentation: Good

4. **MinGPT** (Karpathy)
   - URL: github.com/karpathy/minGPT
   - Simplicity: Excellent
   - Educational value: Very high

---

## Summary Recommendations

### ✅ Do This
- Support CUDA, MPS, CPU (you already have it!)
- Use automatic hardware detection (you already have it!)
- Provide platform-specific configs
- Write clear installation docs for each platform
- Set up GitHub Actions CI/CD
- Add comprehensive README
- Choose permissive license (MIT/Apache 2.0)
- Add CITATION.cff for academic credit

### ❌ Don't Do This
- Don't promise support for platforms you can't test
- Don't hardcode device selection
- Don't ignore documentation
- Don't skip testing before release
- Don't forget to add license

### 🎯 Priority for Release

1. **Must Have** (before going public):
   - LICENSE
   - README with installation instructions
   - Working examples/notebooks
   - Basic tests passing

2. **Should Have** (v0.1.0):
   - Platform-specific docs
   - CITATION.cff
   - CI/CD setup
   - Contributing guidelines

3. **Nice to Have** (future):
   - Pre-trained weights
   - Interactive demos
   - Video tutorials
   - Paper/preprint

