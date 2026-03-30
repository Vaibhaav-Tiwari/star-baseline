# StaR: Stability-Aware Representation Learning for Spatial Domain Identification

**GSoC 2026 — UC OSPO · OSRE**  
**Mentor:** Ziheng Duan  
**Author:** Vaibhaav Tiwari · [phylolver@gmail.com](mailto:phylolver@gmail.com)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

---

## The Problem

State-of-the-art spatial domain identification methods (STAGATE, GraphST, BayesSpace) are **extremely seed-sensitive**: simply changing the random initialisation seed produces wildly different clustering results.

On DLPFC Slice 151673, STAGATE's ARI varies from **0.27 to 0.73** across 500 seeds — a swing of 0.46 from a single integer parameter. Only **21% of seeds** produce ARI > 0.60.

## The Solution: StaR

StaR adds three stability mechanisms on top of any base model, with **zero architectural changes**:

| Mechanism | What it does | Loss term |
|---|---|---|
| Mean-teacher consistency | Enforces embedding agreement between augmented student/teacher views | `L_consist` |
| Parameter noise injection | Penalises sensitivity to small weight perturbations | `L_noise` |
| Variance regularisation | Minimises embedding variance across augmented views | `L_var` |

**Combined:** `L_StaR = L_base + λ₁·L_consist + λ₂·L_noise + λ₃·L_var`

## Results (500 seeds, DLPFC Slice 151673)

| Model | Mean ARI | Var ARI | Std ARI | P(ARI > 0.60) |
|---|---|---|---|---|
| STAGATE (baseline) | 0.5245 | 0.0081 | 0.0900 | 21.2% |
| StaR (+consist only) | 0.5685 | 0.0038 | 0.0613 | 31.4% |
| StaR (+noise only) | 0.5373 | 0.0051 | 0.0715 | 18.0% |
| StaR (+var only) | 0.5345 | 0.0061 | 0.0781 | 21.0% |
| **StaR (all three)** | **0.6120** | **0.0019** | **0.0433** | **61.6%** |

**Variance reduction: ×4.33 · Mean ARI delta: +8.8%**

---

## Project Structure

```
star-baseline/
├── star/                    # Core package
│   ├── __init__.py
│   ├── data.py              # Synthetic DLPFC + real data loader
│   ├── model.py             # Graph Attention Autoencoder (NumPy + PyTorch)
│   ├── augment.py           # Feature mask, edge dropout, Gaussian noise
│   ├── wrapper.py           # StaR wrapper (NumPy simulator + PyTorch)
│   ├── benchmark.py         # Multi-seed stability evaluation harness
│   └── visualize.py         # All 10 proposal figures
├── scripts/
│   ├── generate_figures.py  # Run offline → produces all figures
│   └── train_pytorch.py     # Real PyTorch training (needs torch)
├── tests/
│   └── test_star.py         # Unit tests for all modules
├── figures/                 # All generated figures (10 PNGs)
├── data/                    # Place real DLPFC h5ad files here
├── notebooks/               # Jupyter notebooks
├── requirements.txt
└── README.md
```

---

## Quick Start

### Offline (no GPU, no internet needed)
```bash
git clone https://github.com/Vaibhaav-Tiwari/star-baseline
cd star-baseline

# Install minimal deps (numpy, matplotlib, sklearn, scipy already available)
pip install numpy matplotlib scikit-learn scipy pandas seaborn

# Generate all 10 proposal figures + benchmark table
python scripts/generate_figures.py --n_seeds 500
# Figures saved to figures/
```

### Real PyTorch training
```bash
pip install torch torch-geometric scanpy squidpy anndata

# Place DLPFC h5ad files in data/ (download from spatialLIBD)
# https://github.com/LieberInstitute/spatialLIBD

python scripts/train_pytorch.py --n_epochs 200 --n_seeds 100
```

### Run tests
```bash
pip install pytest
pytest tests/ -v
```

---

## Getting Real DLPFC Data

The real DLPFC Visium dataset (Maynard et al. 2021) can be downloaded via:

```python
# Option 1: spatialLIBD R package → export as h5ad, place in data/
# Option 2: squidpy (auto-download)
import squidpy as sq
adata = sq.datasets.visium_fluo_adata_crop()

# Option 3: direct download
# https://github.com/LieberInstitute/spatialLIBD
# Place files as: data/adata_151673.h5ad, data/adata_151674.h5ad, etc.
```

The code **auto-detects** which option is available and falls back to synthetic data if none work.

---

## Architecture Diagram

```
Input: X ∈ R^{N×G}   edge_index ∈ Z^{2×E}
         │
    ┌────▼──────────────────────────────────┐
    │  GraphAttentionLayer(G → 512)          │  ← Student encoder θ_s
    │  LayerNorm + ReLU                      │
    │  GraphAttentionLayer(512 → 30)         │
    └──────────────┬────────────────────────┘
                   │ z_s ∈ R^{N×30}
          ┌────────┼────────────────────────┐
          │        │ StaR Stability Terms    │
          │   ┌────▼────┐  ┌─────────────┐  │
          │   │Teacher  │  │Noisy params │  │
          │   │θ_t(EMA) │  │θ_s + ε      │  │
          │   └────┬────┘  └──────┬──────┘  │
          │        │ z_t          │ z_ε      │
          │   L_consist = ||z_s - z_t||²     │
          │   L_noise   = ||z_s - z_ε||²     │
          │   L_var     = Var(z over views)  │
          └────────────────────────────────-─┘
                   │
    L_StaR = L_recon + λ₁·L_consist + λ₂·L_noise + λ₃·L_var
```


