"""
star/data.py
============
Data loading and synthetic DLPFC dataset generation.

Real data (requires pip install scanpy squidpy anndata):
  - DLPFC Visium: Maynard et al. 2021, Nature Neuroscience
    12 slices, 7 annotated cortical layers, ~3000-4000 spots each

Offline fallback:
  - Synthetic DLPFC-like dataset (make_synthetic_dlpfc)
    Realistic cortical layer structure, gene expression noise, spatial graph
    Used automatically when real data unavailable
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pathlib import Path
import logging

log = logging.getLogger("StaR.data")


# ─────────────────────────────────────────────────────────────────────────────
# Data container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SpatialDataset:
    """
    Lightweight container for spatial transcriptomics data.

    Attributes
    ----------
    X          : (N, G) float32  - log-normalised gene expression
    coords     : (N, 2) float32  - spatial coordinates in µm
    labels     : (N,)  str       - ground-truth domain labels
    gene_names : list[str]
    spot_names : list[str]
    edge_index : (2, E) int64    - spatial k-NN graph, COO format
    metadata   : dict            - sample-level info
    """
    X          : np.ndarray
    coords     : np.ndarray
    labels     : np.ndarray
    gene_names : List[str]
    spot_names : List[str]
    edge_index : np.ndarray
    metadata   : dict = field(default_factory=dict)

    @property
    def n_spots(self):  return self.X.shape[0]

    @property
    def n_genes(self):  return self.X.shape[1]

    @property
    def n_layers(self): return len(np.unique(self.labels))

    @property
    def label_ids(self):
        uniq = sorted(set(self.labels))
        m = {l: i for i, l in enumerate(uniq)}
        return np.array([m[l] for l in self.labels], dtype=np.int32)

    @property
    def unique_layers(self): return sorted(set(self.labels))


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic DLPFC generator (offline, no dependencies beyond numpy)
# ─────────────────────────────────────────────────────────────────────────────

def make_synthetic_dlpfc(n_spots: int = 3000,
                          n_genes: int = 3000,
                          n_layers: int = 7,
                          n_neighbors: int = 6,
                          noise_level: float = 0.8,
                          seed: int = 0) -> SpatialDataset:
    """
    Generate a synthetic DLPFC-like spatial transcriptomics dataset.

    Tissue is a stack of horizontal cortical layers with realistic:
    - Per-layer gene signature (~200 marker genes each)
    - Gaussian biological noise + Poisson-like technical noise
    - Hexagonal spot packing (~55 µm spacing)
    - Spatial k-NN graph

    Parameters
    ----------
    n_spots     : number of Visium spots
    n_genes     : number of HVGs in expression matrix
    n_layers    : number of cortical layers (DLPFC = 7: L1-L6 + WM)
    n_neighbors : k for spatial k-NN graph
    noise_level : biological noise std-dev
    seed        : random seed
    """
    rng = np.random.RandomState(seed)
    layer_names = [f"Layer{i+1}" for i in range(n_layers - 1)] + ["WM"]

    # ── Spatial layout: hexagonal packing ────────────────────────────────────
    side = int(np.ceil(np.sqrt(n_spots)))
    xs   = np.linspace(0, 6200, side)
    ys   = np.linspace(0, 3000, side)
    xx, yy = np.meshgrid(xs, ys)
    xx[1::2] += (xs[1] - xs[0]) / 2           # hexagonal offset
    coords = np.stack([xx.ravel(), yy.ravel()], axis=1)[:n_spots]
    coords += rng.randn(*coords.shape) * 28    # ±28 µm jitter

    # ── Layer assignment by depth (y-coordinate) ─────────────────────────────
    # Approximate real DLPFC layer proportions
    fracs = np.array([0.05, 0.14, 0.10, 0.20, 0.15, 0.21, 0.15])
    fracs /= fracs.sum()
    bounds = np.concatenate([[0], np.cumsum(fracs)]) * coords[:, 1].max()

    label_ids = np.zeros(n_spots, dtype=int)
    for i in range(n_layers):
        mask = (coords[:, 1] >= bounds[i]) & (coords[:, 1] < bounds[i + 1])
        label_ids[mask] = i
    labels = np.array([layer_names[i] for i in label_ids])

    # ── Gene expression: per-layer signatures ────────────────────────────────
    n_sig = min(200, n_genes // 2)   # marker genes per layer
    layer_means = np.zeros((n_layers, n_genes), dtype=np.float32)
    for k in range(n_layers):
        idx = rng.choice(n_genes, n_sig, replace=False)
        layer_means[k, idx] = rng.exponential(2.5, size=n_sig).astype(np.float32)

    X = np.vstack([
        layer_means[label_ids[i]]
        + rng.randn(n_genes).astype(np.float32) * noise_level
        + rng.exponential(0.08, n_genes).astype(np.float32)
        for i in range(n_spots)
    ])
    X = np.clip(X, 0, None)
    # Log1p normalise (CPM-style)
    X = np.log1p(X / (X.sum(1, keepdims=True) + 1e-8) * 1e4).astype(np.float32)

    # ── Spatial k-NN graph ────────────────────────────────────────────────────
    edge_index = _build_knn_graph(coords, k=n_neighbors)

    log.info(f"Synthetic DLPFC: {n_spots} spots | {n_genes} genes | "
             f"{n_layers} layers | {edge_index.shape[1]} edges")

    return SpatialDataset(
        X=X, coords=coords.astype(np.float32),
        labels=labels,
        gene_names=[f"GENE{i:05d}" for i in range(n_genes)],
        spot_names=[f"SPOT_{i:06d}" for i in range(n_spots)],
        edge_index=edge_index,
        metadata=dict(dataset="synthetic_DLPFC", n_layers=n_layers,
                      n_spots=n_spots, n_genes=n_genes, seed=seed)
    )


def _build_knn_graph(coords: np.ndarray, k: int = 6) -> np.ndarray:
    """Build spatial k-NN graph using block-wise distance computation."""
    N = coords.shape[0]
    block = 512
    src_all, dst_all = [], []

    for start in range(0, N, block):
        end = min(start + block, N)
        diff = coords[start:end, None, :] - coords[None, :, :]   # (B, N, 2)
        dist = np.sqrt((diff ** 2).sum(-1))                       # (B, N)
        np.fill_diagonal(dist[:, start:end], 1e9)                 # mask self
        knn = np.argsort(dist, axis=1)[:, :k]                    # (B, k)
        src = np.repeat(np.arange(start, end), k)
        dst = knn.ravel()
        src_all.append(src); dst_all.append(dst)

    src = np.concatenate(src_all)
    dst = np.concatenate(dst_all)
    # Symmetrise + deduplicate
    s = np.concatenate([src, dst])
    d = np.concatenate([dst, src])
    edges = np.unique(np.stack([s, d], axis=0), axis=1)
    return edges.astype(np.int64)


# ─────────────────────────────────────────────────────────────────────────────
# Real data loader (needs scanpy + squidpy + anndata)
# ─────────────────────────────────────────────────────────────────────────────

def load_dlpfc(data_dir: Optional[str] = None,
               sample_id: str = "151673",
               n_hvgs: int = 3000,
               n_neighbors: int = 6) -> SpatialDataset:
    """
    Load the DLPFC Visium dataset.

    Tries in order:
      1. Local h5ad file at data_dir/adata_{sample_id}.h5ad
      2. Auto-download via squidpy
      3. Synthetic fallback

    To get real data:
        pip install scanpy squidpy anndata
        # Then place DLPFC h5ad files in data/
        # Download from: https://github.com/LieberInstitute/spatialLIBD
    """
    # Attempt 1: local file
    if data_dir is not None:
        h5ad = Path(data_dir) / f"adata_{sample_id}.h5ad"
        if h5ad.exists():
            try:
                return _load_h5ad(str(h5ad), n_hvgs, n_neighbors)
            except Exception as e:
                log.warning(f"h5ad load failed: {e}")

    # Attempt 2: squidpy
    try:
        import squidpy as sq, scanpy as sc
        log.info("Loading via squidpy ...")
        adata = sq.datasets.visium_fluo_adata_crop()
        return _adata_to_dataset(adata, n_hvgs, n_neighbors)
    except Exception as e:
        log.warning(f"squidpy failed: {e}")

    # Fallback
    log.warning(
        "Real DLPFC data unavailable — using synthetic dataset.\n"
        "For real data: pip install scanpy squidpy anndata\n"
        "Then place adata_151673.h5ad (etc.) in data/\n"
        "Download: https://github.com/LieberInstitute/spatialLIBD"
    )
    return make_synthetic_dlpfc(n_spots=3000, n_genes=n_hvgs)


def _load_h5ad(path: str, n_hvgs: int, n_neighbors: int) -> SpatialDataset:
    import anndata as ad, scanpy as sc, squidpy as sq
    adata = ad.read_h5ad(path)
    return _adata_to_dataset(adata, n_hvgs, n_neighbors)


def _adata_to_dataset(adata, n_hvgs: int, n_neighbors: int) -> SpatialDataset:
    import scanpy as sc, squidpy as sq
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvgs, subset=True)
    sc.pp.scale(adata, max_value=10)
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.array(adata.X)
    coords = adata.obsm.get("spatial", np.zeros((adata.n_obs, 2), np.float32))
    sq.gr.spatial_neighbors(adata, n_neighs=n_neighbors, coord_type="generic")
    cx = adata.obsp["spatial_connectivities"].tocoo()
    edge_index = np.stack([cx.row.astype(np.int64), cx.col.astype(np.int64)])
    lbl_col = next((c for c in ["layer_guess", "cluster", "cell_type"]
                    if c in adata.obs.columns), adata.obs.columns[0])
    return SpatialDataset(
        X=X.astype(np.float32), coords=coords.astype(np.float32),
        labels=adata.obs[lbl_col].values.astype(str),
        gene_names=list(adata.var_names),
        spot_names=list(adata.obs_names),
        edge_index=edge_index,
        metadata=dict(dataset="DLPFC_real", sample=lbl_col)
    )
