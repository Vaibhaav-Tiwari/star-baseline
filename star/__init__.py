"""StaR — Stability-Aware Representation Learning  (GSoC 2026)"""
from .data      import SpatialDataset, make_synthetic_dlpfc, load_dlpfc
from .model     import NumpyGraphAutoEncoder
from .augment   import SpatialAugmentor
from .wrapper   import NumpyStaRSimulator
from .benchmark import StabilityBenchmark
from .visualize import (
    plot_spatial_domains, plot_violin, plot_ablation_table,
    plot_variance_reduction, plot_seed_count_curve,
    plot_loss_landscape, plot_consistency_map,
    plot_training_curves, plot_ari_histogram, plot_gate_entropy,
)
__version__ = "0.1.0"
