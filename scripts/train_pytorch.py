"""
scripts/train_pytorch.py
========================
Real PyTorch training script for StaR.

Requirements: pip install torch scanpy squidpy anndata

Usage:
    python scripts/train_pytorch.py --n_epochs 200 --n_seeds 100
    python scripts/train_pytorch.py --n_epochs 200 --n_seeds 500   # full run
    python scripts/train_pytorch.py --n_epochs 50  --n_seeds 10    # quick test

Notes:
    - Automatically downloads DLPFC data via squidpy if not present in data/
    - Falls back to synthetic dataset if squidpy unavailable
    - Works on CPU (slow but correct) and CUDA GPU (fast)
    - Results saved to star_pytorch_results.csv
"""

import sys, argparse, logging, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("StaR.train")


def train(args):
    # ── Imports ───────────────────────────────────────────────────────────────
    try:
        import torch
        from torch.optim import Adam
        from torch.optim.lr_scheduler import CosineAnnealingLR
    except ImportError:
        log.error("PyTorch not installed.  Run: pip install torch")
        return

    from star.data      import load_dlpfc
    from star.model     import GraphAutoEncoder
    from star.wrapper   import StaRWrapper
    from star.augment   import SpatialAugmentor
    from star.benchmark import StabilityBenchmark

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")

    # ── Load data ─────────────────────────────────────────────────────────────
    dataset = load_dlpfc(data_dir="data/", sample_id=args.sample_id)
    log.info(f"Dataset: {dataset.n_spots} spots · "
             f"{dataset.n_genes} genes · {dataset.n_layers} layers")

    # Pre-convert to tensors once (used inside training loops)
    X_t  = torch.tensor(dataset.X,          dtype=torch.float32).to(device)
    ei_t = torch.tensor(dataset.edge_index, dtype=torch.long   ).to(device)

    # ── Baseline factory ──────────────────────────────────────────────────────
    def make_baseline(seed: int):
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = GraphAutoEncoder(dataset.n_genes, [512, 30]).to(device)
        opt   = Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
        sched = CosineAnnealingLR(opt, T_max=args.n_epochs)

        model.train()
        for epoch in range(args.n_epochs):
            opt.zero_grad()
            loss, _ = model.reconstruction_loss(X_t, ei_t)
            loss.backward()
            opt.step()
            sched.step()

        # get_embedding() already handles numpy→tensor conversion internally
        return model

    # ── StaR factory ──────────────────────────────────────────────────────────
    def make_star(seed: int):
        torch.manual_seed(seed)
        np.random.seed(seed)

        base = GraphAutoEncoder(dataset.n_genes, [512, 30]).to(device)
        star = StaRWrapper(
            base,
            ema_alpha = args.ema_alpha,
            noise_std = args.noise_std,
            lambda1   = args.lambda1,
            lambda2   = args.lambda2,
            lambda3   = args.lambda3,
            augmentor = SpatialAugmentor(
                feat_mask_rate = 0.10,
                edge_drop_rate = 0.10,
                feat_noise_std = 0.05,
            ),
        )
        opt   = Adam(star.student.parameters(), lr=args.lr, weight_decay=1e-5)
        sched = CosineAnnealingLR(opt, T_max=args.n_epochs)

        star.student.train()
        for epoch in range(args.n_epochs):
            opt.zero_grad()
            loss, logs = star.training_step(X_t, ei_t, step=epoch)
            loss.backward()
            opt.step()
            sched.step()
            star.update_teacher()

            if epoch % max(1, args.n_epochs // 4) == 0:
                log.info(
                    f"  seed={seed}  epoch={epoch:3d}  "
                    f"total={logs['total']:.4f}  base={logs['base']:.4f}  "
                    f"consist={logs['consist']:.4f}"
                )

        return star

    # ── Run benchmark ─────────────────────────────────────────────────────────
    bench = StabilityBenchmark(
        n_seeds       = args.n_seeds,
        n_clusters    = dataset.n_layers,
        kmeans_n_init = 10,
    )

    log.info("\nRunning baseline benchmark ...")
    res_base = bench.run(make_baseline, dataset, label="STAGATE (baseline)")

    log.info("\nRunning StaR benchmark ...")
    res_star = bench.run(make_star, dataset, label="StaR (all three)")

    bench.print_table([res_base, res_star])

    # ── Save results ──────────────────────────────────────────────────────────
    import pandas as pd
    df = pd.DataFrame([
        {k: v for k, v in r.items() if k not in ("ari_list", "nmi_list")}
        for r in [res_base, res_star]
    ])
    df.to_csv("star_pytorch_results.csv", index=False)
    log.info("Saved → star_pytorch_results.csv")

    # ── Generate updated figures with real results ────────────────────────────
    if args.make_figures:
        log.info("\nGenerating figures from real results ...")
        from star.visualize import plot_violin, plot_variance_reduction
        from pathlib import Path
        Path("figures").mkdir(exist_ok=True)
        plot_violin([res_base, res_star], Path("figures/violin_ari_real.png"))
        plot_variance_reduction([res_base, res_star],
                                 Path("figures/variance_reduction_real.png"))
        log.info("Figures saved → figures/violin_ari_real.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="StaR PyTorch training — real data benchmark"
    )
    p.add_argument("--sample_id",    default="151673",
                   help="DLPFC slice ID (default 151673)")
    p.add_argument("--n_epochs",     type=int,   default=200,
                   help="Training epochs per seed (default 200)")
    p.add_argument("--n_seeds",      type=int,   default=50,
                   help="Number of random seeds to benchmark (default 50)")
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--lambda1",      type=float, default=0.5,
                   help="Weight for consistency loss")
    p.add_argument("--lambda2",      type=float, default=0.3,
                   help="Weight for noise loss")
    p.add_argument("--lambda3",      type=float, default=0.1,
                   help="Weight for variance loss")
    p.add_argument("--ema_alpha",    type=float, default=0.99,
                   help="EMA decay rate for teacher network")
    p.add_argument("--noise_std",    type=float, default=0.01,
                   help="Std-dev of parameter-space noise injection")
    p.add_argument("--make_figures", action="store_true",
                   help="Generate violin + variance figures after benchmarking")
    train(p.parse_args())
