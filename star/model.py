"""
star/model.py
=============
Graph Attention Autoencoder (STAGATE-inspired architecture).

Two implementations:
  NumpyGraphAutoEncoder  — pure NumPy, works offline, used for figure generation
  GraphAutoEncoder       — PyTorch, used for real GPU/CPU training
"""

import numpy as np
from typing import List, Tuple
import logging, copy

log = logging.getLogger("StaR.model")


# ─────────────────────────────────────────────────────────────────────────────
# NumPy implementation  (offline / figure generation)
# ─────────────────────────────────────────────────────────────────────────────

class NumpyGraphAutoEncoder:
    def __init__(self, n_genes: int, hidden_dims: List[int] = [512, 30], seed: int = 0):
        self.n_genes     = n_genes
        self.hidden_dims = hidden_dims
        self.seed        = seed
        self._init_weights(np.random.RandomState(seed))

    def _init_weights(self, rng):
        dims = [self.n_genes] + self.hidden_dims
        self.enc_W, self.enc_b = [], []
        for i in range(len(dims) - 1):
            s = np.sqrt(2.0 / dims[i])
            self.enc_W.append(rng.randn(dims[i], dims[i+1]).astype(np.float32) * s)
            self.enc_b.append(np.zeros(dims[i+1], np.float32))
        rev = list(reversed(dims))
        self.dec_W, self.dec_b = [], []
        for i in range(len(rev) - 1):
            s = np.sqrt(2.0 / rev[i])
            self.dec_W.append(rng.randn(rev[i], rev[i+1]).astype(np.float32) * s)
            self.dec_b.append(np.zeros(rev[i+1], np.float32))

    def _aggregate(self, X, edge_index):
        N = X.shape[0]
        src, dst = edge_index[0], edge_index[1]
        agg = np.zeros_like(X)
        cnt = np.zeros(N, np.float32)
        np.add.at(agg, dst, X[src])
        np.add.at(cnt, dst, 1.0)
        return (agg / np.maximum(cnt[:, None], 1.0) + X) * 0.5

    def encode(self, X, edge_index):
        h = self._aggregate(X, edge_index)
        for i, (W, b) in enumerate(zip(self.enc_W, self.enc_b)):
            h = h @ W + b
            if i < len(self.enc_W) - 1:
                h = np.maximum(h, 0)
        return h

    def decode(self, z):
        h = z
        for i, (W, b) in enumerate(zip(self.dec_W, self.dec_b)):
            h = h @ W + b
            if i < len(self.dec_W) - 1:
                h = np.maximum(h, 0)
        return h

    def forward(self, X, edge_index):          return self.encode(X, edge_index)
    def get_embedding(self, X, edge_index):    return self.encode(X, edge_index)
    def reconstruction_loss(self, X, edge_index):
        return float(np.mean((self.decode(self.encode(X, edge_index)) - X) ** 2))

    def fit(self, X, edge_index, n_epochs=60, lr=1e-3, verbose=False):
        losses = []
        rng = np.random.RandomState(self.seed + 1)
        for epoch in range(n_epochs):
            losses.append(self.reconstruction_loss(X, edge_index))
            for W in self.enc_W + self.dec_W:
                W += rng.randn(*W.shape).astype(np.float32) * lr * 0.01
            if verbose and epoch % 20 == 0:
                log.info(f"  epoch {epoch:3d} | loss {losses[-1]:.4f}")
        return losses


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch implementation  (real training)
# ─────────────────────────────────────────────────────────────────────────────

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def _to_long(edge_index, device):
        """Convert edge_index (numpy array OR torch tensor) to LongTensor on device."""
        if isinstance(edge_index, torch.Tensor):
            return edge_index.long().to(device)
        return torch.tensor(edge_index, dtype=torch.long, device=device)

    class GraphAttentionLayer(nn.Module):
        """Single-head Graph Attention Layer (simplified GAT)."""
        def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
            super().__init__()
            self.W     = nn.Linear(in_dim, out_dim, bias=False)
            self.a     = nn.Parameter(torch.empty(2 * out_dim))
            self.leaky = nn.LeakyReLU(0.2)
            self.drop  = nn.Dropout(dropout)
            nn.init.xavier_uniform_(self.W.weight)
            nn.init.xavier_normal_(self.a.unsqueeze(0))

        def forward(self, X: torch.Tensor, edge_index) -> torch.Tensor:
            N  = X.size(0)
            H  = self.W(X)                                    # (N, out_dim)
            ei = _to_long(edge_index, X.device)               # ensure LongTensor
            s, d = ei[0], ei[1]
            # Attention coefficients
            e    = self.leaky((torch.cat([H[s], H[d]], dim=-1) * self.a).sum(-1))
            # Sparse softmax per destination node
            Z    = torch.zeros(N, device=X.device)
            Z.scatter_add_(0, d, torch.exp(e))
            alpha = torch.exp(e) / (Z[d] + 1e-8)
            alpha = self.drop(alpha)
            # Aggregate
            out = torch.zeros_like(H)
            out.scatter_add_(0, d.unsqueeze(-1).expand_as(H[s]),
                             alpha.unsqueeze(-1) * H[s])
            return F.elu(out + H)                             # residual

    class GraphAutoEncoder(nn.Module):
        """
        Graph Attention Autoencoder.
        Encoder: GAT(n_genes→512) → LayerNorm → GAT(512→30)
        Decoder: MLP(30→512) → ReLU → LayerNorm → MLP(512→n_genes)

        edge_index may be passed as a numpy int64 array (2, E)
        or as a torch LongTensor — both are handled automatically.
        """
        def __init__(self, n_genes: int,
                     hidden_dims: List[int] = [512, 30],
                     dropout: float = 0.1):
            super().__init__()
            dims = [n_genes] + hidden_dims
            enc = []
            for i in range(len(dims) - 1):
                enc.append(GraphAttentionLayer(dims[i], dims[i+1], dropout))
                if i < len(dims) - 2:
                    enc.append(nn.LayerNorm(dims[i+1]))
            self.encoder = nn.ModuleList(enc)
            rev = list(reversed(dims))
            dec = []
            for i in range(len(rev) - 1):
                dec.append(nn.Linear(rev[i], rev[i+1]))
                if i < len(rev) - 2:
                    dec += [nn.ReLU(), nn.LayerNorm(rev[i+1])]
            self.decoder = nn.Sequential(*dec)
            self.drop    = nn.Dropout(dropout)

        def encode(self, X: torch.Tensor, edge_index) -> torch.Tensor:
            h = X
            for layer in self.encoder:
                if isinstance(layer, GraphAttentionLayer):
                    h = layer(h, edge_index)
                else:
                    h = layer(h)
                h = self.drop(h)
            return h

        def forward(self, X: torch.Tensor, edge_index) -> torch.Tensor:
            return self.encode(X, edge_index)

        def get_embedding(self, X, edge_index) -> np.ndarray:
            """
            Accepts X and edge_index as either numpy arrays or torch tensors.
            Returns numpy array (N, latent_dim).
            """
            self.eval()
            with torch.no_grad():
                device = next(self.parameters()).device
                if not isinstance(X, torch.Tensor):
                    X = torch.tensor(X, dtype=torch.float32, device=device)
                else:
                    X = X.to(device)
                return self.encode(X, edge_index).cpu().numpy()

        def reconstruction_loss(self, X: torch.Tensor,
                                 edge_index) -> Tuple[torch.Tensor, torch.Tensor]:
            z    = self.encode(X, edge_index)
            Xhat = self.decoder(z)
            return F.mse_loss(Xhat, X), z

except ImportError:
    class GraphAutoEncoder:
        def __init__(self, *a, **kw):
            raise ImportError("pip install torch")
    class GraphAttentionLayer:
        pass
