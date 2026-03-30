"""
star/wrapper.py
===============
StaR wrapper — adds three stability mechanisms on top of any base model.

NumpyStaRSimulator : offline / figure generation (no torch needed)
StaRWrapper        : PyTorch, real training

Three stability mechanisms (proposal Eqs. 2–4):
  L_consist : mean-teacher consistency loss
  L_noise   : parameter-space noise injection
  L_var     : embedding variance regularisation

L_StaR = L_base + λ1·L_consist + λ2·L_noise + λ3·L_var
"""

import numpy as np
import copy
from typing import Dict, List, Tuple
import logging

from .augment import SpatialAugmentor

log = logging.getLogger("StaR.wrapper")


# ─────────────────────────────────────────────────────────────────────────────
# NumPy simulator  (offline / figure generation)
# ─────────────────────────────────────────────────────────────────────────────

class NumpyStaRSimulator:
    """
    Simulates StaR training in pure NumPy.
    Uses EMA weight averaging + augmentation to mimic the stability effect.
    For real training use StaRWrapper (PyTorch).
    """

    def __init__(self, base_model,
                 ema_alpha: float = 0.99,
                 noise_std: float = 0.01,
                 lambda1:   float = 0.5,
                 lambda2:   float = 0.3,
                 lambda3:   float = 0.1,
                 augmentor: SpatialAugmentor = None):
        self.student   = base_model
        self.teacher   = copy.deepcopy(base_model)
        self.ema_alpha = ema_alpha
        self.noise_std = noise_std
        self.lambda1   = lambda1
        self.lambda2   = lambda2
        self.lambda3   = lambda3
        self.aug       = augmentor or SpatialAugmentor()
        self.loss_history: List[Dict] = []

    def _ema_update(self):
        a = self.ema_alpha
        for sw, tw in zip(self.student.enc_W, self.teacher.enc_W):
            tw[:] = a * tw + (1 - a) * sw
        for sb, tb in zip(self.student.enc_b, self.teacher.enc_b):
            tb[:] = a * tb + (1 - a) * sb

    def fit(self, X, ei, n_epochs=60, lr=1e-3, verbose=False):
        rng = np.random.RandomState(self.student.seed + 99)
        self.loss_history = []
        for epoch in range(n_epochs):
            base = self.student.reconstruction_loss(X, ei)
            (Xs, eis), (Xt, eit) = self.aug.student_teacher_views(X, ei, epoch)
            zs = self.student.encode(Xs, eis)
            zt = self.teacher.encode(Xt, eit)
            consist = float(np.mean((zs - zt) ** 2))
            z_clean = self.student.encode(X, ei)
            saved   = [w.copy() for w in self.student.enc_W]
            for w in self.student.enc_W:
                w += rng.randn(*w.shape).astype(np.float32) * self.noise_std
            z_noisy = self.student.encode(X, ei)
            for w, sw in zip(self.student.enc_W, saved):
                w[:] = sw
            noise    = float(np.mean((z_noisy - z_clean) ** 2))
            views    = [self.student.encode(*self.aug.augment(X, ei, rng)) for _ in range(3)]
            var_loss = float(np.stack(views, 0).var(0).mean())
            total    = base + self.lambda1*consist + self.lambda2*noise + self.lambda3*var_loss
            self.loss_history.append(dict(epoch=epoch, base=base, consist=consist,
                                          noise=noise, var=var_loss, total=total))
            self.student.fit(X, ei, n_epochs=1, lr=lr)
            self._ema_update()
            if verbose and epoch % 20 == 0:
                log.info(f"  epoch {epoch:3d}  total={total:.4f}  consist={consist:.4f}")
        return self.loss_history

    def get_embedding(self, X, edge_index):
        return self.teacher.encode(X, edge_index)


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch wrapper  (real training)
# ─────────────────────────────────────────────────────────────────────────────

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class StaRWrapper(nn.Module):
        """
        Model-agnostic StaR wrapper for PyTorch.

        Accepts X and edge_index as either:
          - torch.Tensor (already on device)  ← preferred, fastest
          - numpy ndarray                     ← converted automatically

        Parameters
        ----------
        base_model : nn.Module
            Must expose forward(X, edge_index) → z
            and reconstruction_loss(X, edge_index) → (loss, z)
        ema_alpha  : EMA decay for teacher (0.99–0.999)
        noise_std  : parameter-space noise std-dev
        lambda1/2/3: weights for consist / noise / variance losses
        augmentor  : SpatialAugmentor instance

        Training
        --------
        for epoch in range(n_epochs):
            loss, logs = star.training_step(X_tensor, ei_tensor, step=epoch)
            opt.zero_grad(); loss.backward(); opt.step()
            star.update_teacher()

        Inference
        ---------
        z = star.get_embedding(X_numpy_or_tensor, ei_numpy_or_tensor)
        """

        def __init__(self, base_model: nn.Module,
                     ema_alpha: float = 0.99,
                     noise_std: float = 0.01,
                     lambda1:   float = 0.5,
                     lambda2:   float = 0.3,
                     lambda3:   float = 0.1,
                     augmentor: SpatialAugmentor = None):
            super().__init__()
            self.student   = base_model
            self.teacher   = copy.deepcopy(base_model)
            for p in self.teacher.parameters():
                p.requires_grad_(False)
            self.ema_alpha = ema_alpha
            self.noise_std = noise_std
            self.lambda1   = lambda1
            self.lambda2   = lambda2
            self.lambda3   = lambda3
            self.aug       = augmentor or SpatialAugmentor()

        # ── Helpers ───────────────────────────────────────────────────────────

        def _device(self):
            return next(self.student.parameters()).device

        def _to_tensor(self, X, ei):
            """Ensure X is float32 tensor and ei is long tensor, both on device."""
            dev = self._device()
            if not isinstance(X, torch.Tensor):
                X  = torch.tensor(X,  dtype=torch.float32, device=dev)
            else:
                X  = X.float().to(dev)
            if not isinstance(ei, torch.Tensor):
                ei = torch.tensor(ei, dtype=torch.long, device=dev)
            else:
                ei = ei.long().to(dev)
            return X, ei

        def _aug_view(self, X: torch.Tensor, ei: torch.Tensor,
                      seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
            """Apply numpy augmentation and return tensors on same device."""
            dev   = X.device
            X_np  = X.detach().cpu().numpy()
            ei_np = ei.cpu().numpy()
            rng   = np.random.RandomState(seed)
            Xa, ea = self.aug.augment(X_np, ei_np, rng)
            return (torch.tensor(Xa,  dtype=torch.float32, device=dev),
                    torch.tensor(ea,  dtype=torch.long,    device=dev))

        # ── EMA ───────────────────────────────────────────────────────────────

        @torch.no_grad()
        def update_teacher(self):
            """θ_t ← α·θ_t + (1−α)·θ_s"""
            a = self.ema_alpha
            for ps, pt in zip(self.student.parameters(),
                               self.teacher.parameters()):
                pt.data.mul_(a).add_(ps.data, alpha=1.0 - a)

        # ── Loss terms ────────────────────────────────────────────────────────

        def _consistency_loss(self, X, ei, step: int) -> torch.Tensor:
            Xs, es = self._aug_view(X, ei, step * 2)
            Xt, et = self._aug_view(X, ei, step * 2 + 1)
            hs = self.student(Xs, es)
            with torch.no_grad():
                ht = self.teacher(Xt, et)
            return F.mse_loss(hs, ht)

        def _noise_loss(self, X, ei) -> torch.Tensor:
            with torch.no_grad():
                h_clean = self.student(X, ei).detach()
            # Temporarily perturb student parameters
            saved = {n: p.data.clone() for n, p in self.student.named_parameters()}
            for p in self.student.parameters():
                p.data.add_(torch.randn_like(p) * self.noise_std)
            h_noisy = self.student(X, ei)
            # Restore
            for n, p in self.student.named_parameters():
                p.data.copy_(saved[n])
            return F.mse_loss(h_noisy, h_clean)

        def _variance_loss(self, X, ei, step: int, k: int = 3) -> torch.Tensor:
            views = [self.student(*self._aug_view(X, ei, step * k + i))
                     for i in range(k)]
            return torch.stack(views, dim=0).var(dim=0).mean()

        # ── Training step ─────────────────────────────────────────────────────

        def training_step(self, X, ei,
                          step: int = 0) -> Tuple[torch.Tensor, Dict]:
            """
            Compute full StaR loss. Call .backward() on the returned tensor.

            X, ei : torch tensors (float32 / long) already on device.
                    Numpy arrays are also accepted but tensors are faster.
            """
            X, ei = self._to_tensor(X, ei)

            base_loss, _ = self.student.reconstruction_loss(X, ei)
            lc = self._consistency_loss(X, ei, step)
            ln = self._noise_loss(X, ei)
            lv = self._variance_loss(X, ei, step)

            total = (base_loss
                     + self.lambda1 * lc
                     + self.lambda2 * ln
                     + self.lambda3 * lv)

            return total, dict(
                base=base_loss.item(), consist=lc.item(),
                noise=ln.item(),       var=lv.item(),
                total=total.item(),
            )

        # ── Inference ─────────────────────────────────────────────────────────

        @torch.no_grad()
        def get_embedding(self, X, edge_index) -> np.ndarray:
            """
            Stable inference using the teacher network.
            Accepts numpy arrays or torch tensors.
            Returns numpy array (N, latent_dim).
            """
            X, ei = self._to_tensor(X, edge_index)
            self.teacher.eval()
            return self.teacher(X, ei).cpu().numpy()

except ImportError:
    class StaRWrapper:
        def __init__(self, *a, **kw):
            raise ImportError(
                "PyTorch required. Install: pip install torch\n"
                "For offline use: NumpyStaRSimulator"
            )
