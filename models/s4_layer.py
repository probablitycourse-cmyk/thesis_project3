"""
Trainable S4 layer with optional cross-channel Theta reparameterisation.

Three modes are supported:

  "none"   standard S4 -- the H internal channels evolve independently
  "const"  a single learned mixing matrix Theta shared by every Legendre order
  "order"  Theta^(n) = expm(n * G), i.e. a different mixing per Legendre order

The synthesis view behind "order" is

    u_j(x) ~= sum_n sum_k Theta^(n)_{jk} z_n^(k)(t) g_n(t, x)

Because the Legendre basis is orthonormal, this is implemented cheaply by
folding Theta into the readout vector C before the kernel is built:

    C_eff[:, n] = Theta^(n) @ C[:, n]

which leaves the fast frequency-domain kernel machinery untouched.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .hippo import nplr_decompose
from .ssm_state import SSMState

THETA_MODES = ("none", "const", "order")


class S4Layer(nn.Module):
    """One S4 layer: H independent SSM copies sharing a HiPPO initialisation."""

    def __init__(
        self,
        H: int,
        N: int,
        theta_mode: str = "none",
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        theta_init_scale: float = 0.01,
        train_ssm_state: bool = False,
    ) -> None:
        super().__init__()
        if theta_mode not in THETA_MODES:
            raise ValueError(f"theta_mode must be one of {THETA_MODES}, got {theta_mode!r}")

        self.H, self.N, self.theta_mode = H, N, theta_mode

        # Lambda and q come from HiPPO; trainable when train_ssm_state=True
        # (this is what the official S4 does -- HiPPO is only an initialisation)
        self.state = SSMState(N, trainable=train_ssm_state)

        # readout C, stored as real (H, N, 2) for optimiser compatibility
        self.C_param = nn.Parameter(torch.randn(H, N, 2) / np.sqrt(N))

        # per-channel timescale, log-parameterised so Delta stays positive
        log_dt = torch.rand(H) * (np.log(dt_max) - np.log(dt_min)) + np.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)

        # skip / feedthrough term
        self.D = nn.Parameter(torch.randn(H))

        if theta_mode in ("const", "order"):
            # exp(G) is always invertible, so no singularity risk during training
            self.G = nn.Parameter(torch.randn(H, H) * theta_init_scale)
            if theta_mode == "order":
                self.register_buffer("n_idx", torch.arange(N, dtype=torch.float32))

    # ------------------------------------------------------------------
    def effective_C(self) -> torch.Tensor:
        """Return the (H, N) complex readout after applying Theta."""
        C = torch.view_as_complex(self.C_param)

        if self.theta_mode == "none":
            return C

        if self.theta_mode == "const":
            return torch.matrix_exp(self.G).to(C.dtype) @ C

        # order-dependent: Theta^(n) = expm(n * G)
        Theta = torch.matrix_exp(self.n_idx[:, None, None] * self.G[None])  # (N, H, H)
        return torch.einsum("nij,jn->in", Theta.to(C.dtype), C)

    # ------------------------------------------------------------------
    def kernel(self, L: int) -> torch.Tensor:
        """
        Build the (H, L) real convolution kernel via the bilinear-discretised
        generating function evaluated at the L-th roots of unity, with a
        rank-1 Woodbury correction, followed by an inverse FFT.
        """
        device = self.C_param.device
        C_eff = self.effective_C()
        dt = torch.exp(self.log_dt)

        omega = torch.exp(-2j * np.pi * torch.arange(L, device=device) / L)
        omega = omega.to(torch.complex64)

        dt_ = dt[:, None, None]
        z_ = omega[None, :, None]
        lam_ = self.state.Lambda[None, None, :]
        q_ = self.state.q[None, None, :]
        cc_ = C_eff.conj()[:, None, :]

        R = 1.0 / ((2.0 / dt_) * (1 - z_) / (1 + z_) - lam_)

        k00 = (cc_ * R * q_).sum(-1)
        qRq = (q_.conj() * R * q_).sum(-1)
        k11 = 0.5 * qRq
        k01 = np.sqrt(0.5) * k00
        k10 = np.sqrt(0.5) * qRq

        K_hat = (2.0 / (1 + omega))[None, :] * (k00 - k01 * k10 / (1 + k11))
        return torch.fft.ifft(K_hat, dim=-1).real

    # ------------------------------------------------------------------
    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """u: (batch, H, L) -> y: (batch, H, L)."""
        _, H, L = u.shape
        if H != self.H:
            raise ValueError(f"expected {self.H} channels, got {H}")

        K = self.kernel(L)

        n_fft = 1
        while n_fft < 2 * L:
            n_fft *= 2

        U_f = torch.fft.rfft(u, n=n_fft, dim=-1)
        K_f = torch.fft.rfft(K, n=n_fft, dim=-1)
        y = torch.fft.irfft(U_f * K_f[None], n=n_fft, dim=-1)[..., :L]

        return y + self.D[None, :, None] * u
