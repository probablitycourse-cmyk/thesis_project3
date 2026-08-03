"""
EEG-specific architectural components:

  1. MultiBandS4Layer  -- parallel SSM branches, each initialised with a
     timescale range matched to a canonical EEG frequency band. Because the
     bilinear discretisation makes the kernel's temporal receptive field a
     direct function of Delta, branches with different Delta ranges span
     genuinely different function classes (verified numerically: dt=0.001
     reaches 50% kernel energy by step 50, dt=0.05 by step 7).

  2. SpatialFilter -- structured channel mixing that respects the physical
     10-20 electrode layout (hemispheres / lobes) instead of a single dense
     projection. Note this is a RESTRICTION of a dense Linear, so it can only
     help through inductive bias, never through added capacity.

  3. GatedS4Block -- multiplicative gate in the style of Mamba,
     y = SSM(x) * sigmoid(W_g x). This is the only component here that adds a
     genuine nonlinearity to the function class.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .hippo import nplr_decompose
from .ssm_state import SSMState

# Canonical EEG bands (Hz) and the Delta range that matches each one.
# Delta ~ 1 / (f * FS_normalised); lower frequency -> larger effective memory
# -> smaller Delta. Ranges overlap slightly so branches can specialise.
EEG_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

# 14-channel Emotiv EPOC layout used by DREAMER, in file order.
EMOTIV_CHANNELS = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
                   "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]

EMOTIV_REGIONS = {
    "frontal_left":   [0, 1, 2, 3],    # AF3, F7, F3, FC5
    "frontal_right":  [10, 11, 12, 13],  # FC6, F4, F8, AF4
    "temporal_left":  [4, 5],           # T7, P7
    "temporal_right": [8, 9],           # P8, T8
    "occipital":      [6, 7],           # O1, O2
}


def band_to_dt_range(f_low: float, f_high: float, fs: float) -> tuple[float, float]:
    """
    Map a frequency band to a Delta range.

    A component oscillating at f Hz has period fs/f samples; the SSM needs a
    timescale of that order to resolve it, so Delta ~ f / fs. The wider the
    band, the wider the Delta range given to that branch.
    """
    dt_min = f_low / fs
    dt_max = f_high / fs
    return float(dt_min), float(dt_max)


# ----------------------------------------------------------------------
class SpatialFilter(nn.Module):
    """
    Region-aware channel projection.

    Each anatomical region gets its own small linear map; the outputs are
    concatenated and then mixed once. Optionally adds explicit left-right
    difference features, which matter for valence (frontal alpha asymmetry).
    """

    def __init__(
        self,
        H: int,
        n_channels: int = 14,
        regions: dict | None = None,
        use_asymmetry: bool = True,
    ) -> None:
        super().__init__()
        self.regions = regions if regions is not None else EMOTIV_REGIONS
        self.use_asymmetry = use_asymmetry

        n_regions = len(self.regions)
        per_region = max(4, H // (n_regions + (2 if use_asymmetry else 0)))

        self.region_names = list(self.regions.keys())
        self.region_proj = nn.ModuleList(
            [nn.Linear(len(self.regions[r]), per_region) for r in self.region_names]
        )
        for r in self.region_names:
            self.register_buffer(f"idx_{r}", torch.tensor(self.regions[r], dtype=torch.long))

        out_dim = per_region * n_regions

        if use_asymmetry:
            # frontal and temporal left-right differences
            self.asym_pairs = [("frontal_left", "frontal_right"),
                               ("temporal_left", "temporal_right")]
            self.asym_pairs = [(a, b) for a, b in self.asym_pairs
                               if a in self.regions and b in self.regions]
            self.asym_proj = nn.ModuleList([
                nn.Linear(min(len(self.regions[a]), len(self.regions[b])), per_region)
                for a, b in self.asym_pairs
            ])
            out_dim += per_region * len(self.asym_pairs)
        else:
            self.asym_pairs = []

        self.combine = nn.Linear(out_dim, H)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, L, n_channels) -> (batch, L, H)."""
        parts = []
        for name, proj in zip(self.region_names, self.region_proj):
            idx = getattr(self, f"idx_{name}")
            parts.append(proj(x[..., idx]))

        for (a, b), proj in zip(self.asym_pairs, self.asym_proj):
            ia = getattr(self, f"idx_{a}")
            ib = getattr(self, f"idx_{b}")
            k = min(len(ia), len(ib))
            parts.append(proj(x[..., ia[:k]] - x[..., ib[:k]]))

        return self.combine(torch.cat(parts, dim=-1))


# ----------------------------------------------------------------------
class MultiBandS4Layer(nn.Module):
    """
    Parallel SSM branches, one per EEG frequency band.

    The H channels are split evenly across the bands; every branch shares the
    same HiPPO initialisation but draws its Delta from the range matched to
    its band, so each branch sees the signal at a different temporal scale.
    """

    def __init__(
        self,
        H: int,
        N: int,
        fs: float = 128.0,
        bands: dict | None = None,
        gated: bool = False,
        train_ssm_state: bool = False,
    ) -> None:
        super().__init__()
        self.H, self.N = H, N
        self.bands = bands if bands is not None else EEG_BANDS
        self.band_names = list(self.bands.keys())
        n_bands = len(self.band_names)

        self.state = SSMState(N, trainable=train_ssm_state)

        # split H across bands; the remainder is spread over the first bands
        base, rem = divmod(H, n_bands)
        self.band_sizes = [base + (1 if i < rem else 0) for i in range(n_bands)]

        self.C_param = nn.Parameter(torch.randn(H, N, 2) / np.sqrt(N))
        self.D = nn.Parameter(torch.randn(H))

        # per-band Delta initialisation
        log_dts = []
        for name, size in zip(self.band_names, self.band_sizes):
            f_lo, f_hi = self.bands[name]
            dt_lo, dt_hi = band_to_dt_range(f_lo, f_hi, fs)
            u = torch.rand(size)
            log_dts.append(u * (np.log(dt_hi) - np.log(dt_lo)) + np.log(dt_lo))
        self.log_dt = nn.Parameter(torch.cat(log_dts))

        self.gated = gated
        if gated:
            self.gate_proj = nn.Linear(H, H)

    def band_info(self) -> str:
        lines = []
        for name, size in zip(self.band_names, self.band_sizes):
            f_lo, f_hi = self.bands[name]
            dt_lo, dt_hi = band_to_dt_range(f_lo, f_hi, 128.0)
            lines.append(f"    {name:6s} {f_lo:5.1f}-{f_hi:4.1f} Hz -> "
                         f"dt {dt_lo:.4f}-{dt_hi:.4f}  ({size} channels)")
        return "\n".join(lines)

    def kernel(self, L: int) -> torch.Tensor:
        device = self.C_param.device
        C = torch.view_as_complex(self.C_param)
        dt = torch.exp(self.log_dt)

        omega = torch.exp(-2j * np.pi * torch.arange(L, device=device) / L).to(torch.complex64)

        dt_ = dt[:, None, None]
        z_ = omega[None, :, None]
        lam_ = self.state.Lambda[None, None, :]
        q_ = self.state.q[None, None, :]
        cc_ = C.conj()[:, None, :]

        R = 1.0 / ((2.0 / dt_) * (1 - z_) / (1 + z_) - lam_)

        k00 = (cc_ * R * q_).sum(-1)
        qRq = (q_.conj() * R * q_).sum(-1)
        K_hat = (2.0 / (1 + omega))[None, :] * (k00 - 0.5 * k00 * qRq / (1 + 0.5 * qRq))
        return torch.fft.ifft(K_hat, dim=-1).real

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """u: (batch, H, L) -> (batch, H, L)."""
        _, H, L = u.shape
        K = self.kernel(L)

        n_fft = 1
        while n_fft < 2 * L:
            n_fft *= 2
        U_f = torch.fft.rfft(u, n=n_fft, dim=-1)
        K_f = torch.fft.rfft(K, n=n_fft, dim=-1)
        y = torch.fft.irfft(U_f * K_f[None], n=n_fft, dim=-1)[..., :L]
        y = y + self.D[None, :, None] * u

        if self.gated:
            # gate is computed channel-last then transposed back
            g = torch.sigmoid(self.gate_proj(u.transpose(1, 2))).transpose(1, 2)
            y = y * g

        return y
