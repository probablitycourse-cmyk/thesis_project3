"""
EEGS4Model -- S4 backbone with optional EEG-specific components.

Three independent flags let each idea be ablated separately:

    use_spatial     region-aware channel projection instead of a dense Linear
    use_multiband   parallel SSM branches with per-band Delta ranges
    use_gating      multiplicative Mamba-style gate inside each block

With all three off the model is byte-for-byte equivalent to the plain S4
baseline, so any difference measured is attributable to the flags alone.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .s4_layer import S4Layer
from .s4_model import _split_param_groups
from .eeg_components import MultiBandS4Layer, SpatialFilter, EEG_BANDS


class EEGS4Block(nn.Module):
    """S4 (plain or multi-band) -> GELU -> channel mixing -> residual + norm."""

    def __init__(
        self,
        H: int,
        N: int,
        fs: float = 128.0,
        use_multiband: bool = True,
        use_gating: bool = False,
        theta_mode: str = "none",
        dropout: float = 0.1,
        train_ssm_state: bool = False,
    ) -> None:
        super().__init__()

        if use_multiband:
            self.ssm = MultiBandS4Layer(H, N, fs=fs, gated=use_gating,
                                        train_ssm_state=train_ssm_state)
        else:
            self.ssm = S4Layer(H, N, theta_mode=theta_mode,
                               train_ssm_state=train_ssm_state)
            self.gate_proj = nn.Linear(H, H) if use_gating else None

        self.use_multiband = use_multiband
        self.use_gating = use_gating

        self.mix = nn.Linear(H, H)
        self.norm = nn.LayerNorm(H)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, L, H) -> (batch, L, H)."""
        u = x.transpose(1, 2)
        y = self.ssm(u)

        # MultiBandS4Layer applies its own gate; plain S4Layer needs it here
        if self.use_gating and not self.use_multiband:
            g = torch.sigmoid(self.gate_proj(u.transpose(1, 2))).transpose(1, 2)
            y = y * g

        y = self.act(y).transpose(1, 2)
        y = self.drop(self.mix(y))
        return self.norm(x + y)


class EEGS4Model(nn.Module):
    """Spatial projection -> stack of EEG S4 blocks -> pooling -> readout."""

    def __init__(
        self,
        input_dim: int = 14,
        H: int = 64,
        N: int = 64,
        num_layers: int = 4,
        out_dim: int = 2,
        fs: float = 128.0,
        use_spatial: bool = True,
        use_multiband: bool = True,
        use_gating: bool = False,
        theta_mode: str = "none",
        dropout: float = 0.1,
        pooling: str = "mean",
        train_ssm_state: bool = False,
    ) -> None:
        super().__init__()
        self.train_ssm_state = train_ssm_state
        if pooling not in ("mean", "last", "max"):
            raise ValueError(f"unknown pooling {pooling!r}")
        self.pooling = pooling
        self.use_spatial = use_spatial
        self.use_multiband = use_multiband
        self.use_gating = use_gating

        if use_spatial and input_dim == 14:
            self.in_proj = SpatialFilter(H, n_channels=input_dim)
        else:
            self.in_proj = nn.Linear(input_dim, H)
            self.use_spatial = False

        self.blocks = nn.ModuleList([
            EEGS4Block(H, N, fs=fs, use_multiband=use_multiband,
                       use_gating=use_gating, theta_mode=theta_mode, dropout=dropout,
                       train_ssm_state=train_ssm_state)
            for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(H, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        for block in self.blocks:
            h = block(h)

        if self.pooling == "mean":
            pooled = h.mean(dim=1)
        elif self.pooling == "last":
            pooled = h[:, -1, :]
        else:
            pooled = h.max(dim=1).values

        return self.out_proj(pooled)

    def param_groups(self, lr: float, ssm_mult: float = 0.1, theta_mult: float = 0.1,
                     state_mult: float = 0.01):
        return _split_param_groups(self, lr, ssm_mult, theta_mult, state_mult)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def describe(self) -> str:
        flags = []
        flags.append(f"spatial={'on' if self.use_spatial else 'off'}")
        flags.append(f"multiband={'on' if self.use_multiband else 'off'}")
        flags.append(f"gating={'on' if self.use_gating else 'off'}")
        flags.append(f"train_state={'on' if self.train_ssm_state else 'off'}")
        lines = ["EEGS4Model [" + ", ".join(flags) + "]"]
        if self.use_multiband:
            lines.append("  band -> Delta ranges (fs=128):")
            lines.append(self.blocks[0].ssm.band_info())
        return "\n".join(lines)
