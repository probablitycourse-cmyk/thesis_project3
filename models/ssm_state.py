"""
Trainable SSM state parameters with a hard stability guarantee.

In the official S4 (Gu et al., 2022) the NPLR quantities Lambda, P, Q and B
are all trainable -- HiPPO only supplies their initialisation. Training
Lambda directly is unsafe though: every HiPPO-LegS eigenvalue has real part
exactly -0.5, i.e. it sits right on the stability boundary, so an
unconstrained gradient step can push Re(Lambda) positive and make the kernel
diverge.

This module reparameterises the real part through softplus,

    Re(Lambda) = -softplus(w)

which is negative for every value of w, so stability holds throughout
training while the magnitude stays free. The imaginary part (the oscillation
frequency) is unconstrained.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .hippo import nplr_decompose


def _inverse_softplus(y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """w such that softplus(w) == y, for y > 0."""
    y = torch.clamp(y, min=eps)
    return y + torch.log(-torch.expm1(-y).clamp(min=eps))


class SSMState(nn.Module):
    """
    Holds Lambda and q (== B == the rank-1 factor for HiPPO-LegS).

    trainable=False  -> both are buffers, exactly the HiPPO initialisation
    trainable=True   -> both are parameters; Re(Lambda) is kept negative by
                        the softplus reparameterisation
    """

    def __init__(self, N: int, trainable: bool = False) -> None:
        super().__init__()
        self.N = N
        self.trainable = trainable

        Lambda, q, _ = nplr_decompose(N)
        Lambda_t = torch.tensor(Lambda, dtype=torch.complex64)
        q_t = torch.tensor(q, dtype=torch.complex64)

        if not trainable:
            self.register_buffer("Lambda_buf", Lambda_t)
            self.register_buffer("q_buf", q_t)
            return

        # Re(Lambda) = -softplus(w)  ->  w = inverse_softplus(-Re(Lambda))
        neg_real = (-Lambda_t.real).clamp(min=1e-4)
        self.lambda_re_raw = nn.Parameter(_inverse_softplus(neg_real))
        self.lambda_im = nn.Parameter(Lambda_t.imag.clone())
        self.q_param = nn.Parameter(torch.view_as_real(q_t))

    # ------------------------------------------------------------------
    @property
    def Lambda(self) -> torch.Tensor:
        if not self.trainable:
            return self.Lambda_buf
        re = -torch.nn.functional.softplus(self.lambda_re_raw)
        return torch.complex(re, self.lambda_im)

    @property
    def q(self) -> torch.Tensor:
        if not self.trainable:
            return self.q_buf
        return torch.view_as_complex(self.q_param)

    # ------------------------------------------------------------------
    def stability_margin(self) -> float:
        """Largest (i.e. least negative) real part -- must stay below zero."""
        with torch.no_grad():
            return float(self.Lambda.real.max())

    def extra_repr(self) -> str:
        return f"N={self.N}, trainable={self.trainable}"


def ssm_param_names() -> tuple[str, ...]:
    """Parameter-name suffixes that belong to the SSM state, for lr grouping."""
    return ("lambda_re_raw", "lambda_im", "q_param")
