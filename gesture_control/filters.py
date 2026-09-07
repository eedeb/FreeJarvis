"""Pointer smoothing."""

from __future__ import annotations

import math

import numpy as np


class OneEuroFilter:
    """The 1-Euro filter (Casiez et al., 2012), applied to a 2D point.

    A fixed low-pass filter forces a choice between a jittery cursor and a
    laggy one.  This one adapts: the cutoff frequency rises with the observed
    speed, so slow movement (where jitter is visible and lag is not) is
    filtered hard, and fast movement (where lag is visible and jitter is not)
    passes through almost untouched.
    """

    def __init__(self, min_cutoff: float = 1.5, beta: float = 0.03,
                 d_cutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x_prev: np.ndarray | None = None
        self._dx_prev = np.zeros(2)
        self._t_prev: float | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * max(cutoff, 1e-6))
        return 1.0 / (1.0 + tau / dt)

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = np.zeros(2)
        self._t_prev = None

    def __call__(self, t: float, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self._x_prev is None or self._t_prev is None:
            self._x_prev, self._t_prev = x.copy(), t
            return x

        dt = t - self._t_prev
        if dt <= 0.0 or dt > 0.5:
            # First frame after a stall: restart rather than let a huge dt
            # collapse the filter to a pass-through with a stale derivative.
            self._x_prev, self._t_prev = x.copy(), t
            self._dx_prev = np.zeros(2)
            return x

        dx = (x - self._x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev

        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(dx_hat))
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self._x_prev

        self._x_prev, self._dx_prev, self._t_prev = x_hat, dx_hat, t
        return x_hat
