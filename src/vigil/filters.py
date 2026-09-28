"""Vectorised One Euro filter (Casiez et al. 2012) for jittery keypoints / vertices."""

from __future__ import annotations

import numpy as np


def _alpha(cutoff: np.ndarray | float, dt: float) -> np.ndarray | float:
    tau = 1.0 / (2 * np.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """Filters an array of any shape element-wise. NaN inputs reset that element."""

    def __init__(self, min_cutoff: float = 1.5, beta: float = 0.3, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self) -> None:
        self._x: np.ndarray | None = None
        self._dx: np.ndarray | None = None
        self._t: float | None = None

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if self._x is None or self._x.shape != x.shape or self._t is None or t <= self._t:
            self._x, self._dx, self._t = x.copy(), np.zeros_like(x), t
            return x
        dt = t - self._t
        self._t = t

        fresh = np.isnan(self._x)  # previously missing → start from the new value
        prev = np.where(fresh, x, self._x)
        dx = (x - prev) / dt
        dx_hat = self._dx + _alpha(self.d_cutoff, dt) * (dx - self._dx)
        dx_hat = np.where(np.isnan(dx_hat), 0.0, dx_hat)
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        x_hat = prev + _alpha(cutoff, dt) * (x - prev)

        self._x = x_hat.astype(np.float32)  # NaN where x is NaN → resets next time
        self._dx = dx_hat.astype(np.float32)
        return self._x
