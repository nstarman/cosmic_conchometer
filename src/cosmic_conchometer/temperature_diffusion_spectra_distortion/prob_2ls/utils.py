"""Spectral Distortion."""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__: list[str] = []

if TYPE_CHECKING:
    from scipy.interpolate import CubicSpline


def cubic_global_coeffs_from_ppoly_coeffs(
    spl: CubicSpline,
) -> tuple[float, float, float, float]:
    """Convert PPoly coefficients to global coefficients.

    ::
        c3(x-xi)^3 + c2(x-xi)^2 + c1(x-xi) + c0
        = p3 x^2 + p2 x^2 + p1 x + p0.

        p3 = c3
        p2 = -3 c3 xi + c2
        p1 = 3 c3 xi^2 - 2 c2 xi + c1
        p0 = -c3 xi^3 + c2 xi^2 - c1 xi + c0
    """
    xi = spl.x[:-1]
    c3, c2, c1, c0 = spl.c

    p3 = c3
    p2 = -3 * c3 * xi + c2
    p1 = 3 * c3 * xi**2 - 2 * c2 * xi + c1
    p0 = -c3 * xi**3 + c2 * xi**2 - c1 * xi + c0
    return p3, p2, p1, p0
