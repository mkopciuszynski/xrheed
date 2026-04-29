"""
Peak detection and reconstruction utilities for RHEED analysis.

Provides methods to reconstruct diffraction peaks in reciprocal space
by preserving peak positions and amplitudes while replacing peak shapes
with idealized Gaussians.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from numpy.typing import NDArray


# ---------------------------------------------------------------------
# Internal helpers (pixel space)
# ---------------------------------------------------------------------

def _detect_local_peaks_pixel(
    data: NDArray[np.float64],
    *,
    sigma: float,
    size: int,
    percentile: float,
) -> NDArray[np.float64]:
    """
    Detect significant local maxima in a 2D array (pixel space).

    Returns an array with non-zero values at peak positions,
    storing peak amplitudes.
    """
    pass


def _reconstruct_from_peaks_pixel(
    peaks: NDArray[np.float64],
    *,
    sigma: float,
    amplitude_scale: float,
) -> NDArray[np.float64]:
    """
    Reconstruct signal from peak map using Gaussian spreading (pixel space).
    """
    pass


# ---------------------------------------------------------------------
# Public API (reciprocal / k-space)
# ---------------------------------------------------------------------

def reconstruct_kspace_peaks(
    data: xr.DataArray,
    *,
    sigma_smooth: float,
    sigma_peak: float,
    size: int,
    percentile: float,
    amplitude_scale: float = 1.0,
    reduce: str | None = None,
) -> xr.DataArray:
    """
    Reconstruct diffraction peaks in k-space (e.g., kx–ky).

    Detects peak maxima and replaces them with idealized Gaussian peaks.
    Designed for k-space data where peak positions carry physical meaning.

    Works with:
    - 2D image (kx, ky)
    - 3D stack (e.g., time, kx, ky)

    Parameters are expressed in physical units (k-space).
    """
    pass