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
from scipy.ndimage import gaussian_filter, maximum_filter
from tqdm.auto import tqdm

from ..constants import IMAGE_NDIMS, STACK_NDIMS

# ---------------------------------------------------------------------
# Internal helpers (pixel space)
# ---------------------------------------------------------------------


def _detect_local_peaks_pixel(
    data: NDArray[np.float64],
    *,
    sigma: float,
    size: int,
    percentile: float,
) -> NDArray[np.bool_]:
    """
    Detect significant local maxima in pixel space.

    Returns
    -------
    mask : ndarray of bool
        Boolean mask of significant peak locations.
    """

    nan_mask = np.isnan(data)
    valid_mask = ~nan_mask

    # Smooth only for detection
    smoothed = gaussian_filter(data, sigma=sigma)

    local_max = smoothed == maximum_filter(smoothed, size=size)
    local_max &= valid_mask

    peak_values = smoothed[local_max]
    if peak_values.size == 0:
        return np.zeros_like(data, dtype=bool)

    threshold = np.percentile(peak_values, percentile)
    significant = local_max & (smoothed >= threshold)

    return significant


def _reconstruct_from_peaks_pixel(
    peaks: NDArray[np.float64],
    *,
    sigma: float,
) -> NDArray[np.float64]:
    """
    Reconstruct image from delta-like peaks using Gaussian spreading.

    Notes
    -----
    scipy.ndimage.gaussian_filter conserves total intensity (L1),
    not peak amplitude. Normalization must be applied externally
    if peak heights are to be preserved.
    """
    return gaussian_filter(peaks, sigma=sigma)


def _get_isotropic_scale(data: xr.DataArray) -> float:
    """
    Compute k-space → pixel scale assuming an isotropic (kx, ky) grid.

    Raises if grid is non-uniform or anisotropic.
    """

    if "kx" not in data.coords or "ky" not in data.coords:
        raise ValueError(
            "DataArray must have explicit 'kx' and 'ky' coordinates "
            "to compute isotropic k-space scale."
        )

    kx = data.coords["kx"]
    ky = data.coords["ky"]

    if kx.size < 2 or ky.size < 2:
        raise ValueError("kx and ky must have at least two points.")

    dkx = float(kx[1] - kx[0])
    dky = float(ky[1] - ky[0])

    if not np.isclose(dkx, dky, rtol=1e-3):
        raise ValueError(
            f"kx and ky spacing must be equal (isotropic grid): "
            f"dkx={dkx}, dky={dky}"
        )

    return dkx


def _process_image(
    data: xr.DataArray,
    *,
    sigma_smooth: float,
    min_peak_distance: float,
    peak_percentile: float,
    peak_width: float,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Full processing pipeline for a single image (physical → pixel).

    - Detect peaks on a smoothed image
    - Measure amplitudes from the original image
    - Reconstruct using normalized Gaussian peaks
    """

    scale = _get_isotropic_scale(data)

    sigma_smooth_px = sigma_smooth / scale
    peak_width_px = peak_width / scale
    size_px = max(1, int(round(min_peak_distance / scale)))

    # --- detect peaks (mask only) ---
    peak_mask = _detect_local_peaks_pixel(
        data.values,
        sigma=sigma_smooth_px,
        size=size_px,
        percentile=peak_percentile,
    )

    # --- extract true amplitudes ---
    peaks = np.zeros_like(data.values)
    peaks[peak_mask] = data.values[peak_mask]

    # --- reconstruct with correct normalization ---
    # 2D Gaussian: G_max = 1 / (2πσ²)
    normalization = 2.0 * np.pi * peak_width_px**2

    peak_indices = np.argwhere(peak_mask)

    kx_vals = data.coords["kx"].values
    ky_vals = data.coords["ky"].values

    peak_kx = kx_vals[peak_indices[:, 0]]
    peak_ky = ky_vals[peak_indices[:, 1]]
    peak_amp = data.values[peak_mask]

    peaks_da = xr.DataArray(
        peak_amp,
        dims=("peak",),
        coords={
            "kx": ("peak", peak_kx),
            "ky": ("peak", peak_ky),
        },
        name="peaks",
    )

    reconstructed = (
        _reconstruct_from_peaks_pixel(peaks, sigma=peak_width_px) * normalization
    )

    reconstructed_da = xr.DataArray(
        reconstructed,
        coords=data.coords,
        dims=data.dims,
        attrs=data.attrs,
        name=data.name,
    )

    return (reconstructed_da, peaks_da)


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------


def filter_kspace_peaks(
    data: xr.DataArray,
    *,
    sigma_smooth: float,
    min_peak_distance: float,
    peak_percentile: float,
    peak_width: float,
    show_progress: bool = False,
    return_peaks: bool = False,
):
    """
    Filter k-space data by retaining only significant diffraction peaks.

    Optionally returns detected peak coordinates.
    """

    # --- single image ---
    if data.ndim == IMAGE_NDIMS:
        reconstructed, peaks = _process_image(
            data,
            sigma_smooth=sigma_smooth,
            min_peak_distance=min_peak_distance,
            peak_percentile=peak_percentile,
            peak_width=peak_width,
        )

        if return_peaks:
            return reconstructed, peaks
        return reconstructed

    # --- stack ---
    elif data.ndim == STACK_NDIMS:
        stack_dim = data.dims[0]
        n_frames = data.sizes[stack_dim]

        iterator = range(n_frames)
        if show_progress:
            iterator = tqdm(iterator, desc="Filtering k-space peaks")

        reconstructed_list: list[xr.DataArray] = []
        peaks_list: list[xr.DataArray] = []

        for i in iterator:
            rec, peaks = _process_image(
                data.isel({stack_dim: i}),
                sigma_smooth=sigma_smooth,
                min_peak_distance=min_peak_distance,
                peak_percentile=peak_percentile,
                peak_width=peak_width,
            )

            reconstructed_list.append(rec)
            peaks_list.append(peaks)

        reconstructed_stack = xr.concat(reconstructed_list, dim=stack_dim)

        if not return_peaks:
            return reconstructed_stack

        return reconstructed_stack, peaks_list

    else:
        raise ValueError("Input must be a 2D image or 3D stack.")


def plot_detected_peaks(
    data: xr.DataArray,
    peaks: xr.DataArray,
    *,
    ax=None,
    vmin=None,
    vmax=None,
    **scatter_kwargs,
):
    """
    Plot k-space image with detected peaks overlaid.
    """

    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()

    if scatter_kwargs is None:
        scatter_kwargs = {}

    data.plot(
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        add_colorbar=False,
    )

    ax.set_title("")

    scatter_kwargs.setdefault("marker", "o")
    scatter_kwargs.setdefault("s", 50)
    scatter_kwargs.setdefault("edgecolors", "white")
    scatter_kwargs.setdefault("linewidths", 0.5)
    scatter_kwargs.setdefault("facecolors", "none")

    ax.scatter(
        peaks.coords["ky"].values,
        peaks.coords["kx"].values,
        **scatter_kwargs,
    )

    ax.set_aspect("equal")
    ax.set_xlabel("$k_y$ (1/Å)")
    ax.set_ylabel("$k_x$ (1/Å)")

    return ax
