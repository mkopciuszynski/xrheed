import logging

import numpy as np
import xarray as xr
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter, gaussian_filter1d  # type: ignore

from ..constants import IMAGE_NDIMS

logger = logging.getLogger(__name__)


def gaussian_filter_profile(
    profile: xr.DataArray,
    sigma: float = 1.0,
) -> xr.DataArray:
    """
    Apply a 1D Gaussian filter to a 1D xarray.DataArray profile.

    Parameters
    ----------
    profile : xr.DataArray
        1D data profile to be filtered.
    sigma : float, optional
        Standard deviation for Gaussian kernel, in the same units as the profile coordinate (default is 1.0).

    Returns
    -------
    xr.DataArray
        The filtered profile as a new DataArray.
    """
    logger.debug("gaussian_filter_profile called: sigma=%s", sigma)
    assert isinstance(profile, xr.DataArray), "profile must be an xarray.DataArray"
    assert profile.ndim == 1, "profile must have only one dimension"

    values: NDArray = profile.values

    # Calculate the spacing between coordinates
    coords: NDArray = profile.coords[profile.dims[0]].values
    if len(coords) < 2:
        raise ValueError(
            "profile coordinate must have at least two points to determine spacing"
        )
    spacing: float = float(coords[1] - coords[0])
    if abs(spacing) < 1e-5:
        raise ValueError("profile coordinate spacing cannot be zero")

    sigma_px: float = sigma / spacing

    filtered_values: NDArray = gaussian_filter1d(values, sigma=sigma_px)

    filtered_profile: xr.DataArray = xr.DataArray(
        filtered_values,
        coords=profile.coords,
        dims=profile.dims,
        attrs=profile.attrs,
        name=profile.name,
    )
    return filtered_profile


def high_pass_filter(
    rheed_data: xr.DataArray,
    threshold: float = 0.1,
    sigma: float = 1.0,
) -> xr.DataArray:
    """
    Vectorized high-pass filter to a RHEED image or stack using Gaussian filtering.

    Parameters
    ----------
    rheed_data : xr.DataArray
        RHEED image (2D) or image stack (3D) to be filtered.
        The stack must have the first dimension as the stacking dimension.
    threshold : float, optional
        Threshold for the high-pass filter (default is 0.1).
        Scales the blurred image before subtraction.
    sigma : float, optional
        Standard deviation for the Gaussian kernel in screen units (default is 1.0).

    Returns
    -------
    xr.DataArray
        High-pass filtered RHEED image or stack.
    """

    logger.debug(
        "high_pass_filter called: ndim=%s threshold=%s sigma=%s",
        getattr(rheed_data, "ndim", None),
        threshold,
        sigma,
    )

    # Validate input
    if not isinstance(rheed_data, xr.DataArray):
        raise TypeError("rheed_data must be an xarray.DataArray")
    if "screen_scale" not in rheed_data.attrs:
        raise ValueError("rheed_data must have 'screen_scale' attribute")

    # --- Convert sigma to pixels ---
    sigma_px: float = sigma * rheed_data.ri.screen_scale

    # --- Spatial dims ---
    if rheed_data.ndim < IMAGE_NDIMS:
        raise ValueError("Data must be at least 2D")

    spatial_dims = list(rheed_data.dims[-2:])

    # --- Apply vectorized ---
    filtered = xr.apply_ufunc(
        _high_pass_single_image,
        rheed_data,
        kwargs={
            "sigma_px": sigma_px,
            "threshold": threshold,
        },
        input_core_dims=[spatial_dims],
        output_core_dims=[spatial_dims],
        vectorize=True,
    )

    # --- Preserve + annotate attrs ---
    filtered.attrs.update(rheed_data.attrs)
    filtered.attrs.update(
        {
            "hp_filter": True,
            "hp_threshold": threshold,
            "hp_sigma": sigma,
        }
    )

    return filtered


def _high_pass_single_image(
    image: np.ndarray,
    sigma_px: float,
    threshold: float,
) -> np.ndarray:
    """
    Apply high-pass filter to a single 2D image.
    """

    img = image.astype(np.float32)

    blurred = gaussian_filter(img, sigma=sigma_px, mode="nearest")
    img = img - threshold * blurred

    # shift to positive
    img -= img.min()

    # clip to dtype
    if np.issubdtype(image.dtype, np.integer):
        info = np.iinfo(image.dtype)
        img = np.clip(img, info.min, info.max).astype(image.dtype)
    else:
        img = img.astype(image.dtype)

    return img
