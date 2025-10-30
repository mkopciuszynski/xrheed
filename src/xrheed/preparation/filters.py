import numpy as np
import xarray as xr
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter, gaussian_filter1d  # type: ignore

from ..constants import IMAGE_NDIMS, STACK_NDIMS
import logging

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
    Apply a high-pass filter to a RHEED image or stack using Gaussian filtering.

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

    sigma_px: float = sigma * rheed_data.ri.screen_scale

    # --- Handle single image ---
    if rheed_data.ndim == IMAGE_NDIMS:
        filtered_values = _apply_hp_filter(rheed_data.values, threshold, sigma_px)
        filtered = rheed_data.copy()
        filtered.values = filtered_values

    # --- Handle stack ---
    elif rheed_data.ndim == STACK_NDIMS:
        stack_dim = rheed_data.dims[0]
        filtered_slices = []

        for i in range(rheed_data.sizes[stack_dim]):
            slice_values = rheed_data.isel({stack_dim: i}).values
            filtered_slice = _apply_hp_filter(slice_values, threshold, sigma_px)
            da_slice = rheed_data.isel({stack_dim: i}).copy()
            da_slice.values = filtered_slice
            filtered_slices.append(da_slice)

        filtered = xr.concat(filtered_slices, dim=stack_dim)
        filtered = filtered.assign_coords({stack_dim: rheed_data[stack_dim]})

    else:
        raise ValueError(
            f"rheed_data must be {IMAGE_NDIMS}D or {STACK_NDIMS}D (stack), "
            f"got {rheed_data.ndim}D"
        )

    # Set attributes for high-pass filtering
    filtered.attrs.update(
        {
            "hp_filter": True,
            "hp_threshold": threshold,
            "hp_sigma": sigma,
        }
    )
    logger.info(
        "high_pass_filter: applied hp filter (threshold=%s sigma=%s) to data with ndim=%s",
        threshold,
        sigma,
        rheed_data.ndim,
    )

    return filtered


def _apply_hp_filter(
    image_values: NDArray, threshold: float, sigma_px: float
) -> NDArray:
    """
    Helper function to apply high-pass filter to a single 2D image array.

    Returns clipped uint8 array.
    """
    logger.debug(
        "_apply_hp_filter: sigma_px=%s threshold=%s image_shape=%s",
        sigma_px,
        threshold,
        getattr(image_values, "shape", None),
    )
    blurred = gaussian_filter(image_values, sigma=sigma_px)
    hp_image = image_values - threshold * blurred
    hp_image -= hp_image.min()
    hp_image = np.clip(hp_image, 0, 255).astype(np.uint8)
    return hp_image
