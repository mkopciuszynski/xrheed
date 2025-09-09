import numpy as np
import xarray as xr
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter, gaussian_filter1d  # type: ignore


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
    rheed_image: xr.DataArray, threshold: float = 0.1, sigma: float = 1.0
) -> xr.DataArray:
    """
    Apply a high-pass filter to a RHEED image using Gaussian filtering.

    Parameters
    ----------
    rheed_image : xr.DataArray
        RHEED image data to be filtered.
    threshold : float, optional
        Threshold for the high-pass filter (default is 0.1).
        This value scales the blurred image before subtraction,
        effectively controlling the strength of the filter.
        A higher value will result in a stronger high-pass effect.
    sigma : float, optional
        Standard deviation for the Gaussian kernel, in the same units as the image coordinate
        (default is 1.0).

    Returns
    -------
    xr.DataArray
        The high-pass filtered RHEED image as a new DataArray.
    """
    # Validate input
    assert isinstance(
        rheed_image, xr.DataArray
    ), "rheed_image must be an xarray.DataArray"
    assert rheed_image.ndim == 2, "rheed_image must have two dimensions"
    assert (
        "screen_scale" in rheed_image.attrs
    ), "rheed_image must have 'screen_scale' attribute"

    # Create a copy of the input image to avoid modifying the original
    high_pass_image: xr.DataArray = rheed_image.copy()

    sigma_px: float = sigma * rheed_image.ri.screen_scale

    rheed_image_values: NDArray = rheed_image.values

    # Apply Gaussian filter to the image
    blurred_image_values: NDArray = gaussian_filter(rheed_image_values, sigma=sigma_px)

    high_pass_image_values: NDArray = (
        rheed_image_values - threshold * blurred_image_values
    )
    high_pass_image_values -= high_pass_image_values.min()

    # Clip to valid uint8 range and cast
    high_pass_image_values = np.clip(high_pass_image_values, 0, 255).astype(np.uint8)

    high_pass_image.values = high_pass_image_values

    # Set attributes for the high-pass filtered image
    high_pass_image.attrs["hp_filter"] = True
    high_pass_image.attrs["hp_threshold"] = threshold
    high_pass_image.attrs["hp_sigma"] = sigma

    return high_pass_image
