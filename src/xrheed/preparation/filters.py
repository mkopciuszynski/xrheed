import xarray as xr
from scipy.ndimage import gaussian_filter1d


def gaussian_filter_profile(
    profile: xr.DataArray,
    sigma: float = 1.0,
):
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

    values = profile.values

    # Calculate the spacing between coordinates
    coords = profile.coords[profile.dims[0]].values
    if len(coords) < 2:
        raise ValueError(
            "profile coordinate must have at least two points to determine spacing"
        )
    spacing = float(coords[1] - coords[0])
    if spacing == 0:
        raise ValueError("profile coordinate spacing cannot be zero")

    sigma_px = sigma / spacing

    filtered_values = gaussian_filter1d(values, sigma=sigma_px)

    filtered_profile = xr.DataArray(
        filtered_values,
        coords=profile.coords,
        dims=profile.dims,
        attrs=profile.attrs,
        name=profile.name,
    )

    return filtered_profile
