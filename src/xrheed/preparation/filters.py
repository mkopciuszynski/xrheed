import xarray as xr
import numpy as np
from scipy.ndimage import gaussian_filter1d

def gaussian_filter_profile(
        profile: xr.DataArray,
        sigma: float = 1.0,
    ):

    assert isinstance(profile, xr.DataArray)
    assert profile.ndim == 1, "profile must have only one dimension"
    
    values = profile.values

    # Calculate the spacing between coordinates
    coords = profile.coords[profile.dims[0]].values
    spacing = coords[1] - coords[0]

    sigma_px = sigma / spacing

    values = gaussian_filter1d(values, sigma=sigma_px)

    filtered_profile = xr.DataArray(values,
                                    profile.coords,
                                    profile.dims,
                                    attrs=profile.attrs)

    return filtered_profile

