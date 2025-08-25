import numpy as np
import xarray as xr
from scipy import ndimage


def convert_x_to_kx(
    x_coords_mm: np.ndarray,
    ewald_sphere_radius: float,
    screen_sample_distance_mm: float,
) -> np.ndarray:
    """Convert x coordinates from mm to kx [1/Å] using the Ewald sphere radius and screen-sample distance.
    Parameters
    ----------
    x_coords_mm : np.ndarray
        Array of x coordinates in millimeters (mm).
    ewald_sphere_radius : float
        Radius of the Ewald sphere in reciprocal space (1/Å).
    screen_sample_distance_mm : float
        Distance from the sample to the screen in millimeters (mm).

    Returns
    -------
    np.ndarray
        Converted x coordinates in kx [1/Å].
    """

    kx = (x_coords_mm / screen_sample_distance_mm) * ewald_sphere_radius

    return kx


def transform_to_kxky(
    rheed_image: xr.DataArray,
    rotate: bool = False,
) -> xr.DataArray:
    """
    Transform the RHEED image to kx-ky coordinates.

    Parameters
    ----------
    rotate : bool, optional
        If True, rotate the transformed image (default: True).

    Returns
    -------
    xr.DataArray
        Transformed image in kx-ky coordinates.
    """

    # prepare the data for calculations
    screen_sample_distance = rheed_image.ri.screen_sample_distance
    beta = rheed_image.ri.beta
    alpha = rheed_image.ri.alpha

    ewald_radius = np.sqrt(rheed_image.ri.beam_energy) * 0.5123

    k = ewald_radius
    kk = k**2

    # new coordinates for transformation
    kx = np.linspace(-10, 10, 1024)
    ky = np.linspace(-10, 10, 1024)

    KX, KY = np.meshgrid(kx, ky, indexing="ij")

    # take into acount the theta angle
    KY = KY - k * (1 - np.cos(np.deg2rad(beta)))

    tr = (KY + k) ** 2 + KX**2

    # make nans points outside Ewald sphere
    ind = tr > kk
    KX[ind] = np.nan
    KY[ind] = np.nan

    kr = np.sqrt(kk - (k - abs(KY)) ** 2)
    th = np.arcsin(kr / k)
    rho = screen_sample_distance * np.tan(th)

    px_mm = rho * KX / kr
    py_mm = -np.sqrt(rho**2 - px_mm**2)

    # relation between old and new
    x = xr.DataArray(px_mm, dims=["kx", "ky"], coords={"kx": kx, "ky": ky})
    y = xr.DataArray(py_mm, dims=["kx", "ky"], coords={"kx": kx, "ky": ky})

    trans_image = rheed_image.interp(x=x, y=y, method="linear")
    trans_image = trans_image.T

    if rotate:
        nan_mask = ~np.isnan(trans_image.values)
        trans_image = trans_image.fillna(0)

        # Rotate the mask and data
        rotated_nan_mask = (
            ndimage.rotate(nan_mask.astype(int), angle=30 - alpha, reshape=False) > 0
        )
        trans_image.data = ndimage.rotate(trans_image.data, 30 - alpha, reshape=False)

        # Apply the mask to restore NaN values in the rotated DataArray
        trans_image = trans_image.where(rotated_nan_mask)

    return trans_image
