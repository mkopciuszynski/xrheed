import numpy as np
import xarray as xr
from scipy import ndimage

from .base import convert_gx_gy_to_sx_sy


def transform_image_to_kxky(
    rheed_image: xr.DataArray,
    rotate: bool = False,
    mirror: bool = False,
) -> xr.DataArray:
    """
    Transform the RHEED image to kx-ky coordinates.

    Parameters
    ----------
    rotate : bool, optional
        If True, rotate the transformed image (default: True).
    mirror : bool, optional
        If True, add mirrored image (default: False).

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

    # new coordinates for transformation
    # TODO add the parameter that allows to set kx, ky
    kx = np.arange(-10, 10, 0.01, dtype=np.float32)
    ky = np.arange(-10, 10, 0.01, dtype=np.float32)

    gx, gy = np.meshgrid(kx, ky, indexing="ij")

    sx_to_kx, sy_to_ky = convert_gx_gy_to_sx_sy(
        gx,
        gy,
        ewald_radius=ewald_radius,
        beta=beta,
        screen_sample_distance=screen_sample_distance,
        remove_outside=False,
    )

    # relation between old and new
    sx = xr.DataArray(sx_to_kx, dims=["kx", "ky"], coords={"kx": kx, "ky": ky})
    sy = xr.DataArray(sy_to_ky, dims=["kx", "ky"], coords={"kx": kx, "ky": ky})

    trans_image = rheed_image.interp(sx=sx, sy=sy, method="linear")

    if mirror:
        da_mirror = trans_image.isel(kx=slice(None, None, -1)).assign_coords(
            kx=trans_image.kx
        )
        trans_image = trans_image.fillna(da_mirror)

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

    trans_image.attrs = rheed_image.attrs

    return trans_image
