import numpy as np
import xarray as xr
from typing import Union
from numpy.typing import NDArray
from scipy import ndimage  # type: ignore

from .base import convert_gx_gy_to_sx_sy
from ..constants import IMAGE_NDIMS, STACK_NDIMS


def transform_image_to_kxky(
    rheed_data: xr.DataArray,
    rotate: bool = False,
    point_symmetry: bool = False,
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
    screen_sample_distance: float = rheed_data.ri.screen_sample_distance
    ewald_radius: float = rheed_data.ri.ewald_sphere_radius

    beta: float = rheed_data.ri.beta

    # new coordinates for transformation
    # TODO add the parameter that allows to set kx, ky
    kx: NDArray[np.float32] = np.arange(-10, 10, 0.01, dtype=np.float32)
    ky: NDArray[np.float32] = np.arange(-10, 10, 0.01, dtype=np.float32)

    gx: NDArray[np.float32]
    gy: NDArray[np.float32]

    gx, gy = np.meshgrid(kx, ky, indexing="ij")

    sx_to_kx: NDArray[np.float32]
    sy_to_ky: NDArray[np.float32]

    sx_to_kx, sy_to_ky = convert_gx_gy_to_sx_sy(
        gx,
        gy,
        ewald_radius=ewald_radius,
        beta=beta,
        screen_sample_distance=screen_sample_distance,
        remove_outside=False,
    )

    # relation between old and new
    sx: xr.DataArray = xr.DataArray(
        sx_to_kx, dims=["kx", "ky"], coords={"kx": kx, "ky": ky}
    )
    sy: xr.DataArray = xr.DataArray(
        sy_to_ky, dims=["kx", "ky"], coords={"kx": kx, "ky": ky}
    )

    if rheed_data.ndim == IMAGE_NDIMS:

        alpha: float = rheed_data.ri.alpha
        
        transformed: xr.DataArray = rheed_data.interp(sx=sx, sy=sy, method="linear")

        if rotate:
            trans_image_rotated = _rotate_trans_image(transformed, alpha)
            transformed = trans_image_rotated

        if point_symmetry:
            trans_image_rotated = _rotate_trans_image(transformed, 180)
            transformed = xr.where(np.isnan(transformed), trans_image_rotated, transformed)
        
        transformed.attrs = rheed_data.attrs
        return transformed

    if rheed_data.ndim == STACK_NDIMS and "alpha" in rheed_data.coords:

        transformed_slices = []
        alpha: NDArray= rheed_data.ri.alpha
        

        for i in range(rheed_data.sizes["alpha"]):
            image = rheed_data.isel(alpha=i)
            transformed = image.interp(sx=sx, sy=sy, method="linear")

            transformed = _rotate_trans_image(transformed, float(alpha[i]))
            if point_symmetry:
                rotated_180 = _rotate_trans_image(transformed, 180)
                transformed = xr.where(np.isnan(transformed), rotated_180, transformed)

            transformed_slices.append(transformed)

        transformed_stack = xr.concat(transformed_slices, dim="alpha")
        transformed_stack = transformed_stack.assign_coords(alpha=rheed_data.alpha)

        transformed_stack.attrs = rheed_data.attrs
        return transformed_stack

    

    return transformed


def _rotate_trans_image(
    trans_image: xr.DataArray, angle: float, mode: str = "nearest"
) -> xr.DataArray:
    """
    Rotate a 2D xarray.DataArray around its center by a given angle.

    Parameters
    ----------
    rheed_image : xr.DataArray
        2D image-like DataArray to rotate.
    angle : float
        Rotation angle in degrees (counter-clockwise).
    mode : str
        How to handle values outside boundaries ('constant', 'nearest', 'reflect', ...).

    Returns
    -------
    rotated : xr.DataArray
        Rotated DataArray with NaNs preserved.
    """
    if trans_image.ndim != 2:
        raise ValueError("rotate_xarray expects a 2D DataArray")

    # Assert that coordinates exist
    if "kx" not in trans_image.coords or "ky" not in trans_image.coords:
        raise ValueError("rotate_xarray requires coordinates 'kx' and 'ky'")

    # Assert that kx and ky are identical
    if not np.allclose(trans_image["kx"].values, trans_image["ky"].values):
        raise ValueError("rotate_xarray requires kx and ky coordinates to be identical")

    # Build mask for NaNs
    nan_mask: NDArray[np.bool_] = ~np.isnan(trans_image.values)
    filled: xr.DataArray = trans_image.fillna(0)

    # Rotate data and mask
    rotated_data: NDArray[np.uint8] = ndimage.rotate(
        filled.values, angle, reshape=False, mode=mode, order=1
    ).astype(np.uint8)

    rotated_mask: NDArray[np.bool_] = (
        ndimage.rotate(
            nan_mask.astype(np.uint8), angle, reshape=False, mode=mode, order=0
        )
        > 0
    ).astype(np.bool)

    # Wrap back into DataArray, reusing same coords/dims
    rotated = xr.DataArray(
        rotated_data,
        coords=trans_image.coords,
        dims=trans_image.dims,
        attrs=trans_image.attrs,
        name=trans_image.name,
    )

    return rotated.where(rotated_mask)
