import numpy as np
import xarray as xr
from numpy.typing import NDArray
from scipy import ndimage  # type: ignore

from .base import convert_gx_gy_to_sx_sy


def transform_image_to_kxky(
    rheed_image: xr.DataArray,
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
    screen_sample_distance: float = rheed_image.ri.screen_sample_distance
    beta: float = rheed_image.ri.beta
    alpha: float = rheed_image.ri.alpha

    ewald_radius: float = np.sqrt(rheed_image.ri.beam_energy) * 0.5123

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

    trans_image: xr.DataArray = rheed_image.interp(sx=sx, sy=sy, method="linear")

    if rotate:
        trans_image_rotated = _rotate_trans_image(trans_image, alpha)
        trans_image = trans_image_rotated

    if point_symmetry:
        trans_image_rotated = _rotate_trans_image(trans_image, 180)
        trans_image = xr.where(np.isnan(trans_image), trans_image_rotated, trans_image)

    trans_image.attrs = rheed_image.attrs

    return trans_image


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
