import numpy as np
import xarray as xr
from numpy.typing import NDArray
from scipy import ndimage  # type: ignore
import logging

from ..constants import IMAGE_NDIMS, STACK_NDIMS
from .base import convert_gx_gy_to_sx_sy


logger = logging.getLogger(__name__)


def transform_image_to_kxky(
    rheed_data: xr.DataArray,
    rotate: bool = True,
    point_symmetry: bool = False,
) -> xr.DataArray:
    """
    Transform a RHEED image or stack into kx-ky coordinates.

    Parameters
    ----------
    rheed_data : xr.DataArray
        RHEED image or stack with coordinates ('sx', 'sy'), optionally 'alpha'.
    rotate : bool, optional
        If True, rotate the transformed image(s) by the incident angle alpha.
    point_symmetry : bool, optional
        If True, combine with a 180°-rotated copy to enforce point symmetry.

    Returns
    -------
    xr.DataArray
        Transformed image or stack in kx-ky coordinates.
    """

    # --- Physical and geometric parameters ---
    screen_sample_distance = rheed_data.ri.screen_sample_distance
    ewald_radius = rheed_data.ri.ewald_radius
    beta = rheed_data.ri.beta
    alpha = rheed_data.ri.alpha

    logger.info(
        "transform_image_to_kxky: rheed_data.ndim=%s rotate=%s point_symmetry=%s",
        rheed_data.ndim,
        rotate,
        point_symmetry,
    )

    # --- Target coordinate grid ---
    kx: NDArray[np.float32] = np.arange(-10, 10, 0.01, dtype=np.float32)
    ky: NDArray[np.float32] = np.arange(-10, 10, 0.01, dtype=np.float32)
    gx, gy = np.meshgrid(kx, ky, indexing="ij")

    logger.debug("transform_image_to_kxky: grid shapes kx=%s ky=%s gx=%s gy=%s", kx.shape, ky.shape, gx.shape, gy.shape)

    sx_to_kx, sy_to_ky = convert_gx_gy_to_sx_sy(
        gx,
        gy,
        ewald_radius=ewald_radius,
        beta=beta,
        screen_sample_distance=screen_sample_distance,
        remove_outside=False,
    )

    sx = xr.DataArray(sx_to_kx, dims=["kx", "ky"], coords={"kx": kx, "ky": ky})
    sy = xr.DataArray(sy_to_ky, dims=["kx", "ky"], coords={"kx": kx, "ky": ky})

    logger.debug("transform_image_to_kxky: converted sx/sy shapes %s %s", sx.shape, sy.shape)

    # --- Helper to process a single image ---
    def _transform_single_image(image: xr.DataArray, angle: float) -> xr.DataArray:
        """Transform and optionally rotate a single RHEED frame."""
        transformed = image.interp(sx=sx, sy=sy, method="linear")
        if rotate:
            transformed = _rotate_trans_image(transformed, angle)
        if point_symmetry:
            rotated_180 = _rotate_trans_image(transformed, 180)
            transformed = xr.where(np.isnan(transformed), rotated_180, transformed)
        transformed.attrs = image.attrs
        return transformed

    # --- Handle single image ---
    if rheed_data.ndim == IMAGE_NDIMS:
        logger.info("transform_image_to_kxky: processing single image case with alpha=%.3f", float(alpha))
        return _transform_single_image(rheed_data, float(alpha))

    # --- Handle stack with alpha coordinate ---
    if rheed_data.ndim == STACK_NDIMS and "alpha" in rheed_data.coords:
        logger.info("transform_image_to_kxky: processing stack with alpha coordinate, size=%d", rheed_data.sizes["alpha"])
        transformed_slices = [
            _transform_single_image(rheed_data.isel(alpha=i), float(alpha[i]))
            for i in range(rheed_data.sizes["alpha"])
        ]
        transformed_stack = xr.concat(transformed_slices, dim="alpha")
        transformed_stack = transformed_stack.assign_coords(alpha=rheed_data.alpha)
        transformed_stack.attrs = rheed_data.attrs
        return transformed_stack

    raise ValueError(
        f"Unsupported ndim={rheed_data.ndim}, expected {IMAGE_NDIMS} (image) or {STACK_NDIMS} (stack)"
    )


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

    logger.info("_rotate_trans_image: angle=%.3f mode=%s input_shape=%s", angle, mode, trans_image.shape)

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

    logger.debug("_rotate_trans_image: rotated shape=%s", rotated.shape)

    return rotated.where(rotated_mask)
