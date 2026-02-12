import logging

import numpy as np
import xarray as xr
from numpy.typing import NDArray
from scipy import ndimage  # type: ignore
from tqdm.auto import tqdm
from typing import Union
import warnings

from ..constants import IMAGE_NDIMS, STACK_NDIMS
from .base import convert_gx_gy_to_sx_sy

logger = logging.getLogger(__name__)


def transform_image_to_kxky(
    rheed_data: xr.DataArray,
    rotate: bool = True,
    point_symmetry: bool = False,
    n_fold_symmetry: Union[int, None] = None,
) -> xr.DataArray:
    """
    Transform a RHEED image or stack into kx-ky coordinates.

    Parameters
    ----------
    rheed_data : xr.DataArray
        RHEED image or stack with coordinates ('sx', 'sy'), optionally 'azimuthal_angle'.
    rotate : bool, optional
        If True, rotate the transformed image(s) by the azimuthal angle.
    point_symmetry : bool, optional
        If True, combine with a 180deg-rotated copy to enforce point symmetry.

    Returns
    -------
    xr.DataArray
        Transformed image or stack in kx-ky coordinates.
    """

    # --- Physical and geometric parameters ---
    screen_sample_distance = rheed_data.ri.screen_sample_distance
    ewald_radius = rheed_data.ri.ewald_radius
    incident_angle = rheed_data.ri.incident_angle
    azimuthal_angle = rheed_data.ri.azimuthal_angle

    logger.info(
        "transform_image_to_kxky: rheed_data.ndim=%s rotate=%s n_fold_symmetry=%s",
        rheed_data.ndim,
        rotate,
        n_fold_symmetry,
    )

    # --- Target coordinate grid ---
    # TODO add kwargs to set this by user
    kx: NDArray[np.float32] = np.arange(-10, 10, 0.01, dtype=np.float32)
    ky: NDArray[np.float32] = np.arange(-10, 10, 0.01, dtype=np.float32)
    gx, gy = np.meshgrid(kx, ky, indexing="ij")

    logger.debug(
        "transform_image_to_kxky: grid shapes kx=%s ky=%s gx=%s gy=%s",
        kx.shape,
        ky.shape,
        gx.shape,
        gy.shape,
    )

    sx_to_kx, sy_to_ky = convert_gx_gy_to_sx_sy(
        gx,
        gy,
        ewald_radius=ewald_radius,
        incident_angle=incident_angle,
        screen_sample_distance=screen_sample_distance,
        remove_outside=False,
    )

    sx = xr.DataArray(sx_to_kx, dims=["kx", "ky"], coords={"kx": kx, "ky": ky})
    sy = xr.DataArray(sy_to_ky, dims=["kx", "ky"], coords={"kx": kx, "ky": ky})

    logger.debug(
        "transform_image_to_kxky: converted sx/sy shapes %s %s", sx.shape, sy.shape
    )

    
    def _transform_single_image(image: xr.DataArray, 
                                angle: float) -> xr.DataArray:
        
        transformed_base = image.interp(sx=sx, sy=sy, method="linear")

        # Primary rotation - according to azimuthal orientation
        if rotate:
            transformed_base = _rotate_trans_image(transformed_base, angle)

        # n-fold rotational symmetry
        if n_fold_symmetry is not None and n_fold_symmetry > 1:
            
            symmetry_images = []
            symmetry_images.append(transformed_base)

            rotation_step = 360.0 / n_fold_symmetry

            for i in range(1, n_fold_symmetry):
                rot_angle = i * rotation_step
                rotated = _rotate_trans_image(transformed_base, rot_angle)
                symmetry_images.append(rotated)
        
            symmetry_stack = xr.concat(symmetry_images, dim="n")
            transformed = symmetry_stack.max("n")
        else:
            transformed = transformed_base

        transformed.attrs = image.attrs
        return transformed

    # --- Handle single image ---
    if rheed_data.ndim == IMAGE_NDIMS:
        logger.info(
            "transform_image_to_kxky: processing single image case with alpha=%.3f",
            float(azimuthal_angle),
        )
        return _transform_single_image(rheed_data, float(azimuthal_angle))

    # --- Handle stack with alpha coordinate ---
    if rheed_data.ndim == STACK_NDIMS and "alpha" in rheed_data.coords:
        logger.info(
            "transform_image_to_kxky: processing stack with alpha coordinate, size=%d",
            rheed_data.sizes["alpha"],
        )
        transformed_slices = []
        for i in tqdm(range(rheed_data.sizes["alpha"]), desc="Transforming slices"):
            transformed_slices.append(
                _transform_single_image(
                    rheed_data.isel(alpha=i), float(azimuthal_angle[i])
                )
            )
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

    logger.debug(
        "called _rotate_trans_image: angle=%.3f mode=%s input_shape=%s",
        angle,
        mode,
        trans_image.shape,
    )

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
