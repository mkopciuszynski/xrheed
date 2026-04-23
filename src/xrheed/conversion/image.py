import logging
import warnings
from typing import cast

import numpy as np
import xarray as xr
from numpy.typing import NDArray
from scipy import ndimage  # type: ignore
from tqdm.auto import tqdm

from ..constants import DEFAULT_K_VECT, IMAGE_NDIMS, MIRROR_ROT_DEG, STACK_NDIMS
from .base import convert_gx_gy_to_sx_sy

logger = logging.getLogger(__name__)


def transform_image_to_kxky(
    rheed_image: xr.DataArray,
    *,
    k_vect: np.ndarray | None = None,
    rotate: bool = True,
    point_symmetry: bool = False,
) -> xr.DataArray:
    """
    Transform a single RHEED image into kx-ky coordinates.

    Notes
    -----
    - Rotation is applied only if `rotate=True` AND azimuthal angle is available.
    - kx and ky are assumed identical and defined by `k_vect`.
    """

    # --- BACKWARD COMPATIBILITY: stack support ---
    if rheed_image.ndim == STACK_NDIMS:
        warnings.warn(
            "Passing a stack to transform_image_to_kxky is deprecated. "
            "Use transform_stack_to_kxky instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return transform_stack_to_kxky(
            rheed_image,
            k_vect=k_vect,
            rotate=rotate,
            point_symmetry=point_symmetry,
        )

    if rheed_image.ndim != IMAGE_NDIMS:
        raise ValueError(
            f"Unsupported ndim={rheed_image.ndim}, expected {IMAGE_NDIMS} (image)"
        )

    # --- k-vector ---
    if k_vect is None:
        k_vect = DEFAULT_K_VECT
    k_vect = np.asarray(k_vect, dtype=np.float32)

    gx, gy = np.meshgrid(k_vect, k_vect, indexing="ij")

    # --- geometry parameters ---
    ri = rheed_image.ri
    sx_to_kx, sy_to_ky = convert_gx_gy_to_sx_sy(
        gx,
        gy,
        ewald_radius=ri.ewald_radius,
        incident_angle=ri.incident_angle,
        screen_sample_distance=ri.screen_sample_distance,
        remove_outside=False,
    )

    sx = xr.DataArray(sx_to_kx, dims=("kx", "ky"), coords={"kx": k_vect, "ky": k_vect})
    sy = xr.DataArray(sy_to_ky, dims=("kx", "ky"), coords={"kx": k_vect, "ky": k_vect})

    # --- rotation logic ---
    rotate_angle: float | None = None
    if rotate and hasattr(ri, "azimuthal_angle") and ri.azimuthal_angle is not None:
        rotate_angle = float(ri.azimuthal_angle)

    return _transform_frame_kxky(
        rheed_image,
        sx=sx,
        sy=sy,
        rotate_angle=rotate_angle,
        point_symmetry=point_symmetry,
    )


def transform_stack_to_kxky(
    rheed_stack: xr.DataArray,
    *,
    stack_dim: str | None = None,
    azimuthal_angle_coord: str = "alpha",
    k_vect: np.ndarray | None = None,
    rotate: bool = True,
    point_symmetry: bool = False,
    show_progress: bool = False,
) -> xr.DataArray:
    """
    Transform a stack of RHEED images into kx-ky coordinates.
    """

    if rheed_stack.ndim < STACK_NDIMS:
        raise ValueError("Expected stack (>=3D). Use transform_image_to_kxky.")

    if stack_dim is None:
        stack_dim = cast(str, rheed_stack.dims[0])

    if stack_dim not in rheed_stack.dims:
        raise ValueError(f"Invalid stack_dim='{stack_dim}'")

    # --- k-vector ---
    if k_vect is None:
        k_vect = DEFAULT_K_VECT
    k_vect = np.asarray(k_vect, dtype=np.float32)

    gx, gy = np.meshgrid(k_vect, k_vect, indexing="ij")

    # --- compute sx, sy once ---
    ref = rheed_stack.isel({stack_dim: 0})
    ri = ref.ri

    sx_to_kx, sy_to_ky = convert_gx_gy_to_sx_sy(
        gx,
        gy,
        ewald_radius=ri.ewald_radius,
        incident_angle=ri.incident_angle,
        screen_sample_distance=ri.screen_sample_distance,
        remove_outside=False,
    )

    sx = xr.DataArray(sx_to_kx, dims=("kx", "ky"), coords={"kx": k_vect, "ky": k_vect})
    sy = xr.DataArray(sy_to_ky, dims=("kx", "ky"), coords={"kx": k_vect, "ky": k_vect})

    angles: xr.DataArray | None = None
    if rotate:
        if azimuthal_angle_coord not in rheed_stack.coords:
            raise ValueError(
                f"Missing coordinate '{azimuthal_angle_coord}' required for rotation"
            )
        angles = rheed_stack[azimuthal_angle_coord]

    iterator = range(rheed_stack.sizes[stack_dim])
    if show_progress:
        iterator = tqdm(iterator, desc="Transforming stack")

    results: list[xr.DataArray] = []

    for i in iterator:
        img = rheed_stack.isel({stack_dim: i})

        rotate_angle: float | None = None
        if rotate and angles is not None:
            rotate_angle = float(angles.isel({stack_dim: i}))

        transformed = _transform_frame_kxky(
            img,
            sx=sx,
            sy=sy,
            rotate_angle=rotate_angle,
            point_symmetry=point_symmetry,
        )

        coord_val = rheed_stack[stack_dim].values[i].item()
        transformed = transformed.expand_dims({stack_dim: [coord_val]})
        results.append(transformed)

    out = xr.concat(
        results,
        dim=stack_dim,
        coords="minimal",
        compat="override",
        join="override",
    )
    out.attrs = rheed_stack.attrs
    return out


def _transform_frame_kxky(
    frame: xr.DataArray,
    *,
    sx: xr.DataArray,
    sy: xr.DataArray,
    rotate_angle: float | None = None,
    point_symmetry: bool = False,
) -> xr.DataArray:
    """
    Transform a single 2D RHEED frame into kx-ky coordinates.
    """

    if frame.ndim != IMAGE_NDIMS:
        raise ValueError("_transform_frame_kxky expects a 2D DataArray")

    if not np.issubdtype(frame.dtype, np.floating):
        frame = frame.astype(np.float32)

    transformed = frame.interp(sx=sx, sy=sy, method="linear")

    if rotate_angle is not None:
        transformed = _rotate_trans_image(transformed, rotate_angle)

    if point_symmetry:
        rotated_180 = _rotate_trans_image(transformed, MIRROR_ROT_DEG)
        transformed = xr.where(np.isnan(transformed), rotated_180, transformed)

    transformed.attrs = frame.attrs
    return transformed


def _rotate_trans_image(
    trans_image: xr.DataArray,
    angle: float,
    mode: str = "constant",
) -> xr.DataArray:
    """
    Rotate a 2D xarray.DataArray around its center by a given angle.

    Notes
    -----
    - Assumes `trans_image` is already floating point (float32).
    - NaN regions are preserved using an explicit validity mask.
    """

    if trans_image.ndim != IMAGE_NDIMS:
        raise ValueError("_rotate_trans_image expects a 2D DataArray")

    logger.debug(
        "called _rotate_trans_image: angle=%.3f mode=%s input_shape=%s",
        angle,
        mode,
        trans_image.shape,
    )

    if "kx" not in trans_image.coords or "ky" not in trans_image.coords:
        raise ValueError("Rotation requires 'kx' and 'ky' coordinates")

    if not np.allclose(trans_image["kx"].values, trans_image["ky"].values):
        raise ValueError("kx and ky coordinates must be identical for rotation")

    valid_mask: NDArray[np.bool_] = ~np.isnan(trans_image.values)

    filled = trans_image.fillna(0.0)

    rotated_data = ndimage.rotate(
        filled.values,
        angle,
        reshape=False,
        order=3,
        mode=mode,
    )

    rotated_mask: NDArray[np.bool_] = ndimage.rotate(
        valid_mask, angle, reshape=False, order=0, mode=mode
    ).astype(bool)

    rotated = xr.DataArray(
        rotated_data,
        coords=trans_image.coords,
        dims=trans_image.dims,
        attrs=trans_image.attrs,
        name=trans_image.name,
    )

    return rotated.where(rotated_mask)
