"""
RHEED data loading API.

This module provides a unified entry point for loading RHEED data.

Two loading modes are supported:

1. Plugin-based loading (recommended)
   - supports single images and stacks
   - uses registered plugins
   - reproducible and metadata-driven

2. Manual loading (beginner-friendly)
   - supports ONLY single-image loading
   - user provides geometry explicitly
   - still produces a canonical DataArray
"""

from pathlib import Path
from typing import Optional, Sequence, Union, Iterable
import logging

import numpy as np
import xarray as xr
from PIL import Image

from .plugins import PLUGINS, LoadRheedBase
from .constants import (
    IMAGE_DIMS,
    IMAGE_NDIMS,
    STACK_NDIMS,
    CANONICAL_STACK_DIMS,
)

logger = logging.getLogger(__name__)

__all__ = ["load_data"]


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------


def _normalize_paths(
    path: Union[str, Path, Sequence[Union[str, Path]]],
) -> list[Path]:
    """Normalize path argument to a non-empty list of Path objects."""
    if isinstance(path, (str, Path)):
        paths = [Path(path)]
    else:
        paths = [Path(p) for p in path]

    if not paths:
        raise ValueError("No input paths provided")

    return paths


def _validate_single_image_da(da: xr.DataArray) -> None:
    """Validate that a DataArray represents a canonical single RHEED image."""
    if da.ndim != IMAGE_NDIMS:
        raise ValueError(f"Invalid image ndim={da.ndim}, expected {IMAGE_NDIMS}")
    if set(da.dims) != IMAGE_DIMS:
        raise ValueError(f"Invalid image dims {set(da.dims)}, expected {IMAGE_DIMS}")


# ----------------------------------------------------------------------
# Plugin-based loading
# ----------------------------------------------------------------------


def _load_plugin_images(
    paths: Iterable[Path],
    *,
    plugin: str,
    **kwargs,
) -> list[xr.DataArray]:
    """Load one or more images via a registered plugin."""
    if plugin not in PLUGINS:
        raise ValueError(f"Unknown plugin: {plugin}")

    plugin_cls = PLUGINS[plugin]
    loader = plugin_cls()

    dataarrays: list[xr.DataArray] = []

    for p in paths:
        if not loader.is_file_accepted(p):
            raise ValueError(f"File not accepted by plugin '{plugin}': {p}")

        da = loader.load_single_image(p, **kwargs)
        _validate_single_image_da(da)
        dataarrays.append(da)

    return dataarrays


# ----------------------------------------------------------------------
# Manual loading (single image ONLY)
# ----------------------------------------------------------------------


def _load_manual_single_image(
    path: Path,
    *,
    screen_scale: float,
    screen_center_sy_px: Optional[int] = None,
    screen_center_sx_px: Optional[int] = None,
    screen_sample_distance: Optional[float] = None,
    beam_energy: Optional[float] = None,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
) -> xr.DataArray:
    """
    Manual loading path.

    This is NOT a plugin and does NOT implement the plugin abstract API.
    The user explicitly supplies geometry and (optional) acquisition parameters.

    Only single-image loading is supported.
    """

    # --- Load image ---
    image = Image.open(path).convert("L")
    image_np = np.asarray(image, dtype=np.uint8)
    h, w = image_np.shape

    # --- Resolve geometry ---
    cx = screen_center_sx_px if screen_center_sx_px is not None else w // 2
    cy = screen_center_sy_px if screen_center_sy_px is not None else h // 2
    px_to_mm = float(screen_scale)

    sx = (np.arange(w) - cx) / px_to_mm
    sy = (cy - np.arange(h)) / px_to_mm

    image_np = np.flipud(image_np)
    sy = np.flip(sy)

    # Assemble attrs directly from manual arguments
    attrs = {
        "screen_scale": screen_scale,
        "screen_center_sx_px": cx,
        "screen_center_sy_px": cy,
        "screen_sample_distance": screen_sample_distance,
        "beam_energy": beam_energy,
        "alpha": alpha,
        "beta": beta,
    }

    # --- Create DataArray ---
    da = xr.DataArray(
        image_np,
        dims=("sy", "sx"),
        coords={"sy": sy, "sx": sx},
        attrs=attrs,
    )

    return da


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


def load_data(
    path: Union[str, Path, Sequence[Union[str, Path]]],
    plugin: Optional[str] = None,
    *,
    stack_dim: Optional[str] = None,
    stack_coords: Optional[Sequence] = None,
    # ---- manual-loading arguments  ----
    screen_scale: Optional[float] = None,
    screen_center_sy_px: Optional[int] = None,
    screen_center_sx_px: Optional[int] = None,
    screen_sample_distance: Optional[float] = None,
    beam_energy: Optional[float] = None,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    **kwargs,
) -> xr.DataArray:
    """
    Load RHEED image data.

    Parameters
    ----------
    path
        Path or paths to image files.
    plugin
        Name of a registered plugin (recommended).
        If None, manual loading is used.
    stack_dim
        Dimension name for stacking multiple images (plugin mode only).
    stack_coords
        Optional coordinate values for the stack dimension.

    Returns
    -------
    xr.DataArray
        Canonical RHEED DataArray.
    """

    paths = _normalize_paths(path)

    # ------------------------------------------------------------------
    # 1. Manual loading path (single image only)
    # ------------------------------------------------------------------
    if plugin is None:
        if len(paths) != 1:
            raise ValueError(
                "Manual loading supports only a single image. "
                "Use a plugin for multi-image loading."
            )

        if screen_scale is None:
            raise ValueError("Manual loading requires screen_scale to be provided")

        logger.info(
            "Using manual loading path (beginner mode). "
            "Plugin-based loading is recommended for reproducibility."
        )

        return _load_manual_single_image(
            paths[0],
            screen_scale=screen_scale,
            screen_center_sy_px=screen_center_sy_px,
            screen_center_sx_px=screen_center_sx_px,
            screen_sample_distance=screen_sample_distance,
            beam_energy=beam_energy,
            alpha=alpha,
            beta=beta,
        )

    # ------------------------------------------------------------------
    # 2. Plugin-based loading path
    # ------------------------------------------------------------------
    dataarrays = _load_plugin_images(
        paths,
        plugin=plugin,
        **kwargs,
    )

    # ------------------------------------------------------------------
    # 3. Single image → return directly
    # ------------------------------------------------------------------
    if len(dataarrays) == 1:
        if stack_dim is not None or stack_coords is not None:
            raise ValueError("stack_dim / stack_coords provided for a single image")
        return dataarrays[0]

    # ------------------------------------------------------------------
    # 4. Multiple images → stacking required
    # ------------------------------------------------------------------
    if stack_dim is None:
        raise ValueError("stack_dim must be provided when loading multiple images")

    if stack_dim not in CANONICAL_STACK_DIMS:
        logger.warning(
            f"Non-canonical stack dimension '{stack_dim}'. "
            "This is allowed but discouraged."
        )

    stacked = xr.concat(dataarrays, dim=stack_dim)

    if stacked.ndim != STACK_NDIMS:
        raise ValueError(
            f"Stacked data has ndim={stacked.ndim}, expected {STACK_NDIMS}"
        )

    # ------------------------------------------------------------------
    # 5. Assign stack coordinates (structural only)
    # ------------------------------------------------------------------
    if stack_coords is not None:
        if len(stack_coords) != len(dataarrays):
            raise ValueError("Length of stack_coords does not match number of images")
        stacked = stacked.assign_coords({stack_dim: stack_coords})

    # ------------------------------------------------------------------
    # 6. Conservative promotion of acquisition parameters
    # ------------------------------------------------------------------
    for key in ("alpha", "beta"):
        values = [da.attrs.get(key) for da in dataarrays]

        # Promote only if all values exist and they vary
        if all(v is not None for v in values):
            if len(set(values)) > 1 and key not in stacked.coords:
                stacked = stacked.assign_coords({key: (stack_dim, values)})

    return stacked
