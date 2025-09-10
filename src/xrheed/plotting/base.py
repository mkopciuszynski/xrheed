from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from numpy.typing import NDArray


def plot_image(
    rheed_image: xr.DataArray,
    ax: Optional[Axes] = None,
    auto_levels: float = 0.0,
    show_center_lines: bool = True,
    show_specular_spot: bool = False,
    **kwargs,
) -> Axes:
    """
    Plot a RHEED image using matplotlib.

    Parameters
    ----------
    rheed_image : xr.DataArray
        The RHEED image to plot.
    ax : matplotlib.axes.Axes or None, optional
        The axes to plot on. If None, a new figure and axes are created.
    auto_levels : float, optional
        If > 0, automatically set vmin/vmax for contrast enhancement.
    show_center_lines : bool, optional
        If True, show center lines at x=0 and y=0.
    show_specular_spot : bool, optional
        If True, overlay the specularly reflected spot on the image.
    **kwargs
        Additional keyword arguments passed to xarray plot.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plotted image.
    """

    if auto_levels > 0.0:
        vmin, vmax = _set_auto_levels(rheed_image, auto_levels)
        kwargs["vmin"] = vmin
        kwargs["vmax"] = vmax

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    rheed_image.plot(ax=ax, add_colorbar=False, cmap="gray", **kwargs)

    if show_center_lines:
        ax.axhline(y=0.0, linewidth=1.0, linestyle="--")
        ax.axvline(x=0.0, linewidth=1.0, linestyle="--")

    if show_specular_spot:
        specular_y = (
            -np.tan(np.deg2rad(rheed_image.ri.beta))
            * rheed_image.ri.screen_sample_distance
        )
        ax.scatter(
            0.0, specular_y, marker="o", edgecolors="c", facecolors="none", s=100
        )

    roi_width: float = rheed_image.ri.screen_roi_width
    roi_height: float = rheed_image.ri.screen_roi_height

    ax.set_aspect(1)
    ax.set_xlim(-roi_width, roi_width)
    ax.set_ylim(-roi_height, 10)
    ax.set_xlabel("Screen X (mm)")
    ax.set_ylabel("Screen Y (mm)")

    return ax


def _set_auto_levels(
    image: xr.DataArray, auto_levels: float = 5.0
) -> tuple[float, float]:
    """
    Calculate vmin and vmax for displaying an image with enhanced contrast,
    using a region of interest defined by screen dimensions.

    Parameters
    ----------
    image : xr.DataArray
        The input image (2D xarray DataArray) with RHEED screen ROI attributes.
    auto_levels : float, optional
        Percentage of pixels to clip at both low and high ends. Higher values increase contrast.

    Returns
    -------
    tuple[float, float]
        Suggested display levels (vmin, vmax) for the image.
    """

    # Extract ROI based on screen dimensions from the xarray accessor
    screen_roi_width: float = image.ri.screen_roi_width
    screen_roi_height: float = image.ri.screen_roi_height

    roi_image = image.sel(
        sx=slice(-screen_roi_width, screen_roi_width), sy=slice(-screen_roi_height, 0)
    )

    # Flatten, exclude NaNs
    values: NDArray[np.uint8] = roi_image.values.ravel()
    values = values[~np.isnan(values)]

    # Compute clipped percentiles
    low_percentile: float = auto_levels
    high_percentile: float = 100 - auto_levels

    vmin: float = float(np.percentile(values, low_percentile))
    vmax: float = float(np.percentile(values, high_percentile))

    return vmin, vmax
