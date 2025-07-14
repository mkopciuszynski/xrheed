import matplotlib.pyplot as plt


import numpy as np
import xarray as xr


def plot_image(
    rheed_image: xr.DataArray,
    ax: plt.Axes | None = None,
    auto_levels: float = 0.0,
    show_center_lines: bool = True,
    **kwargs,
):
    """Plot a RHEED image."""

    if auto_levels > 0.0:
        vmin, vmax = _set_auto_levels(rheed_image, auto_levels)
        kwargs["vmin"] = vmin
        kwargs["vmax"] = vmax

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    if show_center_lines:
        ax.axhline(y=0.0, linewidth=0.5)
        ax.axvline(x=0.0, linewidth=0.5)

    rheed_image.plot(ax=ax, add_colorbar=False, cmap="gray", **kwargs)

    roi_width = rheed_image.ri.screen_roi_width
    roi_height = rheed_image.ri.screen_roi_height

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
    auto_levels : float
        Percentage of pixels to clip at both low and high ends.
        Higher values increase contrast.

    Returns
    -------
    vmin, vmax : tuple of floats
        Suggested display levels for the image.
    """

    # Extract ROI based on screen dimensions from the xarray accessor
    screen_roi_width = image.ri.screen_roi_width
    screen_roi_height = image.ri.screen_roi_height

    roi_image = image.sel(
        x=slice(-screen_roi_width, screen_roi_width), y=slice(-screen_roi_height, 0)
    )

    # Flatten, exclude NaNs
    values = roi_image.values.ravel()
    values = values[~np.isnan(values)]

    # Compute clipped percentiles
    low_percentile = auto_levels
    high_percentile = 100 - auto_levels

    vmin = np.percentile(values, low_percentile)
    vmax = np.percentile(values, high_percentile)

    return float(vmin), float(vmax)
