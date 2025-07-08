import matplotlib.pyplot as plt

from scipy import ndimage

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

    roi_width = rheed_image.R.screen_roi_width
    roi_height = rheed_image.R.screen_roi_height

    ax.set_aspect(1)
    ax.set_xlim(-roi_width, roi_width)
    ax.set_ylim(-roi_height, 10)
    ax.set_xlabel("Screen X (mm)")
    ax.set_ylabel("Screen Y (mm)")

    return ax




def _set_auto_levels(
    image: xr.DataArray,
    auto_contrast: float = 5.0
) -> tuple[float, float]:
    """
    Compute vmin and vmax for plotting, setting vmax such that a given
    percentage of pixels are overexposed.

    Parameters:
    -----------
    image : xr.DataArray
        Image data.
    auto_contrast : float
        Percentage of overexposed pixels allowed (e.g., 5.0 for 5%).

    Returns:
    --------
    vmin, vmax : tuple of floats
        Suggested min and max values for plotting.
    """


    screen_roi_width = image.R.screen_roi_width
    screen_roi_height = image.R.screen_roi_height
    roi_image = image.sel(
        x=slice(-screen_roi_width, screen_roi_width),
        y=slice(-screen_roi_height, 0)
    )

    # Extract the raw data as a NumPy array
    data_flat = roi_image.values.ravel()

    # Remove NaNs if present
    data_flat = data_flat[~np.isnan(data_flat)]

    # Compute vmin as the minimum value, vmax from the given percentile
    vmin = float(np.min(data_flat))
    vmax = float(np.percentile(data_flat, 100 - auto_contrast))

    return vmin, vmax
