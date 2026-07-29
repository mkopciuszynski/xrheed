import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from numpy.typing import NDArray


def _set_auto_levels(
    image: xr.DataArray,
    auto_levels: float | tuple[float, float] = 5.0,
) -> tuple[float, float]:
    """
    Calculate vmin and vmax for displaying an image with enhanced contrast,
    using a region of interest defined by screen dimensions.

    Parameters
    ----------
    image : xr.DataArray
        The input image (2D xarray DataArray) with RHEED screen ROI attributes.
    auto_levels : float or tuple[float, float], optional
        - If float: Percentage of pixels to clip at both low and high ends (e.g., 5.0 -> 5th and 95th percentiles).
        - If tuple: Explicit low and high percentiles (e.g., (1.0, 99.0)).

    Returns
    -------
    tuple[float, float]
        Suggested display levels (vmin, vmax) for the image.
    """
    if isinstance(auto_levels, (int, float)):
        low_percentile = float(auto_levels)
        high_percentile = 100.0 - float(auto_levels)
    else:
        low_percentile, high_percentile = auto_levels

    # Extract ROI based on screen dimensions from the xarray accessor
    screen_roi_width: float = image.ri.screen_roi_width
    screen_roi_height: float = image.ri.screen_roi_height

    roi_image = image.sel(
        sx=slice(-screen_roi_width, screen_roi_width),
        sy=slice(-screen_roi_height, 0),
    )

    # Flatten, exclude NaNs
    values: NDArray[np.float32] = roi_image.values.ravel()
    values = values[~np.isnan(values)]

    # Fallback if the slice result is empty or all-NaN
    if values.size == 0:
        return float(image.min().item()), float(image.max().item())

    vmin: float = float(np.percentile(values, low_percentile))
    vmax: float = float(np.percentile(values, high_percentile))

    return vmin, vmax


def plot_image(
    rheed_image: xr.DataArray,
    ax: Axes | None = None,
    auto_levels: float | tuple[float, float] = 0.0,
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
    auto_levels : float or tuple[float, float], optional
        - If > 0 or a tuple, automatically set vmin/vmax using percentile clipping inside the ROI.
        - Pass a float `p` to clip the bottom `p%` and top `p%`.
        - Pass a tuple `(p_low, p_high)` for asymmetric bounds.
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

    # Calculate auto levels if user hasn't supplied BOTH vmin and vmax
    if "vmin" not in kwargs or "vmax" not in kwargs:
        if auto_levels:
            vmin_calc, vmax_calc = _set_auto_levels(rheed_image, auto_levels)
        else:
            vmin_calc = float(rheed_image.min().item())
            vmax_calc = float(rheed_image.max().item())

        # Set calculated defaults without overwriting explicit kwargs
        kwargs.setdefault("vmin", vmin_calc)
        kwargs.setdefault("vmax", vmax_calc)

    if "cmap" not in kwargs:
        kwargs.setdefault("cmap", "gray")
    if "add_colorbar" not in kwargs:
        kwargs.setdefault("add_colorbar", False)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    rheed_image.plot(ax=ax, **kwargs)
    ax.set_aspect(1.0)

    if show_center_lines:
        ax.axhline(y=0.0, linewidth=1.0, color="c", linestyle="-", alpha=0.8)
        ax.axvline(x=0.0, linewidth=1.0, color="c", linestyle="-", alpha=0.8)

    beta = rheed_image.ri.beta
    if show_specular_spot and beta is not None:
        specular_y = -np.tan(np.deg2rad(beta)) * rheed_image.ri.screen_sample_distance
        ax.scatter(
            0.0, specular_y, marker="o", edgecolors="c", facecolors="none", s=100
        )
        ax.scatter(
            0.0, -specular_y, marker="o", edgecolors="m", facecolors="none", s=50
        )

    roi_width: float = rheed_image.ri.screen_roi_width
    roi_height: float = rheed_image.ri.screen_roi_height

    ax.set_xlim(-roi_width, roi_width)
    ax.set_ylim(-roi_height, rheed_image.sy.max())
    ax.set_title("")
    ax.set_xlabel("Screen X (mm)")
    ax.set_ylabel("Screen Y (mm)")

    return ax
