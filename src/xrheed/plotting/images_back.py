from __future__ import annotations

from scipy import ndimage

import numpy as np
import xarray as xr

from xrheed.kinematics.Lattice import Lattice
import matplotlib.pyplot as plt


def plot_image(
    rheed_image: xr.DataArray,
    ax: plt.Axes | None = None,
    hp_filter: bool = False,
    auto_levels: float = 0.0,
    show_center_lines: bool = True,
    **kwargs,
) -> plt.Axes:
    """Plot a RHEED image."""
    

    if hp_filter:

        hp_power = rheed_image.R.hp_threshold
        hp_sigma = rheed_image.R.hp_sigma

        blurred_image = ndimage.gaussian_filter(rheed_image, sigma=hp_sigma)
        high_pass_image = rheed_image.values - hp_power * blurred_image
        high_pass_image -= high_pass_image.min()

        rheed_image.values = high_pass_image


    if auto_levels > 0.0:
        vmin, vmax = _set_auto_levels(rheed_image, auto_levels)
        kwargs["vmin"] = vmin
        kwargs["vmax"] = vmax

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    if show_center_lines:
        ax.axhline(y=0.0, linewidth=0.5)
        ax.axvline(x=0.0, linewidth=0.5)

    print(kwargs)

    rheed_image.plot(ax=ax, add_colorbar=False, cmap="gray", **kwargs)

    ax.set_aspect(1)
    ax.set_ylim(rheed_image.screen_size_y, -10)
    ax.set_xlim(-rheed_image.screen_size_x, rheed_image.screen_size_x)

    return ax


def plot_real(lat: Lattice, ax=None, attr=".b", **kwargs):
    if not isinstance(ax, plt.Axes):
        fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(lat.lattice_x, lat.lattice_y, attr)

    return ax


def plot_inverse(lat: Lattice, ax=None, attr=".r", **kwargs):
    if not isinstance(ax, plt.Axes):
        fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(lat.rec_lattice_x, lat.rec_lattice_y, attr)

    return ax


def plot_evald(
    lat: Lattice,
    ax: plt.Axes | None = None,
    phi: float = 0.0,
    theta: float = 1.0,
    screen_size_x: float = 50,
    screen_size_y: float = 60,
    attr=".b",
    **kwargs,
):

    px, py = lat.calculate_evald(
        phi=phi,
        theta=theta,
        screen_size_x=screen_size_x,
        screen_size_y=screen_size_y,
        **kwargs,
    )

    if not isinstance(ax, plt.Axes):
        fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(px, py, attr, **kwargs)

    ax.set_aspect(1)
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

    # Extract the raw data as a NumPy array
    data_flat = image.values.ravel()

    # Remove NaNs if present
    data_flat = data_flat[~np.isnan(data_flat)]

    # Compute vmin as the minimum value, vmax from the given percentile
    vmin = float(np.min(data_flat))
    vmax = float(np.percentile(data_flat, 100 - auto_contrast))

    return vmin, vmax
