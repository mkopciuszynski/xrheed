from __future__ import annotations

import xarray as xr
from xrheed.conversion.base import convert_x_to_kx
import matplotlib.pyplot as plt
import numpy as np


def plot_profile(
    rheed_profile: xr.DataArray,
    ax: plt.Axes | None = None,
    transform_to_kx: bool = True,
    normalize: bool = True,
    **kwargs,
) -> plt.Axes:
    """
    Plot a RHEED intensity profile, optionally normalizing and transforming to kx.

    Parameters
    ----------
    rheed_profile : xr.DataArray
        The RHEED intensity profile to plot.
    ax : matplotlib.axes.Axes or None, optional
        The axes to plot on. If None, a new figure and axes are created.
    transform_to_kx : bool, optional
        If True, transform the x-axis to kx using experimental geometry.
    normalize : bool, optional
        If True, normalize the intensity profile.
    **kwargs
        Additional keyword arguments passed to matplotlib plot.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plotted profile.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(3.5, 2))

    profile = rheed_profile.copy()

    if normalize:
        # Normalize the profile
        normalized = profile - np.min(profile)
        normalized = normalized / np.max(normalized)
        profile.values = normalized.values
        profile.attrs = rheed_profile.attrs.copy()

    if transform_to_kx:

        k_e = profile.ri.ewald_sphere_radius
        screen_sample_distance = profile.ri.screen_sample_distance

        x = profile.coords["x"].data

        kx = convert_x_to_kx(
            x, ewald_sphere_radius=k_e, screen_sample_distance_mm=screen_sample_distance
        )

        ax.plot(
            kx,
            profile,
            **kwargs,
        )
        ax.set_xlabel("$k_x$ (1/Å)")

    else:
        profile.plot(ax=ax, **kwargs)
        ax.set_xlabel("x (mm)")

    if normalize:
        ax.set_ylabel("Normalized Intensity")
    else:
        ax.set_ylabel("Intensity")

    return ax
