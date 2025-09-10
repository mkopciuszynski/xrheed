from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.axes import Axes
from numpy.typing import NDArray

from xrheed.conversion.base import convert_sx_to_ky


def plot_profile(
    rheed_profile: xr.DataArray,
    ax: Optional[Axes] = None,
    transform_to_k: bool = True,
    normalize: bool = True,
    **kwargs,
) -> Axes:
    """
    Plot a RHEED intensity profile, optionally normalizing and transforming to kx.

    Parameters
    ----------
    rheed_profile : xr.DataArray
        The RHEED intensity profile to plot.
    ax : matplotlib.axes.Axes or None, optional
        The axes to plot on. If None, a new figure and axes are created.
    transform_to_k : bool, optional
        If True, transform the sx-axis to ky using experimental geometry.
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

    profile: xr.DataArray = rheed_profile.copy()

    if normalize:
        # Normalize the profile
        normalized: xr.DataArray = profile - profile.min()
        normalized = normalized / normalized.max()

        profile.values = normalized.values
        profile.attrs = rheed_profile.attrs.copy()

    if transform_to_k:
        k_e: float = profile.ri.ewald_sphere_radius
        screen_sample_distance: float = profile.ri.screen_sample_distance

        sx: NDArray = profile.coords["sx"].values

        kx = convert_sx_to_ky(
            sx,
            ewald_sphere_radius=k_e,
            screen_sample_distance_mm=screen_sample_distance,
        )

        ax.plot(
            kx,
            profile,
            **kwargs,
        )
        ax.set_xlabel("$k_y$ (1/Å)")

    else:
        profile.plot(ax=ax, **kwargs)
        ax.set_xlabel("$S_x$ (mm)")

    if normalize:
        ax.set_ylabel("Normalized Intensity")
    else:
        ax.set_ylabel("Intensity")

    return ax
