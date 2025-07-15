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
):
    """Plot a RHEED profile."""
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
