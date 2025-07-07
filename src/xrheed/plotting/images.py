from __future__ import annotations

from xrheed.Rheed import Rheed
from xrheed.kinematics.Lattice import Lattice
import matplotlib.pyplot as plt


def plot_image(
    rheed: Rheed,
    ax: plt.Axes | None = None,
    vmax: float | None = None,
    vmin: float | None = None,
    show_center_lines: bool = True,
    **kwargs,
) -> plt.Axes:
    """Plot a RHEED image."""

    image = rheed.image
    if vmax is None:
        vmax = rheed.vmax
    if vmin is None:
        vmin = rheed.vmin

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    if show_center_lines:
        ax.axhline(y=0.0, linewidth=0.5)
        ax.axvline(x=0.0, linewidth=0.5)

    image.plot(ax=ax, add_colorbar=False, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_aspect(1)
    ax.set_ylim(rheed.screen_size_y, -10)
    ax.set_xlim(-rheed.screen_size_x, rheed.screen_size_x)

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
