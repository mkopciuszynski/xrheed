from __future__ import annotations

from pyrheed.Rheed import Rheed
from pyrheed.ewald.Lattice import Lattice
import matplotlib.pyplot as plt

def plot_image(rheed: Rheed,
               ax: plt.Axes | None = None,
               vmax: float | None = None,
               vmin: float | None = None,
               **kwargs):

    image = rheed.image
    if vmax is None:
        vmax = rheed.vmax
    if vmin is None:
        vmin = rheed.vmin

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    #        ax.set_xlim(self.center_size[0] - self.center_size[2] // 2, self.center_size[0] + self.center_size[2] // 2)
    #        ax.set_ylim(self.center_size[1] + self.center_size[3], self.center_size[1])
    ax.axhline(y=0.0, linewidth=0.5)
    ax.axvline(x=0.0, linewidth=0.5)
    image.plot(ax=ax, add_colorbar=False, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_aspect(1)
    ax.set_ylim(rheed.screen_size_y, -10)
    ax.set_xlim(-rheed.screen_size_x, rheed.screen_size_x)

#    ax.set_xlim(-rheed.size[0]//2, rheed.size[0]//2)
#    ax.set_ylim(-rheed.size[1]+1, 1)
#    ax.set_aspect(aspect=(rheed.size[0] / rheed.size[1]))
    return ax

def plot_profile(self, ax=None, center_size=None, position_size=None, kx_scale=None, vshift=0, asymetry=0,
                 attr='-b', **kwargs):

    profile = self.get_profile(center_size=center_size, position_size=position_size, kx_scale=kx_scale,
                               asymetry=asymetry)
    profile -= profile.min()
    profile /= profile.max()

    kx = self.kx

    if not isinstance(ax, plt.Axes):
        fig, axs = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
        fig.subplots_adjust(hspace=0.2)
        axs[0].imshow(self.roi, cmap="gray")
        axs[1].plot(kx, profile, attr, **kwargs)
    else:
        ax.plot(kx, profile + vshift, attr, **kwargs)

def plot_real(lat: Lattice, ax=None, attr='.b', **kwargs):
    if not isinstance(ax, plt.Axes):
        fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(lat.lattice_x, lat.lattice_y, attr)

    return ax


def plot_inverse(lat: Lattice, ax=None, attr='.r', **kwargs):
    if not isinstance(ax, plt.Axes):
        fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(lat.rec_lattice_x, lat.rec_lattice_y, attr)

    return ax


def plot_evald(lat: Lattice, ax: plt.Axes | None = None,
               phi: float = 0.0, theta: float = 1.0,
               screen_size_x: float = 50, screen_size_y: float = 60,
               attr='.b', **kwargs):

    px, py = lat.calculate_evald(phi=phi, theta=theta,
                                 screen_size_x=screen_size_x,
                                 screen_size_y=screen_size_y,
                                 **kwargs)

    if not isinstance(ax, plt.Axes):
        fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(px, py, attr, **kwargs)

    ax.set_aspect(1)
    return ax
