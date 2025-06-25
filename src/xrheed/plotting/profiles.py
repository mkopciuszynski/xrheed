from __future__ import annotations

import matplotlib.pyplot as plt


def plot_profile(
    self,
    ax=None,
    center_size=None,
    position_size=None,
    kx_scale=None,
    vshift=0,
    asymetry=0,
    attr="-b",
    **kwargs,
):

    profile = self.get_profile(
        center_size=center_size,
        position_size=position_size,
        kx_scale=kx_scale,
        asymetry=asymetry,
    )
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
