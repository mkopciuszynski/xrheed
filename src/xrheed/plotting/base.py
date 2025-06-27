import matplotlib.pyplot as plt


def plot_image(
    image,
    ax: plt.Axes | None = None,
    screen_roi_width: float = 10,
    screen_roi_height: float = 10,
    hp_filter: bool = False,
    auto_levels: bool = False,
    **kwargs,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    if auto_levels:
        vmin = image.min().values
        vmax = image.mean().values
        image = (image - vmin) / (vmax - vmin) * 50

    image.plot(ax=ax, cmap="gray", add_colorbar=False, **kwargs)

    ax.set_xlim(-screen_roi_width, screen_roi_width)
    ax.set_ylim(-screen_roi_height, 5)

    ax.set_xlabel("Screen x [mm]")
    ax.set_ylabel("Screen y [mm]")

    ax.axhline(y=0.0, linewidth=0.5, color="w")
    ax.axvline(x=0.0, linewidth=0.5, color="w")
    ax.set_aspect(1)

    return ax
