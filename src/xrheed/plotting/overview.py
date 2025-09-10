import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.figure import Figure

from xrheed.plotting.base import plot_image


def plot_images(
    rheed_images: list[xr.DataArray],
    ncols: int = 3,
    auto_levels: float = 0.0,
    show_center_lines: bool = True,
    **kwargs,
) -> Figure:
    """Plot a list of RHEED images in a grid layout.
    Parameters
    ----------
    rheed_images : list[xr.DataArray]
        List of RHEED images to plot.
    ncols : int, optional
        Number of columns in the grid layout (default is 3).
    auto_levels : float, optional
        If > 0, automatically set vmin/vmax for contrast enhancement (default is 0.0).
    show_center_lines : bool, optional
        If True, show center lines at x=0 and y=0 (default is False).
    **kwargs
        Additional keyword arguments passed to xarray plot.
    Returns
    -------
    plt.figure
        The figure containing the plotted images.
    """

    n_images: int = len(rheed_images)

    if n_images < 2:
        raise ValueError("At least two images are required to plot in a grid layout.")

    nrows: int = (n_images + ncols - 1) // ncols  # Calculate number of rows needed

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(ncols * 3, nrows * 2),
        sharex=True,
        sharey=True,
    )
    axes = axes.flatten() if nrows > 1 else axes

    for i, image in enumerate(rheed_images):
        ax = axes[i]
        plot_image(
            image,
            ax=ax,
            auto_levels=auto_levels,
            show_center_lines=show_center_lines,
            **kwargs,
        )
        ax.set_title(f"Image {i + 1}")
        if i % ncols == 0:
            ax.set_ylabel("Screen Y (mm)")
        else:
            ax.set_ylabel("")
        if i // ncols == nrows - 1:
            ax.set_xlabel("Screen X (mm)")
        else:
            ax.set_xlabel("")

    # Hide any unused axes
    for i in range(n_images, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()

    return fig
