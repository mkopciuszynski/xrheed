from typing import Sequence, Union

import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.figure import Figure

from ..constants import IMAGE_NDIMS, STACK_NDIMS
from .base import plot_image


def plot_images(
    rheed_data: Union[xr.DataArray, Sequence[xr.DataArray]],
    ncols: int = 3,
    fig_w: float = 3,
    auto_levels: float = 0.0,
    show_center_lines: bool = False,
    show_specular_spot: bool = False,
    **kwargs,
) -> Figure:
    """
    Plot multiple RHEED images in a grid layout.

    Parameters
    ----------
    rheed_data : xr.DataArray or list[xr.DataArray]
        Either:
          - A 3D stack (DataArray with one leading dimension, e.g. 'alpha'),
          - A list of 2D RHEED images.
        Must contain more than one image.
    ncols : int, optional
        Number of columns in the grid layout (default: 3).
    fig_w : float, optional
        Width of single RHEED image (in)
    auto_levels : float, optional
        If > 0, automatically set vmin/vmax for contrast enhancement (default: 0.0).
    show_center_lines : bool, optional
        If True, show center lines at x=0 and y=0 (default: False).
    show_specular_spot : bool, optional
        If True, show specular spot (default: False).
    **kwargs
        Additional keyword arguments passed to `plot_image()`.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the plotted images.
    """

    coord_values = None
    stack_dim = None

    # --- Case 1: list of images ---
    if isinstance(rheed_data, (list, tuple)):
        if not rheed_data:
            raise ValueError("Empty list of images provided.")
        if not all(isinstance(img, xr.DataArray) for img in rheed_data):
            raise TypeError(
                f"All images in the list must be {IMAGE_NDIMS}D DataArrays."
            )
        if len(rheed_data) < 2:
            raise ValueError("At least two images are required to plot a grid layout.")
        images = rheed_data

    # --- Case 2: 3D stack ---
    elif isinstance(rheed_data, xr.DataArray):
        if rheed_data.ndim != STACK_NDIMS:
            raise ValueError(
                f"rheed_data must be a {STACK_NDIMS}D DataArray (stack of images). "
                f"Use `plot_image()` for single {IMAGE_NDIMS}D images."
            )
        stack_dim = rheed_data.dims[0]
        n_frames = rheed_data.sizes[stack_dim]
        if n_frames < 2:
            raise ValueError("Stack must contain at least two images to plot.")
        images = [rheed_data.isel({stack_dim: i}) for i in range(n_frames)]
        coord_values = rheed_data[stack_dim].values

    else:
        raise TypeError("rheed_data must be an xarray.DataArray or list of them.")

    n_images = len(images)
    nrows = (n_images + ncols - 1) // ncols

    first_image = images[0]
    fig_ratio: float = (
        first_image.ri.screen_roi_width * 2.0 / first_image.ri.screen_roi_height
    )
    fig_h: float = fig_w / fig_ratio * 1.2

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(ncols * fig_w, nrows * fig_h),
        sharex=True,
        sharey=True,
    )

    axes = axes.flat if hasattr(axes, "flat") else [axes]

    for i, image in enumerate(images):
        ax = axes[i]
        plot_image(
            image,
            ax=ax,
            auto_levels=auto_levels,
            show_center_lines=show_center_lines,
            show_specular_spot=show_specular_spot,
            **kwargs,
        )

        # Use coordinate value as title if stack
        if coord_values is not None and stack_dim is not None:
            title_value = coord_values[i]
            ax.set_title(f"{stack_dim} = {title_value:2.2f}")
        else:
            ax.set_title(f"Image {i + 1}")

        # Label only edge axes
        if i % ncols == 0:
            ax.set_ylabel("Screen Y (mm)")
        else:
            ax.set_ylabel("")
        if i // ncols == nrows - 1:
            ax.set_xlabel("Screen X (mm)")
        else:
            ax.set_xlabel("")

    # Hide any unused subplots
    for j in range(n_images, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    return fig
