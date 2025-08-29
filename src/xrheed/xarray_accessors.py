import logging

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.patches import Rectangle
from scipy import constants, ndimage

from .conversion.base import convert_sx_to_kx
from .plotting.base import plot_image
from .plotting.profiles import plot_profile
from .preparation.alignment import find_horizontal_center, find_vertical_center

logger = logging.getLogger(__name__)

DEFAULT_SCREEN_ROI_WIDTH = 50.0
DEFAULT_SCREEN_ROI_HEIGHT = 50.0
DEFAULT_BETA = 1.0
DEFAULT_ALPHA = 0.0


@xr.register_dataarray_accessor("ri")
class RHEEDAccessor:
    def __init__(self, xarray_obj: xr.DataArray) -> None:
        self._obj = xarray_obj
        self._center = None

    def _get_attr(self, attr_name: str, default: float | None = None) -> float:
        assert isinstance(self._obj, xr.DataArray)
        if attr_name in self._obj.attrs:
            return self._obj.attrs[attr_name]
        if default is not None:
            return default
        raise AttributeError(
            f"Attribute '{attr_name}' not found and no default provided."
        )

    def _set_attr(self, attr_name: str, value: float) -> None:
        assert isinstance(self._obj, xr.DataArray)
        self._obj.attrs[attr_name] = value

    def __repr__(self):
        screen_scale = self._get_attr("screen_scale")
        beam_energy = self._get_attr("beam_energy")
        screen_sample_distance = self._get_attr("screen_sample_distance")
        beta = self._get_attr("beta", DEFAULT_BETA)
        alpha = self._get_attr("alpha", DEFAULT_ALPHA)

        return (
            f"<RHEEDAccessor>\n"
            f"  Image shape: {self._obj.shape}\n"
            f"  Screen scale: {screen_scale}\n"
            f"  Screen sample distance: {screen_sample_distance:.2f}\n"
            f"  Beata (incident) angle: {beta:.2f} deg\n"
            f"  Alpha (azimuthal) angle: {alpha:.2f} deg\n"
            f"  Beam Energy: {beam_energy} eV\n"
        )

    @property
    def screen_sample_distance(self) -> float:
        """Screen sample distance"""
        return self._get_attr("screen_sample_distance", 1.0)

    @property
    def beta(self) -> float:
        """Polar angle"""
        return self._get_attr("beta", DEFAULT_BETA)

    @beta.setter
    def beta(self, value: float) -> None:
        if not isinstance(value, (int, float)):
            raise ValueError("beta must be a number.")
        self._set_attr("beta", float(value))

    @property
    def alpha(self) -> float:
        """Azimuthal angle"""
        return self._get_attr("alpha", DEFAULT_ALPHA)

    @alpha.setter
    def alpha(self, value: float) -> None:
        if not isinstance(value, (int, float)):
            raise ValueError("alpha must be a number.")
        self._set_attr("alpha", float(value))

    @property
    def screen_scale(self) -> float:
        """Screen scaling px to mm"""
        return self._get_attr("screen_scale", 1.0)

    @screen_scale.setter
    def screen_scale(self, px_to_mm: float) -> None:
        if px_to_mm < 0:
            raise ValueError("screen_scale must be positive.")
        old_px_to_mm = self._get_attr("screen_scale", 1.0)
        self._set_attr("screen_scale", px_to_mm)

        image = self._obj
        image["sx"] = image.sx * old_px_to_mm / px_to_mm
        image["sy"] = image.sy * old_px_to_mm / px_to_mm

    @property
    def screen_width(self) -> float:
        """Screen width in mm"""
        return self._get_attr("screen_width", None)

    @property
    def screen_roi_width(self) -> float:
        return self._get_attr("screen_roi_width", DEFAULT_SCREEN_ROI_WIDTH)

    @screen_roi_width.setter
    def screen_roi_width(self, value: float) -> None:
        if value <= 0:
            raise ValueError("screen_roi_width must be positive.")
        self._set_attr("screen_roi_width", value)

    @property
    def screen_roi_height(self) -> float:
        return self._get_attr("screen_roi_height", DEFAULT_SCREEN_ROI_HEIGHT)

    @screen_roi_height.setter
    def screen_roi_height(self, value: float) -> None:
        if value <= 0:
            raise ValueError("screen_roi_height must be positive.")
        self._set_attr("screen_roi_height", value)

    @property
    def beam_energy(self) -> float:
        """Beam energy in keV"""
        return self._get_attr("beam_energy", None)

    @beam_energy.setter
    def beam_energy(self, value: float) -> None:
        if value <= 0:
            raise ValueError("beam_energy must be positive.")
        self._set_attr("beam_energy", value)

    @property
    def ewald_sphere_radius(self) -> float:
        """Ewald radius in 1/Å"""
        beam_energy = self.beam_energy
        if beam_energy is None:
            raise ValueError("Beam energy is not set.")

        k_e = np.sqrt(2 * constants.m_e * constants.e * beam_energy) / constants.hbar

        return k_e * 1e-10

    def rotate(self, angle: float) -> None:
        image_data = self._obj.data
        image_data = ndimage.rotate(image_data, angle, reshape=False)
        self._obj.data = image_data

    def apply_image_center(
        self, center_x: float = 0.0, center_y: float = 0.0, auto_center: bool = False
    ) -> None:
        image = self._obj

        if auto_center:
            center_x = find_horizontal_center(image)
            image["sx"] = image.sx - center_x
            center_y = find_vertical_center(image)
            image["sy"] = image.sy - center_y
        else:
            image["sx"] = image.sx - center_x
            image["sy"] = image.sy - center_y

        logger.info("The image was shifted to a new center.")

    def get_profile(
        self,
        center: tuple[float, float] | None = None,
        width: float | None = None,
        height: float | None = None,
        plot_origin: bool = False,
    ) -> xr.DataArray:
        """Get a profile of the RHEED image.

        Parameters
        ----------
        center : tuple[float, float] | None, optional
            Center of the profile in (sx, sy) coordinates. If None, the center of the image will be used.
        width : float | None, optional
            Width of the profile. If None, the full width of the image will be used.
        height : float | None, optional
            Height of the profile. If None, the full height of the image will be used.

        Returns
        -------
        xr.DataArray
            The profile of the RHEED image.
        """

        rheed_image = self._obj

        if center is None:
            center = (0.0, 0.0)

        if width is None:
            width = rheed_image.sx.size

        if height is None:
            height = rheed_image.sy.size

        profile = rheed_image.sel(
            sx=slice(center[0] - width / 2, center[0] + width / 2),
            sy=slice(center[1] - height / 2, center[1] + height / 2),
        ).sum("sy")

        # Manually copy attrs
        profile.attrs = rheed_image.attrs.copy()

        profile.attrs["profile_center"] = center
        profile.attrs["profile_width"] = width
        profile.attrs["profile_height"] = height

        if plot_origin:
            # Plot the DataArray
            fig, ax = plt.subplots()

            plot_image(
                rheed_image=rheed_image, ax=ax, auto_levels=0.5, show_center_lines=False
            )

            # Compute bottom-left corner of the box
            start_x = center[0] - width / 2
            start_y = center[1] - height / 2

            # Add the rectangle
            rect = Rectangle(
                (start_x, start_y),
                width,
                height,
                linewidth=1,
                edgecolor="red",
                facecolor="none",
            )
            ax.add_patch(rect)

        return profile

    def plot_image(
        self,
        ax: plt.Axes | None = None,
        auto_levels: float = 0.0,
        show_center_lines: bool = True,
        **kwargs,
    ) -> plt.Axes:
        """Plot RHEED image.

        Parameters
        ----------
        ax : plt.Axes | None, optional
            Axes to plot on. If None, a new figure and axes will be created.
        auto_levels : float, optional
            If greater than 0, apply auto levels to the image.
            The number represents the allowed percentage of overexposed pixels.
            Default is 0.0 (no auto autolevels).
        show_center_lines : bool, optional
            If True, draw horizontal and vertical lines at the center of the image.
            Default is True.
        **kwargs : dict
            Additional keyword arguments passed to the plotting function.
        Returns
        -------
        plt.Axes
            The axes with the plotted image.
        """

        # use a copy of the object to avoid modifying the original data
        rheed_image = self._obj.copy()

        return plot_image(
            rheed_image=rheed_image,
            ax=ax,
            auto_levels=auto_levels,
            show_center_lines=show_center_lines,
            **kwargs,
        )


@xr.register_dataarray_accessor("rp")
class RHEEDProfileAccessor:
    def __init__(self, xarray_obj: xr.DataArray):
        self._obj = xarray_obj

    def __repr__(self):
        center = self._obj.attrs.get("profile_center", "N/A")
        width = self._obj.attrs.get("profile_width", "N/A")
        height = self._obj.attrs.get("profile_height", "N/A")
        return (
            f"<RHEEDProfileAccessor\n"
            f"  Center: sx, sy [mm]: {center} \n"
            f"  Width: {width} mm\n"
            f"  Height: {height} mm\n"
        )

    def convert_to_kx(self) -> xr.DataArray:
        """Convert the profile sx coordinate to kx [1/Å] using the Ewald sphere radius and screen sample distance."""

        if "sx" not in self._obj.coords:
            raise ValueError("The profile must have 'sx' coordinate to convert to kx.")

        k_e = self._obj.ri.ewald_sphere_radius
        screen_sample_distance = self._obj.ri.screen_sample_distance

        sx = self._obj.coords["sx"].data

        kx = convert_sx_to_kx(
            sx,
            ewald_sphere_radius=k_e,
            screen_sample_distance_mm=screen_sample_distance,
        )

        profile_kx = self._obj.assign_coords(sx=kx).rename({"sx": "kx"})

        return profile_kx

    def plot_profile(
        self,
        ax: plt.Axes | None = None,
        transform_to_kx: bool = True,
        normalize: bool = True,
        **kwargs,
    ) -> plt.Axes:
        """Plot a RHEED profile.

        Parameters
        ----------
        ax : plt.Axes | None, optional
            Axes to plot on. If None, a new figure and axes will be created.
        transform_to_kx : bool, optional
            If True, convert the x coordinate to kx [1/Å].
            Default is True.
        normalize : bool, optional
            If True, normalize the profile to the range [0, 1].
            Default is True.
        **kwargs : dict
            Additional keyword arguments passed to the plotting function.

        Returns
        -------
        plt.Axes
            The axes with the plotted profile.
        """

        rheed_profile = self._obj.copy()

        return plot_profile(
            rheed_profile=rheed_profile,
            ax=ax,
            transform_to_kx=transform_to_kx,
            normalize=normalize,
            **kwargs,
        )
