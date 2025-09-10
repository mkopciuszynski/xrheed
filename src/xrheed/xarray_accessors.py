import logging
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from numpy.typing import NDArray
from scipy import constants, ndimage  # type: ignore

from .conversion.base import convert_sx_to_ky
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
        """
        Accessor for RHEED-related metadata and methods.

        Parameters
        ----------
        xarray_obj : xr.DataArray
            The DataArray this accessor is attached to.
        """
        self._obj: xr.DataArray = xarray_obj
        self._center: Optional[Tuple[float, float]] = None

    def _get_attr(self, attr_name: str, default: Optional[float] = None) -> float:
        """
        Retrieve an attribute from the DataArray's attrs dict.

        Parameters
        ----------
        attr_name : str
            Name of the attribute.
        default : float, optional
            Default value to return if the attribute is not found.

        Returns
        -------
        float
            Attribute value (cast to float).

        Raises
        ------
        AttributeError
            If the attribute is missing and no default is provided.
        ValueError
            If the stored attribute cannot be converted to float.
        """
        value = self._obj.attrs.get(attr_name, default)
        if value is None:
            raise AttributeError(
                f"Attribute '{attr_name}' not found and no default provided."
            )
        try:
            return float(value)
        except (TypeError, ValueError):
            raise ValueError(f"Attribute '{attr_name}' must be numeric, got {value!r}.")

    def _set_attr(self, attr_name: str, value: float) -> None:
        """
        Set an attribute in the DataArray's attrs dict.

        Parameters
        ----------
        attr_name : str
            Name of the attribute.
        value : float
            Value to store (cast to float).
        """
        if not isinstance(value, (int, float)):
            raise ValueError(
                f"Attribute '{attr_name}' must be numeric, got {type(value).__name__}."
            )
        self._obj.attrs[attr_name] = float(value)

    def __repr__(self) -> str:
        """
        Return a human-readable summary of the DataArray and key RHEED metadata.
        Missing attributes are shown as 'N/A'.
        """
        screen_scale = self._get_attr("screen_scale", None)
        beam_energy = self._get_attr("beam_energy", None)
        screen_sample_distance = self._get_attr("screen_sample_distance", None)
        beta = self._get_attr("beta", DEFAULT_BETA)
        alpha = self._get_attr("alpha", DEFAULT_ALPHA)

        return (
            f"<RHEEDAccessor>\n"
            f"  Image shape: {self._obj.shape}\n"
            f"  Screen scale: {screen_scale if screen_scale is not None else 'N/A'}\n"
            f"  Screen sample distance: {screen_sample_distance if screen_sample_distance is not None else 'N/A'}\n"
            f"  Beta (incident) angle: {beta:.2f} deg\n"
            f"  Alpha (azimuthal) angle: {alpha:.2f} deg\n"
            f"  Beam Energy: {beam_energy if beam_energy is not None else 'N/A'} eV\n"
        )

    @property
    def screen_sample_distance(self) -> float:
        """Distance from sample to screen in mm."""
        return self._get_attr("screen_sample_distance", 1.0)

    @property
    def beta(self) -> float:
        """Polar (incident) angle in degrees."""
        return self._get_attr("beta", DEFAULT_BETA)

    @beta.setter
    def beta(self, value: float) -> None:
        if not isinstance(value, (int, float)):
            raise ValueError("Beta must be a number.")
        self._set_attr("beta", float(value))

    @property
    def alpha(self) -> float:
        """Azimuthal angle in degrees."""
        return self._get_attr("alpha", DEFAULT_ALPHA)

    @alpha.setter
    def alpha(self, value: float) -> None:
        if not isinstance(value, (int, float)):
            raise ValueError("Alpha must be a number.")
        self._set_attr("alpha", float(value))

    @property
    def screen_scale(self) -> float:
        """Screen scaling factor (pixels to mm)."""
        return self._get_attr("screen_scale", 1.0)

    @screen_scale.setter
    def screen_scale(self, px_to_mm: float) -> None:
        if px_to_mm <= 0:
            raise ValueError("screen_scale must be positive.")

        old_px_to_mm = self._get_attr("screen_scale", 1.0)
        self._set_attr("screen_scale", px_to_mm)

        # Adjust coordinate axes to maintain real-world scale
        self._obj["sx"] = self._obj.sx * old_px_to_mm / px_to_mm
        self._obj["sy"] = self._obj.sy * old_px_to_mm / px_to_mm

    @property
    def screen_width(self) -> Optional[float]:
        """Total screen width in mm."""
        return self._get_attr("screen_width", None)

    @property
    def screen_roi_width(self) -> float:
        """Region-of-interest width in mm."""
        return self._get_attr("screen_roi_width", DEFAULT_SCREEN_ROI_WIDTH)

    @screen_roi_width.setter
    def screen_roi_width(self, value: float) -> None:
        if value <= 0:
            raise ValueError("screen_roi_width must be positive.")
        self._set_attr("screen_roi_width", value)

    @property
    def screen_roi_height(self) -> float:
        """Region-of-interest height in mm."""
        return self._get_attr("screen_roi_height", DEFAULT_SCREEN_ROI_HEIGHT)

    @screen_roi_height.setter
    def screen_roi_height(self, value: float) -> None:
        if value <= 0:
            raise ValueError("screen_roi_height must be positive.")
        self._set_attr("screen_roi_height", value)

    @property
    def beam_energy(self) -> Optional[float]:
        """Beam energy in eV."""
        return self._get_attr("beam_energy", None)

    @beam_energy.setter
    def beam_energy(self, value: float) -> None:
        if value <= 0:
            raise ValueError("beam_energy must be positive.")
        self._set_attr("beam_energy", value)

    @property
    def ewald_sphere_radius(self) -> float:
        """
        Calculate the Ewald sphere radius (1/Å) from the beam energy.

        Raises
        ------
        ValueError
            If beam energy is not set.
        """
        beam_energy = self.beam_energy
        if beam_energy is None:
            raise ValueError("Beam energy is not set.")

        # Ewald sphere radius k = sqrt(2 m e E) / hbar
        k_e = np.sqrt(2 * constants.m_e * constants.e * beam_energy) / constants.hbar

        return k_e * 1e-10  # convert from 1/m to 1/A

    def rotate(self, angle: float) -> None:
        """
        Rotate the image data by a specified angle.

        Parameters
        ----------
        angle : float
            Rotation angle in degrees. Positive values correspond to
            counterclockwise rotation.

        Notes
        -----
        - The rotation is applied in-place to `self._obj.data`.
        - The shape of the array is preserved (`reshape=False`), so some edges may be clipped.
        """
        image_data = self._obj.data
        image_data = ndimage.rotate(image_data, angle, reshape=False)
        self._obj.data = image_data

    def apply_image_center(
        self, center_x: float = 0.0, center_y: float = 0.0, auto_center: bool = False
    ) -> None:
        """
        Shift the image coordinates to a specified center or automatically determine it.

        Parameters
        ----------
        center_x : float, optional
            Horizontal coordinate of the new image center (default is 0.0).
        center_y : float, optional
            Vertical coordinate of the new image center (default is 0.0).
        auto_center : bool, optional
            If True, the center is computed automatically using
            `find_horizontal_center` and `find_vertical_center`.

        Notes
        -----
        - This method modifies `self._obj['sx']` and `self._obj['sy']` in-place.
        - When `auto_center=True`, the provided `center_x` and `center_y` are ignored.
        - Logs an info message after shifting the image.
        """
        image = self._obj

        if auto_center:
            center_x = find_horizontal_center(image)
            center_y = find_vertical_center(image)

        image["sx"] = image.sx - center_x
        image["sy"] = image.sy - center_y

        logger.info("The image was shifted to a new center.")

    def get_profile(
        self,
        center: Optional[Tuple[float, float]] = None,
        width: Optional[float] = None,
        height: Optional[float] = None,
        plot_origin: bool = False,
    ) -> xr.DataArray:
        """Get a profile of the RHEED image.

        Parameters
        ----------
        center : tuple[float, float] , optional
            Center of the profile in (sx, sy) coordinates. If None, the center of the image will be used.
        width : float, optional
            Width of the profile. If None, the full width of the image will be used.
        height : float, optional
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
            width = float(rheed_image.sx.max() - rheed_image.sx.min())

        if height is None:
            height = float(rheed_image.sy.max() - rheed_image.sy.min())

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
        ax: Optional[Axes] = None,
        auto_levels: float = 0.0,
        show_center_lines: bool = True,
        show_specular_spot: bool = True,
        **kwargs,
    ) -> Axes:
        """Plot RHEED image.

        Parameters
        ----------
        ax : Axes, optional
            Axes to plot on. If None, a new figure and axes will be created.
        auto_levels : float, optional
            If greater than 0, apply auto levels to the image.
            The number represents the allowed percentage of overexposed pixels.
            Default is 0.0 (no auto autolevels).
        show_center_lines : bool, optional
            If True, draw horizontal and vertical lines at the center of the image.
            Default is True.
        show_specular_spot : bool, optional
            If True, overlay the specularly reflected spot on the image.
        **kwargs : dict
            Additional keyword arguments passed to the plotting function.
        Returns
        -------
        Axes
            The axes with the plotted image.
        """

        # use a copy of the object to avoid modifying the original data
        rheed_image: xr.DataArray = self._obj.copy()

        return plot_image(
            rheed_image=rheed_image,
            ax=ax,
            auto_levels=auto_levels,
            show_center_lines=show_center_lines,
            show_specular_spot=show_specular_spot,
            **kwargs,
        )


@xr.register_dataarray_accessor("rp")
class RHEEDProfileAccessor:
    def __init__(self, xarray_obj: xr.DataArray):
        """
        Initialize the RHEEDProfileAccessor with a given xarray DataArray.

        Args:
            xarray_obj (xr.DataArray): The DataArray representing a RHEED profile.
        """

        self._obj = xarray_obj

    def __repr__(self):
        """
        Return a string representation of the RHEEDProfileAccessor, summarizing key profile attributes.

        Returns:
            str: A formatted string displaying the profile center, width, and height.
        """

        center = self._obj.attrs.get("profile_center", "N/A")
        width = self._obj.attrs.get("profile_width", "N/A")
        height = self._obj.attrs.get("profile_height", "N/A")
        return (
            f"<RHEEDProfileAccessor\n"
            f"  Center: sx, sy [mm]: {center} \n"
            f"  Width: {width} mm\n"
            f"  Height: {height} mm\n"
        )

    def convert_to_k(self) -> xr.DataArray:
        """
        Permanently convert the profile's screen x-coordinate (`sx`) to the momentum space coordinate $k_y$ [1/Å].

        The conversion is performed using the Ewald sphere radius and the screen-to-sample distance, based on the scattering geometry defined in the xRHEED project.

        Returns:
            xr.DataArray: A new DataArray with the `sx` coordinate replaced by `ky`.

        Raises:
            ValueError: If the profile does not contain an `sx` coordinate.
        """

        if "sx" not in self._obj.coords:
            raise ValueError("The profile must have 'sx' coordinate to convert to ky.")

        k_e: float = self._obj.ri.ewald_sphere_radius
        screen_sample_distance: float = self._obj.ri.screen_sample_distance

        sx: NDArray = self._obj.coords["sx"].values

        ky = convert_sx_to_ky(
            sx,
            ewald_sphere_radius=k_e,
            screen_sample_distance_mm=screen_sample_distance,
        )

        profile_k: xr.DataArray = self._obj.assign_coords(sx=ky).rename({"sx": "ky"})

        return profile_k

    def plot_profile(
        self,
        ax: Optional[Axes] = None,
        transform_to_k: bool = True,
        normalize: bool = True,
        **kwargs,
    ) -> Axes:
        """Plot a RHEED profile.

        Parameters
        ----------
        ax : Axes, optional
            Axes to plot on. If None, a new figure and axes will be created.
        transform_to_k : bool, optional
            If True, convert the screen x coordinate to ky [1/Å].
            Default is True.
        normalize : bool, optional
            If True, normalize the profile to the range [0, 1].
            Default is True.
        **kwargs : dict
            Additional keyword arguments passed to the plotting function.

        Returns
        -------
        Axes
            The axes with the plotted profile.
        """

        rheed_profile: xr.DataArray = self._obj.copy()

        return plot_profile(
            rheed_profile=rheed_profile,
            ax=ax,
            transform_to_k=transform_to_k,
            normalize=normalize,
            **kwargs,
        )
