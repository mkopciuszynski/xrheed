"""
Xarray accessors for RHEED (Reflection High-Energy Electron Diffraction) data.

Accessors
---------

- **ri**: for manipulating and analyzing RHEED images, including plotting and image centering.
- **rp**: for manipulating RHEED intensity profiles.

These accessors extend xarray's `DataArray` objects with domain-specific methods for RHEED analysis.
"""

import logging
from typing import Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from numpy.typing import NDArray
from scipy import ndimage  # type: ignore

from .constants import (
    DEFAULT_ALPHA,
    DEFAULT_BETA,
    DEFAULT_SCREEN_ROI_HEIGHT,
    DEFAULT_SCREEN_ROI_WIDTH,
    IMAGE_DIMS,
    IMAGE_NDIMS,
    K_INV_ANGSTROM,
    STACK_NDIMS,
)
from .conversion.base import convert_sx_to_ky
from .plotting.base import plot_image
from .plotting.profiles import plot_profile
from .preparation.alignment import (
    find_horizontal_center,
    find_vertical_center,
    find_incident_angle,
)

logger = logging.getLogger(__name__)


@xr.register_dataarray_accessor("ri")
class RHEEDAccessor:
    """
    Xarray accessor for RHEED images.

    Provides convenient access to RHEED-specific metadata, image manipulation,
    centering, rotation, and profile extraction methods.
    """

    def __init__(self, xarray_obj: xr.DataArray) -> None:
        self._obj = xarray_obj
        logger.debug(
            "Registered RHEEDAccessor for DataArray with shape %s and coords %s",
            getattr(xarray_obj, "shape", None),
            list(xarray_obj.coords.keys()),
        )

    # ---- Properties ----
    @property
    def screen_sample_distance(self) -> float:
        """Distance between sample and screen in mm. Default: 1.0 mm."""
        da = self._obj
        return float(da.attrs.get("screen_sample_distance", 1.0))

    @property
    def beta(self) -> float:
        """Incident angle (beta) in degrees. Default: DEFAULT_BETA."""
        da = self._obj
        return float(da.attrs.get("beta", DEFAULT_BETA))

    @beta.setter
    def beta(self, value: float) -> None:
        """Set the incident angle (beta) in degrees."""
        if not isinstance(value, (int, float)):
            raise ValueError(f"beta must be numeric, got {value!r}")
        self._obj.attrs["beta"] = float(value)

    @property
    def alpha(self) -> Union[float, NDArray]:
        """
        Azimuthal angle (alpha) in degrees.
        If present as a coordinate, returns the coordinate values; otherwise returns an attribute.
        """
        da = self._obj
        if "alpha" in da.coords:
            return da.coords["alpha"].values
        return float(da.attrs.get("alpha", DEFAULT_ALPHA))

    @alpha.setter
    def alpha(self, value: float) -> None:
        """Set the azimuthal angle (alpha) in degrees."""
        if not isinstance(value, (int, float)):
            raise ValueError(f"alpha must be numeric, got {value!r}")
        self._obj.attrs["alpha"] = float(value)

    @property
    def screen_scale(self) -> float:
        """Screen scale in px/mm. Default: 1.0."""
        da = self._obj
        return float(da.attrs.get("screen_scale", 1.0))

    @screen_scale.setter
    def screen_scale(self, px_to_mm: float) -> None:
        """
        Set the screen scale (px/mm) and update coordinate scaling accordingly.

        Parameters
        ----------
        px_to_mm : float
            New scale in pixels per millimeter. Must be positive.
        """
        if px_to_mm <= 0:
            raise ValueError("screen_scale must be positive.")

        da = self._obj
        old_px_to_mm = self.screen_scale
        da.attrs["screen_scale"] = float(px_to_mm)

        missing = IMAGE_DIMS - da.coords.keys()
        if missing:
            raise ValueError(f"Missing required coordinate(s): {sorted(missing)}")

        da["sx"] = da.sx * old_px_to_mm / px_to_mm
        da["sy"] = da.sy * old_px_to_mm / px_to_mm

    @property
    def screen_width(self) -> Optional[float]:
        """Screen width in mm, if set."""
        val = self._obj.attrs.get("screen_width")
        return float(val) if val is not None else None

    @property
    def screen_roi_width(self) -> float:
        """Width of the region of interest (ROI) on the screen in mm."""
        return float(self._obj.attrs.get("screen_roi_width", DEFAULT_SCREEN_ROI_WIDTH))

    @screen_roi_width.setter
    def screen_roi_width(self, value: float) -> None:
        """Set the screen ROI width in mm."""
        self._obj.attrs["screen_roi_width"] = float(value)

    @property
    def screen_roi_height(self) -> float:
        """Height of the region of interest (ROI) on the screen in mm."""
        return float(
            self._obj.attrs.get("screen_roi_height", DEFAULT_SCREEN_ROI_HEIGHT)
        )

    @screen_roi_height.setter
    def screen_roi_height(self, value: float) -> None:
        """Set the screen ROI height in mm."""
        self._obj.attrs["screen_roi_height"] = float(value)

    @property
    def beam_energy(self) -> Optional[float]:
        """Beam energy in eV, if set."""
        val = self._obj.attrs.get("beam_energy")
        return float(val) if val is not None else None

    @beam_energy.setter
    def beam_energy(self, value: float) -> None:
        """Set the beam energy in eV."""
        self._obj.attrs["beam_energy"] = float(value)

    @property
    def ewald_radius(self) -> float:
        """
        Compute the Ewald sphere radius in reciprocal space (k-space).

        Returns
        -------
        float
            Ewald sphere radius in 1/Å.
        """
        beam_energy = self.beam_energy
        if beam_energy is None:
            raise ValueError("Beam energy is not set.")
        return np.sqrt(beam_energy) * K_INV_ANGSTROM

    # ---- Methods ----
    def __repr__(self) -> str:
        da = self._obj
        return (
            f"<RHEEDAccessor>\n"
            f"  File name: {da.attrs.get('file_name', 'N/A')}\n"
            f"  File creation time: {da.attrs.get('file_ctime', 'N/A')}\n"
            f"  Image shape: {da.shape}\n"
            f"  Screen scale: {self.screen_scale} px/mm\n"
            f"  Screen sample distance: {self.screen_sample_distance} mm\n"
            f"  Beta (incident) angle: {self.beta:.2f} deg\n"
            f"  Alpha (azimuthal) angle: {self.alpha:.2f} deg\n"
            f"  Beam Energy: {self.beam_energy} eV\n"
        )

    def rotate(self, angle: float) -> None:
        """
        Rotate the image or stack of images by a specified angle.

        Parameters
        ----------
        angle : float
            Rotation angle in degrees. Positive values rotate counterclockwise.
        """
        da = self._obj
        logger.debug("rotate called: angle=%s, ndim=%s", angle, da.ndim)
        if da.ndim == IMAGE_NDIMS:
            da.data = ndimage.rotate(da.data, angle, reshape=False)
        elif da.ndim == STACK_NDIMS:
            stack_dim = da.dims[0]
            da.data = np.stack(
                [
                    ndimage.rotate(da.isel({stack_dim: i}).data, angle, reshape=False)
                    for i in range(da.sizes[stack_dim])
                ],
                axis=0,
            )
        else:
            logger.error(
                "rotate: unsupported ndim=%s (expected %s or %s)",
                da.ndim,
                IMAGE_NDIMS,
                STACK_NDIMS,
            )
            raise ValueError(
                f"Expected {IMAGE_NDIMS}D or {STACK_NDIMS}D, got {da.ndim}D"
            )
        logger.info("Rotation applied: angle=%.4f degrees", float(angle))

    def set_center_manual(
        self,
        center_x: Union[float, list[float], np.ndarray] = 0.0,
        center_y: Union[float, list[float], np.ndarray] = 0.0,
        method: Literal["linear", "nearest", "cubic"] = "linear",
    ) -> None:
        """
        Manually shift the image center for a single image or a stack.

        Parameters
        ----------
        center_x : float or sequence
            Horizontal shift(s). Scalar applied to all frames; array-like must match stack length.
        center_y : float or sequence
            Vertical shift(s). Same logic as center_x.
        method : {'linear', 'nearest', 'cubic'}, optional
            Interpolation method for per-frame shifts (default='linear').
        """
        da = self._obj

        def _first_float(val):
            if isinstance(val, float):
                return val
            elif isinstance(val, (list, np.ndarray)):
                return float(val[0])
            else:
                return float(val)

        logger.debug(
            "set_center_manual called: center_x=%.4f, center_y=%.4f, method=%s",
            _first_float(center_x),
            _first_float(center_y),
            method,
        )

        missing = IMAGE_DIMS - da.coords.keys()
        if missing:
            raise ValueError(f"Missing required coordinate(s): {sorted(missing)}")

        if da.ndim == IMAGE_NDIMS:
            da["sx"] = da.sx - center_x
            da["sy"] = da.sy - center_y

        elif da.ndim == STACK_NDIMS:
            stack_dim = da.dims[0]
            n_frames = da.sizes[stack_dim]

            cx = np.atleast_1d(center_x)
            cy = np.atleast_1d(center_y)

            # Broadcast scalars
            if cx.size == 1:
                cx = np.full(n_frames, cx.item())
            if cy.size == 1:
                cy = np.full(n_frames, cy.item())

            if len(cx) != n_frames or len(cy) != n_frames:
                logger.error(
                    "Invalid center lengths: expected %s, got %s and %s",
                    n_frames,
                    len(cx),
                    len(cy),
                )
                raise ValueError(
                    f"center_x/center_y must be scalar or length={n_frames}, got {len(cx)} and {len(cy)}"
                )

            # Normalize shifts relative to first frame
            cx0, cy0 = cx[0], cy[0]
            da["sx"] = da.sx - cx0
            da["sy"] = da.sy - cy0

            cx -= cx0
            cy -= cy0

            # In-place modification of the underlying numpy array for all frames
            for i in range(n_frames):
                if i == 0:
                    continue  # first frame already shifted
                new_coords = {"sx": da.sx - cx[i], "sy": da.sy - cy[i]}
                da.data[i] = (
                    da.isel({stack_dim: i})
                    .interp(new_coords, method=method, kwargs={"fill_value": 0})
                    .data
                )
            logger.info(
                "Manual centering applied to %d frames using method=%s; center_x=%.4f, center_y=%.4f",
                n_frames,
                method,
                float(cx0),
                float(cy0),
            )

        else:
            raise ValueError(
                f"Unsupported ndim={da.ndim}, expected {IMAGE_NDIMS} or {STACK_NDIMS}"
            )

    def set_center_auto(self, update_incident_angle: bool = False) -> None:
        """
        Automatically determine and apply the image center using
        `find_horizontal_center` and `find_vertical_center`.

        Uses the first frame if the data is a stack.
        If update_incident_angle is True, updates the incident angle based on the new center.
        """
        da = self._obj
        image = da[0] if da.ndim == STACK_NDIMS else da

        image_roi = image.ri.get_roi_image()

        center_x = find_horizontal_center(image_roi)
        center_y = find_vertical_center(image_roi, center_x=center_x)

        self.set_center_manual(center_x, center_y)

        logger.info(
            "Applied automatic centering: center_x=%.4f, center_y=%.4f",
            float(center_x),
            float(center_y),
        )

        if update_incident_angle:
            beta = find_incident_angle(da)
            da.ri.beta = beta
            logger.info(
                "Updated incident angle: %.4f",
                float(beta),
            )
        
    def get_roi_image(self) -> xr.DataArray:
        """
        Return a copy of the image restricted to the screen ROI.

        The ROI is defined by the attributes 'screen_roi_width' and
        'screen_roi_height' (in mm).
        """
        da = self._obj

        roi_width: float = self.screen_roi_width
        roi_height: float = self.screen_roi_height

        da_roi = da.sel(
            sx=slice(-roi_width, roi_width),
            sy=slice(-roi_height, None),
        ).copy()
        return da_roi


    def get_profile(
        self,
        center: Tuple[float, float],
        width: float,
        height: float,
        stack_index: int = 0,
        reduce_over: Literal["sy", "sx", "both"] = "sy",
        method: Literal["mean", "sum"] = "mean",
        show_origin: bool = False,
        **kwargs,
    ) -> xr.DataArray:
        """
        Extract an intensity profile from the RHEED image or stack.

        Parameters
        ----------
        center : tuple[float, float]
            Center coordinates (sx, sy) in mm.
        width : float
            Width of the profile window in mm.
        height : float
            Height of the profile window in mm.
        stack_index : int
            Frame index for stacks (default=0).
        reduce_over : {'sy', 'sx', 'both'}
            Dimension(s) over which to reduce intensity (default='sy').
        method : {'mean', 'sum'}
            Reduction method (default='mean').
        show_origin : bool
            If True, display a rectangle showing the profile window.

        Returns
        -------
        xr.DataArray
            Profile data with metadata preserved.
        """
        da = self._obj
        logger.debug(
            "get_profile called: center=%s width=%s height=%s stack_index=%s reduce_over=%s method=%s",
            center,
            width,
            height,
            stack_index,
            reduce_over,
            method,
        )

        cropped = da.sel(
            sx=slice(center[0] - width / 2, center[0] + width / 2),
            sy=slice(center[1] - height / 2, center[1] + height / 2),
        )
        reduce_func = cropped.mean if method == "mean" else cropped.sum

        if reduce_over == "sy":
            profile = reduce_func(dim="sy")
        elif reduce_over == "sx":
            profile = reduce_func(dim="sx")
        elif reduce_over == "both":
            profile = reduce_func(dim=("sy", "sx"))
        else:
            raise ValueError("reduce_over must be 'sy', 'sx', or 'both'")

        profile.attrs = da.attrs.copy()
        profile.attrs.update(
            {
                "profile_center": center,
                "profile_width": width,
                "profile_height": height,
                "reduce_over": reduce_over,
                "reduce_method": method,
            }
        )

        if show_origin:
            fig, ax = plt.subplots()
            self.plot_image(ax=ax, stack_index=stack_index, **kwargs)
            rect = Rectangle(
                (center[0] - width / 2, center[1] - height / 2),
                width,
                height,
                linewidth=1,
                edgecolor="red",
                facecolor="none",
            )
            ax.add_patch(rect)
            logger.debug("Added origin rectangle to plot at center=%s", center)

        return profile

    def plot_image(
        self,
        ax: Optional[Axes] = None,
        auto_levels: float = 0.0,
        show_center_lines: bool = False,
        show_specular_spot: bool = False,
        stack_index: int = 0,
        **kwargs,
    ) -> Axes:
        """
        Plot a RHEED image or stack frame.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, a new figure is created.
        auto_levels : float
            Automatic contrast adjustment level (default=0.0).
        show_center_lines : bool
            If True, show horizontal and vertical center lines.
        show_specular_spot : bool
            If True, highlight the specular spot.
        stack_index : int
            Frame index for stacks (default=0).

        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the plot.
        """
        da = self._obj
        logger.debug(
            "plot_image called: ndim=%s stack_index=%s auto_levels=%s show_center_lines=%s show_specular_spot=%s",
            da.ndim,
            stack_index,
            auto_levels,
            show_center_lines,
            show_specular_spot,
        )
        if da.ndim == STACK_NDIMS:
            da = da.isel({da.dims[0]: stack_index})
        elif da.ndim != IMAGE_NDIMS:
            logger.error("plot_image: unsupported ndim=%s", da.ndim)
            raise ValueError(
                f"Expected {IMAGE_NDIMS}D or {STACK_NDIMS}D, got {da.ndim}D"
            )

        return plot_image(
            rheed_image=da,
            ax=ax,
            auto_levels=auto_levels,
            show_center_lines=show_center_lines,
            show_specular_spot=show_specular_spot,
            **kwargs,
        )


@xr.register_dataarray_accessor("rp")
class RHEEDProfileAccessor:
    """
    Xarray accessor for RHEED intensity profiles.

    Provides profile plotting and conversion to reciprocal space.
    """

    def __init__(self, xarray_obj: xr.DataArray):
        self._obj = xarray_obj
        logger.debug(
            "Registered RHEEDProfileAccessor for DataArray with shape %s",
            getattr(xarray_obj, "shape", None),
        )

    def __repr__(self):
        da = self._obj
        return (
            f"<RHEEDProfileAccessor>\n"
            f"  Center: sx, sy [mm]: {da.attrs.get('profile_center', 'N/A')} \n"
            f"  Width: {da.attrs.get('profile_width', 'N/A')} mm\n"
            f"  Height: {da.attrs.get('profile_height', 'N/A')} mm\n"
            f"  Reduce over: {da.attrs.get('reduce_over', 'N/A')}\n"
            f"  Reduce method: {da.attrs.get('reduce_method', 'N/A')}\n"
        )

    def convert_to_k(self) -> xr.DataArray:
        """
        Convert profile coordinates from screen units (sx) to reciprocal space (ky).

        Returns
        -------
        xr.DataArray
            Profile with 'sx' replaced by 'ky'.
        """
        da = self._obj
        if "sx" not in da.coords:
            raise ValueError("The profile must have 'sx' coordinate to convert to ky.")
        k_e = da.ri.ewald_radius
        screen_sample_distance = da.ri.screen_sample_distance
        logger.debug(
            "convert_to_k: converting sx->ky with ewald_radius=%s, screen_sample_distance=%s",
            k_e,
            screen_sample_distance,
        )
        sx = da.coords["sx"].values
        ky = convert_sx_to_ky(
            sx,
            ewald_radius=k_e,
            screen_sample_distance_mm=screen_sample_distance,
        )
        return da.assign_coords(sx=ky).rename({"sx": "ky"})

    def plot_profile(
        self,
        ax: Optional[Axes] = None,
        transform_to_k: bool = True,
        normalize: bool = True,
        **kwargs,
    ) -> Axes:
        """
        Plot a RHEED intensity profile.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, a new figure is created.
        transform_to_k : bool
            If True, convert sx to ky using the Ewald sphere.
        normalize : bool
            If True, normalize intensity to [0, 1].

        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the plot.
        """
        da = self._obj.copy()
        return plot_profile(
            rheed_profile=da,
            ax=ax,
            transform_to_k=transform_to_k,
            normalize=normalize,
            **kwargs,
        )
