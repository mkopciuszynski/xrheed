"""
This module defines xarray accessors for RHEED (Reflection High-Energy Electron Diffraction) data.

Accessors
---------

- **ri**: for manipulating and analyzing RHEED images, including plotting and image centering.
- **rp**: for manipulating RHEED intensity profiles.

These accessors extend xarray's `DataArray` objects with domain-specific methods for RHEED analysis.
"""

import logging
from typing import Optional, Tuple, Union, Literal

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
    K_INV_ANGSTROM,
    IMAGE_DIMS,
    IMAGE_NDIMS,
    STACK_NDIMS,
)
from .conversion.base import convert_sx_to_ky
from .plotting.base import plot_image
from .plotting.profiles import plot_profile
from .preparation.alignment import find_horizontal_center, find_vertical_center

logger = logging.getLogger(__name__)


@xr.register_dataarray_accessor("ri")
class RHEEDAccessor:
    def __init__(self, xarray_obj: xr.DataArray) -> None:
        self._obj: xr.DataArray = xarray_obj
        self._center: Optional[Tuple[float, float]] = None

    def _get_attr(self, attr_name: str, default: Optional[float] = None) -> float:
        da: xr.DataArray = self._obj
        value = da.attrs.get(attr_name, default)
        if value is None:
            raise AttributeError(
                f"Attribute '{attr_name}' not found and no default provided."
            )
        try:
            return float(value)
        except (TypeError, ValueError):
            raise ValueError(f"Attribute '{attr_name}' must be numeric, got {value!r}.")

    def _set_attr(self, attr_name: str, value: float) -> None:
        if not isinstance(value, (int, float)):
            raise ValueError(
                f"Attribute '{attr_name}' must be numeric, got {type(value).__name__}."
            )
        self._obj.attrs[attr_name] = float(value)

    def __repr__(self) -> str:
        da: xr.DataArray = self._obj
        screen_scale = self._get_attr("screen_scale", None)
        beam_energy = self._get_attr("beam_energy", None)
        screen_sample_distance = self._get_attr("screen_sample_distance", None)
        beta = self._get_attr("beta", DEFAULT_BETA)
        alpha = self._get_attr("alpha", DEFAULT_ALPHA)
        file_name = da.attrs.get("file_name", "N/A")
        file_ctime = da.attrs.get("file_ctime", "N/A")

        return (
            f"<RHEEDAccessor>\n"
            f"  File name: {file_name}\n"
            f"  File creation time: {file_ctime}\n"
            f"  Image shape: {da.shape}\n"
            f"  Screen scale: {screen_scale} px/mm\n"
            f"  Screen sample distance: {screen_sample_distance} mm\n"
            f"  Beta (incident) angle: {beta:.2f} deg\n"
            f"  Alpha (azimuthal) angle: {alpha:.2f} deg\n"
            f"  Beam Energy: {beam_energy} eV\n"
        )

    # ---- Properties ----
    @property
    def screen_sample_distance(self) -> float:
        return self._get_attr("screen_sample_distance", 1.0)

    @property
    def beta(self) -> float:
        return self._get_attr("beta", DEFAULT_BETA)

    @beta.setter
    def beta(self, value: float) -> None:
        self._set_attr("beta", float(value))

    @property
    def alpha(self) -> Union[float, NDArray]:
        da: xr.DataArray = self._obj
        if "alpha" in da.coords:
            return da.coords["alpha"].values
        return self._get_attr("alpha", DEFAULT_ALPHA)

    @alpha.setter
    def alpha(self, value: float) -> None:
        self._set_attr("alpha", float(value))

    @property
    def screen_scale(self) -> float:
        return self._get_attr("screen_scale", 1.0)

    @screen_scale.setter
    def screen_scale(self, px_to_mm: float) -> None:
        da: xr.DataArray = self._obj
        if px_to_mm <= 0:
            raise ValueError("screen_scale must be positive.")

        old_px_to_mm = self._get_attr("screen_scale", 1.0)
        self._set_attr("screen_scale", px_to_mm)

        missing = IMAGE_DIMS - da.coords.keys()
        if missing:
            raise ValueError(f"Missing required coordinate(s): {sorted(missing)}")         

        da["sx"] = da.sx * old_px_to_mm / px_to_mm
        da["sy"] = da.sy * old_px_to_mm / px_to_mm

    @property
    def screen_width(self) -> Optional[float]:
        return self._get_attr("screen_width", None)

    @property
    def screen_roi_width(self) -> float:
        return self._get_attr("screen_roi_width", DEFAULT_SCREEN_ROI_WIDTH)

    @screen_roi_width.setter
    def screen_roi_width(self, value: float) -> None:
        self._set_attr("screen_roi_width", value)

    @property
    def screen_roi_height(self) -> float:
        return self._get_attr("screen_roi_height", DEFAULT_SCREEN_ROI_HEIGHT)

    @screen_roi_height.setter
    def screen_roi_height(self, value: float) -> None:
        self._set_attr("screen_roi_height", value)

    @property
    def beam_energy(self) -> Optional[float]:
        return self._get_attr("beam_energy", None)

    @beam_energy.setter
    def beam_energy(self, value: float) -> None:
        self._set_attr("beam_energy", value)

    @property
    def ewald_sphere_radius(self) -> float:
        beam_energy = self.beam_energy
        if beam_energy is None:
            raise ValueError("Beam energy is not set.")
        return np.sqrt(beam_energy) * K_INV_ANGSTROM

    # ---- Methods ----
    def rotate(self, angle: float) -> None:
        da: xr.DataArray = self._obj
        if da.ndim == IMAGE_NDIMS:
            rotated = ndimage.rotate(da.data, angle, reshape=False)
        elif da.ndim == STACK_NDIMS:
            stack_dim = da.dims[0]
            rotated = np.stack(
                [
                    ndimage.rotate(da.isel({stack_dim: i}).data, angle, reshape=False)
                    for i in range(da.sizes[stack_dim])
                ],
                axis=0,
            )
        else:
            raise ValueError(
                f"Expected {IMAGE_NDIMS}D or {STACK_NDIMS}D array, got {da.ndim}D"
            )
        da.data = rotated

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
            Horizontal shift(s). If scalar, applied to all frames.
            If array-like, must match stack length.
        center_y : float or sequence
            Vertical shift(s). Same logic as center_x.
        method : str, optional
            Interpolation method for per-frame shifts (default = "linear").
        """
        da: xr.DataArray = self._obj

        missing = IMAGE_DIMS - da.coords.keys()
        if missing:
            raise ValueError(f"Missing required coordinate(s): {sorted(missing)}")            

        if da.ndim == IMAGE_NDIMS:
            da['sx'] = da.sx - center_x
            da['sy'] = da.sy - center_y

        elif da.ndim == STACK_NDIMS:
            stack_dim = da.dims[0]
            n_frames = da.sizes[stack_dim]

            cx = np.atleast_1d(center_x)
            cy = np.atleast_1d(center_y)

            if cx.size == 1 and cy.size == 1:
                self._obj = da.assign_coords(sx=da.sx - float(cx), sy=da.sy - float(cy))
                da['sx'] = da.sx - float(cx)
                da['sy'] = da.sy - float(cy)


            else:
                # Broadcast scalars to full-length vectors
                if cx.size == 1:
                    cx = np.full(n_frames, cx.item())
                if cy.size == 1:
                    cy = np.full(n_frames, cy.item())

                if len(cx) != n_frames or len(cy) != n_frames:
                    raise ValueError(
                        f"center_x/center_y must be scalar or length={n_frames}, got {len(cx)} and {len(cy)}"
                    )

                # Normalize shifts relative to first frame
                cx0, cy0 = cx[0], cy[0]
                cx = cx - cx0
                cy = cy - cy0

                da['sx'] = da.sx - float(cx0)
                da['sy'] = da.sy - float(cy0)

                shifted_slices = []
                for i in range(n_frames):
                    new_coords = {"sx": da.sx - cx[i], "sy": da.sy - cy[i]}
                    shifted = da.isel({stack_dim: i}).interp(
                        new_coords, method=method, kwargs={"fill_value": 0}
                    )
                    shifted_slices.append(shifted)

                self._obj = xr.concat(shifted_slices, dim=stack_dim)

        else:
            raise ValueError(
                f"Unsupported ndim={da.ndim}, expected {IMAGE_NDIMS} or {STACK_NDIMS}"
            )

    def set_center_auto(self) -> None:
        """
        Automatically determine and apply the image center using
        `find_horizontal_center` and `find_vertical_center`.
        Uses the first frame if data is a stack.
        """
        da = self._obj

        # Use the first frame if data is a stack.
        image = da[0] if da.ndim == STACK_NDIMS else da

        center_x = find_horizontal_center(image)
        center_y = find_vertical_center(image)

        self.set_center_manual(center_x, center_y)
        logger.info("Applied automatic centering.")

    def get_profile(
        self,
        center: Tuple[float, float],
        width: float,
        height: float,
        stack_index: int = 0,
        reduce_over: Literal["sy", "sx", "both"] = "sy",
        method: Literal["mean", "sum"] = "mean",
        plot_origin: bool = False,
    ) -> xr.DataArray:
        da: xr.DataArray = self._obj
        if da.ndim == STACK_NDIMS:
            da = da.isel({da.dims[0]: stack_index})
        elif da.ndim != IMAGE_NDIMS:
            raise ValueError(
                f"Expected {IMAGE_NDIMS}D or {STACK_NDIMS}D, got {da.ndim}D"
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

        if plot_origin:
            fig, ax = plt.subplots()
            self.plot_image(ax=ax, stack_index=stack_index)
            rect = Rectangle(
                (center[0] - width / 2, center[1] - height / 2),
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
        show_center_lines: bool = False,
        show_specular_spot: bool = False,
        stack_index: int = 0,
        **kwargs,
    ) -> Axes:

        da: xr.DataArray = self._obj

        if da.ndim == STACK_NDIMS:
            da = da.isel({da.dims[0]: stack_index})
        elif da.ndim != IMAGE_NDIMS:
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
    def __init__(self, xarray_obj: xr.DataArray):
        self._obj = xarray_obj

    def __repr__(self):
        da: xr.DataArray = self._obj
        center = da.attrs.get("profile_center", "N/A")
        width = da.attrs.get("profile_width", "N/A")
        height = da.attrs.get("profile_height", "N/A")
        return (
            f"<RHEEDProfileAccessor\n"
            f"  Center: sx, sy [mm]: {center} \n"
            f"  Width: {width} mm\n"
            f"  Height: {height} mm\n"
        )

    def convert_to_k(self) -> xr.DataArray:
        da: xr.DataArray = self._obj
        if "sx" not in da.coords:
            raise ValueError("The profile must have 'sx' coordinate to convert to ky.")
        k_e: float = da.ri.ewald_sphere_radius
        screen_sample_distance: float = da.ri.screen_sample_distance
        sx: NDArray = da.coords["sx"].values
        ky = convert_sx_to_ky(
            sx,
            ewald_sphere_radius=k_e,
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
        da: xr.DataArray = self._obj.copy()
        return plot_profile(
            rheed_profile=da,
            ax=ax,
            transform_to_k=transform_to_k,
            normalize=normalize,
            **kwargs,
        )
