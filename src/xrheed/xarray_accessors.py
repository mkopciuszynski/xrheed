from scipy import ndimage
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from .plotting.base import plot_image
from .preparation.alignment import find_horizontal_center, find_vertical_center

from scipy.signal import savgol_filter
import ruptures as rpt
import logging

logger = logging.getLogger(__name__)

DEFAULT_SCREEN_ROI_WIDTH = 50.0
DEFAULT_SCREEN_ROI_HEIGHT = 50.0
DEFAULT_HP_SIGMA = 30
DEFAULT_HP_POWER = 0.8
DEFAULT_THETA = 1.0


@xr.register_dataarray_accessor("R")
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
        raise AttributeError(f"Attribute '{attr_name}' not found and no default provided.")

    def _set_attr(self, attr_name: str, value: float) -> None:
        assert isinstance(self._obj, xr.DataArray)
        self._obj.attrs[attr_name] = value

    @property
    def image(self) -> xr.DataArray:
        return self._obj

    @property
    def hp_image(self) -> xr.DataArray:
        image = self._obj

        hp_power = self.hp_threshold
        hp_sigma = self.hp_sigma

        blurred_image = ndimage.gaussian_filter(image, sigma=hp_sigma)
        high_pass_image = image - hp_power * blurred_image
        high_pass_image -= high_pass_image.min()

        return high_pass_image

    @property
    def screen_sample_distance(self) -> float:
        """Screen sample distance"""
        return self._get_attr("screen_sample_distance", 1.0)

    @property
    def theta(self) -> float:
        """Polar angle"""
        return self._get_attr("theta", DEFAULT_THETA)

    @theta.setter
    def theta(self, value: float) -> None:
        self._set_attr("theta", value)

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
        image["x"] = image.x * old_px_to_mm / px_to_mm
        image["y"] = image.y * old_px_to_mm / px_to_mm

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

    @property
    def hp_sigma(self) -> float:
        return self._get_attr("hp_sigma", DEFAULT_HP_SIGMA)

    @hp_sigma.setter
    def hp_sigma(self, value: int) -> None:
        if value <= 0:
            raise ValueError("hp_sigma must be positive.")
        self._set_attr("hp_sigma", value)

    @property
    def hp_threshold(self) -> float:
        return self._get_attr("hp_power", DEFAULT_HP_POWER)

    @hp_threshold.setter
    def hp_threshold(self, value: float) -> None:
        if value < 0:
            raise ValueError("hp_threshold must be non-negative.")
        self._set_attr("hp_power", value)

    def rotate(self, angle: float) -> None:
        image_data = self._obj.data
        image_data = ndimage.rotate(image_data, angle, reshape=False)
        self._obj.data = image_data

    def apply_screen_center(self, 
                            center_x: float = 0.0, 
                            center_y: float = 0.0,
                            auto_center: bool= False) -> None:
        image = self._obj

        if auto_center:
            center_x = find_horizontal_center(image)
            center_y = find_vertical_center(image)

        image["x"] = image.x - center_x
        image["y"] = image.y - center_y

        logger.info("The image was shifted to a new center.")

    def apply_hp_filter(self) -> None:
        image = self._obj
        image.data = image.R.hp_image.data
        logger.info("Original data was exchanged for hp filtered image!")

    def plot_image(
        self,
        ax: plt.Axes | None = None,
        hp_filter: bool = False,
        auto_levels: bool = False,
        **kwargs,
        ):

        image = self.hp_image if hp_filter else self.image

        return plot_image(
            image=image,
            ax=ax,
            screen_roi_width=self.screen_roi_width,
            screen_roi_height=self.screen_roi_height,
            hp_filter=hp_filter,
            auto_levels=auto_levels,
            **kwargs,
        )


@xr.register_dataarray_accessor("P")
class ProfileAccessor:

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def set_range(self) -> None:
        pass


