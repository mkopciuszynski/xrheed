from scipy import ndimage
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

from scipy.signal import savgol_filter
import ruptures as rpt


@xr.register_dataarray_accessor("R")
class RHEEDAccessor:    

    def __init__(self, xarray_obj: xr.DataArray) -> None:
        self._obj = xarray_obj
        self._center = None

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

        return self._obj.attrs["screen_sample_distance"]
    
    @property 
    def theta(self) -> float:
        """Polar angle"""

        if 'theta' in self._obj.attrs:
            return self._obj.attrs['theta']
        else:
            print("Warning: theta (polar) angle not found in attributes. Using default value of 1.0")
            return 1.0

    @theta.setter
    def theta(self, value):
        self._obj.attrs['theta'] = value

    @property
    def screen_scale(self) -> float:
        """Screen scaling px to mm"""

        assert isinstance(self._obj, xr.DataArray)
        return self._obj.attrs["screen_scale"]
    
    @screen_scale.setter 
    def screen_scale(self, px_to_mm: float): 

        if px_to_mm < 0: 
            raise ValueError("Cannot be negative")
        
        image = self._obj

        old_px_to_mm = image.attrs['screen_scale']
        image.attrs['screen_scale'] = px_to_mm

        image["x"] = image.x * old_px_to_mm / px_to_mm
        image["y"] = image.y * old_px_to_mm / px_to_mm

    @property
    def screen_width(self) -> float:
        """Screen width in mm"""

        assert isinstance(self._obj, xr.DataArray)
        return self._obj.attrs["screen_width"]
    
    @property 
    def screen_roi_width(self) -> float:
        return self._obj.attrs.get('screen_roi_width', 50.0)
    
    @screen_roi_width.setter
    def screen_roi_width(self, value: float):
        self._obj.attrs['screen_roi_width'] = value

    @property
    def screen_roi_height(self) -> float:
        return self._obj.attrs.get('screen_roi_height', 50.0)
    
    @screen_roi_height.setter
    def screen_roi_height(self, value: float):
        self._obj.attrs['screen_roi_height'] = value
    
    @property
    def beam_energy(self) -> float:
        """Screen width in mm"""

        assert isinstance(self._obj, xr.DataArray)
        return self._obj.attrs["beam_energy"]
    
    @property
    def hp_sigma(self) -> int:
        return self._obj.attrs.get('hp_sigma', 30)
    
    @hp_sigma.setter
    def hp_sigma(self, value: int):
        self._obj.attrs['hp_sigma'] = value
    
    @property
    def hp_threshold(self) -> float:
        return self._obj.attrs.get('hp_power', 0.8)
    
    @hp_threshold.setter
    def hp_threshold(self, value: float):
        self._obj.attrs['hp_power'] = value
    
    def rotate(self, phi: float) -> None:
        image_data = self._obj.data
        image_data = ndimage.rotate(image_data, phi, reshape=False)
        self._obj.data = image_data

    def set_center(self) -> None:
        image = self._obj
        image["x"] = image.x - _horizontal_center(image)
        image["y"] = image.y - _vertical_center(image)

    def apply_hp_filter(self) -> None:
        image = self._obj
        image.data = image.R.hp_image.data
        print("Original data was exchanged for hp filtered image!")


    def plot_image(
            self,
            ax: plt.Axes | None = None,
            hp_filter: bool = False,
            auto_levels: bool = False,
            **kwargs):

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))
        
        if hp_filter:
            image = self.hp_image
        else:
            image = self.image

        if auto_levels:
            vmin = image.min().values
            vmax = image.mean().values
            image = (image - vmin) / (vmax - vmin) * 50

        image.plot(ax=ax, cmap="gray", add_colorbar=False, **kwargs)

        ax.set_xlim(-self.screen_roi_width, self.screen_roi_width)
        ax.set_ylim(-self.screen_roi_height, 5)
            
        ax.set_xlabel("Screen x [mm]") 
        ax.set_ylabel("Screen y [mm]") 
        
        ax.axhline(y=0.0, linewidth=0.5, color='w')
        ax.axvline(x=0.0, linewidth=0.5, color='w')

        ax.set_aspect(1)
        return ax
    

@xr.register_dataarray_accessor("P")
class ProfileAccessor():

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def set_range():
        pass


def _horizontal_center(image: xr.DataArray) -> float:
    # Find the horizontal center
    
    profile = image.sum('y')
    x_max = float(image.x[profile.argmax()])

    # Shift the x-coordinates by the found maximum
    return x_max


def _vertical_center(image: xr.DataArray, 
                     edge_width: float = 5.0) -> float:
    # Shadow edges defines the vertical center 0,0 point of an image


    profile = image.sel(x=slice(-20, 20)).mean("x")
    edge_width_px = int(edge_width*image.R.screen_scale)

    smoothed_data = savgol_filter(profile, 
                                window_length=edge_width_px, 
                                polyorder=1) 

    gradient = np.diff(smoothed_data)

    algo = rpt.Dynp(model="l2").fit(gradient)
    breakpoints = algo.predict(n_bkps=2)


    edge_pos = image.y[breakpoints[0]]
    return edge_pos

