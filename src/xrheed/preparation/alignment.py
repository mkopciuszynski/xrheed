import xarray as xr
import numpy as np

from scipy.signal import savgol_filter
import ruptures as rpt


def find_horizontal_center(image: xr.DataArray) -> float:

    profile = image.sum("y")
    return float(image.x[profile.argmax()])


def find_vertical_center(image: xr.DataArray, shadow_edge_width: float = 5.0) -> float:
    # Shadow edges defines the vertical center 0, 0 point of an image

    profile = image.sel(x=slice(-20, 20)).mean("x")
    edge_width_px = int(shadow_edge_width * image.R.screen_scale)

    smoothed_data = savgol_filter(profile, window_length=edge_width_px, polyorder=1)

    gradient = np.diff(smoothed_data)

    algo = rpt.Dynp(model="l2").fit(gradient)
    breakpoints = algo.predict(n_bkps=2)

    edge_pos = image.y[breakpoints[0]]
    return float(edge_pos)
