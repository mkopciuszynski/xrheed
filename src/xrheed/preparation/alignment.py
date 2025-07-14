import xarray as xr
import numpy as np

import lmfit as lf


from xrheed.preparation.filters import gaussian_filter_profile


def find_horizontal_center(image: xr.DataArray) -> float:

    profile = image.sum("y")
    profile_smoothed = gaussian_filter_profile(profile, sigma=1.0)
    max_pos = profile_smoothed.x.values[np.argmax(profile_smoothed.values)]

    # TODO improve this adding additional horizontal_center search

    return max_pos


def find_vertical_center(image: xr.DataArray, shadow_edge_width: float = 5.0) -> float:
    # Shadow edges defines the vertical center 0, 0 point of an image

    x_range = 20.0
    x_mirror_spot_size = 3.0

    profile = image.where(
        ((image.x >= -x_range) & (image.x <= -x_mirror_spot_size))
        | ((image.x >= x_mirror_spot_size) & (image.x <= x_range)),
        drop=True,
    ).sum(dim="x")

    sigma = shadow_edge_width * 0.1
    profile_smoothed = gaussian_filter_profile(profile, sigma=sigma)

    max_idx = profile_smoothed.argmax()
    subprofile = profile_smoothed.isel(y=slice(max_idx.item(), None))

    # Prepare data for fitting
    x = subprofile["y"].values
    y = subprofile

    y -= y.min()
    y /= y.max()

    sigmoid_model = lf.Model(_linear_plus_sigmoid)

    params = sigmoid_model.make_params(a=0.0, b=0.0, L=1.0, k=0.1, x0=0.0)

    result = sigmoid_model.fit(y, params=params, x=x)
    sigmoid_center = result.params["x0"].value
    sigmoid_k = result.params["k"].value

    return sigmoid_center - sigmoid_k * 3.0


def find_theta(
    image: xr.DataArray,
    x_range: tuple[float, float] = (-3, 3),
    y_range: tuple[float, float] = (-30, 30),
) -> float:
    """
    Find incident theta angle in degrees
    using the position of transmission and mirror spots.

    Parameters:
    -----------
    image : xarray.DataArray
        RHEED image with 'x' and 'y' coordinates.
    x_range : tuple(float, float)
        The range of x to select from the image.
    y_range : tuple(float, float)
        The range of y to select from the image.

    Returns:
    --------
    theta_deg : float
        Angle theta in degrees.
    """

    screen_sample_distance = image.R.screen_sample_distance

    # Sum along y (or x) to get a 1D profile.
    # Here summing over 'y' to get vertical profile along x.
    vertical_profile = image.sel(x=slice(*x_range), y=slice(*y_range)).sum("x")

    # Transmission spot: y > 0
    trans_part = vertical_profile.sel(y=slice(0, 30))
    x_trans = trans_part.y[np.argmax(trans_part.values)].item()

    # Mirror spot: y < 0
    mirr_part = vertical_profile.sel(y=slice(-30, 0))
    x_mirr = mirr_part.y[np.argmax(mirr_part.values)].item()

    # Calculate distance and shadow edge
    spot_distance = x_trans - x_mirr
    shadow_edge = 0.5 * (x_trans + x_mirr)

    # Calculate theta in radians
    theta_rad = np.arctan(0.5 * spot_distance / screen_sample_distance)

    # Convert to degrees
    theta_deg = np.degrees(theta_rad)

    print(f"Transmission spot at: {x_trans:.2f}")
    print(f"Mirror spot at: {x_mirr:.2f}")
    print(f"Spot distance: {spot_distance:.2f}")
    print(f"Shadow edge: {shadow_edge:.2f}")
    print(f"Theta angle: {theta_deg:.2f}")

    return theta_deg


# Define sigmoid function for fitting
def _sigmoid(x, amp, k, x0, back):
    return amp / (1 + np.exp(-k * (x - x0))) + back


# Model: Linear + Sigmoid
def _linear_plus_sigmoid(x, a, b, L, k, x0):
    return a * x + b + L / (1 + np.exp(-k * (x - x0)))
