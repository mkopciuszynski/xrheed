import xarray as xr
import numpy as np

import lmfit as lf


from xrheed.preparation.filters import gaussian_filter_profile


def find_horizontal_center(image: xr.DataArray) -> float:
    """
    Find the horizontal center of a RHEED image by summing along the y-axis and finding the maximum position.

    Parameters
    ----------
    image : xr.DataArray
        RHEED image with 'x' and 'y' coordinates.

    Returns
    -------
    float
        The x-coordinate of the horizontal center.
    """

    profile = image.sum("y")
    profile_smoothed = gaussian_filter_profile(profile, sigma=1.0)
    max_pos = profile_smoothed.x.values[np.argmax(profile_smoothed.values)]

    # TODO improve this adding additional horizontal_center search

    return max_pos


def find_vertical_center(image: xr.DataArray, shadow_edge_width: float = 5.0) -> float:

    """
    Find the vertical center of a RHEED image using the shadow edge and a linear+sigmoid fit.

    Parameters
    ----------
    image : xr.DataArray
        RHEED image with 'x' and 'y' coordinates.
    shadow_edge_width : float, optional
        Estimated width of the shadow edge (default is 5.0).

    Returns
    -------
    float
        The y-coordinate of the vertical center.
    """

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

    screen_sample_distance = image.ri.screen_sample_distance

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
    beta_rad = np.arctan(0.5 * spot_distance / screen_sample_distance)

    # Convert to degrees
    beta_deg = np.degrees(beta_rad)

    print(f"Transmission spot at: {x_trans:.2f}")
    print(f"Mirror spot at: {x_mirr:.2f}")
    print(f"Spot distance: {spot_distance:.2f}")
    print(f"Shadow edge: {shadow_edge:.2f}")
    print(f"Theta angle: {beta_deg:.2f}")

    return beta_deg


# Define sigmoid function for fitting
def _sigmoid(x, amp, k, x0, back):
    """
    Sigmoid function used for fitting shadow edges.

    Parameters
    ----------
    x : array-like
        Input values.
    amp : float
        Amplitude.
    k : float
        Slope.
    x0 : float
        Center position.
    back : float
        Background offset.

    Returns
    -------
    array-like
        Sigmoid function values.
    """
    return amp / (1 + np.exp(-k * (x - x0))) + back


# Model: Linear + Sigmoid
def _linear_plus_sigmoid(x, a, b, L, k, x0):
    """
    Linear plus sigmoid model for fitting shadow edges.

    Parameters
    ----------
    x : array-like
        Input values.
    a : float
        Linear slope.
    b : float
        Linear offset.
    L : float
        Sigmoid amplitude.
    k : float
        Sigmoid slope.
    x0 : float
        Sigmoid center.

    Returns
    -------
    array-like
        Model values.
    """
    return a * x + b + L / (1 + np.exp(-k * (x - x0)))
