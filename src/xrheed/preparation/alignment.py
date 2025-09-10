import lmfit as lf  # type: ignore
import numpy as np
import xarray as xr
from numpy.typing import NDArray

from xrheed.preparation.filters import gaussian_filter_profile


def find_horizontal_center(image: xr.DataArray) -> float:
    """
    Find the horizontal center of a RHEED image by summing along the y-axis and finding the maximum position.

    Parameters
    ----------
    image : xr.DataArray
        RHEED image with 'sx' and 'sy' coordinates.

    Returns
    -------
    float
        The sx-coordinate of the horizontal center.
    """

    profile: xr.DataArray = image.sum("sy")
    profile_smoothed: xr.DataArray = gaussian_filter_profile(profile, sigma=1.0)
    max_pos: float = profile_smoothed.sx.values[np.argmax(profile_smoothed.values)]

    # TODO improve this adding additional horizontal_center search

    return max_pos


def find_vertical_center(image: xr.DataArray, shadow_edge_width: float = 5.0) -> float:
    """
    Find the vertical center of a RHEED image using the shadow edge and a linear+sigmoid fit.

    Parameters
    ----------
    image : xr.DataArray
        RHEED image with 'sx' and 'sy' coordinates.
    shadow_edge_width : float, optional
        Estimated width of the shadow edge (default is 5.0).

    Returns
    -------
    float
        The y-coordinate of the vertical center.
    """

    x_range: float = 20.0
    x_mirror_spot_size: float = 3.0

    profile: xr.DataArray = image.where(
        ((image.sx >= -x_range) & (image.sx <= -x_mirror_spot_size))
        | ((image.sx >= x_mirror_spot_size) & (image.sx <= x_range)),
        drop=True,
    ).sum(dim="sx")

    sigma: float = shadow_edge_width * 0.1
    profile_smoothed: xr.DataArray = gaussian_filter_profile(profile, sigma=sigma)
    max_idx: int = int(np.argmax(profile_smoothed.values))

    subprofile: xr.DataArray = profile_smoothed.isel(sy=slice(max_idx, None))

    # Prepare data for fitting
    sx: NDArray = subprofile["sy"].values
    sy: NDArray = subprofile.values

    sy -= sy.min()
    sy /= sy.max()

    sigmoid_model: lf.Model = lf.Model(_linear_plus_sigmoid)

    params = sigmoid_model.make_params(a=0.0, b=0.0, L=1.0, k=0.1, x0=0.0)

    result = sigmoid_model.fit(sy, params=params, x=sx)
    sigmoid_center: float = result.params["x0"].value
    sigmoid_k: float = result.params["k"].value

    return sigmoid_center - sigmoid_k * 3.0


def find_incident_angle(
    image: xr.DataArray,
    x_range: tuple[float, float] = (-3, 3),
    y_range: tuple[float, float] = (-30, 30),
) -> float:
    """
    Find incident angle in degrees
    using the position of transmission and mirror spots.

    Parameters:
    -----------
    image : xarray.DataArray
        RHEED image with 'sx' and 'sy' coordinates.
    x_range : tuple(float, float)
        The range of x to select from the image.
    y_range : tuple(float, float)
        The range of y to select from the image.

    Returns:
    --------
    beta_deg : float
        Angle beta in degrees.
    """

    screen_sample_distance: float = image.ri.screen_sample_distance

    # Sum along y (or x) to get a 1D profile.
    # Here summing over 'y' to get vertical profile along x.
    vertical_profile: xr.DataArray = image.sel(
        sx=slice(*x_range), sy=slice(*y_range)
    ).sum("sx")

    # Transmission spot: y > 0
    trans_part: xr.DataArray = vertical_profile.sel(sy=slice(0, 30))
    x_trans: NDArray = trans_part.sy[np.argmax(trans_part.values)].item()

    # Mirror spot: y < 0
    mirr_part: xr.DataArray = vertical_profile.sel(sy=slice(-30, 0))
    x_mirr: NDArray = mirr_part.sy[np.argmax(mirr_part.values)].item()

    # Calculate distance and shadow edge
    spot_distance: float = float(x_trans - x_mirr)
    shadow_edge: float = float(0.5 * (x_trans + x_mirr))

    # Calculate beta in radians
    beta_rad: float = np.arctan(0.5 * spot_distance / screen_sample_distance)

    # Convert to degrees
    beta_deg: float = np.degrees(beta_rad)

    print(f"Transmission spot at: {x_trans:.2f}")
    print(f"Mirror spot at: {x_mirr:.2f}")
    print(f"Spot distance: {spot_distance:.2f}")
    print(f"Shadow edge: {shadow_edge:.2f}")
    print(f"Theta angle: {beta_deg:.2f}")

    return beta_deg


# Define sigmoid function for fitting
def _sigmoid(x: NDArray, amp: float, k: float, x0: float, back: float) -> NDArray:
    """
    Sigmoid function used for fitting shadow edges.

    Parameters
    ----------
    x : NDArray
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
    NDArray
        Sigmoid function values.
    """
    return amp / (1 + np.exp(-k * (x - x0))) + back


# Model: Linear + Sigmoid
def _linear_plus_sigmoid(
    x: NDArray, a: float, b: float, L: float, k: float, x0: float
) -> NDArray:
    """
    Linear plus sigmoid model for fitting shadow edges.

    Parameters
    ----------
    x : NDArray
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
    NDArray
        Model values.
    """
    return a * x + b + L / (1 + np.exp(-k * (x - x0)))
