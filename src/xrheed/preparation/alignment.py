import logging
from typing import Optional, Tuple

import lmfit as lf  # type: ignore
import numpy as np
import xarray as xr
from lmfit.models import LinearModel, LorentzianModel  # type: ignore
from numpy.typing import NDArray
from scipy.signal import find_peaks  # type: ignore
from scipy.special import expit  # type: ignore

from xrheed.preparation.filters import gaussian_filter_profile

logger = logging.getLogger(__name__)


def find_horizontal_center(
    image: xr.DataArray,
    n_stripes: int = 10,
    prominence: float = 0.1,
    tolerance: float = 2.0,  # default in mm
) -> float:
    """
    Estimate horizontal (sx) symmetry center of a diffraction image.

    Parameters
    ----------
    image : xr.DataArray
        2D image with 'sx' and 'sy' coordinates.
    n_stripes : int, optional
        Number of horizontal stripes along 'sy' to analyze (default 10).
    prominence : float, optional
        Minimum prominence for peak detection (relative to normalized profile).
    tolerance : float, optional
        Maximum allowed deviation from global center (in sx units, default 0.2 mm).

    Returns
    -------
    float
        Estimated sx coordinate of symmetry center.
    """
    if "sx" not in image.coords or "sy" not in image.coords:
        raise AssertionError("Image must have 'sx' and 'sy' coordinates")

    # --- Global profile and approximate center ---
    global_profile = image.mean(dim="sy")
    smooth_sigma = 2.0 * _spot_sigma_from_profile(global_profile)
    global_profile_smooth = gaussian_filter_profile(global_profile, sigma=smooth_sigma)

    # Normalize
    y = global_profile_smooth.values.astype(float)
    y = (y - y.min()) / np.ptp(y)

    # Detect peaks
    peaks, _ = find_peaks(y, prominence=prominence)
    x_coords = global_profile_smooth.sx.values[peaks]
    heights = y[peaks]

    if x_coords.size == 0:
        raise RuntimeError("No peaks found in global profile")
    elif x_coords.size < 3:
        approx_center = float(global_profile_smooth.idxmax(dim="sx").item())
    else:
        approx_center = float(np.average(x_coords, weights=heights))
    logger.info("Global approx_center: %.4f", approx_center)

    global_max = global_profile_smooth.max()

    ny = int(image.sizes["sy"])
    stripe_height = max(1, ny // int(n_stripes))
    sx_coords = np.asarray(image.sx.values)

    centers = []
    for i in range(n_stripes):
        start = i * stripe_height
        end = ny if i == n_stripes - 1 else (i + 1) * stripe_height
        stripe = image.isel(sy=slice(start, end))
        profile = stripe.mean(dim="sy")
        if profile.size == 0:
            continue

        profile_smooth = gaussian_filter_profile(profile, sigma=smooth_sigma)
        if profile_smooth.max() < global_max * 0.7:
            continue

        y = profile_smooth.values.astype(float)
        y = (y - y.min()) / np.ptp(y)
        peaks, _ = find_peaks(y, prominence=prominence)
        if peaks.size == 0:
            continue
        x_coords = np.sort(sx_coords[peaks])

        # Candidate centers: mean of all + drop-one means
        candidates = [np.mean(x_coords)]
        if x_coords.size > 1:
            for j in range(x_coords.size):
                candidates.append(np.mean(np.delete(x_coords, j)))

        # Pick candidate closest to global approx_center
        center = min(candidates, key=lambda c: abs(c - approx_center))

        # Only accept if within fixed tolerance
        if abs(center - approx_center) <= tolerance:
            centers.append(center)
        else:
            logger.debug(
                "Stripe %d rejected: candidate %.3f vs approx_center %.3f (tol %.3f)",
                i,
                center,
                approx_center,
                tolerance,
            )

    if not centers:
        raise RuntimeError("No valid peaks found in any stripe.")

    center_final = float(np.median(centers))
    logger.info(
        "Estimated horizontal center: %.4f, using %d stripes",
        center_final,
        len(centers),
    )
    return center_final


def find_vertical_center(
    image: xr.DataArray,
    n_stripes: int = 10,
    center_x: float = 0.0,
) -> float:
    """
    Estimate the vertical (sy) center of a RHEED image using the shadow edge.
    The image is divided into vertical stripes along 'sx'; for each stripe,
    a profile along 'sy' is extracted and a linear+sigmoid model is fitted
    to locate the shadow edge. The final center is the median of valid fits.

    Parameters
    ----------
    image : xr.DataArray
        2D RHEED image with 'sx' and 'sy' coordinates.
    n_stripes : int, optional
        Number of vertical stripes along 'sx' to analyze (default 10).
    center_x : float, optional
        Horizontal center (sx) to subtract from coordinates before analysis
        (default 0.0). Useful to align profiles to a previously estimated
        horizontal center.

    Returns
    -------
    float
        Estimated sy coordinate of the vertical center.
    """
    if "sx" not in image.coords or "sy" not in image.coords:
        raise AssertionError("Image must have 'sx' and 'sy' coordinates")

    nx: int = int(image.sizes["sx"])
    stripe_width: int = max(1, nx // n_stripes)

    global_profile = image.mean(dim="sx")
    smooth_sigma: float = 1.0 * _spot_sigma_from_profile(global_profile)

    centers = []
    for i in range(n_stripes):
        start = i * stripe_width
        end = nx if i == n_stripes - 1 else (i + 1) * stripe_width
        stripe = image.isel(sx=slice(start, end))
        if stripe.size == 0:
            continue

        # Collapse stripe into a vertical profile
        profile = stripe.sum(dim="sx")

        # Smooth profile
        profile_smoothed = gaussian_filter_profile(profile, sigma=smooth_sigma)

        # Extract coordinates and values
        sy_coords = profile_smoothed["sy"].values
        vals = profile_smoothed.values.astype(float)

        # Find all local maxima
        peaks, _ = find_peaks(vals, prominence=0.1)
        if peaks.size == 0:
            continue

        # Filter peaks to only those at negative sy
        negative_mask = sy_coords[peaks] < 0
        negative_peaks = peaks[negative_mask]
        if negative_peaks.size == 0:
            continue

        # Take the *last* peak among negative sy (i.e. closest to zero from the left)
        peak_idx = int(negative_peaks[-1])

        # Restrict to the falling edge after that local maximum
        subprofile = profile_smoothed.isel(sy=slice(peak_idx, None))
        if subprofile.size == 0:
            continue
        sy_coords = subprofile["sy"].values
        vals = subprofile.values.astype(float)

        # Add synthetic plateau points before the falling edge
        n_extra = 100
        sy_step = np.median(np.diff(sy_coords))
        sy_extra = sy_coords[0] - sy_step * np.arange(n_extra, 0, -1)
        vals_extra = np.full_like(sy_extra, vals[0])  # flat extension

        # Concatenate synthetic + original data
        sy_coords = np.concatenate([sy_extra, sy_coords])
        vals = np.concatenate([vals_extra, vals])

        if np.ptp(vals) == 0:
            continue

        # Normalize
        vals = (vals - vals.min()) / np.ptp(vals)

        # Fit sigmoid with limited iterations
        sigmoid_model = lf.Model(_linear_plus_sigmoid)
        params = sigmoid_model.make_params(a=0.0, b=0.1, L=1.0, k=-0.5, x0=0.0)

        params["L"].set(min=0.8, max=1.2)
        params["k"].set(min=-2.0, max=-0.2)
        params["x0"].set(min=-10.0, max=10.0)
        params["a"].set(min=-0.2, max=0.2)
        params["b"].set(min=0.0, max=0.2)
        result = sigmoid_model.fit(
            vals,
            params=params,
            x=sy_coords,
            max_nfev=100,  # limit number of iterations
        )

        # Accept only if fit converged and quality is reasonable
        redchi = getattr(result, "redchi", np.inf)
        if result.success and redchi < 0.01:
            x0 = result.params["x0"].value
            k = result.params["k"].value
            # use the edge of the sigmoid about 16%
            center = x0 - np.log(5) / k
            centers.append(center)
            logger.debug(
                "Fit accepted: x0=%.4f, k=%.4f, redchi=%.4g", x0, k, result.redchi
            )
        else:
            logger.debug(
                "Fit rejected: success=%s, redchi=%.4g", result.success, result.redchi
            )

    if not centers:
        raise RuntimeError("No valid vertical centers found in any stripe.")

    center_y = float(np.median(centers))
    logger.info(
        "Vertical center estimated at %.4f, using %d edge profiles",
        center_y,
        len(centers),
    )

    # --- refinement: adjust using reflected and trismission spots if available ---
    sy_mirr, sy_trans = _find_reflection_and_transmission_spots(
        image, center_x=center_x, center_y=center_y
    )

    if sy_trans is not None:
        shadow_edge = 0.5 * (sy_trans + sy_mirr)
        center_y += shadow_edge
        logger.info("Adjust using reflected and trismission spots: %.4f", shadow_edge)
    else:
        logger.debug("Incident angle refinement skipped (%s)")

    return center_y


def find_incident_angle(
    image: xr.DataArray,
    y_range: tuple[float, float] = (-30, 30),
    prominence: float = 0.1,
) -> float:
    """
    Find incident angle in degrees using reflection/transmission spots near sx=0.
    """
    screen_sample_distance: float = image.ri.screen_sample_distance
    logger.debug(
        "find_incident_angle: screen_sample_distance=%.4f, y_range=%s, prominence=%.3f",
        screen_sample_distance,
        y_range,
        prominence,
    )

    sy_mirr, sy_trans = _find_reflection_and_transmission_spots(
        image, y_range=y_range, prominence=prominence
    )

    logger.info("Mirror spot detected at sy=%.4f", sy_mirr)
    if sy_trans is not None:
        logger.info("Transmission spot detected at sy=%.4f", sy_trans)
    beta_deg = _calculate_incident_angle(sy_mirr, sy_trans, screen_sample_distance)
    if sy_trans is not None:
        spot_distance = sy_trans - sy_mirr
        shadow_edge = 0.5 * (sy_trans + sy_mirr)
        logger.info("Spot distance=%.4f, Shadow edge=%.4f", spot_distance, shadow_edge)
        logger.info("Incident angle (deg) from reflection+transmission: %.4f", beta_deg)
    else:
        logger.warning(
            "Transmission spot not detected; using reflection-only estimate. "
            "Incident angle (deg)=%.4f",
            beta_deg,
        )
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
    return a * x + b + L * expit(k * (x - x0))


def _spot_sigma_from_profile(
    profile: xr.DataArray, start_window: int = 5, max_window: int = 50
) -> float:
    """
    Helper: Fit a Lorentzian + linear background around the strongest peak
    in a 1D diffraction profile. Start with a small window and
    iteratively expand until the fit stabilizes.

    Parameters
    ----------
    profile : xr.DataArray
        1D profile with coordinate 'sx' (mm).
    start_window : int
        Initial half-width of the fitting window (points).
        Also used as the step size for expansion.
    max_window : int
        Maximum half-width to try.

    Returns
    -------
    sigma_mm : float
        Lorentzian sigma (HWHM) in mm.
    """
    # choose coordinate name: prefer 'sx', fall back to 'sy'
    if "sx" in profile.coords:
        x = profile["sx"].values.astype(float)
    elif "sy" in profile.coords:
        x = profile["sy"].values.astype(float)
    else:
        raise AssertionError("Profile must have 'sx' or 'sy' coordinate")

    y = profile.values.astype(float)
    n = len(y)

    if x.shape[0] != n:
        raise RuntimeError("Coordinate length does not match profile length")

    i_max = int(np.argmax(y))

    best_sigma = None
    prev_sigma = None

    for half in range(start_window, max_window + 1, start_window):
        left = max(0, i_max - half)
        right = min(n, i_max + half)
        xw = x[left:right]
        yw = y[left:right]

        if len(xw) < 5:
            continue

        # Model: Lorentzian + linear background
        lmod = LorentzianModel(prefix="l_")
        bmod = LinearModel(prefix="b_")
        model = lmod + bmod

        params = model.make_params()
        params["l_center"].set(value=x[i_max], min=xw.min(), max=xw.max())
        params["l_sigma"].set(value=(xw[-1] - xw[0]) / 20, min=np.diff(xw).mean())
        params["l_amplitude"].set(
            value=(yw.max() - yw.min()) * (xw[-1] - xw[0]) / 10, min=0
        )
        params["b_slope"].set(value=0)
        params["b_intercept"].set(value=yw.min())

        try:
            result = model.fit(yw, params, x=xw)
            sigma = float(result.params["l_sigma"].value)
        except Exception:
            continue

        # Check stability: if sigma stops changing much, accept it
        if prev_sigma is not None and abs(sigma - prev_sigma) < 0.05 * sigma:
            best_sigma = sigma
            break

        prev_sigma = sigma
        best_sigma = sigma

    if best_sigma is None:
        raise RuntimeError("No stable fit found")

    return best_sigma


def _find_reflection_and_transmission_spots(
    image: xr.DataArray,
    y_range: tuple[float, float] = (-30, 30),
    prominence: float = 0.1,
    center_x: float = 0.0,
    center_y: float = 0.0,
) -> Tuple[float, Optional[float]]:
    """
    Detect reflection (sy<0) and transmission (sy>0) spots near sx=0.
    Optionally shift coordinates by center_x and center_y.

    Parameters
    ----------
    image : xr.DataArray
        RHEED image with 'sx' and 'sy' coordinates.
    y_range : tuple(float, float), optional
        Range of sy to select for the vertical profile (default -30..30).
    prominence : float, optional
        Minimum prominence for peak detection.
    center_x : float, optional
        Horizontal center to subtract from sx (default 0.0).
    center_y : float, optional
        Vertical center to subtract from sy (default 0.0).

    Returns
    -------
    sy_mirr : float
        Position of the reflection spot (always required).
    sy_trans : float | None
        Position of the transmission spot, or None if not found.
    """
    # --- determine sx range dynamically from reflection profile ---
    profile_for_sigma = image.sel(sy=slice(-20, 0)).sum(dim="sy")
    sigma = _spot_sigma_from_profile(profile_for_sigma) * 0.5
    x_range = (center_x - sigma, center_x + sigma)

    # --- vertical profile near sx=0 ---
    vertical_profile: xr.DataArray = image.sel(
        sx=slice(*x_range), sy=slice(*y_range)
    ).sum("sx")

    sigma = _spot_sigma_from_profile(vertical_profile) * 0.5
    vertical_profile = gaussian_filter_profile(vertical_profile, sigma=sigma)

    sy_coords = vertical_profile.sy.values - center_y
    vals = vertical_profile.values.astype(float)

    if np.ptp(vals) == 0:
        raise RuntimeError("Flat profile: cannot detect spots")

    vals -= vals.min()
    vals /= vals.max()

    peaks, _ = find_peaks(vals, prominence=prominence)
    if peaks.size == 0:
        raise RuntimeError("No peaks detected in vertical profile")

    sy_peaks = sy_coords[peaks]
    vals_peaks = vals[peaks]

    refl_candidates = sy_peaks[sy_peaks < 0]
    trans_candidates = sy_peaks[sy_peaks > 0]

    if refl_candidates.size == 0:
        raise RuntimeError("No reflection spot detected")

    sy_mirr = float(refl_candidates[np.argmax(vals_peaks[sy_peaks < 0])])
    sy_trans = None
    if trans_candidates.size > 0:
        sy_trans = float(trans_candidates[np.argmax(vals_peaks[sy_peaks > 0])])

    return sy_mirr, sy_trans


def _calculate_incident_angle(
    sy_mirr: float, sy_trans: Optional[float], screen_sample_distance: float
) -> float:
    """
    Calculate incident angle beta (deg) from mirror and transmission spot positions.
    If sy_trans is None, use reflection-only estimate.
    """
    if sy_trans is not None:
        spot_distance = sy_trans - sy_mirr
        beta_rad = np.arctan(0.5 * spot_distance / screen_sample_distance)
        beta_deg = np.degrees(beta_rad)
    else:
        beta_rad = np.arctan(sy_mirr / screen_sample_distance)
        beta_deg = np.degrees(beta_rad)
    return beta_deg
