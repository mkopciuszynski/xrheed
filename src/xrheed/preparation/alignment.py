import logging
import lmfit as lf  # type: ignore
import numpy as np
import xarray as xr
from typing import Optional, Tuple
from lmfit.models import LinearModel, LorentzianModel
from numpy.typing import NDArray
from scipy.signal import find_peaks
from xrheed.preparation.filters import gaussian_filter_profile


logger = logging.getLogger(__name__)


def find_horizontal_center(
    image: xr.DataArray,
    n_stripes: int = 20,
    smooth_sigma: Optional[float] = None,
    min_prominence: float = 0.1,
) -> float:
    """
    Estimate horizontal (sx) symmetry center of a diffraction image.

    Parameters
    ----------
    image : xr.DataArray
        2D image with 'sx' and 'sy' coords.
    n_stripes : int, optional
        Number of horizontal stripes along 'sy' to analyze (default 20).
    smooth_sigma : float, optional
        Smoothing sigma in sx units. If None, estimated automatically
        from the global profile using _spot_sigma_from_profile, then scaled 2.0.
    min_prominence : float, optional
        Minimum prominence for peak detection (relative to normalized profile).

    Returns
    -------
    float
        Estimated sx coordinate of symmetry center.
    """
    if "sx" not in image.coords or "sy" not in image.coords:
        raise AssertionError("Image must have 'sx' and 'sy' coordinates")

    # --- Estimate smoothing sigma if not provided ---
    if smooth_sigma is None:
        global_profile = image.sum(dim="sy")
        smooth_sigma = 2.0 * _spot_sigma_from_profile(global_profile)
        logger.debug("Auto smooth_sigma estimated: %.4f", smooth_sigma)

    ny = int(image.sizes["sy"])
    stripe_height = max(1, ny // int(n_stripes))
    sx_coords = np.asarray(image.sx.values)

    # --- First pass: collect maxima from all stripes ---
    stripe_profiles = []
    stripe_maxima = []
    for i in range(n_stripes):
        start = i * stripe_height
        end = ny if i == n_stripes - 1 else (i + 1) * stripe_height
        stripe = image.isel(sy=slice(start, end))
        profile = stripe.sum(dim="sy")
        if profile.size == 0:
            continue
        profile_smooth = gaussian_filter_profile(profile, sigma=smooth_sigma)
        stripe_profiles.append((profile_smooth, sx_coords))
        stripe_maxima.append(profile_smooth.values.max())

    if not stripe_profiles:
        raise RuntimeError("No valid stripes found.")

    avg_max = np.mean(stripe_maxima)
    logger.debug("Average stripe max: %.4f", avg_max)

    centers = []
    for idx, ((profile_smooth, sx_coords), max_val) in enumerate(
        zip(stripe_profiles, stripe_maxima)
    ):
        logger.debug("Stripe %d: max=%.4f", idx, max_val)

        # Skip weak stripes (< 50% of average max)
        if max_val < avg_max * 0.5:
            logger.debug("Stripe %d skipped (too weak)", idx)
            continue

        y = profile_smooth.values.astype(float)

        y_norm = (y - y.min()) / np.ptp(y)
        peaks, _ = find_peaks(y_norm, prominence=min_prominence)

        if peaks.size == 1:
            center = float(sx_coords[peaks[0]])
            centers.append(center)
            logger.debug("Stripe %d: single peak at %.4f", idx, center)
        elif peaks.size == 2:
            x1, x2 = sx_coords[peaks[0]], sx_coords[peaks[1]]
            center = float(0.5 * (x1 + x2))
            centers.append(center)
            logger.debug(
                "Stripe %d: two peaks at %.4f, %.4f → midpoint %.4f",
                idx,
                x1,
                x2,
                center,
            )
        else:
            logger.debug("Stripe %d skipped (found %d peaks)", idx, peaks.size)
            continue

    if not centers:
        raise RuntimeError("No valid peaks found in any stripe.")

    center_final = float(np.median(centers))
    logger.info(
        "Estimated horizontal center: %.4f, using %d selected profile",
        center_final,
        len(centers),
    )
    return center_final


def find_vertical_center(
    image: xr.DataArray,
    shadow_edge_width: float = 5.0,
    n_stripes: int = 20,
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
    shadow_edge_width : float, optional
        Estimated width of the shadow edge (default 5.0).
    n_stripes : int, optional
        Number of vertical stripes along 'sx' to analyze (default 20).

    Returns
    -------
    float
        Estimated sy coordinate of the vertical center.
    """
    if "sx" not in image.coords or "sy" not in image.coords:
        raise AssertionError("Image must have 'sx' and 'sy' coordinates")

    nx = int(image.sizes["sx"])
    stripe_width = max(1, nx // n_stripes)

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
        sigma = shadow_edge_width * 0.1
        profile_smoothed = gaussian_filter_profile(profile, sigma=sigma)

        # Take only the falling edge after the maximum
        max_idx = int(np.argmax(profile_smoothed.values))
        subprofile = profile_smoothed.isel(sy=slice(max_idx, None))
        if subprofile.size == 0:
            continue

        sy_coords = subprofile["sy"].values
        vals = subprofile.values.astype(float)

        if np.ptp(vals) == 0:
            continue

        # Normalize
        vals -= vals.min()
        vals /= vals.max()

        # Fit sigmoid
        sigmoid_model = lf.Model(_linear_plus_sigmoid)
        params = sigmoid_model.make_params(
            a=0.0, b=0.0, L=1.0, k=0.1, x0=float(sy_coords.mean())
        )
        try:
            result = sigmoid_model.fit(vals, params=params, x=sy_coords)
            x0 = result.params["x0"].value
            k = result.params["k"].value
            center = x0 - k
            centers.append(center)
            logger.debug(
                "Stripe %d: fitted x0=%.4f k=%.4f → center=%.4f", i, x0, k, center
            )
        except Exception as e:
            logger.debug("Stripe %d: fit failed (%s)", i, str(e))
            continue

    if not centers:
        raise RuntimeError("No valid vertical centers found in any stripe.")

    center_final = float(np.median(centers))
    logger.info(
        "Vertical center estimated at %.4f (from %d stripes)",
        center_final,
        len(centers),
    )

    # --- refinement: adjust with shadow edge if available ---
    try:
        sy_mirr, sy_trans = _find_reflection_and_transmission_spots(image)
        if sy_trans is not None:
            shadow_edge = 0.5 * (sy_trans + sy_mirr)
            center_final += shadow_edge
            logger.info(
                "Adjusted vertical center with shadow edge %.4f → %.4f",
                shadow_edge,
                center_final,
            )
    except Exception as e:
        logger.debug("Incident angle refinement skipped (%s)", str(e))

    return center_final


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
    return a * x + b + L / (1 + np.exp(-k * (x - x0)))


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
) -> Tuple[float, Optional[float]]:
    """
    Detect reflection (sy<0) and transmission (sy>0) spots near sx=0.

    Parameters
    ----------
    image : xr.DataArray
        RHEED image with 'sx' and 'sy' coordinates.
    y_range : tuple(float, float), optional
        Range of sy to select for the vertical profile (default -30..30).
    prominence : float, optional
        Minimum prominence for peak detection.

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
    x_range = (-sigma, sigma)

    # --- vertical profile near sx=0 ---
    vertical_profile: xr.DataArray = image.sel(
        sx=slice(*x_range), sy=slice(*y_range)
    ).sum("sx")

    sigma = _spot_sigma_from_profile(vertical_profile) * 0.5
    vertical_profile = gaussian_filter_profile(vertical_profile, sigma=sigma)

    sy_coords = vertical_profile.sy.values
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
