from typing import Optional, Tuple
import numpy as np


def convert_x_to_kx(
    x_coords_mm: np.ndarray,
    ewald_sphere_radius: float,
    screen_sample_distance_mm: float,
) -> np.ndarray:
    """Convert x coordinates from mm to kx [1/Å] using the Ewald sphere radius and screen-sample distance.
    Parameters
    ----------
    x_coords_mm : np.ndarray
        Array of x coordinates in millimeters (mm).
    ewald_sphere_radius : float
        Radius of the Ewald sphere in reciprocal space (1/Å).
    screen_sample_distance_mm : float
        Distance from the sample to the screen in millimeters (mm).

    Returns
    -------
    np.ndarray
        Converted x coordinates in kx [1/Å].
    """

    kx = (x_coords_mm / screen_sample_distance_mm) * ewald_sphere_radius

    return kx


def convert_gx_gy_to_sx_sy(
    gx: np.ndarray,
    gy: np.ndarray,
    ewald_radius: float,
    beta: float,
    screen_sample_distance: float,
    remove_outside: Optional[bool] = True,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert reciprocal lattice coordinates (gx, gy) to RHEED screen coordinates (sx, sy)
    using the Ewald sphere construction.

    Parameters
    ----------
    gx : np.ndarray
        Array of reciprocal lattice x-coordinates.
    gy : np.ndarray
        Array of reciprocal lattice y-coordinates.
    ewald_radius : float
        Radius of the Ewald sphere in reciprocal space (1/Å or same units as gx, gy).
    beta : float
        Incident beam angle in degrees relative to the surface normal.
    screen_sample_distance : float
        Distance from the sample to the detector/screen.
    remove_outside : Optional[bool], default=True
        If True, points outside the Ewald sphere are removed.
        If False, points outside are set to NaN.
    **kwargs
        Additional keyword arguments (currently unused).

    Returns
    -------
    sx : np.ndarray
        Array of x-coordinates on the RHEED screen corresponding to input gx, gy.
    sy : np.ndarray
        Array of y-coordinates on the RHEED screen corresponding to input gx, gy.

    Notes
    -----
    - The function assumes a simple planar screen perpendicular to the z-axis.
    - The coordinate transformation accounts for the Ewald sphere geometry
      and the projection of diffraction spots onto the screen.
    - `beta` is the incident angle of the electron beam relative to the sample surface.
    - Points outside the Ewald sphere can be optionally removed or set as NaN
      using the `remove_outside` flag.
    """
    
    # Ewald sphere radius
    k0 = ewald_radius
    # Ewald sphere radius square
    kk = k0**2
    
    # calculate the shift between the center of Ewald sphere and the center of reciprocal lattice
    delta_x = k0 * np.cos(np.deg2rad(beta))

    # shift the center of reciprocal lattice
    kx = gx + delta_x
    ky = gy

    # Check if the kx, ky points are inside Ewald sphere
    kxy2 = kx**2 + ky**2
    ind = kxy2 < kk
    # remove those outside or mark as nans
    if remove_outside:
        kx = kx[ind]
        ky = ky[ind]
    else:
        kx[~ind] = np.nan
        ky[~ind] = np.nan

    # calculate the radius r_k
    rk = np.sqrt(k0**2 - kx**2)

    # calculate theta and phi (cos) values
    phi = np.arccos(ky / rk)
    theta = np.arcsin(rk / k0)

    # calculate the radius on the RHEED screen
    rho = screen_sample_distance * np.tan(theta)

    # calculate the spot positions
    sx = rho * np.cos(phi)
    sy = -rho * np.sin(phi)

    return sx, sy
