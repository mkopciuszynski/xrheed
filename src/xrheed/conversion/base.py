from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


def convert_sx_to_ky(
    x_coords_mm: NDArray,
    ewald_sphere_radius: float,
    screen_sample_distance_mm: float,
) -> NDArray:
    """Convert sx coordinates from mm to ky [1/Å] using the Ewald sphere radius and screen-sample distance.
    Parameters
    ----------
    x_coords_mm : NDArray
        Array of x coordinates in millimeters (mm).
    ewald_sphere_radius : float
        Radius of the Ewald sphere in reciprocal space (1/Å).
    screen_sample_distance_mm : float
        Distance from the sample to the screen in millimeters (mm).

    Returns
    -------
    NDArray
        Converted x coordinates in ky [1/Å].
    """

    kx: NDArray = (x_coords_mm / screen_sample_distance_mm) * ewald_sphere_radius

    return kx


def convert_gx_gy_to_sx_sy(
    gx: NDArray[np.float32],
    gy: NDArray[np.float32],
    ewald_radius: float,
    beta: float,
    screen_sample_distance: float,
    remove_outside: Optional[bool] = True,
    **kwargs,
) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
    """
    Convert reciprocal lattice coordinates (gx, gy) to RHEED screen coordinates (sx, sy)
    using the Ewald sphere construction.

    Parameters
    ----------
    gx : NDArray
        Array of reciprocal lattice x-coordinates.
    gy : NDArray
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
    sx : NDArray
        Array of x-coordinates on the RHEED screen corresponding to input gx, gy.
    sy : NDArray
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
    k0: np.float32 = np.float32(ewald_radius)
    # Ewald sphere radius square
    kk: np.float32 = np.float32(k0**2)

    # calculate the shift between the center of Ewald sphere and the center of reciprocal lattice
    delta_x: np.float32 = np.float32(k0 * np.cos(np.deg2rad(beta)))

    # shift the center of reciprocal lattice
    kx: NDArray[np.float32] = gx + delta_x
    ky: NDArray[np.float32] = gy

    # Check if the kx, ky points are inside Ewald sphere
    kxy2: NDArray[np.float32] = kx**2 + ky**2

    ind: NDArray[np.bool_] = kxy2 < kk

    # remove those outside or mark as nans
    if remove_outside:
        kx = kx[ind]
        ky = ky[ind]
    else:
        kx[~ind] = np.nan
        ky[~ind] = np.nan

    # calculate the radius r_k
    rk: NDArray[np.float32] = np.sqrt(k0**2 - kx**2)

    # calculate theta and phi (cos) values
    phi: NDArray[np.float32] = np.arccos(ky / rk)
    theta: NDArray[np.float32] = np.arcsin(rk / k0)

    # calculate the radius on the RHEED screen
    rho: NDArray[np.float32] = screen_sample_distance * np.tan(theta)

    # calculate the spot positions
    sx: NDArray[np.float32] = rho * np.cos(phi)
    sy: NDArray[np.float32] = -rho * np.sin(phi)

    return sx, sy
