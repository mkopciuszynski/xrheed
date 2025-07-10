import numpy as np

import xarray as xr
from scipy import constants


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
    