import logging
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from numpy.typing import NDArray
from scipy import ndimage  # type: ignore
from tqdm.auto import tqdm

from .cache_utils import smart_cache

if TYPE_CHECKING:
    from .ewald import Ewald

logger = logging.getLogger(__name__)


def generate_spot_structure(
    spot_w: int,
    spot_h: int,
) -> NDArray[np.bool_]:
    """
    Generate a binary elliptical spot structure.
    """

    yy, xx = np.ogrid[:spot_h, :spot_w]

    center_x = spot_w / 2 - 0.5
    center_y = spot_h / 2 - 0.5

    radius_x = spot_w / 2
    radius_y = spot_h / 2

    return ((xx - center_x) ** 2 / radius_x**2) + (
        (yy - center_y) ** 2 / radius_y**2
    ) <= 1


def generate_mask(ewald: "Ewald") -> NDArray[np.bool_]:
    """
    Generate a mask for predicted spot positions in the image.

    Returns
    -------
    NDArray[np.bool_]
        Boolean mask of the same shape as the RHEED image.
    """

    image = ewald.image

    assert image is not None

    screen_scale = ewald.screen_scale
    screen_roi_width = ewald.screen_roi_width
    screen_roi_height = ewald.screen_roi_height

    # Physical origin of image
    origin_x = image.sx.values.min()
    origin_y = image.sy.values.min()

    # Map physical coords to pixel indices
    ppx = np.round((ewald.ew_sx - origin_x) * screen_scale).astype(np.uint32)
    ppy = np.round((ewald.ew_sy - origin_y) * screen_scale).astype(np.uint32)

    # Filter within bounds
    valid = (
        (ppx >= 0)
        & (ppx < image.shape[1])
        & (ppy >= 0)
        & (ppy < image.shape[0])
        & (ewald.ew_sx >= -screen_roi_width)
        & (ewald.ew_sx <= screen_roi_width)
        & (ewald.ew_sy >= -screen_roi_height)
        & (ewald.ew_sy <= 0)
    )

    ppx = ppx[valid]
    ppy = ppy[valid]

    # Build mask
    mask: NDArray[np.bool_] = np.zeros_like(image, dtype=np.bool_)
    mask[ppy, ppx] = True

    # Apply dilation
    mask = ndimage.binary_dilation(mask, structure=ewald.spot_structure).astype(
        np.bool_
    )

    return mask


def calculate_match(ewald: "Ewald", normalize: bool = True) -> np.uint32:
    """
    Calculate the match coefficient between predicted and observed spots.
    """

    assert ewald.image is not None

    image = ewald.image.data

    mask = generate_mask(ewald)

    # Calculate the match coefficient as the sum of masked image intensity
    match_coef = (mask * image).sum(dtype=np.uint32)

    # Optionally normalize
    if normalize:
        norm_coef = np.uint32(
            np.count_nonzero(mask) // np.count_nonzero(ewald.spot_structure)
        )
        match_coef = np.uint32(match_coef // norm_coef)

    return match_coef


@smart_cache
def match_alpha(
    ewald: "Ewald",
    alpha_vector: NDArray,
    normalize: bool = True,
    tqdm_disable: bool = True,
) -> xr.DataArray:
    """
    Calculate match coefficients over a range of azimuthal angles.
    """

    match_vector = np.zeros_like(alpha_vector, dtype=np.uint32)

    for i, alpha in enumerate(tqdm(alpha_vector, disable=tqdm_disable)):
        ewald.ewald_azimuthal_rotation = alpha
        ewald.calculate_ewald()
        match_vector[i] = calculate_match(ewald, normalize=normalize)

    return xr.DataArray(
        match_vector,
        dims=["alpha"],
        coords={"alpha": alpha_vector},
    )


@smart_cache
def match_scale(
    ewald: "Ewald",
    scale_vector: NDArray,
    normalize: bool = True,
    tqdm_disable: bool = True,
) -> xr.DataArray:
    """
    Calculate the match coefficient for a series of lattice scale values.
    """

    match_vector = np.zeros_like(scale_vector, dtype=np.uint32)

    ewald.ewald_roi = ewald._calc_ewald_roi(scale_vector.max())

    ewald._inverse_lattice = ewald._prepare_inverse_lattice()

    for i, scale in enumerate(tqdm(scale_vector, disable=tqdm_disable)):
        ewald.lattice_scale = scale
        ewald.calculate_ewald()
        match_vector[i] = calculate_match(ewald, normalize=normalize)

    return xr.DataArray(
        match_vector,
        dims=["scale"],
        coords={"scale": scale_vector},
    )


@smart_cache
def match_alpha_scale(
    ewald: "Ewald",
    alpha_vector: NDArray,
    scale_vector: NDArray,
    normalize: bool = True,
    flatten: bool = True,
    tqdm_disable: bool = True,
) -> xr.DataArray:
    """
    Calculate the match coefficient for a grid of alpha angles and scale values.
    """

    match_matrix: NDArray[np.uint32] = np.zeros(
        (len(alpha_vector), len(scale_vector)), dtype=np.uint32
    )

    ewald._ewald_roi = ewald._calc_ewald_roi(scale_vector.max())
    ewald._inverse_lattice = ewald._prepare_inverse_lattice()

    for i, scale in enumerate(
        tqdm(scale_vector, disable=tqdm_disable, desc="Matching scales")
    ):
        logger.info(
            "Matching scale %d/%d: lattice_scale=%.2f",
            i + 1,
            len(scale_vector),
            scale,
        )

        ewald.lattice_scale = scale
        ewald.calculate_ewald()

        match_alpha_vals = np.zeros_like(alpha_vector)

        for j, alpha in enumerate(alpha_vector):
            ewald.ewald_azimuthal_rotation = alpha
            ewald.calculate_ewald()
            match_alpha_vals[j] = calculate_match(
                ewald,
                normalize=normalize,
            )

        match_matrix[:, i] = match_alpha_vals

    if flatten:
        mean_profile = match_matrix.mean(axis=0)

        scale_vals = np.arange(match_matrix.shape[1])
        coeffs = np.polyfit(scale_vals, mean_profile, deg=2)
        background_fit = np.poly1d(coeffs)(scale_vals)

        match_matrix = match_matrix - background_fit

    match_matrix -= match_matrix.min()

    return xr.DataArray(
        match_matrix,
        dims=["alpha", "scale"],
        coords={"alpha": alpha_vector, "scale": scale_vector},
    )
