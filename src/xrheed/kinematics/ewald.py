import copy
from logging import warning
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from numpy.typing import NDArray
from scipy import ndimage  # type: ignore
from tqdm.notebook import tqdm

from ..conversion.base import convert_gx_gy_to_sx_sy
from ..plotting.base import plot_image
from .cache_utils import smart_cache
from .lattice import Lattice, rotation_matrix


class Ewald:
    """
    Class for calculating and analyzing the Ewald sphere construction in RHEED.

    This class provides functionality for generating reciprocal lattice spots,
    simulating their appearance on a RHEED screen, and matching them with
    experimental data.
    """

    # Class constants
    SPOT_WIDTH_MM: float = 1.5  #: Default spot width in mm
    SPOT_HEIGHT_MM: float = 5.0  #: Default spot height in mm

    def __init__(self, lattice: Lattice, image: Optional[xr.DataArray] = None) -> None:
        """
        Initialize an Ewald object for RHEED spot calculation.

        Parameters
        ----------
        lattice : Lattice
            Lattice object representing the crystal structure.
        image : Optional[xr.DataArray], optional
            RHEED image data. If None, default values are used.
        """

        self.image: xr.DataArray
        self.beam_energy: float
        self.screen_sample_distance: float
        self.screen_scale: float
        self.screen_roi_width: float
        self.screen_roi_height: float
        self._beta: float
        self._alpha: float
        self._image_data_available: bool

        if image is None:
            warning("RHEED image is not provided, default values are loaded.")
            self.beam_energy = 18.6 * 1000
            self.screen_sample_distance = 309.2
            self.screen_scale = 9.5
            self.screen_roi_width = 60
            self.screen_roi_height = 80

            self._image_data_available = False
            self._beta = 1.0
            self._alpha = 0.0
        else:
            self.image = image.copy()
            self.beam_energy = image.ri.beam_energy
            self.screen_sample_distance = image.ri.screen_sample_distance
            self.screen_scale = image.ri.screen_scale
            self.screen_roi_width = image.ri.screen_roi_width
            self.screen_roi_height = image.ri.screen_roi_height

            self._beta = image.ri.beta
            self._alpha = image.ri.alpha
            self._image_data_available = True

        self._lattice_scale: float = 1.0

        # If set to true, the decorated functions will first try to load cached data
        self.use_cache: bool = True

        # Default Spot size in px used in matching calculations
        self._spot_w_px: int = int(self.SPOT_WIDTH_MM * self.screen_scale)
        self._spot_h_px: int = int(self.SPOT_HEIGHT_MM * self.screen_scale)

        self.spot_structure = self._generate_spot_structure()

        self.shift_x: float = 0.0
        self.shift_y: float = 0.0
        self.fine_scalling: float = 1.0

        # Ewald sphere radius
        self.ewald_radius: float = np.sqrt(self.beam_energy) * 0.5123

        self._ewald_roi: float = self.ewald_radius * (
            self.screen_roi_width / self.screen_sample_distance
        )
        # Lattice and its inverse
        self._lattice: Lattice = copy.deepcopy(lattice)
        self._inverse_lattice: NDArray[np.float32] = self._prepare_inverse_lattice()
        self.label: Optional[str] = lattice.label

        # Mirror symmetry
        self.mirror: bool = False

        self.ew_sx: NDArray[np.float32]
        self.ew_sy: NDArray[np.float32]

        self.calculate_ewald()

    def __repr__(self) -> str:
        """
        Return a formatted string representation of the Ewald object.

        Returns
        -------
        str
            Summary of the Ewald parameters and lattice vectors.
        """
        details = (
            f"Ewald Class Object: {self.label}\n"
            f"  Ewald Radius           : {self.ewald_radius:.2f} 1/Å\n"
            f"  Alpha (rotation angle) : {self.alpha:.2f}°\n"
            f"  Beta (incident angle)  : {self.beta:.2f}°\n"
            f"  Lattice Scale          : {self.lattice_scale:.2f}\n"
            f"  Screen Scale           : {self.screen_scale:.2f} px/mm\n"
            f"  Sample-Screen Distance : {self.screen_sample_distance:.1f} mm\n"
            f"  Screen Shift X         : {self.shift_x:.2f} mm\n"
            f"  Screen Shift Y         : {self.shift_y:.2f} mm\n"
            f"  Reciprocal Vector b1   : [{self._lattice.b1[0]:.2f}, {self._lattice.b1[1]:.2f}] 1/Å\n"
            f"  Reciprocal Vector b2   : [{self._lattice.b2[0]:.2f}, {self._lattice.b2[1]:.2f}] 1/Å\n"
        )
        return details

    def __copy__(self) -> "Ewald":
        """
        Create a shallow copy of the Ewald object.

        Returns
        -------
        Ewald
            A new instance with the same parameters.
        """

        new_ewald = Ewald(self._lattice, self.image)

        new_ewald.alpha = self.alpha
        new_ewald.beta = self.beta
        new_ewald.lattice_scale = self.lattice_scale
        new_ewald.ewald_roi = self.ewald_roi
        new_ewald._spot_w_px = self._spot_w_px
        new_ewald._spot_h_px = self._spot_h_px

        return new_ewald

    @property
    def lattice_scale(self) -> float:
        """float: Scaling factor applied to the lattice vectors."""
        return self._lattice_scale

    @lattice_scale.setter
    def lattice_scale(self, value: float):
        self._lattice_scale = value
        self.calculate_ewald()

    @property
    def alpha(self) -> float:
        """float: Incident beam angle (degrees)."""
        return self._alpha

    @alpha.setter
    def alpha(self, value: float):
        self._alpha = value
        self.calculate_ewald()

    @property
    def beta(self) -> float:
        """float: Region of interest radius on the Ewald sphere."""
        return self._beta

    @beta.setter
    def beta(self, value: float):
        self._beta = value
        self.calculate_ewald()

    @property
    def ewald_roi(self) -> float:
        return self._ewald_roi

    @ewald_roi.setter
    def ewald_roi(self, value: float):
        self._ewald_roi = value
        self._inverse_lattice = self._prepare_inverse_lattice()

    def set_spot_size(self, width: float, height: float):
        """
        Set the spot size used for mask generation.

        Parameters
        ----------
        width : float
            Spot width in mm.
        height : float
            Spot height in mm.
        """
        self._spot_w_px = int(width * self.screen_scale)
        self._spot_h_px = int(height * self.screen_scale)
        self.spot_structure = self._generate_spot_structure()

    def calculate_ewald(self, **kwargs) -> None:
        """
        Calculate the Ewald construction and update spot positions.

        Updates
        -------
        self.ew_sx : NDArray
            Spot x-coordinates (mm).
        self.ew_sy : NDArray
            Spot y-coordinates (mm).
        """

        ewald_radius: float = self.ewald_radius
        alpha: float = self._alpha
        beta: float = self._beta
        screen_sample_distance: float = self.screen_sample_distance

        inverse_lattice: NDArray[np.float32] = self._inverse_lattice.copy()

        if alpha != 0:
            inverse_lattice = inverse_lattice @ rotation_matrix(alpha).T

        gx: NDArray[np.float32] = inverse_lattice[:, 0] / self._lattice_scale
        gy: NDArray[np.float32] = inverse_lattice[:, 1] / self._lattice_scale

        # calculate the spot positions
        sx: NDArray[np.float32]
        sy: NDArray[np.float32]

        sx, sy = convert_gx_gy_to_sx_sy(
            gx, gy, ewald_radius, beta, screen_sample_distance, remove_outside=True
        )

        ind: NDArray[np.bool_] = (
            (sx > -self.screen_roi_width)
            & (sx < self.screen_roi_width)
            & (sy < self.screen_roi_height)
        )

        sx = sx[ind]
        sy = sy[ind]

        if self.mirror:
            if alpha % 60 != 0:
                sx = np.hstack([sx, -sx])
                sy = np.hstack([sy, sy])

        self.ew_sx = sx
        self.ew_sy = sy

    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        show_image: bool = True,
        show_roi: bool = False,
        auto_levels: float = 0.0,
        show_center_lines: bool = False,
        **kwargs,
    ) -> plt.Axes:
        """
        Plot the calculated spot positions and optionally the RHEED image.

        Parameters
        ----------
        ax : Optional[plt.Axes], optional
            Matplotlib axes to plot on. If None, a new figure is created.
        show_image : bool, optional
            If True, plot the RHEED image (default: True).
        show_roi : bool, optional
            If True, overlay the ROI boundary (default: False).
        auto_levels : float, optional
            Contrast enhancement factor for image plotting.
        show_center_lines : bool, optional
            If True, plot center cross lines (default: False).
        **kwargs
            Additional keyword arguments for the scatter plot.

        Returns
        -------
        plt.Axes
            The axes with the plotted data.
        """

        if ax is None:
            fig, ax = plt.subplots()

        if show_image:
            imshow_keys = {"cmap", "vmin", "vmax"}
            plot_image_kwargs = {
                k: kwargs.pop(k) for k in list(kwargs.keys()) if k in imshow_keys
            }
            rheed_image = self.image
            plot_image(
                rheed_image=rheed_image,
                ax=ax,
                auto_levels=auto_levels,
                show_center_lines=show_center_lines,
                **plot_image_kwargs,
            )

            if show_roi:
                ax.set_xlim(rheed_image.sx.min(), rheed_image.sx.max())
                ax.set_ylim(rheed_image.sy.min(), rheed_image.sy.max())

                # Draw vertical lines at x = ±x_width/2
                ax.axvline(
                    x=-self.screen_roi_width, color="red", linestyle="--", linewidth=1
                )
                ax.axvline(
                    x=self.screen_roi_width, color="red", linestyle="--", linewidth=1
                )

                # Draw horizontal lines at y = ±y_width/2
                ax.axhline(
                    y=-self.screen_roi_height, color="red", linestyle="--", linewidth=1
                )
                ax.axhline(y=0.0, color="red", linestyle="--", linewidth=1)

        if "marker" not in kwargs:
            kwargs["marker"] = "|"

        fine_scaling: float = self.fine_scalling

        ax.scatter(
            (self.ew_sx + self.shift_x) * fine_scaling,
            (self.ew_sy + self.shift_y) * fine_scaling,
            **kwargs,
        )

        return ax

    def plot_spots(self, ax=None, show_image: bool = False, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()

        mask = self._generate_mask()
        image = self.image

        if "cmap" not in kwargs:
            kwargs["cmap"] = "gray"

        if show_image:
            ax.imshow(
                mask * image.data,
                origin="lower",
                extent=(image.sx.min(), image.sx.max(), image.sy.min(), image.sy.max()),
                aspect="equal",
                **kwargs,
            )
        else:
            ax.imshow(
                mask,
                origin="lower",
                extent=(image.sx.min(), image.sx.max(), image.sy.min(), image.sy.max()),
                aspect="equal",
                cmap="gray",
            )

        return ax

    def calculate_match(self, normalize: bool = True) -> np.uint32:
        """
        Calculate the match coefficient between predicted and observed spots.

        Parameters
        ----------
        normalize : bool
            If True, normalize the coefficient by the number of spots.

        Returns
        -------
        np.uint32
            Match coefficient.
        """

        assert self._image_data_available, "Image data is not available"

        image = self.image.data

        mask = self._generate_mask()

        # Calculate the match coefficient as the sum of masked image intensity
        match_coef = (mask * image).sum(dtype=np.uint32)

        # Optionally normalize
        if normalize:
            norm_coef = np.uint32(
                np.count_nonzero(mask) // np.count_nonzero(self.spot_structure)
            )
            match_coef = np.uint32(match_coef // norm_coef)

        return match_coef

    @smart_cache
    def match_alpha(
        self, alpha_vector: NDArray, normalize: bool = True
    ) -> xr.DataArray:
        """
        Calculate match coefficients over a range of azimuthal angles.

        Parameters
        ----------
        alpha_vector : NDArray[np.float64]
            Array of alpha (azimuthal) angles in degrees.
        normalize : bool, optional
            If True, normalize the coefficients (default: True).

        Returns
        -------
        xr.DataArray
            Match coefficients with alpha as coordinate.
        """

        match_vector = np.zeros_like(alpha_vector, dtype=np.uint32)

        for i, alpha in enumerate(tqdm(alpha_vector)):
            self.alpha = alpha
            match_vector[i] = self.calculate_match(normalize=normalize)

        return xr.DataArray(
            match_vector, dims=["alpha"], coords={"alpha": alpha_vector}
        )

    @smart_cache
    def match_scale(
        self, scale_vector: NDArray, normalize: bool = True
    ) -> xr.DataArray:
        """
        Calculate the match coefficient for a series of lattice scale values.

        Parameters
        ----------
        scale_vector : NDArray
            Array of scale values to test.
        normalize : bool, optional
            If True, normalize the match coefficient (default: True).

        Returns
        -------
        xr.DataArray
            Match coefficients for each scale value.
        """
        """Here we can calculate the matching for a series of different lattice constants"""

        match_vector = np.zeros_like(scale_vector, dtype=np.uint32)

        self.ewald_roi = (
            self.ewald_radius
            * (self.screen_roi_width / self.screen_sample_distance)
            * scale_vector.max()
        )
        self._inverse_lattice = self._prepare_inverse_lattice()

        for i, scale in enumerate(tqdm(scale_vector)):
            self.lattice_scale = scale
            self.calculate_ewald()
            match_vector[i] = self.calculate_match(normalize=normalize)

        return xr.DataArray(
            match_vector,
            dims=["scale"],
            coords={"scale": scale_vector},
        )

    @smart_cache
    def match_alpha_scale(
        self,
        alpha_vector: NDArray,
        scale_vector: NDArray,
        normalize: bool = True,
        flatten: bool = True,
    ) -> xr.DataArray:
        """
        Calculate the match coefficient for a grid of alpha angles and scale values.

        Parameters
        ----------
        alpha_vector : NDArray
            Array of azimuthal angles to test.
        scale_vector : NDArray
            Array of scale values to test.
        normalize : bool, optional
            If True, normalize the match coefficient (default: True).
        flatten : bool, optional
            If True, the result map is flatten by subtracting quadratic
            background fitted along scale direction
              (default: True).

        Returns
        -------
        xr.DataArray
            Match coefficients for each (alpha, scale) pair.
        """

        match_matrix: NDArray[np.uint32] = np.zeros(
            (len(alpha_vector), len(scale_vector)), dtype=np.uint32
        )

        self._ewald_roi = (
            self.ewald_radius
            * (self.screen_roi_width / self.screen_sample_distance)
            * scale_vector.max()
        )
        self._inverse_lattice = self._prepare_inverse_lattice()

        for i, scale in enumerate(tqdm(scale_vector, desc="Matching scales")):
            self.lattice_scale = scale
            self.calculate_ewald()

            match_phi = np.zeros_like(alpha_vector)
            for j, alpha in enumerate(alpha_vector):
                self.alpha = alpha
                match_phi[j] = self.calculate_match(normalize=normalize)

            match_matrix[:, i] = match_phi

        if flatten:
            # Step 1: Mean over alpha
            mean_profile = match_matrix.mean(axis=0)

            # Step 2: Fit quadratic
            scale_vals = np.arange(match_matrix.shape[1])  # or use actual scale values
            coeffs = np.polyfit(scale_vals, mean_profile, deg=2)
            background_fit = np.poly1d(coeffs)(scale_vals)

            # Step 3: Subtract background
            match_matrix = match_matrix - background_fit

        match_matrix -= match_matrix.min()

        match_matrix_xr = xr.DataArray(
            match_matrix,
            dims=["alpha", "scale"],
            coords={"alpha": alpha_vector, "scale": scale_vector},
        )
        return match_matrix_xr

    def _prepare_inverse_lattice(self) -> NDArray[np.float32]:
        """
        Generate reciprocal lattice points for the current ROI.

        Returns
        -------
        NDArray[np.float32]
            Inverse lattice points as an array of shape (N, 2).
        """
        lattice = self._lattice
        space_size = self._ewald_roi
        inverse_lattice = Lattice.generate_lattice(
            lattice.b1, lattice.b2, space_size=space_size
        )
        return inverse_lattice

    def _generate_spot_structure(self) -> NDArray[np.bool_]:
        """
        Generate a binary elliptical spot structure.

        Returns
        -------
        NDArray[np.bool_]
            Boolean array mask for the spot shape.
        """

        # Define dimensions
        spot_w = self._spot_w_px
        spot_h = self._spot_h_px

        spot_structure = np.zeros((spot_h, spot_w), dtype=bool)

        # Center of the ellipse
        center_x = spot_w / 2 - 0.5
        center_y = spot_h / 2 - 0.5

        # Radii of the ellipse
        radius_x = spot_w / 2
        radius_y = spot_h / 2

        for i in range(spot_h):
            for j in range(spot_w):
                # Check if the point (j, i) is inside the ellipse
                if ((j - center_x) ** 2 / radius_x**2) + (
                    (i - center_y) ** 2 / radius_y**2
                ) <= 1:
                    spot_structure[i, j] = True

        return spot_structure

    # TODO prepare calculate match for a list of phi angles next we will do the same for a list of
    # lattice stalling

    def _generate_mask(self) -> NDArray[np.bool_]:
        """
        Generate a mask for predicted spot positions in the image.

        Returns
        -------
        NDArray[np.bool_]
            Boolean mask of the same shape as the RHEED image.
        """
        image = self.image
        screen_scale = self.screen_scale
        screen_roi_width = self.screen_roi_width
        screen_roi_height = self.screen_roi_height

        # Physical origin of image
        origin_x = image.sx.values.min()
        origin_y = image.sy.values.min()  # bottom edge in mm

        # Map physical coords to pixel indices
        ppx = np.round((self.ew_sx - origin_x) * screen_scale).astype(np.uint32)
        ppy = np.round((self.ew_sy - origin_y) * screen_scale).astype(np.uint32)

        # Filter within bounds
        valid = (
            (ppx >= 0)
            & (ppx < image.shape[1])
            & (ppy >= 0)
            & (ppy < image.shape[0])
            & (self.ew_sx >= -screen_roi_width)
            & (self.ew_sx <= screen_roi_width)
            & (self.ew_sy >= -screen_roi_height)
            & (self.ew_sy <= 0)
        )

        ppx = ppx[valid]
        ppy = ppy[valid]

        # Build mask
        mask: NDArray[np.bool_] = np.zeros_like(image, dtype=np.bool_)
        mask[ppy, ppx] = True

        # Apply dilation
        mask = ndimage.binary_dilation(mask, structure=self.spot_structure).astype(
            np.bool_
        )

        return mask
