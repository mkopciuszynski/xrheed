import copy
from logging import warning
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from numpy.typing import NDArray
from scipy import ndimage
from tqdm.notebook import tqdm

from ..conversion.base import convert_gx_gy_to_sx_sy
from ..plotting.base import plot_image
from .lattice import Lattice, rotation_matrix


class Ewald:
    def __init__(self, lattice: Lattice, image: Optional[xr.DataArray] = None) -> None:
        """
        Initialize an Ewald object for RHEED spot calculation.

        Parameters
        ----------
        lattice : Lattice
            Lattice object representing the crystal structure.
        image : Optional[xr.DataArray], optional
            RHEED image data (default: None).
        """

        if image is None:
            warning("RHEED image is not provided, default values are loaded.")
            self.beam_energy = 18.6 * 1000
            self.screen_sample_distance = 309.2
            self.screen_scale = 9.5
            self.screen_size_w = 60
            self.screen_size_h = 80

            self._image_data_available = False
            self._beta = 1.0
            self._alpha = 0.0
        else:
            self.image = image.copy()
            self.image_data = image.data
            self.beam_energy = image.ri.beam_energy
            self.screen_sample_distance = image.ri.screen_sample_distance
            self.screen_scale = image.ri.screen_scale
            self.screen_size_w = image.ri.screen_roi_width
            self.screen_size_h = image.ri.screen_roi_height

            self._beta = image.ri.beta
            self._alpha = image.ri.alpha
            self._image_data_available = True

        self._lattice_scale: float = 1.0

        # Spot size in px used in matching calculations
        self._spot_w_px: int = 3
        self._spot_h_px: int = 4

        self.spot_structure = self._generate_spot_structure()

        self.shift_x: float = 0.0
        self.shift_y: float = 0.0
        self.fine_scalling: float = 1.0

        # Ewald sphere radius
        self.ewald_radius = np.sqrt(self.beam_energy) * 0.5123

        self._ewald_roi = self.ewald_radius * (
            self.screen_size_w / self.screen_sample_distance
        )
        # Lattice and its inverse
        self._lattice = copy.deepcopy(lattice)
        self._inverse_lattice = self._prepare_inverse_lattice()
        self._inverse_lattice = self._prepare_inverse_lattice()
        self.label = lattice.label

        # Mirror symmetry
        self.mirror = False

        self.calculate_ewald()

    def __repr__(self) -> str:
        """
        Returns a formatted string representation of the Ewald object,
        including its label, key parameters, and reciprocal lattice vectors.
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
        """

        new_ewald = Ewald(self._lattice, self.image)

        new_ewald.alpha = self.alpha
        new_ewald.lattice_scale = self.lattice_scale
        new_ewald.ewald_roi = self.ewald_roi
        new_ewald._spot_w_px = self._spot_w_px
        new_ewald._spot_h_px = self._spot_h_px

        return new_ewald

    @property
    def lattice_scale(self) -> float:
        """
        Get the lattice scaling factor.
        """
        return self._lattice_scale

    @lattice_scale.setter
    def lattice_scale(self, value: float):
        self._lattice_scale = value
        self.calculate_ewald()

    @property
    def alpha(self) -> float:
        """
        Get the azimuthal angle alpha (in degrees).
        """
        return self._alpha

    @alpha.setter
    def alpha(self, value: float):
        self._alpha = value
        self.calculate_ewald()

    @property
    def beta(self) -> float:
        """
        Get the incident angle beta (in degrees).
        """
        return self._beta

    @beta.setter
    def beta(self, value: float):
        self._beta = value
        self.calculate_ewald()

    @property
    def ewald_roi(self) -> float:
        """
        Get the Ewald ROI (region of interest) radius.
        """
        return self._ewald_roi

    @ewald_roi.setter
    def ewald_roi(self, value: float):
        self._ewald_roi = value
        self._inverse_lattice = self._prepare_inverse_lattice()


    def set_spot_size(self, width : int=5, height: int = 10):
        self._spot_w_px = width
        self._spot_h_px = height
        self.spot_structure = self._generate_spot_structure()


    def calculate_ewald(self, **kwargs) -> None:
        """
        Calculate the Ewald construction and spot positions for the current parameters.
        Updates self.sx and self.sy with spot coordinates.
        """

        ewald_radius = self.ewald_radius
        alpha = self._alpha
        beta = self._beta
        screen_sample_distance = self.screen_sample_distance

        inverse_lattice = self._inverse_lattice.copy()

        if alpha != 0:
            inverse_lattice = inverse_lattice @ rotation_matrix(alpha).T

        gx = inverse_lattice[:, 0] / self._lattice_scale
        gy = inverse_lattice[:, 1] / self._lattice_scale

        # calculate the spot positions
        sx, sy = convert_gx_gy_to_sx_sy(
            gx, gy, ewald_radius, beta, screen_sample_distance, remove_outside=True
        )

        ind = (
            (sx > -self.screen_size_w)
            & (sx < self.screen_size_w)
            & (sy < self.screen_size_h)
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
        auto_levels : float, optional
            Contrast enhancement for image plotting.
        show_center_lines : bool, optional
            If True, show center lines at x=0 and y=0.
        **kwargs
            Additional keyword arguments for scatter plot.

        Returns
        -------
        plt.Axes
            The axes with the plotted image and spots.
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

        if "marker" not in kwargs:
            kwargs["marker"] = "|"

        fine_scaling = self.fine_scalling

        ax.scatter(
            (self.ew_sx + self.shift_x) * fine_scaling,
            (self.ew_sy + self.shift_y) * fine_scaling,
            **kwargs,
        )

        return ax

    def calculate_match(self, normalize: bool = True) -> float:
        """
        Calculate the match coefficient between calculated spot positions and the RHEED data.

        This method converts physical coordinates (in mm) of expected spot positions into pixel indices
        using the known screen scale, applies a binary mask with elliptical spot dilation, and computes
        the match coefficient by summing the masked image intensity.

        Parameters
        ----------
        normalize : bool, optional
            If True, normalize the match coefficient (default: True).

        Returns
        -------
        float
            The match coefficient indicating how well the calculated spots match the image.
        """

        assert self._image_data_available, "Image data is not available"

        image = self.image
        image_data = self.image_data
        screen_scale = self.screen_scale

        # Determine the physical origin of the image (top-left corner in mm)
        origin_x = image.sx.values.min()
        origin_y = image.sy.values.max()

        # Convert physical coordinates (mm) to pixel indices (px)
        ppx = np.round((self.ew_sx - origin_x) * screen_scale).astype(np.uint32)
        ppy = np.round((origin_y - self.ew_sy) * screen_scale).astype(np.uint32)

        # Filter out-of-bounds pixel indices
        valid = (
            (ppx >= 0) & (ppx < image_data.shape[1]) &
            (ppy >= 0) & (ppy < image_data.shape[0])
        )

        # Apply mask at valid pixel positions
        ppx = ppx[valid]
        ppy = ppy[valid]
        mask = np.zeros(image_data.shape, dtype=bool)
        mask[ppy, ppx] = True

        # Apply elliptical spot structure to simulate spot spread
        spot_structure = self.spot_structure
        mask = ndimage.binary_dilation(mask, structure=spot_structure)

        # Calculate the match coefficient as the sum of masked image intensity
        match_coef = (mask * image_data).sum()

        # Optionally normalize (if needed in future)
        if normalize:
            match_coef = match_coef / len(ppx)

        return match_coef


    def match_alpha(self, alpha_vector: NDArray, normalize: bool = True) -> xr.DataArray:
        """
        Calculate the match coefficient for a series of phi (azimuthal) angles.

        Parameters
        ----------
        alpha_vector : NDArray
            Array of alpha (azimuthal) angles to test.
        normalize : bool, optional
            If True, normalize the match coefficient (default: True).

        Returns
        -------
        xr.DataArray
            Match coefficients for each phi angle.
        """
        """Here we can calculate the matching for a series of different phi angles"""

        match_vector = np.zeros_like(alpha_vector)

        for i, phi in enumerate(tqdm(alpha_vector)):
            self.alpha = phi
            match_vector[i] = self.calculate_match(normalize=normalize)

        return xr.DataArray(match_vector, dims=["phi"], coords={"phi": alpha_vector})

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

        match_vector = np.zeros_like(scale_vector)

        self.ewald_roi = (
            self.ewald_radius
            * (self.screen_size_w / self.screen_sample_distance)
            * scale_vector.max()
        )
        self._inverse_lattice = self._prepare_inverse_lattice()

        for i, scale in enumerate(tqdm(scale_vector)):
            self.lattice_scale = scale
            self.calculate_ewald()
            match_vector[i] = self.calculate_match(normalize=normalize)

        return xr.DataArray(
            match_vector, dims=["scale"], coords={"scale": scale_vector}
        )

    def match_phi_scale(
        self, phi_vector: NDArray, scale_vector: NDArray, normalize: bool = True
    ) -> xr.DataArray:
        """
        Calculate the match coefficient for a grid of phi angles and scale values.

        Parameters
        ----------
        phi_vector : NDArray
            Array of phi angles to test.
        scale_vector : NDArray
            Array of scale values to test.
        normalize : bool, optional
            If True, normalize the match coefficient (default: True).

        Returns
        -------
        xr.DataArray
            Match coefficients for each (phi, scale) pair.
        """
        """Here we can calculate the matching for a series of different phi angles and lattice constants"""

        match_matrix = np.zeros((len(phi_vector), len(scale_vector)))

        self.ewald_roi = (
            self.ewald_radius
            * (self.screen_size_w / self.screen_sample_distance)
            * scale_vector.max()
        )
        self._inverse_lattice = self._prepare_inverse_lattice()

        for i, scale in enumerate(tqdm(scale_vector, desc="Matching scales")):
            self.lattice_scale = scale
            self.calculate_ewald()

            match_phi = np.zeros_like(phi_vector)
            for j, alpha in enumerate(phi_vector):
                self.alpha = alpha
                match_phi[j] = self.calculate_match(normalize=normalize)

            match_matrix[:, i] = match_phi

        match_matrix_xr = xr.DataArray(
            match_matrix,
            dims=["phi", "scale"],
            coords={"phi": phi_vector, "scale": scale_vector},
        )
        return match_matrix_xr

    def _prepare_inverse_lattice(self) -> NDArray:
        """
        Prepare the inverse lattice points for the current ROI.

        Returns
        -------
        NDArray
            Array of inverse lattice points.
        """
        lattice = self._lattice
        space_size = self._ewald_roi
        inverse_lattice = Lattice.generate_lattice(
            lattice.b1, lattice.b2, space_size=space_size
        )
        return inverse_lattice

    def _generate_spot_structure(self) -> NDArray:
        """
        Generate a boolean mask for the spot structure (ellipse shape).

        Returns
        -------
        np.ndarray
            Boolean mask for spot shape.
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
