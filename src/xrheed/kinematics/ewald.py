import copy
import logging
import warnings
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from numpy.typing import NDArray

from ..constants import IMAGE_NDIMS, K_INV_ANGSTROM, STACK_NDIMS
from ..conversion.base import convert_gx_gy_to_sx_sy
from ..plotting.base import plot_image
from .ewald_matching import (
    calculate_match,
    generate_mask,
    generate_spot_structure,
    match_alpha,
    match_alpha_scale,
    match_scale,
)
from .lattice import Lattice, rotation_matrix

logger = logging.getLogger(__name__)


class Ewald:
    """
    Class for calculating and analyzing the Ewald sphere construction in RHEED.

    This class combines experimental RHEED image metadata with a reciprocal
    lattice model to predict diffraction spot positions on the screen.

    Azimuthal angle convention
    --------------------------
    Three azimuthal angles are distinguished:

    * image_azimuthal_angle
        Experimental azimuth of the RHEED image, read from image metadata.
        This value defines the reference frame and is treated as immutable.

    * ewald_azimuthal_rotation
        User-defined relative rotation of the Ewald construction with respect
        to the image azimuth. This does not modify the image metadata.

    * Ewald azimuthal angles (effective)
        The azimuthal angles actually used in the Ewald construction, derived as

            image_azimuthal_angle ± ewald_azimuthal_rotation

        When mirror symmetry is enabled, both ± rotations are used.

    This separation preserves the experimental reference frame while allowing
    controlled relative rotation of the theoretical Ewald construction.
    """

    SPOT_WIDTH_MM: float = 1.5
    SPOT_HEIGHT_MM: float = 5.0

    NO_IMAGE_DEFAULTS = {
        "beam_energy": 18_600.0,
        "screen_sample_distance": 309.2,
        "screen_scale": 9.5,
        "screen_roi_width": 60.0,
        "screen_roi_height": 80.0,
        "incident_angle": 1.0,
        "azimuthal_angle": 0.0,
    }

    REQUIRED_IMAGE_ATTRS = (
        "beam_energy",
        "screen_sample_distance",
        "screen_scale",
        "incident_angle",
        "azimuthal_angle",
    )

    def __init__(
        self,
        lattice: Lattice,
        image: Optional[xr.DataArray] = None,
        stack_index: int = 0,
    ) -> None:
        self._image_stack: Optional[xr.DataArray] = None
        self._stack_index: int = stack_index

        self.image: Optional[xr.DataArray] = None
        self._image_data_available: bool = False

        if image is None:
            self._initialize_without_image()
        else:
            self._initialize_from_image(image, stack_index)

        self._initialize_geometry()
        self._initialize_lattice(lattice)
        self._initialize_cache()

        self.calculate_ewald()

        logger.info(
            "Initialized Ewald: label=%s image_provided=%s beam_energy=%.1f screen_scale=%.2f",
            self.label,
            self._image_data_available,
            self.beam_energy,
            self.screen_scale,
        )

    def _initialize_without_image(self) -> None:
        logger.warning("RHEED image not provided, default parameters are loaded.")

        self.beam_energy = self.NO_IMAGE_DEFAULTS["beam_energy"]
        self.screen_sample_distance = self.NO_IMAGE_DEFAULTS["screen_sample_distance"]
        self.screen_scale = self.NO_IMAGE_DEFAULTS["screen_scale"]
        self.screen_roi_width = self.NO_IMAGE_DEFAULTS["screen_roi_width"]
        self.screen_roi_height = self.NO_IMAGE_DEFAULTS["screen_roi_height"]

        self._incident_angle = self.NO_IMAGE_DEFAULTS["incident_angle"]
        self._image_azimuthal_angle = self.NO_IMAGE_DEFAULTS["azimuthal_angle"]

    def _initialize_from_image(
        self,
        image: xr.DataArray,
        stack_index: int,
    ) -> None:
        if image.ndim == IMAGE_NDIMS:
            self.image = image.copy()

        elif image.ndim == STACK_NDIMS:
            self._image_stack = image.copy()
            self.image = self._image_stack[stack_index]

        else:
            raise ValueError(
                f"Invalid RHEED image input.\n"
                f"Expected DataArray with ndim={IMAGE_NDIMS} or {STACK_NDIMS}, "
                f"but got ndim={getattr(image, 'ndim', None)}."
            )

        assert self.image is not None

        missing = [
            attr
            for attr in self.REQUIRED_IMAGE_ATTRS
            if getattr(self.image.ri, attr, None) is None
        ]

        if missing:
            raise ValueError(
                "Invalid RHEED image: missing required RI metadata attributes.\n"
                f"Missing attributes: {', '.join(missing)}\n"
            )

        self.beam_energy = float(self.image.ri.beam_energy)
        self.screen_sample_distance = float(self.image.ri.screen_sample_distance)
        self.screen_scale = float(self.image.ri.screen_scale)

        self.screen_roi_width = float(self.image.ri.screen_roi_width)
        self.screen_roi_height = float(self.image.ri.screen_roi_height)

        self._incident_angle = self.image.ri.incident_angle
        self._image_azimuthal_angle = self.image.ri.azimuthal_angle

        self._image_data_available = True

    def _initialize_geometry(self) -> None:
        self._lattice_scale: float = 1.0

        self._spot_w_px = int(self.SPOT_WIDTH_MM * self.screen_scale)
        self._spot_h_px = int(self.SPOT_HEIGHT_MM * self.screen_scale)

        self.spot_structure = generate_spot_structure(
            self._spot_w_px,
            self._spot_h_px,
        )

        self.shift_x = 0.0
        self.shift_y = 0.0
        self.fine_scaling = 1.0

        self.ewald_radius = np.sqrt(self.beam_energy) * K_INV_ANGSTROM

        self._ewald_roi = self._calc_ewald_roi()
        self._ewald_azimuthal_rotation = 0.0

        self.mirror_symmetry = False

    def _initialize_lattice(self, lattice: Lattice) -> None:
        self._lattice = copy.deepcopy(lattice)

        self._inverse_lattice = self._prepare_inverse_lattice()

        self.label = lattice.label

        self.ew_sx: NDArray[np.float32]
        self.ew_sy: NDArray[np.float32]

    def _initialize_cache(self) -> None:
        self.use_cache = True
        self.cache_dir = "cache"
        self.cache_key = None

    def __repr__(self) -> str:
        return (
            f"Ewald Class Object: {self.label}\n"
            f"  Ewald Radius            : {self.ewald_radius:.2f} 1/Å\n"
            f"  Image azimuthal angle:  : {self.image_azimuthal_angle:.2f}°\n"
            f"  Incident angle          : {self.incident_angle:.2f}°\n"
            f"  Real Lattice Scale      : {self.lattice_scale:.2f}\n"
            f"  Screen Scale            : {self.screen_scale:.2f} px/mm\n"
            f"  Sample-Screen Distance  : {self.screen_sample_distance:.1f} mm\n"
            f"  Screen Shift X          : {self.shift_x:.2f} mm\n"
            f"  Screen Shift Y          : {self.shift_y:.2f} mm\n"
            f"  Reciprocal Vector b1    : [{self._lattice.b1[0]:.2f}, {self._lattice.b1[1]:.2f}] 1/Å\n"
            f"  Reciprocal Vector b2    : [{self._lattice.b2[0]:.2f}, {self._lattice.b2[1]:.2f}] 1/Å\n"
        )

    def __copy__(self) -> "Ewald":
        """
        Create a shallow copy of the Ewald object.

        Returns
        -------
        Ewald
            A new instance with the same parameters.
        """

        new_ewald = Ewald(self._lattice, self.image)
        new_ewald.ewald_azimuthal_rotation = self.ewald_azimuthal_rotation
        new_ewald.incident_angle = self.incident_angle
        new_ewald.lattice_scale = self.lattice_scale
        new_ewald.ewald_roi = self.ewald_roi
        new_ewald._spot_w_px = self._spot_w_px
        new_ewald._spot_h_px = self._spot_h_px
        return new_ewald

    @property
    def stack_index(self) -> int:
        """int: Index of the current image in a stack."""
        return self._stack_index

    @stack_index.setter
    def stack_index(self, value: int):
        if self._image_stack is None:
            raise ValueError("Stack index can only be set for 3D image stacks.")
        if not (value < self._image_stack.shape[0]):
            raise ValueError("Stack index out of bounds.")
        self._stack_index = value
        self.image = self._image_stack[self._stack_index]
        self.calculate_ewald()

    @property
    def lattice_scale(self) -> float:
        return self._lattice_scale

    @lattice_scale.setter
    def lattice_scale(self, value: float):
        if abs(self._lattice_scale - value) > 0.5:
            self.ewald_roi = self._calc_ewald_roi(value)
            logging.info("New Ewald roi: %.2f", self.ewald_roi)
        self._lattice_scale = value
        self.calculate_ewald()

    @property
    def image_azimuthal_angle(self) -> float:
        if isinstance(self._image_azimuthal_angle, np.ndarray):
            return self._image_azimuthal_angle[self._stack_index]
        return self._image_azimuthal_angle

    @image_azimuthal_angle.setter
    def image_azimuthal_angle(self, value: float):
        """
        Deprecated setter.

        Historically this modified the azimuthal angle of the image.
        Now the image azimuthal angle is treated as not mutable physical property.
        """
        if isinstance(self._image_azimuthal_angle, np.ndarray):
            raise TypeError(
                "image_azimuthal_angle is derived from stacked image metadata "
                "and cannot be set directly."
            )

        warnings.warn(
            "Setting image_azimuthal_angle is deprecated and will be removed "
            "in a future version. Use ewald_azimuthal_rotation instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        # Interpret old behavior as a relative rotation
        self.ewald_azimuthal_rotation = float(value) - self._image_azimuthal_angle

    @property
    def ewald_azimuthal_rotation(self) -> float:
        return self._ewald_azimuthal_rotation

    @ewald_azimuthal_rotation.setter
    def ewald_azimuthal_rotation(self, value: float):
        self._ewald_azimuthal_rotation = float(value)
        self.calculate_ewald()

    @property
    def ewald_azimuthal_angle(self) -> float:
        return self.image_azimuthal_angle + self._ewald_azimuthal_rotation

    @property
    def incident_angle(self) -> float:
        if isinstance(self._incident_angle, np.ndarray):
            return self._incident_angle[self._stack_index]
        return self._incident_angle

    @incident_angle.setter
    def incident_angle(self, value: float):
        if isinstance(self._incident_angle, np.ndarray):
            raise ValueError("Cannot set incident individually for stack images.")
        self._incident_angle = value
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
        self.spot_structure = generate_spot_structure(
            self._spot_w_px,
            self._spot_h_px,
        )

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

        incident_angle: float = self.incident_angle
        screen_sample_distance: float = self.screen_sample_distance

        # Arrays for reciprocal kx and ky coordinates
        gx: NDArray[np.float32]
        gy: NDArray[np.float32]
        # Arrays for the calculated spot positions
        sx: NDArray[np.float32]
        sy: NDArray[np.float32]

        inverse_lattice: NDArray[np.float32] = self._inverse_lattice.copy()

        # Fine scaling
        inverse_lattice /= self._lattice_scale

        # Effective azimuthal angles used for the Ewald construction
        image_azimuthal_angle: float = self.image_azimuthal_angle
        ewald_azimuthal_rotation: float = self.ewald_azimuthal_rotation

        # Determine which azimuthal angles are used to rotate the reciprocal lattice
        if np.isclose(ewald_azimuthal_rotation, 0.0):
            # No relative rotation: use the image azimuthal angle only
            ewald_azimuthal_angles = [image_azimuthal_angle]
        else:
            # Relative rotation with respect to the image azimuthal angle
            ewald_azimuthal_angles = [image_azimuthal_angle + ewald_azimuthal_rotation]
            if self.mirror_symmetry:
                ewald_azimuthal_angles.insert(
                    0, image_azimuthal_angle - ewald_azimuthal_rotation
                )

        # Apply azimuthal rotations to the inverse lattice and stack results
        rotated_inverse_lattices = [
            inverse_lattice @ rotation_matrix(azimuthal_angle)
            for azimuthal_angle in ewald_azimuthal_angles
        ]

        stacked = np.vstack(rotated_inverse_lattices)
        gx, gy = stacked.T[:2]

        sx, sy = convert_gx_gy_to_sx_sy(
            gx,
            gy,
            ewald_radius,
            incident_angle,
            screen_sample_distance,
            remove_outside=True,
        )

        ind: NDArray[np.bool_] = (
            (sx > -self.screen_roi_width)
            & (sx < self.screen_roi_width)
            & (sy < self.screen_roi_height)
        )

        sx = sx[ind]
        sy = sy[ind]

        self.ew_sx = sx
        self.ew_sy = sy
        logger.debug(
            "calculate_ewald: generated %d spots (mirror=%s) ewald_roi=%.3f",
            sx.size,
            self.mirror_symmetry,
            getattr(self, "_ewald_roi", float("nan")),
        )

    def plot(
        self,
        ax: Optional[Axes] = None,
        show_image: bool = True,
        show_roi: bool = False,
        auto_levels: float = 0.0,
        show_center_lines: bool = False,
        **kwargs,
    ) -> Axes:
        """
        Plot the calculated spot positions and optionally the RHEED image.

        Parameters
        ----------
        ax : Optional[Axes], optional
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
        matplotlib.axes.Axes
            The axes with the plotted data.
        """

        if ax is None:
            fig, ax = plt.subplots()
        logger.debug(
            "plot: show_image=%s show_roi=%s show_center_lines=%s",
            show_image,
            show_roi,
            show_center_lines,
        )

        if show_image:
            if self.image is None:
                raise ValueError("There was no RHEED image attached.")

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

        fine_scaling: float = self.fine_scaling

        ax.scatter(
            (self.ew_sx + self.shift_x) * fine_scaling,
            (self.ew_sy + self.shift_y) * fine_scaling,
            **kwargs,
        )
        logger.info("Plotted %d ewald spots on axes.", getattr(self.ew_sx, "size", 0))

        return ax

    def plot_spots(self, ax=None, show_image: bool = False, **kwargs):
        """
        Plot the spot mask used for spot matching on a RHEED image.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Matplotlib Axes to plot on. If None, a new figure and axes are created.
        show_image : bool, default=False
            If True, overlay the spot mask on the original RHEED image.
            If False, only the mask is displayed.
        **kwargs
            Additional keyword arguments passed to `ax.imshow()`, e.g., `cmap`, `alpha`.

        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the plotted mask (and optionally the image).

        Raises
        ------
        ValueError
            If `show_image=True` but no RHEED image is attached (`self.image is None`).

        Notes
        -----
        - The mask is automatically generated by `self._generate_mask()`.
        - The image coordinates (`sx`, `sy`) are used to set the extent of the plot.
        - The default colormap for the mask is grayscale.
        """

        if ax is None:
            _, ax = plt.subplots()

        mask = self._generate_mask()

        if self.image is None:
            raise ValueError("There was no RHEED image attached.")

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
            logger.debug(
                "plot_spots: show_image=%s mask_shape=%s",
                show_image,
                getattr(mask, "shape", None),
            )
            logger.info("Displayed spot mask on axes.")
        else:
            ax.imshow(
                mask,
                origin="lower",
                extent=(image.sx.min(), image.sx.max(), image.sy.min(), image.sy.max()),
                aspect="equal",
                cmap="gray",
            )

        return ax

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

    def _calc_ewald_roi(self, scale_max: float = 1.0) -> float:
        return float(
            self.ewald_radius
            * (self.screen_roi_width / self.screen_sample_distance)
            * scale_max
        )

    def _generate_mask(self):
        return generate_mask(self)

    def calculate_match(self, normalize: bool = True):
        return calculate_match(self, normalize=normalize)

    def match_alpha(
        self,
        alpha_vector,
        normalize: bool = True,
        tqdm_disable: bool = True,
    ):
        return match_alpha(
            self,
            alpha_vector=alpha_vector,
            normalize=normalize,
            tqdm_disable=tqdm_disable,
        )

    def match_scale(
        self,
        scale_vector,
        normalize: bool = True,
        tqdm_disable: bool = True,
    ):
        return match_scale(
            self,
            scale_vector=scale_vector,
            normalize=normalize,
            tqdm_disable=tqdm_disable,
        )

    def match_alpha_scale(
        self,
        alpha_vector,
        scale_vector,
        normalize: bool = True,
        flatten: bool = True,
        tqdm_disable: bool = True,
    ):
        return match_alpha_scale(
            self,
            alpha_vector=alpha_vector,
            scale_vector=scale_vector,
            normalize=normalize,
            flatten=flatten,
            tqdm_disable=tqdm_disable,
        )
