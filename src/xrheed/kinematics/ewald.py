from logging import warning
import numpy as np
from scipy import ndimage
import xarray as xr
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

import copy
from typing import Optional
from numpy.typing import NDArray

from .lattice import Lattice
from .lattice import rotation_matrix
from ..plotting.base import plot_image


class Ewald:

    def __init__(self, lattice: Lattice, image: Optional[xr.DataArray] = None):

        if image is None:
            warning("RHEED image is not provided, default values are loaded.")
            self.beam_energy = 18.6 * 1000
            self.screen_sample_distance = 309.2
            self.screen_scale = 9.5
            self.screen_size_w = 60
            self.screen_size_h = 80

            self._image_data_available = False
            self._beta = 1.0
        else:
            self.image = image.copy()
            self.image_data = image.data
            self.beam_energy = image.ri.beam_energy
            self.screen_sample_distance = image.ri.screen_sample_distance
            self.screen_scale = image.ri.screen_scale
            self.screen_size_w = image.ri.screen_roi_width
            self.screen_size_h = image.ri.screen_roi_height

            self._beta = image.ri.beta
            self._image_data_available = True

        self._alpha = 0.0
        self._lattice_scale = 1.0

        self._spot_w: int = 3
        self._spot_h: int = 4

        self._spot_structure = self._generate_spot_structure()

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

        # Mirror symmetry
        self.mirror = False

        self.calculate_ewald()

    def __repr__(self) -> str:
        details = (
            f"  Ewald Radius: {self.ewald_radius:.2f} 1/A,\n"
            f"  alpha: {self.alpha:.2f} deg,\n"
            f"  beta: {self.beta:.2f} deg,\n"
            f"  lattice_scale: {self.lattice_scale:.2f},\n"
            f"  screen_scale: {self.screen_scale:.2f} px/mm,\n"
            f"  screen_sample_distance: {self.screen_sample_distance:.1f} mm,\n"
            f"  shift_x: {self.shift_x:.2f} mm,\n"
            f"  shift_y: {self.shift_y:.2f} mm,\n"
            f"  b1 = [{self._lattice.b1[0]:.2f}, {self._lattice.b1[1]:.2f}] 1/A,\n"
            f"  b2 = [{self._lattice.b2[0]:.2f}, {self._lattice.b2[1]:.2f}] 1/A,\n"
        )
        return details

    def __copy__(self):

        new_ewald = Ewald(self._lattice, self.image)

        new_ewald.alpha = self.alpha
        new_ewald.lattice_scale = self.lattice_scale
        new_ewald.ewald_roi = self.ewald_roi
        new_ewald.spot_w = self.spot_w
        new_ewald.spot_h = self.spot_h

        return new_ewald

    @property
    def lattice_scale(self) -> float:
        return self._lattice_scale

    @lattice_scale.setter
    def lattice_scale(self, value: float):

        self._lattice_scale = value
        self.calculate_ewald()

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value: float):
        self._alpha = value
        self.calculate_ewald()

    @property
    def beta(self) -> float:
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

    @property
    def spot_w(self) -> int:
        return self._spot_w

    @spot_w.setter
    def spot_w(self, value: int):
        self._spot_w = value
        self._spot_structure = self._generate_spot_structure()

    @property
    def spot_h(self) -> int:
        return self._spot_h

    @spot_h.setter
    def spot_h(self, value: int):
        self._spot_h = value
        self._spot_structure = self._generate_spot_structure()

    def calculate_ewald(self, **kwargs):

        k = self.ewald_radius
        alpha = self._alpha
        inverse_lattice = self._inverse_lattice.copy()

        kk = k**2

        if alpha != 0:
            inverse_lattice = inverse_lattice @ rotation_matrix(alpha).T

        gx = inverse_lattice[:, 0] / self._lattice_scale
        gy = inverse_lattice[:, 1] / self._lattice_scale

        # move the lattice to the crossing point of the Ewald sphere and specular revlected beam
        gx = gx - k * (1 - np.cos(np.deg2rad(self._beta)))
        tr = (gx + k) ** 2 + gy**2
        # select only the points that are inside the Ewald sphere
        ind = tr < kk
        gx_in = gx[ind]
        gy_in = gy[ind]

        # calculate the radius k_r
        kr = np.sqrt(kk - (k - abs(gx_in)) ** 2)

        # calculate theta and phi (cos) values
        theta = np.arcsin(kr / k)
        phi = np.arccos(gy_in / kr)

        # calculate the radius on the RHEED screen
        rho = self.screen_sample_distance * np.tan(theta)

        # calculate the spot positions
        sx = rho * np.cos(phi)
        sy = rho * np.sin(phi)

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

        self.sx = sx
        self.sy = -sy

    def transform_to_kxky(
        self,
        rotate: bool = True,
    ) -> xr.DataArray:

        # prepare hp_image dataArray
        hp_image = self.image.ri.hp_image

        # prepare the data for calculations
        screen_sample_distance = self.screen_sample_distance
        beta = self.beta
        alpha = self.alpha
        k = self.ewald_radius
        kk = k**2

        # new coordinates for transformation
        kx = np.linspace(-10, 10, 1024)
        ky = np.linspace(-10, 10, 1024)

        KX, KY = np.meshgrid(kx, ky, indexing="ij")

        # take into acount the theta angle
        KY = KY - k * (1 - np.cos(np.deg2rad(beta)))

        tr = (KY + k) ** 2 + KX**2

        # make nans points outside Ewald sphere
        ind = tr > kk
        KX[ind] = np.nan
        KY[ind] = np.nan

        kr = np.sqrt(kk - (k - abs(KY)) ** 2)
        th = np.arcsin(kr / k)
        rho = screen_sample_distance * np.tan(th)

        px_mm = rho * KX / kr
        py_mm = -np.sqrt(rho**2 - px_mm**2)

        # relation between old and new
        x = xr.DataArray(px_mm, dims=["kx", "ky"], coords={"kx": kx, "ky": ky})

        y = xr.DataArray(py_mm, dims=["kx", "ky"], coords={"kx": kx, "ky": ky})

        trans_image = hp_image.interp(x=x, y=y, method="linear")
        trans_image = trans_image.T

        if rotate:
            nan_mask = ~np.isnan(trans_image.values)
            trans_image = trans_image.fillna(0)

            # Rotate the mask and data
            rotated_nan_mask = (
                ndimage.rotate(nan_mask.astype(int), angle=30 - alpha, reshape=False)
                > 0
            )
            trans_image.data = ndimage.rotate(
                trans_image.data, 30 - alpha, reshape=False
            )

            # Apply the mask to restore NaN values in the rotated DataArray
            trans_image = trans_image.where(rotated_nan_mask)

        return trans_image

    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        show_image=True,
        auto_levels: float = 0.0,
        show_center_lines: bool = False,
        **kwargs,
    ):

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
            (self.sx + self.shift_x) * fine_scaling,
            (self.sy + self.shift_y) * fine_scaling,
            **kwargs,
        )

        return ax

    def calculate_match(self, normalize: bool = True) -> float:

        assert self._image_data_available, "Image data is not available"

        image = self.image
        scale = self.screen_scale

        center_x = image.x.values.min()
        center_y = image.y.values.max()
        image_data = self.image_data

        mask = np.zeros(image_data.shape, dtype=bool)

        spot_structure = self._spot_structure

        ppx = ((self.sx - center_x) * scale).astype(int)
        ppy = (-(self.sy - center_y) * scale).astype(int)

        mask[ppy, ppx] = True

        mask = ndimage.binary_dilation(mask, structure=spot_structure)
        match_coef = (mask * image_data).sum()

        if normalize:
            # norm_coef = np.ceil(len(ppx) / 10) * 10
            # norm_coef = len(ppx) * np.count_nonzero(spot_structure)
            match_coef = match_coef / len(ppx)

        return match_coef

    def match_phi(self, phi_vector: NDArray, normalize: bool = True) -> xr.DataArray:
        """Here we can calculate the matching for a series of different phi angles"""

        match_vector = np.zeros_like(phi_vector)

        for i, phi in enumerate(tqdm(phi_vector)):
            self.alpha = phi
            match_vector[i] = self.calculate_match(normalize=normalize)

        return xr.DataArray(match_vector, dims=["phi"], coords={"phi": phi_vector})

    def match_scale(
        self, scale_vector: NDArray, normalize: bool = True
    ) -> xr.DataArray:
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

    def _prepare_inverse_lattice(self):
        lattice = self._lattice
        space_size = self._ewald_roi
        inverse_lattice = Lattice.generate_lattice(
            lattice.b1, lattice.b2, space_size=space_size
        )
        return inverse_lattice

    def _generate_spot_structure(self) -> np.ndarray:

        # Define dimensions
        spot_w = self.spot_w
        spot_h = self.spot_h

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
