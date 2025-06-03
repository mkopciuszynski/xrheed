from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import copy
from matplotlib.axes import Axes

from numpy.typing import NDArray
from typing import Optional, List, Tuple, Literal, Union

Vector = NDArray[np.float64]

AllowedCubicTypes = Literal["SC", "BCC", "FCC"]
AllowedPlanes = Literal["111", "110", "100"]

class Lattice:
    """
    Represents a 2D lattice defined by two basis vectors (a1 and a2), or constructed from a specified plane of a cubic crystal.

    This class provides methods for:
      - Creating a lattice from custom basis vectors or from common cubic crystal planes (e.g., FCC (111)).
      - Generating both real-space and reciprocal-space (inverse) lattices.
      - Rotating and scaling the lattice.
      - Plotting the real and reciprocal lattices.

    Attributes:
        a1 (Vector): First lattice basis vector in real space.
        a2 (Vector): Second lattice basis vector in real space.
        b1 (Vector): First reciprocal lattice vector.
        b2 (Vector): Second reciprocal lattice vector.
        real_lattice (NDArray): Array of real-space lattice points.
        inverse_lattice (NDArray): Array of reciprocal-space lattice points.
    """

    def __init__(
        self,
        a1: Union[List[float], Vector],
        a2: Union[List[float], Vector]
    ) -> None:
        """
        Initializes a Lattice object with two basis vectors.

        Args:
            a1 (List[float] | Vector): The first basis vector of the lattice, as a list of floats or a Vector object.
            a2 (List[float] | Vector): The second basis vector of the lattice, as a list of floats or a Vector object.

        Raises:
            ValueError: If the provided vectors are invalid or cannot be validated.
        """
        self.a1 = self._validate_vector(a1)
        self.a2 = self._validate_vector(a2)

        self.b1, self.b2 = Lattice._calc_inverse_vectors(self.a1, self.a2)

        self.real_lattice = Lattice.generate_lattice(self.a1, self.a2)
        self.inverse_lattice = Lattice.generate_lattice(self.b1, self.b2)

    def __copy__(self) -> Lattice:
        """
        Create a shallow copy of the Lattice object.

        Returns:
            Lattice: A shallow copy of the Lattice object.
        """
        cls = self.__class__
        new_lattice = cls.__new__(cls)
        new_lattice.a1 = self.a1.copy()
        new_lattice.a2 = self.a2.copy()
        new_lattice.b1 = self.b1.copy()
        new_lattice.b2 = self.b2.copy()
        new_lattice.real_lattice = self.real_lattice.copy()
        new_lattice.inverse_lattice = self.inverse_lattice.copy()
        return new_lattice

    def __deepcopy__(self, memo: dict) -> Lattice:
        """
        Create a deep copy of the Lattice object.

        Args:
            memo (dict): Memoization dictionary for deep copy.

        Returns:
            Lattice: A deep copy of the Lattice object.
        """
        cls = self.__class__
        new_lattice = cls.__new__(cls)
        memo[id(self)] = new_lattice
        new_lattice.a1 = copy.deepcopy(self.a1, memo)
        new_lattice.a2 = copy.deepcopy(self.a2, memo)
        new_lattice.b1 = copy.deepcopy(self.b1, memo)
        new_lattice.b2 = copy.deepcopy(self.b2, memo)
        new_lattice.real_lattice = copy.deepcopy(self.real_lattice, memo)
        new_lattice.inverse_lattice = copy.deepcopy(self.inverse_lattice, memo)
        return new_lattice

    @classmethod
    def from_bulk_cubic(
        cls,
        a: float = 1.0,
        cubic_type: AllowedCubicTypes = "FCC",
        plane: AllowedPlanes = "111",
    ) -> Lattice:
        """
        Create a 2D lattice from a bulk cubic crystal.

        Args:
            a (float): Lattice constant.
            cubic_type (str): Type of cubic crystal ('SC', 'BCC', 'FCC').
            plane (str): Miller indices of the plane ('111', '110', '100').

        Returns:
            Lattice: A Lattice object constructed from the specified cubic crystal and plane.

        Raises:
            NotImplementedError: If the specified cubic type or plane is not supported.
        """

        if cubic_type not in {'SC', 'FCC', 'BCC'}:
            raise ValueError("Unsupported cubic_type. Use 'SC', 'FCC', or 'BCC'.")
        if plane not in {'100', '110', '111'}:
            raise ValueError("Unsupported plane. Use '100', '110', or '111'.")
            
        if (cubic_type, plane) == ('SC', '100'):
            a1 = np.array([a, 0, 0])
            a2 = np.array([0, a, 0])

        elif (cubic_type, plane) == ('SC', '110'):
            a1 = np.array([a * np.sqrt(2), 0, 0])
            a2 = np.array([0, a, 0])

        elif (cubic_type, plane) == ('SC', '111'):
            a_surf = a * np.sqrt(2)
            a1 = np.array([a_surf, 0, 0])
            a2 = np.array([a_surf * 0.5, a_surf * np.sqrt(3) / 2, 0])

        elif (cubic_type, plane) == ('FCC', '100'):
            a1 = np.array([a, 0, 0])
            a2 = np.array([a * 0.5, a * 0.5, 0])

        elif (cubic_type, plane) == ('FCC', '110'):
            a1 = np.array([a * np.sqrt(2), 0, 0])
            a2 = np.array([0, a, 0])

        elif (cubic_type, plane) == ('FCC', '111'):
            a_surf = a / np.sqrt(2)
            a1 = np.array([a_surf, 0, 0])
            a2 = np.array([a_surf * 0.5, a_surf * np.sqrt(3) / 2, 0])

        elif (cubic_type, plane) == ('BCC', '100'):
            a1 = np.array([a, 0, 0])
            a2 = np.array([0, a, 0])

        elif (cubic_type, plane) == ('BCC', '110'):
            a1 = np.array([a * np.sqrt(2), 0, 0])
            a2 = np.array([0, a, 0])

        elif (cubic_type, plane) == ('BCC', '111'):
            a_surf = np.sqrt(6) * a / 3
            a1 = np.array([a_surf, 0, 0])
            a2 = np.array([a_surf * 0.5, a_surf * np.sqrt(3) / 2, 0])   

        else:
            raise ValueError(f"Unsupported combination: {cubic_type} {plane}")
        
        return cls(a1, a2)

    def rotate(self, phi: float = 0.0) -> None:
        """
        Rotate the lattice by a given angle (in degrees).

        Args:
            phi (float): Rotation angle in degrees.
        """
        self.a1 = np.dot(rotation_matrix(phi), self.a1)
        self.a2 = np.dot(rotation_matrix(phi), self.a2)

        self.b1, self.b2 = Lattice._calc_inverse_vectors(self.a1, self.a2)

        self.real_lattice = Lattice.generate_lattice(self.a1, self.a2)
        self.inverse_lattice = Lattice.generate_lattice(self.b1, self.b2)

    def scale(self, lattice_scale: float = 1.0) -> None:
        """
        Scale the lattice vectors by a given factor.

        Args:
            lattice_scale (float): Scaling factor for the lattice vectors.
        """
        self.a1 = self.a1 * lattice_scale
        self.a2 = self.a2 * lattice_scale
        self.b1 = self.b1 / lattice_scale
        self.b2 = self.b2 / lattice_scale

        self.real_lattice = Lattice.generate_lattice(self.a1, self.a2)
        self.inverse_lattice = Lattice.generate_lattice(self.b1, self.b2)

    def plot_real(
        self,
        ax: Optional[Axes] = None,
        **kwargs
    ) -> Axes:
        """
        Plot the real-space lattice points.

        Args:
            ax (plt.Axes, optional): Matplotlib Axes object to plot on. If None, a new figure and axes are created.
            **kwargs: Additional keyword arguments passed to plt.plot.

        Returns:
            Axes: The matplotlib Axes object used for plotting.
        """
        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(self.real_lattice[:, 0], self.real_lattice[:, 1], ".", **kwargs)

        ax.set_xlim(-20, 20)
        ax.set_ylim(-10, 10)
        ax.set_aspect(1)
        return ax

    def plot_inverse(
        self,
        ax: Optional[Axes] = None,
        **kwargs
    ) -> Axes:
        """
        Plot the reciprocal-space (inverse) lattice points.

        Args:
            ax (plt.Axes, optional): Matplotlib Axes object to plot on. If None, a new figure and axes are created.
            **kwargs: Additional keyword arguments passed to plt.plot.

        Returns:
            Axes: The matplotlib Axes object used for plotting.
        """
        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(self.inverse_lattice[:, 0], self.inverse_lattice[:, 1], ".", **kwargs)
        ax.plot(0, 0, 'or')

        ax.set_xlabel('gx [1/A]')
        ax.set_ylabel('gy [1/A]')

        ax.set_xlim(-10, 10)
        ax.set_ylim(-7, 7)
        ax.set_aspect(1)
        return ax

    def __str__(self) -> str:
        """
        Return a string representation of the lattice basis vectors.

        Returns:
            str: String representation of a1 and a2.
        """
        return (f"a1 = [{self.a1[0]}, {self.a1[1]}]\n"
                f"a2 = [{self.a2[0]}, {self.a2[1]}]")

    @staticmethod
    def hex_lattice(a: float) -> Tuple[Vector, Vector]:
        """
        Generate basis vectors for a 2D hexagonal lattice.

        Args:
            a (float): Lattice constant.

        Returns:
            Tuple[Vector, Vector]: Two basis vectors for the hexagonal lattice.
        """
        a1 = np.array([a, 0.0, 0.0])
        a2 = np.array([a * 0.5, a * 0.8660, 0.0])

        return a1, a2

    @staticmethod
    def _validate_vector(vector: Union[List[float], Vector]) -> Vector:
        """
        Validate that the vector is a list or ndarray of size (2,) or (3,).

        Args:
            vector (List[float] | Vector): Input vector.

        Returns:
            Vector: Validated 3D vector.

        Raises:
            ValueError: If the input is not a list or ndarray, or has invalid shape.
        """
        if isinstance(vector, list):
            vector = np.array(vector, dtype=float)
        elif isinstance(vector, np.ndarray):
            vector = vector.astype(float)
        else:
            raise ValueError("Vector must be a list or ndarray.")

        if vector.shape == (2,):
            vector = np.append(vector, 0.0)

        if vector.shape != (3,):
            raise ValueError("Vector must be of size (2,) or (3,).")
        return vector

    @staticmethod
    def _calc_inverse_vectors(a1: Vector, a2: Vector) -> Tuple[Vector, Vector]:
        """
        Calculate the reciprocal lattice vectors for a 2D lattice.

        Args:
            a1 (Vector): First real-space basis vector.
            a2 (Vector): Second real-space basis vector.

        Returns:
            Tuple[Vector, Vector]: Two reciprocal lattice vectors.
        """
        n = np.array([0.0, 0.0, 1.0])
        surf: float = abs(np.dot(a1, np.cross(a2, n)))
        b1 = 2 * np.pi / surf * np.cross(a2, n)
        b2 = 2 * np.pi / surf * np.cross(n, a1)
        return b1, b2

    @staticmethod
    def generate_lattice(
        v1: Vector,
        v2: Vector,
        space_size: float = 70.0
    ) -> NDArray[np.float64]:
        """
        Generate a grid of lattice points within a specified space size.

        Args:
            v1 (Vector): First lattice vector.
            v2 (Vector): Second lattice vector.
            space_size (float): The size of the rectangular area in which to generate lattice points.

        Returns:
            NDArray: Array of lattice points within the specified area.
        """
        vec_num_x = int(space_size * 2 / max(abs(v1[0]), abs(v2[0])))
        vec_num_y = int(space_size * 2 / max(abs(v1[1]), abs(v2[1])))

        # Generate a grid of coefficients for the linear combinations
        i_vals = np.arange(-vec_num_x, vec_num_x)
        j_vals = np.arange(-vec_num_y, vec_num_y)
        mi, mj = np.meshgrid(i_vals, j_vals)
        mi = mi.flatten()
        mj = mj.flatten()

        # Generate lattice points using linear combinations (vectorized)
        lattice = np.outer(mi, v1) + np.outer(mj, v2)

        # Filter points that are within the circle (vectorized)
        distances = np.linalg.norm(lattice, axis=1)
        lattice = lattice[distances <= space_size]

        return lattice

def rotation_matrix(phi: float = 0.0) -> NDArray[np.float64]:
    """
    Generates a 3D rotation matrix for a given angle phi (in degrees) about the z-axis.

    Args:
        phi (float): Rotation angle in degrees.

    Returns:
        np.ndarray: 3x3 rotation matrix.
    """
    phi = np.radians(phi)
    return np.array([
        [np.cos(phi), -np.sin(phi), 0.0],
        [np.sin(phi), np.cos(phi), 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=float)
