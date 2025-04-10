from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from numpy.typing import NDArray
from typing import Optional, List, Tuple


Vector = NDArray[np.float64]

class Lattice:
    """ 
    This class defines the lattice tools that allows to create a 2D lattice 
    by providing the basic vectors.
    """

    def __init__(self, 
                 a1: List[float] | Vector, 
                 a2: List[float] | Vector) -> None:
        
        self.a1 = self._validate_vector(a1)
        self.a2 = self._validate_vector(a2)

        self.b1, self.b2 = Lattice._calc_inverse_vectors(self.a1, self.a2)

        self.real_lattice = Lattice.generate_lattice(self.a1, self.a2)
        self.inverse_lattice = Lattice.generate_lattice(self.b1, self.b2)


    def copy(self):
        cls = self.__class__
        new_lattice = cls.__new__(cls)
        new_lattice.a1 = self.a1.copy()
        new_lattice.a2 = self.a2.copy()
        new_lattice.b1 = self.b1.copy()
        new_lattice.b2 = self.b2.copy()
        new_lattice.real_lattice = self.real_lattice.copy()
        new_lattice.inverse_lattice = self.inverse_lattice.copy()
        return new_lattice

    @classmethod
    def from_bulk(cls, 
                  a: float = 1.0, 
                  plane: Optional[str] = None):
    # TODO add other planes
        if plane == '111' or None:
            a_surf = a * 0.7071  # 1/sqrt(2

            a1, a2 = Lattice._hex_lattice(a_surf)
            return cls(a1, a2)
        else:
            raise(ValueError("Only (111) plane is supported."))


    @classmethod
    def from_surface(cls, 
                     a: float=1.0,
                       plane: Optional[str] = None):
        
        if plane == '111' or plane is None:
            a1, a2 = Lattice._hex_lattice(a)
            return cls(a1, a2)
        else:
        #TODO add other symetries
            raise(ValueError("Only (111) plane is supported."))

    def rotate(self, phi: float = 0.0):

        self.a1 = rotation_matrix(phi) @ self.a1
        self.a2 = rotation_matrix(phi) @ self.a2

        self.b1, self.b2 = Lattice._calc_inverse_vectors(self.a1, self.a2)

        self.real_lattice = Lattice.generate_lattice(self.a1, self.a2)
        self.inverse_lattice = Lattice.generate_lattice(self.b1, self.b2)
        

    def scale(self, lattice_scale: float = 1.0) -> None:
        self.a1 = self.a1 * lattice_scale
        self.a2 = self.a2 * lattice_scale
        self.b1 = self.b1 / lattice_scale
        self.b2 = self.b2 / lattice_scale
        
        self.real_lattice = Lattice.generate_lattice(self.a1, self.a2)
        self.inverse_lattice = Lattice.generate_lattice(self.b1, self.b2)



    def plot_real(self, 
             ax: Optional[plt.Axes] = None,
             **kwargs):
        
        if ax is None: 
            fig, ax = plt.subplots() 
        
        ax.plot(self.real_lattice[:,0], 
                self.real_lattice[:,1], ".", **kwargs) 
        
        ax.set_xlim(-20, 20)
        ax.set_ylim(-10, 10)
        ax.set_aspect(1)

    def plot_inverse(self, 
             ax: Optional[plt.Axes] = None,
             **kwargs):
        
        if ax is None: 
            fig, ax = plt.subplots() 
        
        ax.plot(self.inverse_lattice[:,0], 
                self.inverse_lattice[:,1], ".", **kwargs) 
        
        ax.plot(0,0, 'or')

        ax.set_xlabel('gx [1/A]')
        ax.set_ylabel('gy [1/A]')
        
        ax.set_xlim(-10, 10)
        ax.set_ylim(-7, 7)
        ax.set_aspect(1)

    def __str__(self):
        return (f"a1 = [{self.a1[0]}, {self.a1[1]}]\n"
                f"a2 = [{self.a2[0]}, {self.a2[1]}]")


    @staticmethod
    def _hex_lattice(a) -> Tuple[Vector, Vector]:

        a1 = np.array([a, 0.0, 0.0])
        a2 = np.array([a * 0.5, a * 0.8660, 0])

        return a1, a2

    @staticmethod
    def _validate_vector(vector: List[float] | Vector) -> Vector:
        #TODO cleanup this mess
        """Validate that the vector is a list of size (2,) or (3,)."""
        if isinstance(vector, list):
            vector = np.array(vector)
        elif isinstance(vector, np.ndarray):
            pass
        else:
            raise ValueError("Vector must be a list or ndarray.")
        
        if vector.shape == (2,):
            vector = np.append(vector, 0)
        
        if vector.shape != (3,):
            raise ValueError("Vector must be of size (2,) or (3,).")
        return vector

    @staticmethod
    def _calc_inverse_vectors(a1: Vector, 
                              a2: Vector) -> Tuple[Vector, Vector]:

        n = np.array([0, 0, 1])
        surf: float = abs(np.dot(a1, np.cross(a2, n)))
        b1 = 2 * np.pi / surf * np.cross(a2, n)
        b2 = 2 * np.pi / surf * np.cross(n, a1)
        return b1, b2

    @staticmethod
    def generate_lattice(v1: Vector, 
                         v2: Vector, 
                         space_size: float=70) -> NDArray:

        vec_num_x = int(space_size*2 / max(abs(v1[0]), abs(v2[0])))
        vec_num_y = int(space_size*2 / max(abs(v1[1]), abs(v2[1])))

        x = np.zeros(vec_num_x * vec_num_y, dtype=float)
        y = np.zeros_like(x)
        
        ind = 0
        for i in range(0, vec_num_x):
            for j in range(0, vec_num_y):
                x[ind] = v1[0] * i + v2[0] * j
                y[ind] = v1[1] * i + v2[1] * j
                ind += 1

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

    

def rotation_matrix(phi):
    """ Generates a rotation matrix for a given phi angle (in degrees)."""
    phi = np.radians(phi)
    return np.array([[np.cos(phi), -np.sin(phi), 0],
                     [np.sin(phi), np.cos(phi), 0],
                     [0, 0, 1]])
