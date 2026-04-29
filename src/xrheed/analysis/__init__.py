"""
Submodule `analysis` provides tools for extracting physically meaningful
features from RHEED data.

This includes peak detection, peak reconstruction, and profile fitting
in both 2D diffraction images (kx-ky space) and 1D intensity profiles.

"""

from .peaks import reconstruct_peaks_kxky, reconstruct_peaks_stack
# future:
# from .profiles import fit_profile_peaks

__all__ = [
    "reconstruct_peaks_kxky",
    "reconstruct_peaks_stack",
]