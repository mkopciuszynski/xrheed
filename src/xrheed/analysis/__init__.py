"""
Submodule `analysis` provides tools for extracting physically meaningful
features from RHEED data.

"""

from .peaks import filter_kspace_peaks

# future:
# from .profiles import fit_profile_peaks

__all__ = [
    "filter_kspace_peaks",
]
