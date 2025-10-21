"""
Global constants for xrheed.
"""

# -----------------------------
# Canonical dimension names
# -----------------------------
CANONICAL_STACK_DIMS = {"alpha", "beta", "coverage", "time", "temperature", "current"}

# -----------------------------
# Expected dimensionality
# -----------------------------
IMAGE_NDIMS = 2  # Single image: (sy, sx)
STACK_NDIMS = 3  # Stacked images: (extra_dim, sy, sx)

# Core dimensions of any RHEED image
IMAGE_DIMS = {"sy", "sx"}

# -----------------------------
# Default analysis parameters
# -----------------------------
DEFAULT_SCREEN_ROI_WIDTH = 50.0  # mm
DEFAULT_SCREEN_ROI_HEIGHT = 50.0  # mm
DEFAULT_BETA = 1.0  # degrees
DEFAULT_ALPHA = 0.0  # degrees

# -----------------------------
# Physical constants
# -----------------------------
K_INV_ANGSTROM = 0.5123167219534328  # Å⁻¹ * sqrt(E/eV)
ANGSTROM = 10**-10  # Å
