"""Automatic stain correction for scanned Sanborn maps."""

from .spots import restore_rgb
from .files import restore_file

__all__ = ["restore_rgb", "restore_file"]
__version__ = "0.1.0"
