"""Lazy, memory-mapped access to ScanImage TIFF stacks."""

from .gapped_memmap import GappedMemmap
from .memmap_tiff_si import (MemmapTiffSI, UnknownLayoutError,
                             check_layout_readable)
from .slices_to_offsets import slices_to_offsets

__all__ = [
    "GappedMemmap",
    "MemmapTiffSI",
    "UnknownLayoutError",
    "check_layout_readable",
    "slices_to_offsets",
]
