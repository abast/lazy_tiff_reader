"""Lazy, memory-mapped access to ScanImage TIFF stacks."""

from .gapped_memmap import GappedMemmap
from .memmap_tiff_si import (MemmapTiffSI, UnknownLayoutError,
                             check_layout_readable)
from .mroi_si_tiff import (MosaicLayout, MroiGeometry, StaleAnnotationWarning,
                           acq_line_mask_from_annotations,
                           annotation_describes_geometry, assemble_mosaic,
                           erode_line_mask, line_masks_for_geometry,
                           load_mroi_geometry, mroi_mosaic_layout,
                           open_si_stack, read_si_metadata, roi_y_slices,
                           sf_skip_lines, shrink_line_mask, split_rois,
                           stack_tyx)
from .slices_to_offsets import slices_to_offsets

__all__ = [
    "GappedMemmap",
    "MemmapTiffSI",
    "UnknownLayoutError",
    "check_layout_readable",
    "slices_to_offsets",
    # MROI layer: page -> tiles / masks / mosaic (moved from
    # mesoscope.utils.mroi_si_tiff on 2026-08-07; the layering is unchanged --
    # this module sits ON TOP of the page readers above).
    "MosaicLayout",
    "MroiGeometry",
    "StaleAnnotationWarning",
    "acq_line_mask_from_annotations",
    "annotation_describes_geometry",
    "assemble_mosaic",
    "erode_line_mask",
    "line_masks_for_geometry",
    "load_mroi_geometry",
    "mroi_mosaic_layout",
    "open_si_stack",
    "read_si_metadata",
    "roi_y_slices",
    "sf_skip_lines",
    "shrink_line_mask",
    "split_rois",
    "stack_tyx",
]
