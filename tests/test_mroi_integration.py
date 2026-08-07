"""End-to-end MROI checks against a real ScanImage acquisition.

ScanImage metadata lives in a proprietary file header that ``tifffile`` parses
but cannot write, so these cannot run on a synthesised file -- the pure
arithmetic is covered without data in test_mroi_mosaic.py, and this module
covers the metadata-to-pixels path that only a real acquisition exercises.

Point it at a file and run::

    set MESOSCOPE_TEST_TIFF=F:\\Virginia_meso\\<folder>\\<scan>_00001.tif
    pytest tests/test_mroi_integration.py

Skipped when the variable is unset, so the default suite stays hardware-free.

The reformat round-trip tests that used to live here (single-ROI mosaic
no-op, split-writes-one-file-per-roi) stayed in the ``mesoscope`` repo with
``reformat_si_tiff`` itself -- they test the converter, not this layer.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import tifffile

from lazy_tiff_reader.mroi_si_tiff import (
    assemble_mosaic,
    load_mroi_geometry,
    mroi_mosaic_layout,
    roi_y_slices,
    split_rois,
)

TIFF = os.environ.get("MESOSCOPE_TEST_TIFF")

pytestmark = pytest.mark.skipif(
    not TIFF or not os.path.isfile(TIFF),
    reason="set MESOSCOPE_TEST_TIFF to a ScanImage TIFF to run these",
)


@pytest.fixture(scope="module")
def geometry():
    return load_mroi_geometry(TIFF)


@pytest.fixture(scope="module")
def page():
    """First page of the acquisition, as stored on disk."""
    with tifffile.TiffFile(TIFF) as tif:
        return tif.pages[0].asarray()


def test_geometry_is_self_consistent(geometry):
    """Tile heights plus flyto gaps must account for the page exactly."""
    covered = int(geometry.roi_heights.sum())
    gaps = (geometry.n_rois - 1) * geometry.sf_skip
    assert covered + gaps == geometry.total_y


def test_y_slices_cover_the_page(geometry):
    slices = roi_y_slices(geometry)
    assert len(slices) == geometry.n_rois
    assert slices[0][1] == 0
    assert slices[-1][2] == geometry.total_y


def test_split_tiles_match_declared_dims(geometry, page):
    for i, tile in enumerate(split_rois(page, geometry)):
        assert tile.shape[-1] == int(geometry.dims_xy[i, 0])
        assert tile.shape[-2] == int(geometry.dims_xy[i, 1])


def test_mosaic_contains_every_tile_verbatim(geometry, page):
    """The mosaic must move pixels, never alter them."""
    if geometry.n_rois > 1:
        try:
            layout = mroi_mosaic_layout(geometry)
        except ValueError as exc:            # tiles differ in pixel size
            pytest.skip(f"not mosaicable: {exc}")
    else:
        layout = mroi_mosaic_layout(geometry)

    mosaic = assemble_mosaic(page, geometry, layout)
    assert mosaic.shape == layout.shape
    assert mosaic.dtype == page.dtype

    for i, tile in enumerate(split_rois(page, geometry)):
        row, col = int(layout.origins[i, 0]), int(layout.origins[i, 1])
        placed = mosaic[row : row + tile.shape[-2], col : col + tile.shape[-1]]
        np.testing.assert_array_equal(placed, tile)
