"""Tests for MROI spatial reassembly.

ScanImage stacks MROI scanfields along Y in ACQUISITION order with flyto blank
lines between them. Where the ROIs actually sat in the sample is recorded
separately, in microns. These tests pin the arithmetic that puts them back:
get it wrong by one line and every downstream ROI is silently shifted.

Everything here is synthetic -- no acquisition files, no F:\\ or /nrs, so it
runs anywhere.
"""

from __future__ import annotations

import numpy as np
import pytest

from lazy_tiff_reader.mroi_si_tiff import (
    MroiGeometry,
    assemble_mosaic,
    mroi_mosaic_layout,
    roi_y_slices,
    sf_skip_lines,
    split_rois,
)
# Builders live in the package so mesoscope's tests can use them too.
from lazy_tiff_reader.testing import make_geometry, make_page, side_by_side


# --------------------------------------------------------------------------- #
# Layout arithmetic
# --------------------------------------------------------------------------- #


def test_side_by_side_rois_are_placed_side_by_side():
    """The whole point: ROIs stacked in Y on disk sit next to each other in the sample."""
    layout = mroi_mosaic_layout(side_by_side())

    assert layout.shape == (50, 200)                 # one row, two tiles wide
    np.testing.assert_array_equal(layout.origins, [[0, 0], [0, 100]])
    assert layout.pixel_size_xy_um == (1.0, 1.0)
    assert layout.origin_um == (-100.0, -25.0)


def test_stacked_rois_stay_stacked():
    """ROIs genuinely above one another in the sample keep that relationship."""
    geometry = make_geometry(
        centers_um=[[0.0, -25.0], [0.0, 25.0]],
        sizes_um=[[100.0, 50.0], [100.0, 50.0]],
        dims_xy=[[100, 50], [100, 50]],
        sf_skip=8,
    )
    layout = mroi_mosaic_layout(geometry)

    assert layout.shape == (100, 100)
    np.testing.assert_array_equal(layout.origins, [[0, 0], [50, 0]])


def test_single_roi_is_the_identity():
    """A single-ROI file must be untouched -- that is why 'mosaic' is the default."""
    geometry = make_geometry(
        centers_um=[[0.0, 0.0]],
        sizes_um=[[700.0, 100.0]],
        dims_xy=[[700, 100]],
        sf_skip=0,
    )
    layout = mroi_mosaic_layout(geometry)

    assert layout.shape == (100, 700)
    np.testing.assert_array_equal(layout.origins, [[0, 0]])

    page = make_page(geometry)
    np.testing.assert_array_equal(assemble_mosaic(page, geometry), page)


def test_gap_between_rois_becomes_canvas_not_padding():
    """A real spatial gap is preserved; it is signal-free, not tile padding."""
    geometry = make_geometry(
        centers_um=[[-100.0, 0.0], [100.0, 0.0]],   # 200 um apart, 100 um wide
        sizes_um=[[100.0, 50.0], [100.0, 50.0]],
        dims_xy=[[100, 50], [100, 50]],
        sf_skip=8,
    )
    layout = mroi_mosaic_layout(geometry)

    assert layout.shape == (50, 300)                 # 100 + 100 gap + 100
    np.testing.assert_array_equal(layout.origins, [[0, 0], [0, 200]])

    mosaic = assemble_mosaic(make_page(geometry), geometry, fill_value=0)
    assert (mosaic[:, 100:200] == 0).all()           # the gap really is empty
    assert (mosaic[:, :100] == 1).all()
    assert (mosaic[:, 200:] == 2).all()


def test_mismatched_pixel_size_is_refused_not_guessed():
    """Two sampling densities cannot share one canvas; say so instead of mis-scaling."""
    geometry = make_geometry(
        centers_um=[[-50.0, 0.0], [50.0, 0.0]],
        sizes_um=[[100.0, 50.0], [100.0, 50.0]],
        dims_xy=[[100, 50], [200, 50]],              # roi1 at 0.5 um/px
        sf_skip=8,
    )
    with pytest.raises(ValueError, match="disagree on pixel size"):
        mroi_mosaic_layout(geometry)


def test_small_pixel_size_differences_are_tolerated():
    """Rounded metadata should not defeat the mosaic."""
    geometry = make_geometry(
        centers_um=[[-50.0, 0.0], [50.0, 0.0]],
        sizes_um=[[100.0, 50.0], [100.5, 50.0]],     # 0.5% off
        dims_xy=[[100, 50], [100, 50]],
        sf_skip=8,
    )
    layout = mroi_mosaic_layout(geometry)            # must not raise
    assert layout.shape[0] == 50


# --------------------------------------------------------------------------- #
# Pixel movement
# --------------------------------------------------------------------------- #


def test_mosaic_drops_flyto_lines():
    """The blank lines between tiles must not survive into the output."""
    geometry = side_by_side(gap_lines=8)
    page = make_page(geometry)

    assert (page == 999).any(), "fixture should contain flyto lines"
    mosaic = assemble_mosaic(page, geometry)
    assert not (mosaic == 999).any(), "flyto lines leaked into the mosaic"


def test_mosaic_preserves_every_pixel_of_every_tile():
    """Content must be moved, not resampled: exact pixel equality per tile."""
    geometry = side_by_side()
    rng = np.random.default_rng(0)
    page = np.full((geometry.total_y, 100), 999, dtype=np.uint16)
    tiles = []
    for i, y0, y1 in roi_y_slices(geometry):
        tile = rng.integers(0, 4000, size=(y1 - y0, 100), dtype=np.uint16)
        page[y0:y1] = tile
        tiles.append(tile)

    mosaic = assemble_mosaic(page, geometry)
    np.testing.assert_array_equal(mosaic[:, :100], tiles[0])
    np.testing.assert_array_equal(mosaic[:, 100:], tiles[1])


def test_split_rois_returns_tiles_without_gaps():
    geometry = side_by_side()
    tiles = split_rois(make_page(geometry), geometry)

    assert len(tiles) == 2
    for i, tile in enumerate(tiles):
        assert tile.shape == (50, 100)
        assert (tile == i + 1).all(), "tile picked up a neighbour or a flyto line"


def test_leading_axes_are_preserved():
    """Real data is 5D TZCYX; the reassembly must not care how many leading axes."""
    geometry = side_by_side()
    page = make_page(geometry)
    stack = np.broadcast_to(page, (3, 2, 1) + page.shape).copy()

    mosaic = assemble_mosaic(stack, geometry)
    assert mosaic.shape == (3, 2, 1, 50, 200)
    np.testing.assert_array_equal(mosaic[0, 0, 0], assemble_mosaic(page, geometry))

    tiles = split_rois(stack, geometry)
    assert tiles[0].shape == (3, 2, 1, 50, 100)


def test_dtype_is_preserved():
    geometry = side_by_side()
    for dtype in (np.uint16, np.int16, np.float32):
        page = make_page(geometry).astype(dtype)
        assert assemble_mosaic(page, geometry).dtype == dtype


def test_page_height_must_match_geometry():
    """Guard against silently mosaicking a page the geometry does not describe."""
    geometry = side_by_side()
    page = make_page(geometry)[:-1]
    with pytest.raises(ValueError, match="does not match geometry.total_y"):
        split_rois(page, geometry)


# --------------------------------------------------------------------------- #
# The flyto-gap inference this all rests on
# --------------------------------------------------------------------------- #


def test_sf_skip_inferred_from_geometry_when_metadata_disagrees():
    """With no usable flyto metadata the gap is derived from the line budget."""
    # 2 ROIs of 50 lines in a 108-line page -> the gap must be 8.
    assert sf_skip_lines({}, n_rois=2, total_y=108, roi_heights=[50, 50]) == 8


def test_sf_skip_uses_metadata_when_it_checks_out():
    frame_data = {
        "SI.hScan2D.flytoTimePerScanfield": 0.0008,
        "SI.hRoiManager.linePeriod": 0.0001,
    }  # -> 8 lines
    assert sf_skip_lines(frame_data, n_rois=2, total_y=108, roi_heights=[50, 50]) == 8


def test_sf_skip_rejects_an_impossible_layout():
    with pytest.raises(ValueError, match="MROI Y layout mismatch"):
        sf_skip_lines({}, n_rois=3, total_y=107, roi_heights=[50, 50, 50])


def test_single_roi_has_no_gap():
    assert sf_skip_lines({}, n_rois=1, total_y=50, roi_heights=[50]) == 0


def test_roi_y_slices_skip_the_gaps():
    geometry = side_by_side(gap_lines=8)
    assert roi_y_slices(geometry) == [(0, 0, 50), (1, 58, 108)]
