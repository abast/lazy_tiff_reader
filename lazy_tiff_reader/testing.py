"""Synthetic MROI geometry and page builders for tests.

These lived inline in ``tests/test_mroi_mosaic.py`` (originally in the
``mesoscope`` repo). They are shipped inside the package -- not left in
``tests/`` -- because ``mesoscope``'s own test suite builds synthetic MROI
pages with them too, and a test directory is not importable across repos.

Nothing here touches a file: these construct :class:`MroiGeometry` values and
matching in-memory pages, so a test can exercise the layout / mask / mosaic
arithmetic without a TIFF on disk.
"""

from __future__ import annotations

import numpy as np

from .mroi_si_tiff import MroiGeometry, roi_y_slices

__all__ = ["make_geometry", "make_page", "side_by_side"]


def make_geometry(centers_um, sizes_um, dims_xy, *, sf_skip, line_averaging=1):
    """An MroiGeometry with total_y consistent with the tiles and the gap."""
    dims_xy = np.asarray(dims_xy, dtype=int)
    n = dims_xy.shape[0]
    total_y = int(dims_xy[:, 1].sum() + (n - 1) * sf_skip)
    sizes = np.asarray(sizes_um, dtype=float)
    return MroiGeometry(
        dims_xy=dims_xy,
        center_xy_um=np.asarray(centers_um, dtype=float),
        size_xy_um=sizes,
        sf_skip=sf_skip,
        total_y=total_y,
        line_averaging=line_averaging,
        pixel_size_um=float(sizes[0, 0] / dims_xy[0, 0]),
        objective_resolution=161.275,
    )


def side_by_side(gap_lines=8):
    """Two 100x50 um ROIs abutting in X, at 1 um/px. The headline case."""
    return make_geometry(
        centers_um=[[-50.0, 0.0], [50.0, 0.0]],
        sizes_um=[[100.0, 50.0], [100.0, 50.0]],
        dims_xy=[[100, 50], [100, 50]],
        sf_skip=gap_lines,
    )


def make_page(geometry, fill=None):
    """A page with each ROI band filled with a distinct value; gaps = 999."""
    page = np.full(
        (geometry.total_y, int(geometry.dims_xy[:, 0].max())), 999, dtype=np.uint16
    )
    for i, y0, y1 in roi_y_slices(geometry):
        value = (i + 1) if fill is None else fill[i]
        page[y0:y1, : int(geometry.dims_xy[i, 0])] = value
    return page
