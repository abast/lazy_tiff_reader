"""ScanImage MROI TIFF geometry and stack open helpers.

MROI pages concatenate ROI scanfields along the slow (Y / line) axis with
flyto blank lines between tiles. This module reads RoiGroups metadata,
computes Y-band slices, and opens a time-series reader that works when
``MemmapTiffSI`` mis-parses multi-frame single-slice acquisitions.

Public API
----------
- :func:`load_mroi_geometry`
- :func:`roi_y_slices`
- :func:`sf_skip_lines`
- :func:`acq_line_mask_from_annotations`
- :func:`annotation_describes_geometry`
- :func:`line_masks_for_geometry`
- :func:`open_si_stack`
- :class:`MroiGeometry`

Spatial reassembly (undo the on-disk Y stacking):

- :func:`mroi_mosaic_layout` : where each tile belongs on a canvas, in pixels
- :func:`assemble_mosaic`    : paste tiles from a page into that canvas
- :func:`split_rois`         : the other option -- one array per ROI
- :class:`MosaicLayout`
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import tifffile

from .gapped_memmap import GappedMemmap
from .memmap_tiff_si import MemmapTiffSI
from .utils.get_si_tiff_n_pages import get_si_tiff_n_pages
from .utils.read_si_framedata_params import read_si_framedata_params

PathLike = Union[str, Path]


class StaleAnnotationWarning(UserWarning):
    """The header's xroiProps annotations describe a different acquisition."""


@lru_cache(maxsize=8)
def _read_si_metadata_cached(path: str, _stat: Tuple[int, int]) -> Dict[str, Any]:
    sm = read_si_framedata_params(path)
    if sm is None:
        with tifffile.TiffFile(path) as tif:
            return dict(tif.scanimage_metadata or {})
    return sm


def read_si_metadata(tiff_path: PathLike) -> Dict[str, Any]:
    """ScanImage ``FrameData`` + ``RoiGroups`` for one TIFF.

    Delegates to :func:`lazy_tiff_reader...read_si_framedata_params`, which
    reads the header blocks directly. ``tifffile.scanimage_metadata`` is only a
    fallback for non-ScanImage files: it hands the whole FrameData blob to
    ``matlabstr2py``, which never returns once xroiProps waveforms inflate that
    blob past ~20 MB.

    Results are cached per (path, size, mtime); geometry and line masks are
    read separately but come from the same multi-hundred-MB header scan. The
    returned dict is shared between callers -- treat it as read-only.
    """
    path = os.fspath(tiff_path)
    try:
        st = os.stat(path)
        key = (st.st_size, st.st_mtime_ns)
    except OSError:
        key = (-1, -1)
    return _read_si_metadata_cached(str(path), key)


@dataclass(frozen=True)
class MroiGeometry:
    """Enabled-ROI layout for one ScanImage MROI TIFF page."""

    dims_xy: np.ndarray          # (N, 2) int: [X_px, Y_lines] on disk
    center_xy_um: np.ndarray     # (N, 2) float
    size_xy_um: np.ndarray       # (N, 2) float
    sf_skip: int
    total_y: int
    line_averaging: int
    pixel_size_um: float
    objective_resolution: float

    @property
    def n_rois(self) -> int:
        return int(self.dims_xy.shape[0])

    @property
    def roi_heights(self) -> np.ndarray:
        return self.dims_xy[:, 1].astype(int)


def _as_scanfield(roi: dict) -> dict:
    """Return the first scanfield dict from a RoiGroups ROI entry."""
    sf = roi["scanfields"]
    if isinstance(sf, list):
        return sf[0]
    return sf


def _roi_enabled(sf: dict) -> bool:
    en = sf.get("enable", 1)
    if isinstance(en, (list, tuple, np.ndarray)):
        return bool(en[0]) if len(en) else True
    return bool(en)


def _line_averaging(frame_data: dict) -> int:
    for key in (
        "SI.hScan2D.lineAverageFactor",
        "SI.hScan2D.LineAveragingLineCount",
        "SI.hAcq.lineAverageFactor",
    ):
        v = frame_data.get(key)
        if v is not None:
            try:
                n = int(v)
            except (TypeError, ValueError):
                continue
            if n >= 1:
                return n
    return 1


def sf_skip_lines(
    frame_data: dict,
    n_rois: int,
    total_y: int,
    roi_heights: Sequence[int],
) -> int:
    """Flyto gap (lines) between consecutive ROI tiles.

    Computes ``round(flytoTimePerScanfield / linePeriod)``, then validates
    against ``sum(roi_heights) + (n_rois-1)*gap == total_y``. On mismatch,
    infers the gap from geometry. Never gates on ``scannerType == 'Resonant'``.
    """
    heights = [int(h) for h in roi_heights]
    if n_rois < 1:
        raise ValueError("n_rois must be >= 1")
    sum_h = int(sum(heights))
    if n_rois == 1:
        if sum_h != int(total_y):
            raise ValueError(
                f"Single-ROI height {sum_h} != total_y {total_y}"
            )
        return 0

    flyto = frame_data.get("SI.hScan2D.flytoTimePerScanfield")
    line_period = frame_data.get("SI.hRoiManager.linePeriod")
    gap_meta = 0
    if flyto is not None and line_period is not None and float(line_period) > 0:
        gap_meta = int(round(float(flyto) / float(line_period)))

    expected = sum_h + (n_rois - 1) * gap_meta
    if expected == int(total_y):
        return gap_meta

    remainder = int(total_y) - sum_h
    if remainder < 0 or remainder % (n_rois - 1) != 0:
        raise ValueError(
            f"MROI Y layout mismatch: total_y={total_y}, sum_roi_h={sum_h}, "
            f"n_rois={n_rois}, flyto_gap_meta={gap_meta} (expected Y={expected})"
        )
    return remainder // (n_rois - 1)


def load_mroi_geometry(
    tiff_path: PathLike,
    *,
    total_y: int | None = None,
) -> MroiGeometry:
    """Load enabled-ROI geometry from ScanImage RoiGroups + FrameData.

    Parameters
    ----------
    tiff_path
        Path to a ScanImage TIFF.
    total_y
        Page height in lines. If None, read from the first page shape.
    """
    path = str(tiff_path)
    sm = read_si_metadata(path)
    if not sm or "RoiGroups" not in sm or "FrameData" not in sm:
        raise RuntimeError(f"Missing ScanImage RoiGroups/FrameData: {path}")
    if total_y is None:
        with tifffile.TiffFile(path) as tif:
            total_y = int(tif.pages[0].shape[0])
    frame_data = sm["FrameData"]
    rois = sm["RoiGroups"]["imagingRoiGroup"]["rois"]
    if not isinstance(rois, list):
        rois = [rois]

    obj_res = float(frame_data["SI.objectiveResolution"])
    line_avg = _line_averaging(frame_data)
    # Measured SI 2026.1 RGG resonant files retain the native ROI pixel grid
    # on disk even with LineAveragingLineCount > 1. The 2023 corpus instead
    # stores Y / line_avg rows. Select by metadata, never by whichever fits.
    native_saved_rows = (
        frame_data.get("SI.VERSION_MAJOR") == 2026
        and frame_data.get("SI.VERSION_MINOR") == 1
        and frame_data.get("SI.hScan2D.scannerType") == "RGG"
        and frame_data.get("SI.hScan2D.scanMode") == "resonant"
    )

    dims = []
    centers = []
    sizes = []
    for roi in rois:
        sf = _as_scanfield(roi)
        if not _roi_enabled(sf):
            continue
        px = np.asarray(sf["pixelResolutionXY"], dtype=float)
        # pixelResolutionXY = [X_px, Y_lines], with version-specific saved Y.
        x_px = int(round(px[0]))
        y_px = int(round(px[1] / (1 if native_saved_rows else line_avg)))
        dims.append([x_px, y_px])
        centers.append(np.asarray(sf["centerXY"], dtype=float) * obj_res)
        sizes.append(np.asarray(sf["sizeXY"], dtype=float) * obj_res)

    if not dims:
        raise RuntimeError(f"No enabled ROIs in {path}")

    dims_xy = np.asarray(dims, dtype=int)
    center_xy_um = np.round(np.asarray(centers, dtype=float), 1)
    size_xy_um = np.round(np.asarray(sizes, dtype=float), 1)
    skip = sf_skip_lines(
        frame_data, len(dims), int(total_y), dims_xy[:, 1]
    )
    pixel_size_um = float(size_xy_um[0, 0] / dims_xy[0, 0])

    return MroiGeometry(
        dims_xy=dims_xy,
        center_xy_um=center_xy_um,
        size_xy_um=size_xy_um,
        sf_skip=skip,
        total_y=int(total_y),
        line_averaging=line_avg,
        pixel_size_um=pixel_size_um,
        objective_resolution=obj_res,
    )


def _annotation_field(frame_data: dict, actuator: str, field: str) -> np.ndarray:
    key = f"SI.xroiProps.Annotations.{actuator}.{field}"
    if key not in frame_data:
        return np.array([], dtype=np.int64)
    # A one-element annotation is written as a bare scalar, which asarray turns
    # into a 0-d array that indexes as ``arr[0] -> IndexError``. Every caller
    # here wants a sequence of entries, so normalise the rank.
    return np.atleast_1d(np.asarray(frame_data[key], dtype=np.int64))


def annotation_line_period(
    line_start: np.ndarray,
    line_end: np.ndarray,
) -> int:
    """Waveform samples per acquired line, from the annotation indices.

    This is the spacing between consecutive ``LineStart`` entries. It cannot be
    recovered from a ROI's ``ROIStart``/``ROIEnd`` span, because that span also
    covers flyto (and, for the last ROI, frame flyback) dead time during which
    no line is stored. Returns 0 when neither field can supply it.
    """
    ls = np.atleast_1d(np.asarray(line_start, dtype=np.int64))
    le = np.atleast_1d(np.asarray(line_end, dtype=np.int64))
    if ls.size > 1:
        steps = np.diff(ls)
        steps = steps[steps > 0]
        if steps.size:
            return int(np.median(steps))
    if ls.size and le.size:
        span = int(le[0]) - int(ls[0]) + 1
        if span > 0:
            return span
    return 0


def acq_line_mask_from_annotations(
    frame_data: dict,
    roi_height: int,
    *,
    actuator: str = "Beam",
    annotation_roi_index: int = 0,
) -> Optional[np.ndarray]:
    """Boolean (H,) mask of valid acquisition lines inside one scanfield tile.

    Uses ``SI.xroiProps.Annotations.{actuator}`` LineStart/LineEnd and
    FrameFlyBackStart/FrameFlyBackEnd, mapped from waveform sample indices onto
    the ``roi_height`` lines stored per scanfield on disk. Returns ``None`` when
    annotation fields are absent.

    Sample indices are converted to stored-line indices by dividing by the line
    period (see :func:`annotation_line_period`), NOT by ``span / roi_height``:
    a ROI's annotated span also covers flyto dead time, so that ratio
    overestimates the line period and folds every stored line into the top of
    the tile -- marking the rest blank when it is real data.
    """
    roi_height = int(roi_height)
    if roi_height <= 0:
        raise ValueError(f"roi_height must be positive, got {roi_height}")

    roi_start = _annotation_field(frame_data, actuator, "ROIStart")
    roi_end = _annotation_field(frame_data, actuator, "ROIEnd")
    line_start = _annotation_field(frame_data, actuator, "LineStart")
    line_end = _annotation_field(frame_data, actuator, "LineEnd")
    if roi_start.size == 0 or line_start.size == 0:
        return None

    i = int(annotation_roi_index)
    if i < 0 or i >= roi_start.size:
        raise IndexError(
            f"annotation_roi_index {i} out of range for {actuator} "
            f"(n={roi_start.size})"
        )

    rs = int(roi_start[i])
    re = int(roi_end[i])
    span = re - rs + 1
    if span <= 0:
        return None

    period = annotation_line_period(line_start, line_end)
    if period <= 0:
        return None
    mask = np.zeros(roi_height, dtype=bool)

    in_roi = (line_start >= rs) & (line_start <= re)
    rel_start = line_start[in_roi] - rs
    page_bins = rel_start // period
    # Lines past the tile belong to a later scanfield sharing this annotation
    # entry; drop them rather than clipping, which would pile them onto the
    # last stored line.
    page_bins = page_bins[(page_bins >= 0) & (page_bins < roi_height)]
    mask[page_bins] = True

    fb_start = _annotation_field(frame_data, actuator, "FrameFlyBackStart")
    fb_end = _annotation_field(frame_data, actuator, "FrameFlyBackEnd")
    if fb_start.size > i and fb_end.size > i:
        y0 = (int(fb_start[i]) - rs) // period
        y1 = (int(fb_end[i]) - rs) // period
        y0 = max(int(y0), 0)
        y1 = min(int(y1), roi_height - 1)
        if y1 >= y0:
            mask[y0 : y1 + 1] = False

    if not mask.any():
        return None
    return mask


def erode_line_mask(mask: np.ndarray, margin: int) -> np.ndarray:
    """Blank ``margin`` extra lines on each side of every blanked run.

    The xroiProps annotations mark which lines were being acquired, but the
    lines bordering a blanked run are usually partially blanked -- the beam is
    already ramping. Dropping a line or two past each boundary removes them
    without touching low-signal lines elsewhere, which an intensity threshold
    would discard.
    """
    m = np.asarray(mask, dtype=bool)
    margin = int(margin)
    if margin <= 0 or m.size == 0:
        return m
    out = m.copy()
    for shift in range(1, margin + 1):
        out[shift:] &= m[:-shift]   # blanked run grows forward
        out[:-shift] &= m[shift:]   # and backward
    return out


def shrink_line_mask(mask: np.ndarray, spatial_bin_factor: int) -> np.ndarray:
    """OR-merge a line mask after mean-spatial-binning by ``factor``."""
    factor = int(spatial_bin_factor)
    mask = np.asarray(mask, dtype=bool)
    if factor <= 1:
        return mask
    h = (mask.shape[0] // factor) * factor
    if h == 0:
        return mask[:0]
    return mask[:h].reshape(-1, factor).any(axis=1)


def annotation_describes_geometry(
    frame_data: dict,
    geometry: MroiGeometry,
    *,
    actuator: str = "Beam",
) -> Tuple[bool, str]:
    """Does the xroiProps annotation describe THIS acquisition?

    ``hSI.xroiProps`` is written into every ScanImage header, but it is only
    refreshed while the XROI waveform filter is armed. Once XROI is disarmed the
    struct keeps its last built value and the header then carries a well-formed
    description of a DIFFERENT acquisition -- there is no staleness flag to read.
    Both checks below are things the annotation cannot get wrong when it really
    describes the page: one ROI entry per scanfield, and an active extent equal
    to the page height.

    Returns ``(ok, reason)``; ``reason`` is empty when ``ok``.
    """
    f = lambda name: _annotation_field(frame_data, actuator, name)
    roi_start, line_start = f("ROIStart"), f("LineStart")
    fb_start = f("FrameFlyBackStart")
    if roi_start.size == 0 or line_start.size == 0:
        return False, "no xroiProps line annotations in the header"

    period = annotation_line_period(line_start, f("LineEnd"))
    if period <= 0:
        return False, "cannot derive the annotation line period"
    if fb_start.size == 0:
        return False, "no FrameFlyBackStart, so the frame extent is unknown"

    # One frame's worth: ROI entries up to the first frame's flyback.
    n_roi_entries = int((roi_start <= int(fb_start[0])).sum())
    if n_roi_entries != geometry.n_rois:
        return False, (
            f"annotation covers {n_roi_entries} scanfield(s) per frame but the "
            f"page holds {geometry.n_rois}"
        )

    active_lines = (int(fb_start[0]) - int(roi_start[0])) // period
    if active_lines != geometry.total_y:
        return False, (
            f"annotation describes {active_lines} active lines per frame but "
            f"the page is {geometry.total_y} lines"
        )
    return True, ""


def line_masks_for_geometry(
    tiff_path: PathLike,
    geometry: MroiGeometry,
    *,
    actuator: str = "Beam",
) -> List[Optional[np.ndarray]]:
    """Per-scanfield acquisition line masks from xroiProps + optional intensity gate.

    Flyto gaps between tiles are already excluded by :func:`roi_y_slices`; this
    targets blank/frame-flyback lines still present inside each ``roi_height``
    band on disk.

    Annotations that do not describe this page are DISCARDED (with a warning),
    not applied: they would carve real signal out of every tile, and nothing
    downstream can tell the difference because the surviving lines are real.
    Geometry alone is sufficient for a fully-acquired page.
    """
    path = str(tiff_path)
    sm = read_si_metadata(path)
    if not sm or "FrameData" not in sm:
        return [None] * geometry.n_rois
    frame_data = sm["FrameData"]

    ok, reason = annotation_describes_geometry(
        frame_data, geometry, actuator=actuator
    )
    if not ok:
        warnings.warn(
            f"{Path(path).name}: ignoring {actuator} line annotations -- "
            f"{reason}. Every stored line is treated as acquired; flyto gaps "
            f"are still excluded geometrically.",
            StaleAnnotationWarning,
            stacklevel=2,
        )
        return [None] * geometry.n_rois

    n_ann = _annotation_field(frame_data, actuator, "ROIStart").size
    masks: List[Optional[np.ndarray]] = []
    for roi_index in range(geometry.n_rois):
        ann_i = min(roi_index, n_ann - 1) if n_ann > 0 else 0
        mask = acq_line_mask_from_annotations(
            frame_data,
            int(geometry.roi_heights[roi_index]),
            actuator=actuator,
            annotation_roi_index=ann_i,
        )
        masks.append(mask)
    return masks


def roi_y_slices(geometry: MroiGeometry) -> List[Tuple[int, int, int]]:
    """Return ``(roi_index, y0, y1)`` bands with flyto gaps skipped between ROIs."""
    out: List[Tuple[int, int, int]] = []
    sl = 0
    heights = geometry.roi_heights
    for i, h in enumerate(heights):
        y0 = sl
        y1 = sl + int(h)
        if y1 > geometry.total_y:
            raise ValueError(
                f"ROI {i} slice [{y0}:{y1}] exceeds total_y={geometry.total_y}"
            )
        out.append((i, y0, y1))
        sl = y1
        if i < len(heights) - 1:
            sl += geometry.sf_skip
    if sl != geometry.total_y:
        raise ValueError(
            f"ROI slices end at Y={sl}, expected total_y={geometry.total_y}"
        )
    return out


# --------------------------------------------------------------------------- #
# Spatial reassembly
#
# On disk, MROI scanfields are stacked along Y with flyto blank lines between
# them, in acquisition order -- which has nothing to do with where they sat in
# the sample. Two ROIs imaged SIDE BY SIDE come back one above the other. These
# helpers put them back where they belong, using the ROI centres and sizes in
# microns that ScanImage already records.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MosaicLayout:
    """Where each ROI tile belongs on a reassembled canvas.

    Attributes
    ----------
    shape
        ``(H, W)`` of the canvas in pixels.
    origins
        ``(N, 2)`` int array of ``(row, col)`` top-left pixel for each ROI, in
        the same order as :class:`MroiGeometry`.
    pixel_size_xy_um
        ``(x_um_per_px, y_um_per_px)`` the layout was computed at.
    origin_um
        ``(x_um, y_um)`` sample coordinate of canvas pixel ``(0, 0)``, so a
        caller can map canvas pixels back to stage coordinates.
    """

    shape: Tuple[int, int]
    origins: np.ndarray
    pixel_size_xy_um: Tuple[float, float]
    origin_um: Tuple[float, float]

    @property
    def n_rois(self) -> int:
        return int(self.origins.shape[0])


def _tile_pixel_sizes(geometry: MroiGeometry) -> np.ndarray:
    """Per-ROI ``(x_um_per_px, y_um_per_px)``; shape (N, 2)."""
    dims = geometry.dims_xy.astype(float)
    if np.any(dims <= 0):
        raise ValueError(f"non-positive pixel dims in geometry: {geometry.dims_xy}")
    return geometry.size_xy_um.astype(float) / dims


def mroi_mosaic_layout(
    geometry: MroiGeometry,
    *,
    pixel_size_rtol: float = 0.02,
) -> MosaicLayout:
    """Compute where each ROI tile sits on a common canvas.

    Tiles are placed by their recorded centre and size in microns, so ROIs that
    were side by side in the sample end up side by side here, whatever order
    they were scanned in.

    A mosaic only means anything if the tiles share a sampling density -- one
    canvas has one pixel size. Tiles differing by more than ``pixel_size_rtol``
    are rejected rather than silently resampled or mis-scaled.

    Raises
    ------
    ValueError
        If tiles disagree on pixel size beyond ``pixel_size_rtol``.
    """
    px = _tile_pixel_sizes(geometry)
    px_ref = px[0]
    if not np.allclose(px, px_ref, rtol=pixel_size_rtol, atol=0.0):
        detail = ", ".join(
            f"roi{i}={p[0]:.4g}x{p[1]:.4g}" for i, p in enumerate(px)
        )
        raise ValueError(
            "ROIs disagree on pixel size, so they cannot share one canvas "
            f"(rtol={pixel_size_rtol}): {detail} um/px. "
            "Use split_rois() to keep them as separate images."
        )
    x_um_px, y_um_px = float(px_ref[0]), float(px_ref[1])

    half = geometry.size_xy_um.astype(float) / 2.0
    lo = geometry.center_xy_um.astype(float) - half     # (N, 2) top-left in um
    hi = geometry.center_xy_um.astype(float) + half

    origin_x, origin_y = float(lo[:, 0].min()), float(lo[:, 1].min())

    cols = np.round((lo[:, 0] - origin_x) / x_um_px).astype(int)
    rows = np.round((lo[:, 1] - origin_y) / y_um_px).astype(int)
    origins = np.stack([rows, cols], axis=1)

    # Size the canvas from the tiles as PLACED, so it always contains them --
    # deriving it from the micron extent alone can round a pixel short.
    height = int((origins[:, 0] + geometry.dims_xy[:, 1]).max())
    width = int((origins[:, 1] + geometry.dims_xy[:, 0]).max())

    return MosaicLayout(
        shape=(height, width),
        origins=origins,
        pixel_size_xy_um=(x_um_px, y_um_px),
        origin_um=(origin_x, origin_y),
    )


def split_rois(page: np.ndarray, geometry: MroiGeometry) -> List[np.ndarray]:
    """Cut one MROI page into its ROI tiles, dropping the flyto gaps.

    ``page`` may have any leading axes; the last two are ``(Y, X)``. Returns one
    array per ROI, each ``(..., roi_height, roi_width)``.
    """
    if page.ndim < 2:
        raise ValueError(f"page needs at least 2 dims (Y, X); got {page.shape}")
    page_y, page_x = page.shape[-2], page.shape[-1]
    if page_y != geometry.total_y:
        raise ValueError(
            f"page Y={page_y} does not match geometry.total_y={geometry.total_y}"
        )

    tiles: List[np.ndarray] = []
    for i, y0, y1 in roi_y_slices(geometry):
        x_px = int(geometry.dims_xy[i, 0])
        if x_px > page_x:
            raise ValueError(
                f"ROI {i} width {x_px} exceeds page width {page_x}"
            )
        tiles.append(page[..., y0:y1, :x_px])
    return tiles


def assemble_mosaic(
    page: np.ndarray,
    geometry: MroiGeometry,
    layout: Optional[MosaicLayout] = None,
    *,
    fill_value: int = 0,
) -> np.ndarray:
    """Paste an MROI page's tiles onto a canvas at their true sample positions.

    ``page`` may have any leading axes; the last two are ``(Y, X)``. The result
    is ``(..., H, W)`` for the ``layout`` canvas. Pixels no ROI covers are
    ``fill_value`` -- ROIs need not tile the canvas exactly, and gaps between
    them are real, not padding.
    """
    if layout is None:
        layout = mroi_mosaic_layout(geometry)

    tiles = split_rois(page, geometry)
    lead = page.shape[:-2]
    out = np.full(lead + layout.shape, fill_value, dtype=page.dtype)

    for i, tile in enumerate(tiles):
        row, col = int(layout.origins[i, 0]), int(layout.origins[i, 1])
        h, w = tile.shape[-2], tile.shape[-1]
        out[..., row : row + h, col : col + w] = tile
    return out


def _prefer_gapped(tiff_path: str) -> bool:
    """True when MemmapTiffSI would collapse multi-frame time into T=1 (or 0)."""
    n_pages = get_si_tiff_n_pages(tiff_path)
    if n_pages <= 1:
        return False
    try:
        mm = MemmapTiffSI(tiff_path, allow_truncated=True)
    except Exception:
        return True
    # Truncated / mis-parsed TZ layout: use GappedMemmap for page-as-time.
    if mm.shape[0] <= 1 and n_pages > max(mm.n_zplanes, 1) * mm.n_channels:
        return True
    return False


def open_si_stack(tiff_path: PathLike):
    """Open a ScanImage TIFF for time-series access.

    Returns
    -------
    reader
        ``GappedMemmap`` with shape ``(T, C, Y, X)`` or ``MemmapTiffSI`` with
        shape ``(T, Z, C, Y, X)``.
    layout : str
        ``'gapped'`` or ``'memmap'``.
    """
    path = str(tiff_path)
    if _prefer_gapped(path):
        return GappedMemmap(path, method="scanimage"), "gapped"
    return MemmapTiffSI(path, allow_truncated=True), "memmap"


def stack_tyx(reader, layout: str, *, channel: int = 0, z: int = 0):
    """Return a ``(T, Y, X)`` view/array for one channel (and z if memmap)."""
    if layout == "gapped":
        # (T, C, Y, X)
        return reader[:, channel, :, :]
    # MemmapTiffSI: (T, Z, C, Y, X)
    return reader[:, z, channel, :, :]
