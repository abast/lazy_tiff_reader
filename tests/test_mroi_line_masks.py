"""Tests for the xroiProps annotation -> stored-line mask mapping.

The annotations index the scan WAVEFORM in samples; the mask indexes the LINES
ScanImage actually stored. Converting between them needs the line period. A
ROI's annotated span is not a substitute: it also covers the flyto dead time
after the tile (and, for the last ROI, the frame flyback), so using
``span / roi_height`` as the period overestimates it by the flyto fraction and
folds every stored line into the top of the tile. The rest of the tile then
reads as blank and is dropped from the statistics -- silently, because the
surviving lines are real and pass every downstream sanity check.

That happened on real dye-pool acquisitions: 102-line tiles kept 68 lines
(40 for the last ROI), discarding a third of the signal, and the ROIs ramp
across Y so the loss biased the profiles rather than merely shrinking N.

Everything here is synthetic -- no acquisition files, so it runs anywhere.
"""

from __future__ import annotations

import numpy as np
import pytest

from lazy_tiff_reader.mroi_si_tiff import (
    MroiGeometry,
    acq_line_mask_from_annotations,
    annotation_describes_geometry,
    annotation_line_period,
)

# Geometry of the real acquisitions these tests were written against.
PERIOD = 14        # waveform samples per acquired line
ROI_HEIGHT = 102   # lines stored per scanfield
GAP_LINES = 50     # flyto lines between scanfields (not stored)
N_ROIS = 5
FLYBACK = 2212     # frame flyback, samples, after the last scanfield


def build_annotations(
    *,
    n_rois=N_ROIS,
    roi_height=ROI_HEIGHT,
    period=PERIOD,
    gap_lines=GAP_LINES,
    flyback=FLYBACK,
    drop_lines=(),
):
    """FrameData annotations for one frame of an ``n_rois`` MROI acquisition.

    ``drop_lines`` omits those stored-line indices of every tile from
    ``LineStart``, standing in for lines that were genuinely not acquired.
    """
    span = (roi_height + gap_lines) * period
    roi_start, roi_end, line_start = [], [], []
    for i in range(n_rois):
        rs = 1 + i * span
        roi_start.append(rs)
        # The last scanfield's span runs to the end of the frame flyback.
        roi_end.append(
            rs + roi_height * period + flyback - 1
            if i == n_rois - 1
            else rs + span - 1
        )
        line_start += [
            rs + k * period for k in range(roi_height) if k not in set(drop_lines)
        ]
    fb_start = 1 + (n_rois - 1) * span + roi_height * period
    return {
        "SI.xroiProps.Annotations.Beam.ROIStart": roi_start,
        "SI.xroiProps.Annotations.Beam.ROIEnd": roi_end,
        "SI.xroiProps.Annotations.Beam.LineStart": line_start,
        "SI.xroiProps.Annotations.Beam.LineEnd": [
            s + period - 1 for s in line_start
        ],
        "SI.xroiProps.Annotations.Beam.FrameFlyBackStart": [fb_start],
        "SI.xroiProps.Annotations.Beam.FrameFlyBackEnd": [fb_start + flyback - 1],
    }


# --------------------------------------------------------------------------- #
# Line period
# --------------------------------------------------------------------------- #


def test_line_period_comes_from_line_spacing_not_the_roi_span():
    fd = build_annotations()
    ls = fd["SI.xroiProps.Annotations.Beam.LineStart"]
    le = fd["SI.xroiProps.Annotations.Beam.LineEnd"]
    assert annotation_line_period(ls, le) == PERIOD

    # The span-based figure the old code used is inflated by the flyto lines.
    span = ROI_HEIGHT * PERIOD + GAP_LINES * PERIOD
    assert span / ROI_HEIGHT == pytest.approx(20.86, abs=0.01)


def test_line_period_falls_back_to_one_line_length():
    assert annotation_line_period([1], [PERIOD]) == PERIOD


def test_line_period_is_zero_when_nothing_can_supply_it():
    assert annotation_line_period([], []) == 0


# --------------------------------------------------------------------------- #
# The regression: every stored line of every tile is valid
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("roi_index", range(N_ROIS))
def test_fully_acquired_tile_keeps_every_stored_line(roi_index):
    mask = acq_line_mask_from_annotations(
        build_annotations(), ROI_HEIGHT, annotation_roi_index=roi_index
    )
    assert mask is not None
    assert mask.all(), (
        f"ROI {roi_index}: dropped lines {np.where(~mask)[0].tolist()} "
        "that were all acquired"
    )


def test_the_bottom_of_the_tile_is_not_mistaken_for_flyto():
    """The old span-based scale kept only lines 0..67 of 102."""
    mask = acq_line_mask_from_annotations(build_annotations(), ROI_HEIGHT)
    assert mask[68:].all()
    assert mask[ROI_HEIGHT - 1]


def test_last_roi_is_not_penalised_for_the_frame_flyback():
    """The last ROI's span swallows the flyback; it kept only 40 of 102."""
    mask = acq_line_mask_from_annotations(
        build_annotations(), ROI_HEIGHT, annotation_roi_index=N_ROIS - 1
    )
    assert int(mask.sum()) == ROI_HEIGHT


# --------------------------------------------------------------------------- #
# Lines that really were not acquired
# --------------------------------------------------------------------------- #


def test_unacquired_lines_are_blanked_at_the_right_positions():
    dropped = tuple(range(40, 50))
    mask = acq_line_mask_from_annotations(
        build_annotations(drop_lines=dropped), ROI_HEIGHT
    )
    assert np.where(~mask)[0].tolist() == list(dropped)


def test_flyback_inside_a_tile_is_blanked():
    """A tile short enough that the frame flyback lands inside it."""
    fd = build_annotations(n_rois=1, roi_height=60, gap_lines=0)
    mask = acq_line_mask_from_annotations(fd, 80)
    assert mask[:60].all()
    assert not mask[60:].any()


# --------------------------------------------------------------------------- #
# One-element annotations arrive as bare scalars
# --------------------------------------------------------------------------- #


def test_scalar_annotations_do_not_crash_and_keep_real_lines():
    """A single-entry annotation is written as a scalar, not a 1-element list."""
    n_lines = 200
    fd = {
        "SI.xroiProps.Annotations.Beam.ROIStart": 1,
        "SI.xroiProps.Annotations.Beam.ROIEnd": 5012,
        "SI.xroiProps.Annotations.Beam.LineStart": [
            1 + k * PERIOD for k in range(n_lines)
        ],
        "SI.xroiProps.Annotations.Beam.LineEnd": [
            PERIOD + k * PERIOD for k in range(n_lines)
        ],
        "SI.xroiProps.Annotations.Beam.FrameFlyBackStart": 2801,
        "SI.xroiProps.Annotations.Beam.FrameFlyBackEnd": 5012,
    }
    mask = acq_line_mask_from_annotations(fd, ROI_HEIGHT)
    assert mask is not None
    assert mask.all()


def test_missing_annotations_return_none():
    assert acq_line_mask_from_annotations({}, ROI_HEIGHT) is None


def test_annotation_roi_index_out_of_range_is_an_error():
    with pytest.raises(IndexError):
        acq_line_mask_from_annotations(
            build_annotations(), ROI_HEIGHT, annotation_roi_index=N_ROIS
        )


# --------------------------------------------------------------------------- #
# Does the annotation describe THIS page at all?
#
# hSI.xroiProps is written into every header but only refreshed while the XROI
# waveform filter is armed. Disarmed, it keeps its last build -- so the header
# carries a well-formed description of a different acquisition, with no
# staleness flag. Confirmed live on 2026-08-06: the session had 5 active ROIs
# while xroiProps.mroi_specs held one ROI and the annotation 200 lines.
# --------------------------------------------------------------------------- #


def geometry(n_rois=N_ROIS, roi_height=ROI_HEIGHT, gap=GAP_LINES):
    dims = np.array([[630, roi_height]] * n_rois, dtype=int)
    return MroiGeometry(
        dims_xy=dims,
        center_xy_um=np.zeros((n_rois, 2)),
        size_xy_um=np.array([[630.0, float(roi_height)]] * n_rois),
        sf_skip=gap,
        total_y=n_rois * roi_height + (n_rois - 1) * gap,
        line_averaging=1,
        pixel_size_um=1.0,
        objective_resolution=161.275,
    )


def test_matching_annotation_is_accepted():
    ok, reason = annotation_describes_geometry(build_annotations(), geometry())
    assert ok, reason
    assert reason == ""


def test_annotation_for_a_different_roi_count_is_rejected():
    """The real 260806 case: one 200-line scanfield vs a 5-tile page."""
    stale = {
        "SI.xroiProps.Annotations.Beam.ROIStart": 1,
        "SI.xroiProps.Annotations.Beam.ROIEnd": 5012,
        "SI.xroiProps.Annotations.Beam.LineStart": [
            1 + k * PERIOD for k in range(200)
        ],
        "SI.xroiProps.Annotations.Beam.LineEnd": [
            PERIOD + k * PERIOD for k in range(200)
        ],
        "SI.xroiProps.Annotations.Beam.FrameFlyBackStart": 2801,
        "SI.xroiProps.Annotations.Beam.FrameFlyBackEnd": 5012,
    }
    ok, reason = annotation_describes_geometry(stale, geometry())
    assert not ok
    assert "1 scanfield(s) per frame but the page holds 5" in reason


def test_annotation_with_the_wrong_line_count_is_rejected():
    """Right number of tiles, wrong tile height -- still not this page."""
    fd = build_annotations(roi_height=60)
    ok, reason = annotation_describes_geometry(fd, geometry(roi_height=102))
    assert not ok
    assert "active lines per frame" in reason


def test_absent_annotation_is_rejected_with_a_reason():
    ok, reason = annotation_describes_geometry({}, geometry())
    assert not ok
    assert "no xroiProps line annotations" in reason
