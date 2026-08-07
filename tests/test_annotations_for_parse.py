"""
Tests for parsing the SI.xroiProps.annotations_for staleness fingerprint.

The fingerprint is written by MROI_API's publish_annotations next to the
Annotations block; it states what the annotation claims about the page plus
when it was built, so a reader can verify agreement instead of trusting a
possibly-stale annotation. Before this whitelist entry no committed tool
could read it at all -- the parser matched Annotations.<x>.<y> but not
annotations_for.<x>.

The synthetic FrameData below is copied verbatim from a real header
(``F:/Virginia_meso/test/agent_si_annotation_check_00001.tif``, grabbed
2026-08-07), so these tests pin the actual on-disk rendering -- cell array
of chars for roi_names, char (not numeric) posix/iso timestamps -- without
needing the data share mounted. A live-file test runs when F: is available.
"""

import re
from pathlib import Path

import pytest

from lazy_tiff_reader.utils.read_si_framedata_params import (
    _ANNOTATION_PATTERN,
    _FINGERPRINT_PATTERN,
    _SI_PARAMS,
    _parse_si_value,
    read_si_framedata_params,
)

# Verbatim lines from the real header named in the module docstring.
REAL_HEADER_LINES = b"""\
SI.xroiProps.annotations_for.roi_names = {'ROI1' 'ROI2' 'ROI3' 'ROI4'}
SI.xroiProps.annotations_for.n_rois = 4
SI.xroiProps.annotations_for.lines_per_frame = 408
SI.xroiProps.annotations_for.beam_line_period_samples = 14
SI.xroiProps.annotations_for.built_at_posix = '1786121298.815'
SI.xroiProps.annotations_for.built_at_iso = '2026-08-07T12:48:18.815-04:00'
"""

LIVE_TIFF = Path(r"F:\Virginia_meso\test\agent_si_annotation_check_00001.tif")


def _scan(frame_raw):
    """Run the same combined regex pass read_si_framedata_params uses."""
    wanted = b"|".join(re.escape(k).encode() for k in _SI_PARAMS)
    pattern = (rb"^(" + wanted + rb"|" + _ANNOTATION_PATTERN
               + rb"|" + _FINGERPRINT_PATTERN
               + rb")[ \t]*=[ \t]*([^\n\r]*)")
    out = {}
    for m in re.finditer(pattern, frame_raw, re.MULTILINE):
        key = m.group(1).decode("ascii")
        out[key] = _parse_si_value(
            m.group(2).rstrip(),
            unwrap_single=".Annotations." not in key,
        )
    return out


def test_fingerprint_fields_all_extracted():
    parsed = _scan(REAL_HEADER_LINES)
    assert set(parsed) == {
        "SI.xroiProps.annotations_for.roi_names",
        "SI.xroiProps.annotations_for.n_rois",
        "SI.xroiProps.annotations_for.lines_per_frame",
        "SI.xroiProps.annotations_for.beam_line_period_samples",
        "SI.xroiProps.annotations_for.built_at_posix",
        "SI.xroiProps.annotations_for.built_at_iso",
    }


def test_roi_names_cell_array_parses_to_list_of_str():
    parsed = _scan(REAL_HEADER_LINES)
    assert parsed["SI.xroiProps.annotations_for.roi_names"] == [
        "ROI1", "ROI2", "ROI3", "ROI4",
    ]


def test_single_roi_cell_stays_a_list():
    # A one-ROI group must not collapse to a bare string: callers iterate.
    assert _parse_si_value(b"{'ROI1'}") == ["ROI1"]
    assert _parse_si_value(b"{}") == []


def test_timestamps_stay_char_exact():
    # The char rendering is the point: the header writer quantizes nested
    # NUMERIC fields to ~6 sig figs (+/-5000 s on a posix stamp), so the
    # writer emits char and the parser must hand it back exactly, not
    # coerce it through float.
    parsed = _scan(REAL_HEADER_LINES)
    assert parsed["SI.xroiProps.annotations_for.built_at_posix"] == "1786121298.815"
    assert (parsed["SI.xroiProps.annotations_for.built_at_iso"]
            == "2026-08-07T12:48:18.815-04:00")


def test_counts_parse_to_ints():
    parsed = _scan(REAL_HEADER_LINES)
    assert parsed["SI.xroiProps.annotations_for.n_rois"] == 4
    assert parsed["SI.xroiProps.annotations_for.lines_per_frame"] == 408
    assert parsed["SI.xroiProps.annotations_for.beam_line_period_samples"] == 14


@pytest.mark.skipif(not LIVE_TIFF.exists(), reason="data share not mounted")
def test_live_file_fingerprint_agrees_with_annotation():
    meta = read_si_framedata_params(str(LIVE_TIFF))
    fd = meta["FrameData"]
    fp_names = fd["SI.xroiProps.annotations_for.roi_names"]
    n_rois = fd["SI.xroiProps.annotations_for.n_rois"]
    assert n_rois == len(fp_names)
    # The fingerprint must agree with the Annotations block it describes.
    roi_starts = fd["SI.xroiProps.Annotations.Beam.ROIStart"]
    assert len(roi_starts) == n_rois
    # And the posix stamp must round-trip as written, to the millisecond.
    posix = fd["SI.xroiProps.annotations_for.built_at_posix"]
    assert isinstance(posix, str)
    float(posix)  # convertible, but only on the caller's say-so
