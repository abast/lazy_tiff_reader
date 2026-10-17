"""Measured modern saved-grid semantics must not change the historical corpus."""
from pathlib import Path

import numpy as np
import pytest
import tifffile

from lazy_tiff_reader import mroi_si_tiff as reader


def metadata(major=2026, minor=1, scanner="RGG", mode="resonant"):
    return {
        "FrameData": {
            "SI.VERSION_MAJOR": major, "SI.VERSION_MINOR": minor,
            "SI.hScan2D.scannerType": scanner, "SI.hScan2D.scanMode": mode,
            "SI.hScan2D.LineAveragingLineCount": 2,
            "SI.objectiveResolution": 1,
        },
        "RoiGroups": {"imagingRoiGroup": {"rois": {"scanfields": {
            "pixelResolutionXY": [630, 1092], "centerXY": [0, 0],
            "sizeXY": [630, 1092],
        }}}},
    }


@pytest.mark.parametrize("major,minor,scanner,mode,height", [
    (2026, 1, "RGG", "resonant", 1092),
    (2023, 1, "RGG", "resonant", 546),
    (None, None, "RGG", "resonant", 546),
    (2026, 2, "RGG", "resonant", 546),
    (2026, 1, "Other", "resonant", 546),
    (2026, 1, "RGG", "linear", 546),
])
def test_saved_rows_are_metadata_qualified(monkeypatch, major, minor, scanner, mode, height):
    monkeypatch.setattr(reader, "read_si_metadata", lambda _: metadata(major, minor, scanner, mode))
    geometry = reader.load_mroi_geometry("unused", total_y=height)
    np.testing.assert_array_equal(geometry.dims_xy, [[630, height]])
    assert geometry.line_averaging == 2
    # Do not switch rules to make an inconsistent file fit.
    with pytest.raises(ValueError, match="Single-ROI height"):
        reader.load_mroi_geometry("unused", total_y=546 if height == 1092 else 1092)


def test_actual_si2026_lineavg2_saved_grid():
    path = Path("F:/data/meso4/meso4_20261016T191755/si/fresh_matrix_00001.tif")
    if not path.is_file():
        pytest.skip("actual SI 2026.1 lineavg2 rig fixture unavailable")
    frame_data = reader.read_si_metadata(path)["FrameData"]
    assert frame_data["SI.VERSION_MAJOR"] == 2026
    assert frame_data["SI.VERSION_MINOR"] == 1
    assert frame_data["SI.hScan2D.LineAveragingLineCount"] == 2
    geometry = reader.load_mroi_geometry(path)
    np.testing.assert_array_equal(geometry.dims_xy, [[630, 1092]])
    assert geometry.line_averaging == 2
    assert geometry.sf_skip == 0
    with tifffile.TiffFile(path) as tif:
        assert len(tif.pages) == 6
        for page in tif.pages:
            pixels = page.asarray()
            assert pixels.shape == (1092, 630)
            (tile,) = reader.split_rois(pixels, geometry)
            np.testing.assert_array_equal(tile, pixels)
