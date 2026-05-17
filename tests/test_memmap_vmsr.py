"""
Tests for MemmapTiffSI: zero-copy memory-mapped ScanImage TIFF reader.

These tests run against real ScanImage TIFF files in BASE_PATH. If the
directory is missing or empty, all tests skip (rather than fail) so the
suite is portable across machines that don't have the data share mounted.
"""

from pathlib import Path

import numpy as np
import pytest
from ScanImageTiffReader import ScanImageTiffReader as _SIReader

from lazy_tiff_reader import MemmapTiffSI


BASE_PATH = Path(r'F:\Virginia_meso\memmap_test')


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiff_path():
    """Path to a ScanImage TIFF file. Skips the test if none is available."""
    if not BASE_PATH.exists():
        pytest.skip(f"Test data directory not found: {BASE_PATH}")
    candidates = sorted(BASE_PATH.glob("*.tif"))
    if not candidates:
        pytest.skip(f"No .tif files in {BASE_PATH}")
    return candidates[0]


@pytest.fixture(scope="module")
def mm(tiff_path):
    """A constructed MemmapTiffSI, shared by all read-only tests."""
    return MemmapTiffSI(str(tiff_path))


@pytest.fixture(scope="module")
def si_reader(tiff_path):
    """
    Vidrio's official ScanImageTiffReader, used as ground truth for the
    bit-exact correctness tests. Opened once per module, closed on teardown.
    """
    with _SIReader(str(tiff_path)) as r:
        yield r


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_constructs_from_scanimage_tiff(mm):
    """Object instantiates and exposes the documented 5D shape contract."""
    assert mm.ndim == 5
    assert len(mm.shape) == 5
    T, Z, C, Y, X = mm.shape
    assert T >= 1 and Z >= 1 and C >= 1 and Y >= 1 and X >= 1
    assert np.issubdtype(mm.dtype, np.integer)


def test_repr_contains_shape_and_filename(mm, tiff_path):
    """__repr__ should be informative for interactive use."""
    r = repr(mm)
    assert "MemmapTiffSI" in r
    assert str(mm.shape) in r
    assert tiff_path.name in r


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def test_metadata_has_framedata(mm):
    """FrameData dict must be present with at least the channelSave key."""
    meta = mm.metadata
    assert isinstance(meta, dict)
    assert 'FrameData' in meta
    assert 'SI.hChannels.channelSave' in meta['FrameData']


def test_acquisition_parameters_extracted(mm):
    """At least one of the well-known acquisition parameters is populated."""
    params = mm.acquisition_parameters
    assert isinstance(params, dict)
    expected_any = {'frame_rate', 'volume_rate', 'num_slices', 'frames_per_slice'}
    assert expected_any & params.keys(), (
        f"Expected at least one of {expected_any} in acquisition_parameters, "
        f"got keys={list(params.keys())}"
    )


def test_resolution_xyz_is_3_tuple(mm):
    """resolution_xyz is always (x, y, z); populated components are positive floats."""
    res = mm.resolution_xyz
    assert isinstance(res, tuple) and len(res) == 3
    for v in res:
        assert v is None or (isinstance(v, float) and v > 0)


# ---------------------------------------------------------------------------
# Indexing shapes
# ---------------------------------------------------------------------------

def test_indexing_returns_expected_shapes(mm):
    """Standard int-indexing collapses dims as numpy does."""
    T, Z, C, Y, X = mm.shape
    assert mm[0, 0, 0].shape == (Y, X)
    assert mm[0, 0].shape == (C, Y, X)
    assert mm[0].shape == (Z, C, Y, X)
    assert mm[:].shape == mm.shape


def test_slicing_partial_T_and_C(mm):
    """Mixing slices and ints preserves slice dims and drops int dims."""
    T, Z, C, Y, X = mm.shape
    if T < 2:
        pytest.skip("Need T >= 2 for this test")
    out = mm[:2, :, :1]
    assert out.shape == (2, Z, 1, Y, X)


# ---------------------------------------------------------------------------
# Zero-copy invariant
# ---------------------------------------------------------------------------

def test_contiguous_access_is_a_view(mm):
    """A single-page index must remain a view into the underlying np.memmap."""
    arr = mm[0, 0, 0]
    base = arr
    saw_memmap = False
    # Walk the .base chain; np.memmap should appear somewhere along it.
    while base is not None:
        if isinstance(base, np.memmap):
            saw_memmap = True
            break
        base = getattr(base, 'base', None)
    assert saw_memmap, (
        "mm[0,0,0] is not backed by the underlying np.memmap "
        "(zero-copy invariant violated)"
    )


# ---------------------------------------------------------------------------
# Bit-exact correctness vs ScanImageTiffReader (Vidrio's reference reader)
# ---------------------------------------------------------------------------

def _page_index(mm, t, z, c):
    """Convert (t, z, c) into the linear page index used by ScanImage."""
    _, Z, C, _, _ = mm.shape
    return t * Z * C + z * C + c


def _ref_page(si_reader, page):
    """Read a single page from the ScanImageTiffReader ground truth."""
    return si_reader.data(beg=page, end=page + 1)[0]


def test_first_page_matches_scanimage_reader(mm, si_reader):
    """mm[0,0,0] must be bit-identical to ScanImageTiffReader's page 0."""
    np.testing.assert_array_equal(mm[0, 0, 0], _ref_page(si_reader, 0))


def test_last_page_matches_scanimage_reader(mm, si_reader):
    """mm[T-1, Z-1, C-1] must match the last page (catches off-by-one in n_volumes)."""
    T, Z, C, _, _ = mm.shape
    last_page = _page_index(mm, T - 1, Z - 1, C - 1)
    np.testing.assert_array_equal(
        mm[T - 1, Z - 1, C - 1], _ref_page(si_reader, last_page)
    )


def test_middle_page_matches_scanimage_reader(mm, si_reader):
    """A middle (t, z, c) must be bit-identical (catches stride-arithmetic bugs)."""
    T, Z, C, _, _ = mm.shape
    t, z, c = T // 2, Z // 2, C // 2
    page = _page_index(mm, t, z, c)
    np.testing.assert_array_equal(mm[t, z, c], _ref_page(si_reader, page))


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------

def test_non_scanimage_file_raises_runtimeerror(tmp_path):
    """Non-ScanImage input must raise RuntimeError, not return junk metadata."""
    bogus = tmp_path / "not_a_scanimage_file.tif"
    bogus.write_bytes(b"\x00" * 128)
    with pytest.raises(RuntimeError, match="Not a ScanImage BigTIFF"):
        MemmapTiffSI(str(bogus))


# TODO: truncation handling
# -------------------------
# Verify behavior when a file has an incomplete final volume (e.g. acquisition
# stopped mid-volume so npages % (n_zplanes * n_channels) != 0):
#   1. Default ctor (allow_truncated=False) should raise AssertionError with a
#      message naming pages_per_volume and the remainder.
#   2. allow_truncated=True should construct successfully, print a WARNING,
#      and yield shape such that n_volumes * Z * C == npages - remainder.
#
# Implementation note: do this by copying the real TIFF to tmp_path and
# truncating off `stride * k - delta` bytes (k=1, delta=a few bytes inside
# the final page) using Path.write_bytes / os.truncate. Skip if the test
# file has only one volume (truncating would leave zero complete volumes
# and trigger a different error path).
