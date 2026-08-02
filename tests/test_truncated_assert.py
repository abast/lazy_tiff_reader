"""
Regression tests for the truncated-file assertion path in MemmapTiffSI.

A truncated ScanImage acquisition has a page count that is not an exact
multiple of ``n_zplanes * n_channels * pages_per_z_cycle``. When
``allow_truncated=False`` (the default) the constructor must raise a *clear*
``AssertionError`` telling the user to pass ``allow_truncated=True``.

This locks a specific past bug: the assertion message referenced an undefined
local ``n_flyback``, so instead of the intended ``AssertionError`` the
constructor raised ``NameError: name 'n_flyback' is not defined`` on the
truncated path -- an opaque failure that also got silently swallowed by
downstream callers wrapping construction in ``except Exception``.

These tests use a real, known-truncated ScanImage file and skip when the data
share is not mounted, matching the convention in ``test_memmap_vmsr.py``.
"""

from pathlib import Path

import pytest

from lazy_tiff_reader import MemmapTiffSI


# A real acquisition that was stopped mid-volume: 82 pages, 2 complete volumes,
# 24 extra pages. npages % pages_per_volume != 0 -> hits the truncation path.
TRUNCATED_TIFF = Path(r"F:\Virginia_meso\260702_frc_test_data\file_00002.tif")


@pytest.fixture(scope="module")
def truncated_path():
    if not TRUNCATED_TIFF.exists():
        pytest.skip(f"Truncated test file not found: {TRUNCATED_TIFF}")
    return TRUNCATED_TIFF


def test_refusal_is_diagnosable_never_a_nameerror(truncated_path):
    """A file we cannot lay out must refuse with a message that says WHY.

    This is the actual regression guard, and it is deliberately not tied to one
    refusal reason. The bug it locks is that MemmapTiffSI formatted the unbound
    name ``n_flyback`` into its assert message, so the refusal raised
    ``NameError`` -- carrying no page arithmetic, matching no ``except
    AssertionError``, and getting swallowed by callers that catch Exception
    (mesotools analysis/sources.py) into a silent degrade.

    This particular file reaches a DIFFERENT refusal since frame-averaging was
    handled properly (2026-08-02): it has numFramesPerVolume = 28, which the
    logAverageFactor of 7 divides, but numFramesPerVolumeWithFlyback = 29, which
    it does not. Whether the flyback frame is averaged in, written unaveraged, or
    dropped is UNMEASURED -- see the open question in
    arco/plans/si_tiff_reading.plan.md -- so the reader refuses rather than
    guessing a layout. Guessing is what produced every bug this suite pins.
    """
    with pytest.raises((AssertionError, ValueError)) as exc:
        MemmapTiffSI(str(truncated_path))
    msg = str(exc.value)
    assert "allow_truncated" in msg or "logAverageFactor" in msg, (
        f"refusal must explain itself; got: {msg}")
    # The page arithmetic must survive into the message either way.
    assert any(tok in msg for tok in ("numFramesPerVolume", "pages_per_volume")), (
        f"refusal must carry the page arithmetic; got: {msg}")


def test_allow_truncated_does_not_paper_over_an_unknown_layout(truncated_path):
    """allow_truncated means "stopped mid-volume", never "I cannot read this".

    Discarding a remainder is only meaningful once the per-volume page count is
    known. When the layout itself is undetermined there is no remainder to
    discard, so the flag must NOT rescue the file -- otherwise a silently wrong
    volume split reaches the caller wearing a warning about truncation.
    """
    with pytest.raises(ValueError, match="logAverageFactor"):
        MemmapTiffSI(str(truncated_path), allow_truncated=True)
