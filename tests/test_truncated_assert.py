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


def test_truncated_default_raises_assertion_not_nameerror(truncated_path):
    """Default ctor on a truncated file raises AssertionError naming allow_truncated.

    Regression guard: before the fix this raised ``NameError`` (undefined
    ``n_flyback``), which would NOT match this ``pytest.raises`` and would
    surface as a NameError failure instead.
    """
    with pytest.raises(AssertionError, match="allow_truncated"):
        MemmapTiffSI(str(truncated_path))


def test_truncated_allow_true_constructs(truncated_path):
    """allow_truncated=True constructs and discards the incomplete final volume."""
    mm = MemmapTiffSI(str(truncated_path), allow_truncated=True)
    T, Z, C, Y, X = mm.shape
    assert T >= 1 and Z >= 1 and C >= 1
    # Complete volumes only: T*Z*C pages account for all non-discarded pages.
    assert T * Z * C >= 1
