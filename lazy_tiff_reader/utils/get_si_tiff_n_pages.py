"""Utility to calculate number of pages in ScanImage TIFF files."""

import os
import struct
import tifffile

# ScanImage BigTIFF custom header magic (bytes 16-19, little-endian)
_SI_BIGTIFF_MAGIC = 117637889  # 0x07030301


def _is_scanimage_bigtiff(tiff_path: str) -> bool:
    """Return True if the file has a ScanImage BigTIFF header (fast, no IFD scan)."""
    try:
        with open(tiff_path, 'rb') as f:
            hdr = f.read(4)
            if len(hdr) < 4 or hdr[:2] != b'II':
                return False
            if struct.unpack_from('<H', hdr, 2)[0] != 43:
                return False  # not BigTIFF
            f.seek(16)
            si_hdr = f.read(8)
            if len(si_hdr) < 4:
                return False
            magic = struct.unpack_from('<I', si_hdr, 0)[0]
            return magic == _SI_BIGTIFF_MAGIC
    except OSError:
        return False


def get_si_tiff_n_pages(tiff_path: str) -> int:
    """
    Calculate number of pages in a ScanImage TIFF file.

    Uses IFD-based offset calculation with strict validation.
    Requires that the file has a uniform stride pattern where
    (file_size - offset_0) % stride == 0.

    Parameters
    ----------
    tiff_path : str
        Path to ScanImage TIFF file

    Returns
    -------
    int
        Number of pages in the file

    Raises
    ------
    RuntimeError
        If (file_size - offset_0) % stride != 0, indicating file corruption
        or non-standard TIFF structure
    FileNotFoundError
        If the TIFF file does not exist
    ValueError
        If the file is not a valid TIFF or has fewer than 2 pages

    Notes
    -----
    This function uses the IFD (Image File Directory) offsets rather than
    image data offsets. This gives clean modulo arithmetic:

        file_size = offset_0 + npages × stride

    Where each page contributes exactly one stride consisting of:
        [IFD] [Tag value gap] [Image data]

    Examples
    --------
    >>> npages = get_si_tiff_n_pages('/path/to/file.tif')
    >>> print(f'File has {npages} frames')
    File has 500 frames
    """
    # Get file size
    file_size = os.path.getsize(tiff_path)

    # Check SI magic without using tif.scanimage_metadata (which hangs for
    # ScanImage 2023.1+ files with large FrameData blobs).
    if not _is_scanimage_bigtiff(tiff_path):
        raise RuntimeError("Not a scanimage tiff file")

    # Open TIFF and get IFD offsets
    with tifffile.TiffFile(tiff_path) as tif:
        # Use IFD offsets (not image data offsets)
        offset_0 = tif.pages[0].offset  # First IFD position
        try:
            offset_1 = tif.pages[1].offset  # Second IFD position
        except IndexError:
            return 1
        stride = offset_1 - offset_0    # Bytes per page

    # Validate that modulo is zero
    remainder = (file_size - offset_0) % stride

    if remainder != 0:
        raise RuntimeError(
            f"File structure validation failed: "
            f"(file_size - offset_0) % stride = {remainder} (expected 0). "
            f"This may indicate file corruption or non-standard TIFF structure. "
            f"Details: file_size={file_size:,}, offset_0={offset_0:,}, "
            f"stride={stride:,}"
        )

    # Calculate number of pages
    npages = (file_size - offset_0) // stride

    return npages
