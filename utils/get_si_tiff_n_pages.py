"""Utility to calculate number of pages in ScanImage TIFF files."""

import os
import tifffile


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

    # Open TIFF and get IFD offsets
    with tifffile.TiffFile(tiff_path) as tif:
        if tif.scanimage_metadata is None:
            raise RuntimeError("Not a scanimage tiff file")
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
