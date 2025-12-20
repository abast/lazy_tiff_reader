"""
MemmapTiffSI: Zero-copy memory-mapped access to ScanImage TIFF files

Uses numpy stride tricks to create direct views into memory-mapped files,
eliminating all copy overhead for full-frame access.
"""

import os
import numpy as np
import tifffile
from numpy.lib.stride_tricks import as_strided
from utils.get_si_tiff_n_pages import get_si_tiff_n_pages


class MemmapTiffSI:
    """
    Zero-copy memory-mapped access to ScanImage TIFF files using stride tricks.

    This class leverages the regular structure of ScanImage TIFF files to create
    a strided numpy view directly over the memory-mapped file, avoiding all
    copying overhead present in copy-based implementations.

    Key advantages:
    - No copying: Direct views into memmap (no data copying)
    - Simpler code: Standard numpy slicing instead of offset calculation
    - Faster: 4-50x speedup for full-frame access vs copy-based approaches

    Limitations:
    - ScanImage TIFFs only (requires regular stride structure)
    - Some indexing patterns may create copies (standard numpy behavior for non-contiguous access)

    Parameters
    ----------
    tiff_path : str
        Path to ScanImage TIFF file

    Attributes
    ----------
    shape : tuple
        Shape of the data array (T, C, Y, X)
    dtype : np.dtype
        Data type of the image data
    ndim : int
        Number of dimensions (always 4)

    Examples
    --------
    >>> mm = MemmmapTiffSI('data.tif')
    >>> frame = mm[0]              # Get first frame: (C, Y, X)
    >>> frames = mm[0:10]          # Get frame range: (10, C, Y, X)
    >>> channel = mm[:, 0, :, :]   # Get channel 0 all frames: (T, Y, X)
    """

    def __init__(self, tiff_path):
        """
        Initialize MemmmapTiffSI instance.

        Parameters
        ----------
        tiff_path : str
            Path to ScanImage TIFF file
        """
        self._tiff_path = tiff_path
        self._mmap = None
        self._data = None

        # Extract metadata from TIFF
        with tifffile.TiffFile(tiff_path) as tif:
            # Get ScanImage metadata
            si_meta = tif.scanimage_metadata
            frame_data = si_meta['FrameData']
            channels_saved = frame_data['SI.hChannels.channelSave']

            # Handle both single channel (int) and multi-channel (list)
            if isinstance(channels_saved, (list, tuple)):
                self.nchannels = len(channels_saved)
            else:
                self.nchannels = 1  # Single channel stored as int

            # Get page shape and dtype
            page = tif.pages[0]
            self.height, self.width = page.shape
            self.dtype = np.dtype(page.dtype)

            # Calculate stride from first two pages
            self.data_offset_0 = page.dataoffsets[0]
            offset_1 = tif.pages[1].dataoffsets[0]
            self.stride = offset_1 - self.data_offset_0

            # Calculate number of frames
            npages = get_si_tiff_n_pages(tiff_path)
            self.nframes = npages // self.nchannels

            # Store final shape
            self._shape = (self.nframes, self.nchannels, self.height, self.width)

    def _ensure_mmap(self):
        """Lazy load the memory map of the entire file"""
        if self._mmap is None:
            # Memory map the entire file as bytes
            file_size = os.path.getsize(self._tiff_path)
            self._mmap = np.memmap(
                self._tiff_path,
                dtype='uint8',
                mode='r',
                shape=(file_size,)
            )

    def _create_strided_view(self):
        """Create strided numpy view over the memory-mapped file"""
        self._ensure_mmap()

        if self._data is not None:
            return  # Already created

        # First, view the uint8 memmap as the correct dtype
        # This converts byte-level memmap to pixel-level array
        mmap_as_dtype = self._mmap.view(self.dtype)

        # Calculate the offset in dtype units (not bytes)
        dtype_offset = self.data_offset_0 // self.dtype.itemsize

        # Calculate strides in dtype units (not bytes)
        # Strides describe how many dtype elements to skip, not bytes
        stride_in_dtype = self.stride // self.dtype.itemsize
        row_stride_in_dtype = self.width  # pixels per row

        if self.nchannels > 1:
            # Multi-channel: channels are interleaved pages
            # Page order: [F0C0, F0C1, F0C2, F0C3, F1C0, F1C1, ...]
            strides = (
                self.stride * self.nchannels,  # Frame stride: skip all channel pages (in bytes)
                self.stride,                    # Channel stride: next page (in bytes)
                self.width * self.dtype.itemsize,  # Row stride (in bytes)
                self.dtype.itemsize             # Pixel stride (in bytes)
            )
        else:
            # Single channel
            strides = (
                self.stride,                    # Frame stride: next page (in bytes)
                0,                               # Channel stride: N/A (only 1 channel)
                self.width * self.dtype.itemsize,  # Row stride (in bytes)
                self.dtype.itemsize             # Pixel stride (in bytes)
            )

        # Create strided view starting at first frame's data
        # Use the dtype-viewed memmap starting at the data offset
        self._data = as_strided(
            mmap_as_dtype[dtype_offset:],
            shape=self._shape,
            strides=strides,
            writeable=False
        )

    def __getitem__(self, key):
        """
        Get data by indexing directly into strided view.

        No copying for full-frame access. Returns direct views into memmap.

        Parameters
        ----------
        key : int, slice, or tuple
            Index/slice into (T, C, Y, X) dimensions

        Returns
        -------
        np.ndarray
            View into memory-mapped data (or copy if non-contiguous)

        Examples
        --------
        >>> mm[0]                    # Single frame: (C, Y, X)
        >>> mm[0:10]                 # Frame range: (T, C, Y, X)
        >>> mm[0, 0]                 # Frame 0, channel 0: (Y, X)
        >>> mm[:, 0, :, :]           # All frames, channel 0: (T, Y, X)
        """
        if self._data is None:
            self._create_strided_view()

        # Direct indexing into strided view - no copying for contiguous access!
        return self._data[key]

    @property
    def shape(self):
        """Shape of the data array (T, C, Y, X)"""
        return self._shape

    @property
    def ndim(self):
        """Number of dimensions (always 4)"""
        return 4

    def __repr__(self):
        return (f"MemmapTiffSI(shape={self.shape}, dtype={self.dtype}, "
                f"file='{os.path.basename(self._tiff_path)}')")
