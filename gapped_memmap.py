"""
GappedMemmap: Memory-mapped access to TIFF files with IFD gaps

Handles ScanImage TIFFs where frame data is separated by IFD metadata,
making the data appear as a contiguous numpy array without loading into RAM.
"""

import os
import numpy as np
import tifffile


class GappedMemmap(np.ndarray):
    """
    A numpy array subclass that memory-maps a TIFF file with IFD gaps
    between frames, making it appear as a contiguous array.

    Handles both single-channel and multi-channel (interleaved) ScanImage TIFFs.
    Array shape is always 4D: (T, C, Y, X) where T=frames, C=channels.
    For single-channel TIFFs, C=1.

    Usage:
        data = GappedMemmap('/path/to/large.tif')
        # Shape is always (T, C, Y, X)

        # Control metadata extraction method:
        data_auto = GappedMemmap(path, method='auto')        # Auto-detect (default)
        data_fast = GappedMemmap(path, method='scanimage')   # Fast for ScanImage
        data_slow = GappedMemmap(path, method='series')      # Slow, works for all

        # Basic indexing:
        frame = data[100]                    # Single frame, all channels: (C, Y, X)
        frames = data[100:200]               # Frame range: (T, C, Y, X)

        # Channel indexing:
        frame_ch0 = data[100, 0]             # Frame 100, channel 0: (Y, X)
        frames_ch0 = data[100:200, 0]        # Frames 100-200, channel 0: (T, Y, X)

        # Spatial ROI:
        roi = data[50, :, :100, :100]        # Frame 50, all channels, spatial ROI: (C, 100, 100)
        roi_ch1 = data[50, 1, :100, :100]    # Frame 50, channel 1, spatial ROI: (100, 100)
    """

    @staticmethod
    def _get_metadata_from_series(tif, series):
        """Get metadata using series (slower, should work for all TIFFs)

        This method:
        - Accesses tif.series (slow for large files)
        - Iterates through all pages to get offsets (slow)
        - Works for any TIFF file format
        """
        shape = series.shape
        axes = series.axes

        # Get all page offsets and sizes
        offsets = []
        sizes = []
        for page in tif.pages:
            offsets.append(page.dataoffsets[0])
            sizes.append(page.databytecounts[0])

        # Detect if this is multi-channel interleaved data
        nchannels = 1
        if 'C' in axes:
            c_idx = axes.index('C')
            nchannels = shape[c_idx]

        # Always create 4D shape (T, C, Y, X)
        if 'C' in axes:
            # Multi-channel: shape is already correct (T, C, Y, X)
            final_shape = shape
        else:
            # Single-channel: add channel dimension
            # shape is (T, Y, X), convert to (T, 1, Y, X)
            final_shape = (shape[0], 1, shape[1], shape[2])

        return final_shape, nchannels, offsets, sizes, axes

    @staticmethod
    def _get_metadata_from_scanimage(tif, tiff_path, page_shape):
        """Get metadata using ScanImage metadata and pattern calculation (faster)

        This method assumes:
        - ScanImage metadata exists with channel info
        - IFD gaps are consistent (pattern-based offset calculation)

        Much faster than _get_metadata_from_series because:
        - Uses ScanImage metadata instead of series (~170ms saved)
        - Calculates page count from file size instead of len(tif.pages) (~170ms saved)
        - Generates offsets from pattern instead of iterating pages (~360ms saved)
        """
        si_meta = tif.scanimage_metadata
        frame_data = si_meta['FrameData']
        channels_saved = frame_data['SI.hChannels.channelSave']
        nchannels = len(channels_saved)

        # Calculate offsets from pattern (much faster than iterating all pages)
        # ScanImage TIFFs have consistent gaps between pages
        offset_0 = tif.pages[0].dataoffsets[0]
        size_0 = tif.pages[0].databytecounts[0]
        offset_1 = tif.pages[1].dataoffsets[0]
        stride = offset_1 - offset_0  # page_size + gap

        # Calculate page count from file size (much faster than len(tif.pages))
        file_size = os.path.getsize(tiff_path)
        npages = (file_size - offset_0) // stride
        if offset_0 + npages * stride + size_0 <= file_size:
            npages += 1

        # Calculate shape from page count and channels
        nframes = npages // nchannels
        final_shape = (nframes, nchannels, page_shape[0], page_shape[1])
        axes = 'ZCYX'  # ScanImage convention

        # Generate all offsets: offset[n] = offset[0] + n * stride
        offsets = [offset_0 + i * stride for i in range(npages)]
        sizes = [size_0] * npages

        return final_shape, nchannels, offsets, sizes, axes

    def __new__(cls, tiff_path, method='auto'):
        """
        Create a GappedMemmap instance.

        Parameters
        ----------
        tiff_path : str
            Path to the TIFF file
        method : {'auto', 'scanimage', 'series'}, optional
            Method for extracting metadata:
            - 'auto': Try 'scanimage' first, fall back to 'series' (default)
            - 'scanimage': Use ScanImage metadata and pattern calculation (fast, ScanImage only)
            - 'series': Use tifffile series (slower, works for all TIFFs)
        """
        # Open TIFF to get metadata
        with tifffile.TiffFile(tiff_path) as tif:
            # Get basic info from first page
            dtype = tif.pages[0].dtype
            page_shape = tif.pages[0].shape

            # Choose metadata extraction method
            if method == 'auto':
                # Try ScanImage fast path first
                if (hasattr(tif, 'scanimage_metadata') and tif.scanimage_metadata and
                    'FrameData' in tif.scanimage_metadata and
                    'SI.hChannels.channelSave' in tif.scanimage_metadata['FrameData']):
                    final_shape, nchannels, offsets, sizes, axes = cls._get_metadata_from_scanimage(
                        tif, tiff_path, page_shape
                    )
                else:
                    # Fall back to series
                    series = tif.series[0]
                    final_shape, nchannels, offsets, sizes, axes = cls._get_metadata_from_series(tif, series)

            elif method == 'scanimage':
                final_shape, nchannels, offsets, sizes, axes = cls._get_metadata_from_scanimage(
                    tif, tiff_path, page_shape
                )

            elif method == 'series':
                series = tif.series[0]
                final_shape, nchannels, offsets, sizes, axes = cls._get_metadata_from_series(tif, series)

            else:
                raise ValueError(f"Unknown method '{method}'. Must be 'auto', 'scanimage', or 'series'")

        # Create a "view" array (doesn't allocate memory)
        obj = np.ndarray.__new__(cls, final_shape, dtype)

        # Store metadata for __getitem__
        obj._tiff_path = tiff_path
        obj._offsets = np.array(offsets)
        obj._sizes = np.array(sizes)
        obj._page_shape = page_shape
        obj._nchannels = nchannels
        obj._axes = axes
        obj._dtype = dtype
        obj._mmap = None  # Will hold the memmap of entire file

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._tiff_path = getattr(obj, '_tiff_path', None)
        self._offsets = getattr(obj, '_offsets', None)
        self._sizes = getattr(obj, '_sizes', None)
        self._page_shape = getattr(obj, '_page_shape', None)
        self._nchannels = getattr(obj, '_nchannels', None)
        self._axes = getattr(obj, '_axes', None)
        self._dtype = getattr(obj, '_dtype', None)
        self._mmap = getattr(obj, '_mmap', None)

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

    def _load_page(self, page_idx):
        """Load a single TIFF page by its page index"""
        self._ensure_mmap()
        offset = self._offsets[page_idx]
        size = self._sizes[page_idx]

        # Extract page data from memmap and reshape
        page_bytes = self._mmap[offset:offset + size]
        page = np.frombuffer(page_bytes, dtype=self._dtype).reshape(self._page_shape)
        return page.copy()  # Return a copy so it's writeable

    def _load_frame(self, frame_idx):
        """
        Load a single frame with all channels, returning shape (C, Y, X)

        For single-channel data, C=1. For multi-channel, C=nchannels.
        """
        if self._nchannels == 1:
            # Single channel: load page and add channel dimension
            page = self._load_page(frame_idx)
            return page[np.newaxis, ...]  # Add channel dimension: (Y, X) -> (1, Y, X)
        else:
            # Multi-channel: pages are interleaved, load all channels
            channels = []
            for c in range(self._nchannels):
                page_idx = frame_idx * self._nchannels + c
                channels.append(self._load_page(page_idx))
            return np.stack(channels, axis=0)  # Shape: (C, Y, X)

    def __getitem__(self, key):
        """Custom indexing that extracts data while skipping IFD gaps

        Array shape is always (T, C, Y, X) where T=frames, C=channels
        """
        # Single integer index: data[100] -> (C, Y, X)
        if isinstance(key, int):
            return self._load_frame(key)

        # Single slice: data[100:200] -> (T, C, Y, X)
        if isinstance(key, slice):
            frame_indices = range(*key.indices(self.shape[0]))
            frames = [self._load_frame(i) for i in frame_indices]
            return np.stack(frames, axis=0)

        # Tuple indexing: (frame_key, channel_key, y_key, x_key)
        if isinstance(key, tuple):
            frame_key = key[0]
            remaining_key = key[1:] if len(key) > 1 else ()

            if isinstance(frame_key, int):
                # Single frame: load and apply remaining indices
                frame = self._load_frame(frame_key)  # Shape: (C, Y, X)
                if remaining_key:
                    return frame[remaining_key]
                return frame

            elif isinstance(frame_key, slice):
                # Multiple frames
                frame_indices = range(*frame_key.indices(self.shape[0]))
                frames = []
                for i in frame_indices:
                    frame = self._load_frame(i)  # Shape: (C, Y, X)
                    if remaining_key:
                        frame = frame[remaining_key]
                    frames.append(frame)
                return np.stack(frames, axis=0)

        raise NotImplementedError(f"Indexing type {type(key)} not supported")

    def __repr__(self):
        return (f"GappedMemmap(shape={self.shape}, dtype={self.dtype}, "
                f"file='{self._tiff_path}')")

    def close(self):
        """Close the memory-mapped file"""
        if self._mmap is not None:
            del self._mmap
            self._mmap = None

