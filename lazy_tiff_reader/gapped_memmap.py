"""
GappedMemmap: Memory-mapped access to TIFF files with IFD gaps

Handles ScanImage TIFFs where frame data is separated by IFD metadata,
making the data appear as a contiguous numpy array without loading into RAM.
"""

import os
import numpy as np
import tifffile
from .slices_to_offsets import slices_to_offsets
from .utils.copy_byte_spans_between_buffers import copy_byte_spans_between_buffers
from .utils.get_si_tiff_n_pages import get_si_tiff_n_pages


class GappedMemmap:
    """
    Memory-mapped access to TIFF files with IFD gaps between frames.

    Provides efficient random access to large TIFF files by computing exact
    byte offsets and reading only the requested data, avoiding expensive
    copies and unnecessary I/O.

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

        # Indexing returns numpy arrays:
        frame = data[100]                    # Single frame, all channels: (C, Y, X)
        frames = data[100:200]               # Frame range: (T, C, Y, X)

        # Channel indexing:
        frame_ch0 = data[100, 0]             # Frame 100, channel 0: (Y, X)
        frames_ch0 = data[100:200, 0]        # Frames 100-200, channel 0: (T, Y, X)

        # Spatial ROI (very efficient - reads only requested region):
        roi = data[50, :, :100, :100]        # Frame 50, all channels, spatial ROI: (C, 100, 100)
        roi_ch1 = data[50, 1, :100, :100]    # Frame 50, channel 1, spatial ROI: (100, 100)

    Note:
        GappedMemmap is NOT a numpy.ndarray subclass. Indexing returns numpy arrays,
        but operations like `np.mean(data)` require explicit slicing: `np.mean(data[:])`.
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
        # Handle both single channel (int) and multi-channel (list)
        if isinstance(channels_saved, (list, tuple)):
            nchannels = len(channels_saved)
        else:
            nchannels = 1  # Single channel stored as int

        # Calculate page count using efficient validated method
        # This validates file structure: (file_size - ifd_offset_0) % stride == 0
        npages = get_si_tiff_n_pages(tiff_path)

        # Calculate offsets from pattern (much faster than iterating all pages)
        # ScanImage TIFFs have consistent gaps between pages
        offset_0 = tif.pages[0].dataoffsets[0]
        size_0 = tif.pages[0].databytecounts[0]
        offset_1 = tif.pages[1].dataoffsets[0]
        stride = offset_1 - offset_0  # page_size + gap

        # Calculate shape from page count and channels
        nframes = npages // nchannels
        final_shape = (nframes, nchannels, page_shape[0], page_shape[1])
        axes = 'ZCYX'  # ScanImage convention

        # Generate all offsets: offset[n] = offset[0] + n * stride
        offsets = [offset_0 + i * stride for i in range(npages)]
        sizes = [size_0] * npages

        return final_shape, nchannels, offsets, sizes, axes

    def __init__(self, tiff_path, method='auto'):
        """
        Initialize a GappedMemmap instance.

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
                    final_shape, nchannels, offsets, sizes, axes = self._get_metadata_from_scanimage(
                        tif, tiff_path, page_shape
                    )
                else:
                    # Fall back to series
                    series = tif.series[0]
                    final_shape, nchannels, offsets, sizes, axes = self._get_metadata_from_series(tif, series)

            elif method == 'scanimage':
                final_shape, nchannels, offsets, sizes, axes = self._get_metadata_from_scanimage(
                    tif, tiff_path, page_shape
                )

            elif method == 'series':
                series = tif.series[0]
                final_shape, nchannels, offsets, sizes, axes = self._get_metadata_from_series(tif, series)

            else:
                raise ValueError(f"Unknown method '{method}'. Must be 'auto', 'scanimage', or 'series'")

        # Store metadata as public attributes
        self.shape = final_shape
        self.dtype = dtype
        self._tiff_path = tiff_path
        self._offsets = np.array(offsets)
        self._sizes = np.array(sizes)
        self._page_shape = page_shape
        self._nchannels = nchannels
        self._axes = axes
        self._mmap = None  # Will hold the memmap of entire file
        self._bytes_read = 0  # Track IO for efficiency testing


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

    def __getitem__(self, key):
        """
        Extract data by computing exact byte offsets and copying bytes.

        This implementation computes exact byte offsets before reading,
        enabling efficient spatial slicing and pixel access.

        Returns
        -------
        np.ndarray
            Requested data as a regular numpy array

        Examples
        --------
        >>> gm[0]                    # Single frame: (C, Y, X)
        >>> gm[0:10]                 # Frame range: (T, C, Y, X)
        >>> gm[0, 0]                 # Frame 0, channel 0: (Y, X)
        >>> gm[:, 0, 0, 0]           # Single pixel from all frames: (T,)
        >>> gm[0, 0, :100, :100]     # Spatial ROI: (100, 100)
        """
        self._ensure_mmap()

        # Step 1: Normalize key to 4-tuple (frame, channel, y, x)
        normalized_key = self._normalize_key(key)

        # Step 2: Compute output shape
        output_shape = self._compute_output_shape(normalized_key)

        # Step 3: Compute byte offsets using standalone function
        offsets_in, offsets_out, lengths = slices_to_offsets(
            key, self._offsets, self.shape, self._nchannels, self.dtype.itemsize
        )

        # Step 4: Allocate buffer and copy
        output = np.empty(output_shape, dtype=self.dtype, order='C')

        # Create uint8 view of output for byte-level copying
        output_bytes = output.view('uint8').ravel()
        copy_byte_spans_between_buffers(self._mmap, output_bytes, offsets_in, offsets_out, lengths)

        return output

    def __repr__(self):
        return (f"GappedMemmap(shape={self.shape}, dtype={self.dtype}, "
                f"file='{self._tiff_path}')")

    def _normalize_key(self, key):
        """
        Convert any indexing key to a 4-tuple of (frame_key, channel_key, y_key, x_key).

        Each element can be: int, slice, list, np.ndarray, or None

        Examples
        --------
        >>> gm._normalize_key(5)
        (5, slice(None), slice(None), slice(None))

        >>> gm._normalize_key((slice(0, 10), 0))
        (slice(0, 10), 0, slice(None), slice(None))
        """
        # Convert to tuple
        if not isinstance(key, tuple):
            key = (key,)

        # Pad with slice(None) to length 4
        key = key + (slice(None),) * (4 - len(key))

        # Validate length
        if len(key) > 4:
            raise IndexError(f"Too many indices: {len(key)} (maximum 4)")

        return key

    def _expand_key(self, key, max_size):
        """
        Convert key to list of indices.

        Examples
        --------
        >>> _expand_key(5, 100)
        [5]

        >>> _expand_key(slice(0, 10, 2), 100)
        [0, 2, 4, 6, 8]

        >>> _expand_key([1, 3, 5], 100)
        [1, 3, 5]
        """
        if isinstance(key, int):
            # Handle negative indices
            if key < 0:
                key = max_size + key
            if key < 0 or key >= max_size:
                raise IndexError(f"Index {key} out of bounds for size {max_size}")
            return [key]
        elif isinstance(key, slice):
            return list(range(*key.indices(max_size)))
        elif isinstance(key, (list, np.ndarray)):
            indices = np.asarray(key)
            # Handle negative indices
            indices = np.where(indices < 0, max_size + indices, indices)
            # Validate bounds
            if np.any(indices < 0) or np.any(indices >= max_size):
                raise IndexError(f"Index out of bounds for size {max_size}")
            return indices.tolist()
        else:
            raise TypeError(f"Unsupported key type: {type(key)}")

    def _compute_output_shape(self, key):
        """
        Compute output array shape from normalized key.

        Parameters
        ----------
        key : tuple
            4-element tuple (frame_key, channel_key, y_key, x_key)

        Returns
        -------
        tuple
            Output shape (some dimensions may be dropped for int indices)
        """
        dims = []
        for i, (idx, max_size) in enumerate(zip(key, self.shape)):
            if isinstance(idx, int):
                # Int index: dimension is dropped
                continue
            elif isinstance(idx, slice):
                # Slice: compute length
                start, stop, step = idx.indices(max_size)
                length = len(range(start, stop, step))
                dims.append(length)
            elif isinstance(idx, (list, np.ndarray)):
                # Fancy indexing: length is number of elements
                dims.append(len(idx))
            else:
                raise TypeError(f"Unsupported index type: {type(idx)}")

        return tuple(dims)

    def _is_contiguous_rows(self, y_indices, x_indices):
        """
        Check if y and x indices represent contiguous memory.

        Returns True if we can read in row-sized chunks.
        """
        # Check if x covers full row
        x_is_full = (len(x_indices) == self.shape[3] and
                     x_indices == list(range(self.shape[3])))

        if not x_is_full:
            return False

        # Check if y is contiguous
        if len(y_indices) < 2:
            return True

        y_is_contiguous = all(y_indices[i+1] == y_indices[i] + 1
                              for i in range(len(y_indices) - 1))

        return y_is_contiguous

    def _compute_row_chunks(self, page_base_offset, y_indices, x_indices,
                            dest_offset, dtype_size):
        """
        Compute chunks for contiguous row reads (OPTIMIZED PATH).

        For example: gm[0, 0, 100:200, :] reads rows 100-199 as one chunk.
        """
        width = self.shape[3]
        row_size_bytes = width * dtype_size

        # Since y is contiguous and x is full width, read as single chunk
        y_start = y_indices[0]
        num_rows = len(y_indices)

        src_offset = page_base_offset + y_start * row_size_bytes
        length = num_rows * row_size_bytes

        return [(src_offset, dest_offset, length)]

    def _compute_pixel_chunks(self, page_base_offset, y_indices, x_indices,
                              dest_offset, dtype_size):
        """
        Compute chunks for pixel-by-pixel reads.

        For example: gm[:, 0, 0, 0] reads one pixel from each page.
        """
        chunks = []
        width = self.shape[3]

        for y in y_indices:
            for x in x_indices:
                # Compute byte offset for this pixel
                pixel_offset = (y * width + x) * dtype_size
                src_offset = page_base_offset + pixel_offset

                chunks.append((src_offset, dest_offset, dtype_size))
                dest_offset += dtype_size

        return chunks

    def _compute_chunks(self, key, output_shape):
        """
        Compute list of (src_offset, dst_offset, length) byte chunks to copy.

        This is where optimization happens - detecting when we can read
        contiguous rows vs individual pixels.

        Parameters
        ----------
        key : tuple
            4-element normalized key (frame_key, channel_key, y_key, x_key)
        output_shape : tuple
            Output array shape

        Returns
        -------
        List[Tuple[int, int, int]]
            List of (source_offset, dest_offset, length) in bytes
        """
        frame_key, channel_key, y_key, x_key = key

        # Expand keys to index arrays
        frame_indices = self._expand_key(frame_key, self.shape[0])
        channel_indices = self._expand_key(channel_key, self.shape[1])
        y_indices = self._expand_key(y_key, self.shape[2])
        x_indices = self._expand_key(x_key, self.shape[3])

        # Compute chunks
        chunks = []
        dtype_size = self.dtype.itemsize

        # Iterate through output array positions
        dest_offset = 0  # Track position in output buffer

        for frame_idx in frame_indices:
            for channel_idx in channel_indices:
                # Compute page index
                if self._nchannels == 1:
                    page_idx = frame_idx
                else:
                    page_idx = frame_idx * self._nchannels + channel_idx

                page_base_offset = self._offsets[page_idx]

                # Determine if we can read full rows or need pixel-by-pixel
                if self._is_contiguous_rows(y_indices, x_indices):
                    # OPTIMIZATION: Read contiguous row chunks
                    chunks_for_page = self._compute_row_chunks(
                        page_base_offset, y_indices, x_indices, dest_offset, dtype_size
                    )
                else:
                    # FALLBACK: Read individual pixels
                    chunks_for_page = self._compute_pixel_chunks(
                        page_base_offset, y_indices, x_indices, dest_offset, dtype_size
                    )

                chunks.extend(chunks_for_page)

                # Track bytes for efficiency testing
                self._bytes_read += sum(c[2] for c in chunks_for_page)

                # Update destination offset
                dest_offset += len(y_indices) * len(x_indices) * dtype_size

        return chunks

    def close(self):
        """Close the memory-mapped file"""
        if self._mmap is not None:
            del self._mmap
            self._mmap = None

