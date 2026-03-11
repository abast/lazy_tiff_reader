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
    metadata : dict or None
        Full ScanImage metadata from the TIFF (tifffile scanimage_metadata). None if not SI.
    n_volumes, n_zplanes, n_channels : int
        Dimensions T, Z, C from metadata (Z=1 for 2D acquisitions).
    resolution_xyz : tuple of float
        (x, y, z) in µm per pixel from TIFF tags and SI.hStackManager.stackZStepSize.
    acquisition_parameters : dict
        Key FrameData values (e.g. frame_rate, volume_rate, z_step_size).
    shape : tuple
        Shape of the data array (T, Z, C, Y, X)
    dtype : np.dtype
        Data type of the image data
    ndim : int
        Number of dimensions (always 5)

    Examples
    --------
    >>> mm = MemmapTiffSI('data.tif')
    >>> mm.shape
    (10, 1, 2, 512, 512)   # T, Z, C, Y, X
    >>> frame = mm[0, 0]   # First volume, first Z: (C, Y, X)
    >>> mm[0, 0, 0]        # t=0, z=0, c=0: (Y, X)
    """

    def __init__(self, tiff_path):
        """
        Initialize MemmapTiffSI instance.

        Parameters
        ----------
        tiff_path : str
            Path to ScanImage TIFF file
        """
        self._tiff_path = tiff_path
        self._mmap = None
        self._data = None
        self._si_metadata = None
        self._resolution_xyz = [0.0, 0.0, 0.0]
        self._acquisition_parameters = {}

        with tifffile.TiffFile(tiff_path) as tif:
            si_meta = tif.scanimage_metadata
            if si_meta is not None:
                self._si_metadata = si_meta
            frame_data = (si_meta or {}).get('FrameData') or {}

            # Channels
            channels_saved = frame_data.get('SI.hChannels.channelSave', 1)
            if isinstance(channels_saved, (list, tuple)):
                n_channels = len(channels_saved)
            else:
                n_channels = 1
            self.n_channels = n_channels
            self.nchannels = n_channels  # alias

            # Page shape and dtype
            page = tif.pages[0]
            self.height, self.width = page.shape
            self.dtype = np.dtype(page.dtype)
            self.data_offset_0 = page.dataoffsets[0]
            offset_1 = tif.pages[1].dataoffsets[0]
            self.stride = offset_1 - self.data_offset_0

            npages = get_si_tiff_n_pages(tiff_path)

            # T, Z dimensions from metadata
            n_zplanes = frame_data.get('SI.hStackManager.actualNumSlices', 1)
            n_volumes = frame_data.get('SI.hStackManager.actualNumVolumes', 0)
            if n_volumes <= 0 or n_zplanes <= 0:
                n_zplanes = 1
                n_volumes = npages // n_channels
            else:
                if npages != n_volumes * n_zplanes * n_channels:
                    n_zplanes = 1
                    n_volumes = npages // n_channels
            self.n_zplanes = n_zplanes
            self.n_volumes = n_volumes

            self._shape = (n_volumes, n_zplanes, n_channels, self.height, self.width)

            # Resolution: x, y from TIFF tags; z from FrameData
            self._extract_resolution(tif, frame_data)

            # Acquisition parameters from FrameData
            self._acquisition_parameters = self._extract_acquisition_parameters(frame_data)

    def _extract_resolution(self, tif, frame_data):
        """Extract resolution_xyz (x, y, z) in µm from TIFF tags and FrameData."""
        try:
            p0 = tif.pages[0]
            if hasattr(p0, 'tags') and 'XResolution' in p0.tags:
                x_res = p0.tags['XResolution'].value
                y_res = p0.tags['YResolution'].value
                unit = getattr(p0.tags.get('ResolutionUnit'), 'value', 1)
                if unit == 2:  # inch
                    self._resolution_xyz[0] = 25400.0 / (float(x_res[0]) / float(x_res[1]))
                    self._resolution_xyz[1] = 25400.0 / (float(y_res[0]) / float(y_res[1]))
                elif unit == 3:  # cm
                    self._resolution_xyz[0] = 10000.0 / (float(x_res[0]) / float(x_res[1]))
                    self._resolution_xyz[1] = 10000.0 / (float(y_res[0]) / float(y_res[1]))
        except Exception:
            pass
        z_step = frame_data.get('SI.hStackManager.stackZStepSize')
        if z_step is not None:
            try:
                self._resolution_xyz[2] = float(z_step)
            except (TypeError, ValueError):
                pass

    def _extract_acquisition_parameters(self, frame_data):
        """Extract key acquisition parameters from FrameData."""
        out = {}
        keys = {
            'frame_rate': 'SI.hRoiManager.scanFrameRate',
            'volume_rate': 'SI.hRoiManager.scanVolumeRate',
            'z_step_size': 'SI.hStackManager.stackZStepSize',
        }
        for name, key in keys.items():
            if key in frame_data:
                out[name] = frame_data[key]
        return out

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
        """Create 5D strided view (T, Z, C, Y, X). Page order: t*Z*C + z*C + c."""
        self._ensure_mmap()

        if self._data is not None:
            return

        mmap_as_dtype = self._mmap.view(self.dtype)
        dtype_offset = self.data_offset_0 // self.dtype.itemsize
        itemsize = self.dtype.itemsize
        T, Z, C, H, W = self._shape
        stride_page = self.stride  # bytes per page

        # Strides in bytes: [t, z, c, y, x] -> page k = t*Z*C + z*C + c, then y, x within page
        strides = (
            Z * C * stride_page,
            C * stride_page,
            stride_page,
            W * itemsize,
            itemsize,
        )
        self._data = as_strided(
            mmap_as_dtype[dtype_offset:],
            shape=self._shape,
            strides=strides,
            writeable=False,
        )

    def __getitem__(self, key):
        """
        Get data by indexing directly into the 5D strided view.

        No copying for contiguous access. Returns views into memmap.

        Parameters
        ----------
        key : int, slice, or tuple
            Index/slice into (T, Z, C, Y, X) dimensions

        Returns
        -------
        np.ndarray
            View into memory-mapped data (or copy if non-contiguous)

        Examples
        --------
        >>> mm[0, 0]           # First volume, first Z: (C, Y, X)
        >>> mm[0, 0, 0]        # t=0, z=0, c=0: (Y, X)
        >>> mm[:, 0, :, :, :]  # All T, z=0: (T, C, Y, X)
        """
        if self._data is None:
            self._create_strided_view()
        return self._data[key]

    @property
    def metadata(self):
        """
        ScanImage metadata dict from the TIFF (parsed by tifffile from ImageDescription).
        None if the file was not recognized as ScanImage.
        """
        return self._si_metadata

    @property
    def resolution_xyz(self):
        """(x, y, z) resolution in µm per pixel from TIFF tags and stack Z step."""
        return tuple(self._resolution_xyz)

    @property
    def acquisition_parameters(self):
        """Key acquisition parameters from FrameData (e.g. frame_rate, volume_rate, z_step_size)."""
        return dict(self._acquisition_parameters)

    @property
    def shape(self):
        """Shape of the data array (T, Z, C, Y, X)."""
        return self._shape

    @property
    def ndim(self):
        """Number of dimensions (always 5)."""
        return 5

    def __repr__(self):
        return (f"MemmapTiffSI(shape={self.shape}, dtype={self.dtype}, "
                f"file='{os.path.basename(self._tiff_path)}')")
