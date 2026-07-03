"""
MemmapTiffSI: Zero-copy memory-mapped access to ScanImage TIFF files

Uses numpy stride tricks to create direct views into memory-mapped files,
eliminating all copy overhead for full-frame access.
"""

import os
import numpy as np
import tifffile
from numpy.lib.stride_tricks import as_strided
from .utils.get_si_tiff_n_pages import get_si_tiff_n_pages
from .utils.read_si_framedata_params import read_si_framedata_params


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
    metadata : dict
        ScanImage metadata from the TIFF (FrameData keys extracted via byte-level search).
    n_volumes, n_zplanes, n_channels : int
        Dimensions T, Z, C from metadata (Z=1 for 2D acquisitions). ``n_zplanes``
        is the count of *real* Z planes; flyback frames are excluded.
    n_flyback_frames : int
        Number of FastZ flyback frames per volume that are stored on disk
        (0 when FastZ is off or the scanner discarded them before saving).
        These frames are skipped automatically by the strided view so they
        are not visible through ``mm[...]`` indexing.
    resolution_xyz : tuple of (float or None)
        (x, y, z) in µm per pixel from TIFF tags and SI.hStackManager.stackZStepSize.
        Components are None when not available in the file.
    acquisition_parameters : dict
        Key FrameData values (e.g. frame_rate, volume_rate, z_step_size).
    shape : tuple
        Shape of the data array (T, Z, C, Y, X) - flyback frames excluded.
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

    def __init__(self, tiff_path, allow_truncated=False):
        """
        Initialize MemmapTiffSI instance.

        Parameters
        ----------
        tiff_path : str
            Path to ScanImage TIFF file
        allow_truncated : bool
            If True, allow the file to have incomplete volumes at the end
            (e.g., acquisition stopped mid-volume). Extra pages are discarded
            and a warning is printed.
        """
        self._tiff_path = tiff_path
        self._mmap = None
        self._data = None
        self._si_metadata = None
        self._resolution_xyz = [None, None, None]
        self._acquisition_parameters = {}

        # Fast SI metadata read: targeted byte-level search avoids calling
        # matlabstr2py on the full FrameData blob. ScanImage 2023.1+ writes
        # >20 MB FrameData (galvo waveforms, etc.) that causes matlabstr2py
        # to hang; this path reads only the specific keys we need.
        si_meta = read_si_framedata_params(tiff_path)
        if si_meta is None:
            raise RuntimeError(
                f"Not a ScanImage BigTIFF file: {tiff_path}"
            )
        self._si_metadata = si_meta
        frame_data = si_meta['FrameData']

        with tifffile.TiffFile(tiff_path) as tif:
            # Channels
            channels_saved = frame_data['SI.hChannels.channelSave']
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

            # Stride between pages: use page 1 if it exists; otherwise fall
            # back to the page's own byte count (single-frame file).
            if len(tif.pages) > 1:
                self.stride = tif.pages[1].dataoffsets[0] - self.data_offset_0
            else:
                self.stride = int(page.databytecounts[0])

            npages = get_si_tiff_n_pages(tiff_path)

            # T, Z dimensions from metadata
            n_zplanes = frame_data.get('SI.hStackManager.actualNumSlices', 1)

            # On-disk pages per volume per channel. ScanImage writes this
            # explicitly as numFramesPerVolumeWithFlyback in recent versions
            # (>= 2023.x); it is the authoritative count that always matches
            # the file layout. The SI.hFastZ.discardFlybackFrames boolean is
            # NOT reliable -- in some configurations it reports True even
            # though flyback frames are still written to disk.
            #
            # Fall back to numFramesPerVolume (sans flyback) or actualNumSlices
            # for legacy SI files that don't write the *WithFlyback key.
            pages_per_z_cycle = frame_data.get(
                'SI.hStackManager.numFramesPerVolumeWithFlyback'
            )
            if pages_per_z_cycle is None:
                # Legacy heuristic: trust numDiscardFlybackFrames when FastZ is on.
                fastz_enable = bool(frame_data.get('SI.hFastZ.enable', False))
                n_flyback_meta = int(
                    frame_data.get('SI.hFastZ.numDiscardFlybackFrames', 0) or 0
                )
                pages_per_z_cycle = n_zplanes + (n_flyback_meta if fastz_enable else 0)
            pages_per_z_cycle = int(pages_per_z_cycle)
            self.n_flyback_frames = max(0, pages_per_z_cycle - n_zplanes)

            # Always compute n_volumes from actual page count in the file.
            # Metadata n_volumes (SI.hStackManager.actualNumVolumes) is unreliable
            # (ScanImage writes default/garbage values for infinite acquisitions).
            pages_per_volume = pages_per_z_cycle * n_channels
            remainder = npages % pages_per_volume
            n_volumes = npages // pages_per_volume

            if remainder != 0:
                if allow_truncated:
                    print(f"WARNING: {os.path.basename(tiff_path)} has {npages} pages, "
                          f"{n_volumes} complete volumes, "
                          f"{remainder} extra pages discarded.")
                else:
                    assert False, (
                        f"Page count {npages} is not divisible by pages_per_volume="
                        f"(n_zplanes={n_zplanes} + n_flyback={self.n_flyback_frames}) "
                        f"* n_channels={n_channels} = {pages_per_volume}. "
                        f"File may be truncated. Use allow_truncated=True to discard extra pages."
                    )
            self.n_zplanes = n_zplanes
            self.n_volumes = n_volumes
            self._pages_per_z_cycle = pages_per_z_cycle  # used by strided view

            # User-facing shape excludes flyback (real Z planes only).
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
            pass  # resolution stays None
        # Z-step preference order:
        #   1. actualStackZStepSize  -- z step actually achieved during acquisition
        #      (correct for arbitrary / fast-Z stacks where stackZStepSize is just
        #      the *requested* step and may differ from what was scanned).
        #   2. stackZStepSize        -- requested z step (legacy / single-actuator).
        z_step = (frame_data.get('SI.hStackManager.actualStackZStepSize')
                  or frame_data.get('SI.hStackManager.stackZStepSize'))
        if z_step is not None:
            try:
                self._resolution_xyz[2] = float(z_step)
            except (TypeError, ValueError):
                pass  # z resolution stays None

    def _extract_acquisition_parameters(self, frame_data):
        """Extract key acquisition parameters from FrameData."""
        out = {}
        keys = {
            'frame_rate': 'SI.hRoiManager.scanFrameRate',
            'volume_rate': 'SI.hRoiManager.scanVolumeRate',
            'z_step_size_requested': 'SI.hStackManager.stackZStepSize',
            'z_step_size': 'SI.hStackManager.actualStackZStepSize',
            'frames_per_slice': 'SI.hStackManager.framesPerSlice',
            'num_slices_requested': 'SI.hStackManager.numSlices',
            'num_slices': 'SI.hStackManager.actualNumSlices',
            'num_frames_per_volume': 'SI.hStackManager.numFramesPerVolume',
            'stack_definition': 'SI.hStackManager.stackDefinition',
            'stack_mode': 'SI.hStackManager.stackMode',
            'fastz_enable': 'SI.hFastZ.enable',
            'fastz_discard_flyback_frames': 'SI.hFastZ.discardFlybackFrames',
            'fastz_num_discard_flyback_frames': 'SI.hFastZ.numDiscardFlybackFrames',
        }
        for name, key in keys.items():
            if key in frame_data:
                out[name] = frame_data[key]
        # Fall back to requested z step if actual is absent.
        if 'z_step_size' not in out and 'z_step_size_requested' in out:
            out['z_step_size'] = out['z_step_size_requested']
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
        """Create 5D strided view (T, Z, C, Y, X). Page order within one volume:
        for z in range(Z + n_flyback): for c in range(C): page.

        The view stride along T jumps over a full Z cycle (real planes + any
        FastZ flyback frames) so that ``mm[t]`` always lands on the first real
        plane of volume ``t``. The final view is then sliced to ``[:, :Z]`` so
        flyback frames are unreachable through the user-facing shape.
        """
        self._ensure_mmap()

        if self._data is not None:
            return

        mmap_as_dtype = self._mmap.view(self.dtype)
        dtype_offset = self.data_offset_0 // self.dtype.itemsize
        itemsize = self.dtype.itemsize
        T, Z, C, H, W = self._shape
        Z_total = self._pages_per_z_cycle    # Z + n_flyback_frames
        stride_page = self.stride            # bytes per page

        # Strides in bytes: [t, z, c, y, x]
        # page k within volume = z*C + c (z indexed across the FULL Z cycle, incl. flyback)
        # next volume starts Z_total*C pages later
        strides = (
            Z_total * C * stride_page,
            C * stride_page,
            stride_page,
            W * itemsize,
            itemsize,
        )
        full_view = as_strided(
            mmap_as_dtype[dtype_offset:],
            shape=(T, Z_total, C, H, W),
            strides=strides,
            writeable=False,
        )
        # Drop the trailing flyback frames; same strides, shrunk Z extent.
        # When n_flyback_frames == 0 this is a no-op.
        self._data = full_view[:, :Z]

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
        """ScanImage metadata dict (FrameData keys extracted via byte-level search)."""
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
