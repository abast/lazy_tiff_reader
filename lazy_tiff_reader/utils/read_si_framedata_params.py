"""
Fast extraction of ScanImage BigTIFF parameters without full matlabstr2py parse.

Background
----------
Every ScanImage BigTIFF file has a custom 16-byte header inserted at bytes 16-31,
immediately after the standard BigTIFF header. Its layout is:

    bytes  0-15  standard BigTIFF header  (byte order mark, magic=43, IFD offset)
    bytes 16-19  ScanImage magic number   (0x07030301)
    bytes 20-23  format version           (3 or 4)
    bytes 24-27  size0                    number of bytes in the FrameData block
    bytes 28-31  size1                    number of bytes in the RoiData block
    bytes 32 ..  FrameData block          MATLAB-style key=value text, null-terminated
    bytes 32+size0 .. RoiData block       JSON

FrameData contains non-varying SI acquisition parameters as MATLAB-style text, e.g.:
    SI.hChannels.channelSave = [1 2]
    SI.hStackManager.actualNumSlices = 9
    ...

Problem: since commit 35b375b ("Add XROI properties to hSI"), XROI stores its
full per-frame scan waveforms (galvo XY, beam B/Bpb, FastZ) in hSI.xroiProps
before each acquisition. Because xroiProps is a SetObservable SI model property,
mdlGetHeaderString() includes it in FrameData — ballooning it from ~18 KB to
~23 MB. tifffile's read_scanimage_metadata() passes the entire blob to
matlabstr2py(), which hangs on 20+ MB of input.

Fix: read the raw FrameData bytes ourselves and regex-search for only the
specific keys MemmapTiffSI needs. I/O for 23 MB takes ~7 ms; 6 regex passes
over 23 MB take ~50 ms total. No matlabstr2py call needed.
"""

import re
import struct
from typing import Any, Dict, Optional

# Vidrio's custom magic number written at bytes 16-19 of every ScanImage BigTIFF.
# Little-endian uint32: 0x07 0x03 0x03 0x01 -> 0x01030307 = 117637889.
# Presence of this magic confirms the file is a ScanImage BigTIFF (not just any BigTIFF).
_SI_BIGTIFF_MAGIC = 117637889  # 0x07030301 little-endian

# The SI parameters MemmapTiffSI uses to determine array shape and acquisition info.
# Add to this list if you need additional keys available via arr.metadata['FrameData'].
_SI_PARAMS = [
    'SI.hChannels.channelSave',          # which channels were saved, e.g. [1 2]
    'SI.hStackManager.actualNumSlices',  # Z planes per volume
    'SI.hStackManager.actualNumVolumes', # number of volumes (T dimension)
    'SI.hStackManager.stackZStepSize',   # µm between Z planes
    'SI.hRoiManager.scanFrameRate',      # frames per second
    'SI.hRoiManager.scanVolumeRate',     # volumes per second
]


def _parse_si_value(raw: bytes) -> Any:
    """
    Parse a single MATLAB-style value from the raw bytes of one assignment's RHS.

    Handles the value types that appear in ScanImage FrameData:
      true / false          -> Python bool
      'some string'         -> Python str  (strips the single quotes)
      [1 2 3] or [1;2;3]   -> Python list of int/float (semicolons = row separator,
                               treated as flat list here since we only need 1D params)
      42 / 3.14             -> Python int or float
      anything else         -> str as-is
    """
    s = raw.decode('ascii', errors='replace').strip()

    if s == 'true':
        return True
    if s == 'false':
        return False

    if s.startswith("'") and s.endswith("'"):
        return s[1:-1]

    if s.startswith('[') and s.endswith(']'):
        inner = s[1:-1].strip()
        if not inner:
            return []
        # Semicolons separate rows in MATLAB matrices; flatten to a list.
        tokens = inner.replace(';', ' ').split()
        values = []
        for t in tokens:
            try:
                values.append(int(t))
            except ValueError:
                try:
                    values.append(float(t))
                except ValueError:
                    values.append(t)
        # Unwrap single-element lists to a scalar (e.g. actualNumSlices = [9] -> 9)
        return values if len(values) != 1 else values[0]

    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass

    return s


def read_si_framedata_params(tiff_path: str) -> 'Optional[Dict[str, Any]]':
    """
    Read ScanImage FrameData parameters from a BigTIFF file using targeted
    byte-level search. Avoids calling matlabstr2py on the full FrameData blob,
    which hangs for ScanImage files where xroiProps has been populated with
    waveform data (>20 MB).

    Parameters
    ----------
    tiff_path : str
        Path to ScanImage BigTIFF file.

    Returns
    -------
    dict or None
        {'FrameData': {key: value, ...}, 'version': 3|4}
        Same structure as tifffile's scanimage_metadata, but containing only
        the keys listed in _SI_PARAMS. Returns None for non-ScanImage files.
    """
    try:
        with open(tiff_path, 'rb') as f:
            # ---- Standard BigTIFF identification (bytes 0-3) ----
            # Byte order mark: b'II' = little-endian (Intel), b'MM' = big-endian.
            # ScanImage always writes little-endian.
            # Magic number 43 (0x002B) identifies BigTIFF vs classic TIFF (42).
            header = f.read(4)
            if len(header) < 4 or header[:2] != b'II':
                return None
            magic_tiff = struct.unpack_from('<H', header, 2)[0]
            if magic_tiff != 43:
                return None  # not BigTIFF

            # ---- ScanImage custom header (bytes 16-31) ----
            # Sits between the standard BigTIFF header and the FrameData block.
            # struct layout: uint32 magic, uint32 version, uint32 size0, uint32 size1
            f.seek(16)
            si_header = f.read(16)
            if len(si_header) < 16:
                return None
            magic, version, size0, size1 = struct.unpack('<IIII', si_header)
            if magic != _SI_BIGTIFF_MAGIC or version not in (3, 4):
                return None

            # ---- FrameData block (bytes 32 .. 32+size0) ----
            # Contains all non-varying SI acquisition parameters as MATLAB-style
            # key = value text, one assignment per line, null-terminated.
            # In files with xroiProps populated, this is ~23 MB of text because
            # the full galvo/beam waveforms are serialized here as float arrays.
            # Reading 23 MB sequentially takes ~7 ms on a local or USB drive.
            f.seek(32)
            raw = f.read(size0)
    except OSError:
        return None

    # ---- Targeted parameter extraction ----
    # Instead of parsing the full MATLAB struct (which hangs matlabstr2py on
    # large inputs), regex-search the raw bytes once per needed key.
    # Pattern: b'SI.hChannels.channelSave = <capture everything to end of line>'
    frame_data: dict[str, Any] = {}
    for key in _SI_PARAMS:
        pattern = re.escape(key).encode() + rb'[ \t]*=[ \t]*([^\n\r]+)'
        m = re.search(pattern, raw)
        if m:
            frame_data[key] = _parse_si_value(m.group(1).rstrip())

    return {'FrameData': frame_data, 'version': version}
