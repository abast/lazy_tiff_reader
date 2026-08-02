"""
Fast extraction of ScanImage BigTIFF metadata without a full matlabstr2py parse.

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

RoiData is the JSON MROI scanfield description, ``{"RoiGroups": {...}}``, giving
each ROI's centerXY / sizeXY / pixelResolutionXY. MROI callers need it together
with the FrameData keys, so both blocks are parsed here.

Problem: since commit 35b375b ("Add XROI properties to hSI"), XROI stores its
full per-frame scan waveforms (galvo XY, beam B/Bpb, FastZ) in hSI.xroiProps
before each acquisition. Because xroiProps is a SetObservable SI model property,
mdlGetHeaderString() includes it in FrameData - ballooning it from ~18 KB to
100+ MB. tifffile's read_scanimage_metadata() passes the entire blob to
matlabstr2py(), which hangs on that much input.

Fix: read the raw FrameData and RoiData bytes ourselves, then pull out only the
keys callers need in a single regex pass. On a 163 MB blob the read takes
~0.2 s and the scan ~1 s; matlabstr2py on the same blob never returns.
"""

import json
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
    'SI.hChannels.channelSave',             # which channels were saved, e.g. [1 2]
    'SI.hStackManager.actualNumSlices',     # actual # Z planes per volume (excl. flyback)
    'SI.hStackManager.actualNumVolumes',    # actual # volumes (T dimension)
    'SI.hStackManager.numSlices',           # requested # Z planes
    'SI.hStackManager.stackZStepSize',      # requested um between Z planes
    'SI.hStackManager.actualStackZStepSize',# achieved um between Z planes (preferred for arbitrary/fast-Z)
    'SI.hStackManager.framesPerSlice',      # frames acquired per Z slice
    'SI.hStackManager.numFramesPerVolume',          # real frames per volume per channel
    'SI.hStackManager.numFramesPerVolumeWithFlyback',# real + flyback pages per volume per channel
    'SI.hStackManager.stackDefinition',     # 'uniform' / 'arbitrary'
    'SI.hStackManager.stackMode',           # 'slow' / 'fast'
    # FRAME averaging (distinct from the LINE averaging keys further down).
    # ScanImage writes ONE page per logAverageFactor ACQUIRED frames, so every
    # frame count above is in frames, not pages, and must be divided by this to
    # get a page count. Its absence from this whitelist is why MemmapTiffSI
    # silently never divided: .get() returned the default of 1 on every file.
    'SI.hScan2D.logAverageFactor',          # acquired frames averaged into one page
    'SI.hRoiManager.scanFrameRate',         # frames per second
    'SI.hRoiManager.scanVolumeRate',        # volumes per second
    # FastZ flyback. ScanImage's discardFlybackFrames flag is NOT reliable
    # (it's True even when flyback frames are still written to disk on some
    # configurations). MemmapTiffSI prefers numFramesPerVolumeWithFlyback as
    # the source of truth for on-disk per-volume page count; the keys below
    # are kept for legacy SI versions / diagnostics.
    'SI.hFastZ.enable',                     # FastZ on/off
    'SI.hFastZ.discardFlybackFrames',       # UNRELIABLE - see comment above
    'SI.hFastZ.numDiscardFlybackFrames',    # # flyback frames per Z cycle
    'SI.hFastZ.waveformType',               # 'step' / 'sawtooth'
    # MROI geometry. Scanfield tiles are concatenated along Y with flyto blank
    # lines between them; these convert RoiData's normalized units to microns
    # and give the inter-tile gap as round(flytoTimePerScanfield / linePeriod).
    'SI.objectiveResolution',               # um per DEGREE of scan angle (161.275 on meso4)
    'SI.hScan2D.flytoTimePerScanfield',     # seconds spent moving between tiles
    'SI.hRoiManager.linePeriod',            # seconds per scanned line
    'SI.hScan2D.lineAverageFactor',         # lines averaged into one stored line
    'SI.hScan2D.LineAveragingLineCount',    # same, older SI spelling
    'SI.hAcq.lineAverageFactor',            # same, older SI spelling
    # The inter-tile gap applies only on a RESONANT scanner: SI's own
    # FrameScanDataView and getMroiFrameSequence multiply it by an isResonant
    # guard (getMroiDataFromTiff omits the guard -- an inconsistency among SI's
    # three loaders). scanMode is the modern field, scannerType the legacy one.
    'SI.hScan2D.scanMode',                  # 'resonant' / 'linear'
    'SI.hScan2D.scannerType',               # 'RGG' / 'Resonant' (legacy files)
    # scanFramePeriod + flybackTimePerFrame give an INDEPENDENT check on the
    # inter-tile gap, and are what resolved the 50-vs-48 disagreement:
    #   scanFramePeriod / linePeriod
    #     == sum(tile_scan_lines) + (n-1)*gap + frame_flyback_lines
    # holds exactly (358 / 608 / 858 line periods at 1 / 2 / 3 tiles) only when
    # both transits round UP to an EVEN number of line periods. See
    # analysis/core/mroi_layout.py::scan_lines_for_transit.
    'SI.hRoiManager.scanFramePeriod',       # seconds per frame (all tiles + flyback)
    'SI.hScan2D.flybackTimePerFrame',       # seconds of frame flyback
    # MROI is on in every meso4 configuration, including single-ROI ones, which
    # is why linesPerFrame / pixelsPerLine can never be trusted on this rig.
    'SI.hRoiManager.mroiEnable',            # 1 when tiles are concatenated on one page
]

# Per-actuator acquisition line annotations, e.g.
# 'SI.xroiProps.Annotations.Beam.LineStart'. These mark which stored lines were
# actually being acquired (vs. blank flyback), so MROI callers can mask them out.
# Matched generically so any actuator name works; the sibling xroiProps waveform
# blobs -- the reason FrameData is so large -- are deliberately left unparsed.
_ANNOTATION_PATTERN = rb'SI\.xroiProps\.Annotations\.[A-Za-z0-9_]+\.[A-Za-z0-9_]+'


def _parse_si_value(raw: bytes, unwrap_single: bool = True) -> Any:
    """
    Parse a single MATLAB-style value from the raw bytes of one assignment's RHS.

    Handles the value types that appear in ScanImage FrameData:
      true / false          -> Python bool
      'some string'         -> Python str  (strips the single quotes)
      [1 2 3] or [1;2;3]   -> Python list of int/float (semicolons = row separator,
                               treated as flat list here since we only need 1D params)
      42 / 3.14             -> Python int or float
      anything else         -> str as-is

    ``unwrap_single`` collapses a one-element list to a scalar (e.g.
    actualNumSlices = [9] -> 9). Callers that index the result as an array --
    the xroiProps annotations -- pass False to keep it a list.
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
        if unwrap_single and len(values) == 1:
            return values[0]
        return values

    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass

    return s


def _read_si_blocks(tiff_path: str) -> 'Optional[tuple]':
    """
    Read the raw FrameData and RoiData blocks from a ScanImage BigTIFF.

    Returns ``(frame_raw, roi_raw, version)``, or None if the file is not a
    ScanImage BigTIFF.
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

            # ---- FrameData, then RoiData, back to back from byte 32 ----
            # In files with xroiProps populated FrameData is 100+ MB of text
            # because the full galvo/beam waveforms are serialized there.
            # Reading it sequentially is fast; only parsing it is not.
            f.seek(32)
            frame_raw = f.read(size0)
            roi_raw = f.read(size1)
    except OSError:
        return None

    return frame_raw, roi_raw, version


def read_si_framedata_params(tiff_path: str) -> 'Optional[Dict[str, Any]]':
    """
    Read ScanImage metadata from a BigTIFF using targeted byte-level search.
    Avoids calling matlabstr2py on the full FrameData blob, which hangs for
    ScanImage files where xroiProps has been populated with waveform data.

    Parameters
    ----------
    tiff_path : str
        Path to ScanImage BigTIFF file.

    Returns
    -------
    dict or None
        ``{'FrameData': {key: value, ...}, 'RoiGroups': {...}, 'version': 3|4}``
        Same structure as tifffile's scanimage_metadata, but FrameData holds
        only the keys in _SI_PARAMS plus any xroiProps line annotations.
        ``RoiGroups`` is absent when the file has no RoiData block.
        Returns None for non-ScanImage files.
    """
    blocks = _read_si_blocks(tiff_path)
    if blocks is None:
        return None
    frame_raw, roi_raw, version = blocks

    # ---- Targeted parameter extraction ----
    # Instead of parsing the full MATLAB struct (which hangs matlabstr2py on
    # large inputs), scan the raw bytes ONCE for every key of interest. A pass
    # per key would mean re-scanning 100+ MB a few dozen times, and any key
    # absent from the file costs a full scan on its own.
    wanted = b'|'.join(re.escape(key).encode() for key in _SI_PARAMS)
    pattern = (rb'^(' + wanted + rb'|' + _ANNOTATION_PATTERN
               + rb')[ \t]*=[ \t]*([^\n\r]*)')

    frame_data: Dict[str, Any] = {}
    for m in re.finditer(pattern, frame_raw, re.MULTILINE):
        key = m.group(1).decode('ascii')
        frame_data[key] = _parse_si_value(
            m.group(2).rstrip(),
            unwrap_single='.Annotations.' not in key,
        )

    result: Dict[str, Any] = {'FrameData': frame_data, 'version': version}

    # ---- MROI scanfield geometry (RoiData JSON) ----
    # Null-terminated like FrameData; trailing padding is not valid JSON.
    if roi_raw:
        try:
            roi_meta = json.loads(roi_raw.split(b'\x00', 1)[0].decode('utf-8'))
        except (ValueError, UnicodeDecodeError):
            roi_meta = None
        if isinstance(roi_meta, dict):
            result.update(roi_meta)

    return result
