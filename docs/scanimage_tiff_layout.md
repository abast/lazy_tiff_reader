# Reading a ScanImage TIFF -- the complete guide

**Authoritative reference. Read this before writing or fixing any ScanImage TIFF
reader.** Everything here is backed by a measurement on the meso4 rig; nothing is
assumed from ScanImage documentation.

This file is organised as a **reading guide**, not a bug list. Section 1 is a
decision procedure you can follow top-to-bottom to get correct pixels out of any
file this rig produces, plus a runnable reference implementation. Section 2 is
the **toggle reference**: one entry per acquisition knob, what it does to the
bytes on disk, and how to detect it from the file alone. Everything after that
is detail you consult when section 1 sends you there.

Last verified: **2026-08-02** against a **25-case** corpus (ScanImage 2023.1.1,
`TIFF_FORMAT_VERSION = 4`, meso4 / "ginny", vDAQ R1, RGG resonant scanner at
12012.9 Hz). Re-verified after the flyback / fly-to / frame-flyback timing
sweeps.

> **The numbers 512, 700, 200, 158 and 50 that appear below are THIS corpus's
> geometry, not constants.** Every corpus case was acquired with one ROI shape at
> one zoom, so `linesPerFrame` happens to read 512 and pages happen to be 700
> wide throughout. Change the ROI, the zoom or the pixel density and all of them
> move. What generalises is the RELATIONSHIP -- that the declared fields do not
> match the page, and which field to use instead. Never assert a value; assert
> the relationship. The one place a value was hardcoded (a gap of 50 in a test)
> broke the moment the fly-to time was swept, which is what the sweep is for.

This file exists because we had no documentation of any of it, so every consumer
re-derived the arithmetic independently and each got it wrong differently. Four
of the first 17 acquired configurations were mis-read by shipped code, **two of
them silently**.

Related: [`../sync/WIRING.md`](../../mesotools/sync/WIRING.md) (the model for this file's
evidence discipline), [`../sync/si_frames.py`](../../mesotools/sync/si_frames.py),
[`../../lazy_tiff_reader/`](../),
[`../analysis/core/mroi_layout.py`](../../mesotools/analysis/core/mroi_layout.py) (see
section 8 for which to use when),
[`../../plans/si_tiff_reading.plan.md`](../../plans/si_tiff_reading.plan.md).

## Evidence marking

Every claim carries one of:

| Mark | Meaning |
|---|---|
| **MEASURED (manifest)** | Recorded in `F:/data/meso4/si_parser_corpus/manifest.json` -- ScanImage's own readback plus what landed on disk. |
| **MEASURED (corpus headers)** | Read directly out of the corpus TIFF `Software` tags / page descriptions on 2026-08-02. Re-checkable with the snippets given. |
| **MEASURED (SI source)** | Read out of ScanImage's own MATLAB tree at a cited `file:line`. Authoritative for what SI does, not for what our files contain. |
| **MEASURED (plan)** | Recorded in `plans/si_tiff_reading.plan.md` from a source other than this corpus (live readback, archived files). |
| **INFERRED** | Reasoning on top of the above. Not measured. Treat as a hypothesis. |
| **OPEN** | Not answered. Do not resolve it by reasoning -- acquire the case. |

---

## 0. What you are holding

A ScanImage acquisition is a **BigTIFF** file with a Vidrio-specific block glued
between the BigTIFF header and the first page. MEASURED (corpus headers) and
implemented in
[`lazy_tiff_reader/utils/read_si_framedata_params.py`](../lazy_tiff_reader/utils/read_si_framedata_params.py):

```
bytes  0-15   standard BigTIFF header (byte order 'II', magic 43, IFD offset)
bytes 16-19   ScanImage magic 0x07030301  (little-endian uint32 117637889)
bytes 20-23   TIFF_FORMAT_VERSION         (3 or 4; corpus is 4)
bytes 24-27   size0 = length of the FrameData block
bytes 28-31   size1 = length of the RoiData block
bytes 32              FrameData   MATLAB-style "SI.<path> = <value>" text, NUL-terminated
bytes 32+size0        RoiData     JSON: {"RoiGroups": {"imagingRoiGroup": {...}, ...}}
```

Three places carry information, and they are not interchangeable:

| Where | What is in it | How to read it |
|---|---|---|
| **FrameData** (also exposed as page 0's `Software` TIFF tag) | every `SI.*` acquisition parameter; **constant for the whole file** | regex the `SI.x.y = v` lines. Corpus: 727 kB. A real session file: **~23 MB** -- see section 9. |
| **RoiData** (also page 0's `Artist` tag) | the imaging ROI group: per-tile `centerXY`, `sizeXY`, `pixelResolutionXY`, in DEGREES | `json.loads`. Corpus: ~1.8 kB. |
| **per-page `ImageDescription`** | `frameNumbers`, `frameTimestamps_sec`, `acquisitionNumbers`, `epoch`, ... -- **varies per page** | regex per page. See section 7. |

There is **no volume structure in the file format**. A ScanImage TIFF is a flat
sequence of pages; you reconstruct volumes from FrameData. That reconstruction is
section 1.

MEASURED (corpus headers): every page in a file has the same shape and the same
byte size, and the byte stride between consecutive page data offsets is constant.
That is what makes the zero-copy strided view in `MemmapTiffSI` possible, and
what `test_the_rollover_fixture_is_faithful` asserts before it splits a file.

---

## 1. THE DECISION PROCEDURE

Follow these in order. Do not skip step 4.

1. **Open the header without parsing the ROI blob.**
   Read page 0's `Software` tag (or the raw FrameData block). Do **NOT** call
   `tifffile.TiffFile.scanimage_metadata` -- it hands the whole blob to
   `matlabstr2py` and does not return on a real session file (section 9).

2. **Read these fields** (exact names, all under `SI.`):

   | Field | Symbol below | Fallback if absent |
   |---|---|---|
   | `hStackManager.actualNumSlices` | `n_planes` | 1 |
   | `hStackManager.framesPerSlice` | `fps` | 1 |
   | `hStackManager.numFramesPerVolume` | `real` | `n_planes * fps` |
   | `hStackManager.numFramesPerVolumeWithFlyback` | `fpzc` | `n_planes * fps + hFastZ.numDiscardFlybackFrames` |
   | `hScan2D.logAverageFactor` | `avg` | 1 |
   | `hChannels.channelSave` | `C = numel(...)` | 1 |

   Also take `n_pages` from the **file** (`len(tif.pages)`), and `page_shape`
   from **page 0's actual shape**. Never from `linesPerFrame` / `pixelsPerLine`
   (section 3).

3. **Compute the grouping.**

   ```
   pages_per_z_cycle = fpzc // avg          # incl. flyback, per channel
   real_frames       = real // avg          # excl. flyback, per channel
   flyback_pages     = pages_per_z_cycle - real_frames
   pages_per_volume  = pages_per_z_cycle * C
   n_volumes         = n_pages // pages_per_volume
   leftover          = n_pages - n_volumes * pages_per_volume
   ```

4. **Check the refusal condition. This is not optional.**

   ```
   if avg > 1 and (fpzc % avg or real % avg):  ->  REFUSE the file
   ```

   When `logAverageFactor` does not divide the frame counts, `//` is meaningless:
   averaged pages straddle volume boundaries and a saved page physically blends
   one volume's flyback frame with the next volume's first real frame. There is
   no post-hoc repair and no "best effort" answer. Raise a typed error
   (`lazy_tiff_reader.UnknownLayoutError`) and let it propagate. See section 2.5.

   This check is a property of the **file**, not of your reading strategy --
   including when the caller supplies its own layout (section 6). Any reader with
   its own layout path must call
   `lazy_tiff_reader.check_layout_readable(path)` first.

5. **Decide what `leftover != 0` means, and say which.**
   It is *either* a truncated acquisition (SI stopped mid-volume) *or* a wrong
   layout *or* a multi-file rollover (section 5). Collapsing these was bug B2.
   `allow_truncated` must mean "the acquisition stopped mid-volume", never "I do
   not understand this layout".

6. **Group.** Volume `t` occupies pages `[t * pages_per_volume,
   (t+1) * pages_per_volume)`. Within a volume the order is **frame-major,
   channel-minor**:

   ```
   page(t, f, c) = (t * pages_per_z_cycle + f) * C + c
   ```

   MEASURED (corpus headers), pinned by
   `sync/tests/test_tiff_readers_identity.py::test_expected_page_index_matches_the_frameNumbers_in_the_header`
   and, at the pixel level, by `::test_memmap_frame_is_the_page_the_manifest_predicts`.

7. **Drop the flyback pages.** They are ordinary pages on disk and they are the
   **trailing** part of each z cycle, so the real frames are exactly
   `f = 0 .. real_frames - 1` and you never index them. Confirmed against
   ScanImage's own loader, which builds the per-frame z list by APPENDING NaN
   markers for them:
   `stackZs = [obj.zs(:)' nan(1, header.SI.hFastZ.numDiscardFlybackFrames)]`
   -- MEASURED (SI source),
   `+scanimage/+guis/+scanimagedataview/FrameScanDataView.m:384`, and the flyback
   frames are then removed by `fbfs = isnan(tframeZs)` at line 391.
   Pinned by `test_tiff_readers_identity.py::test_memmap_never_returns_a_flyback_page`.

8. **Interpret dimension `f`.** It is **every real frame**, not the plane count:
   `real_frames == n_planes * fps / avg`. Ordering is plane-major with the
   `framesPerSlice` repeats consecutive -- for 3 planes x 2 frames the order is
   `z1 z1 z2 z2 z3 z3`. MEASURED (manifest): `hStackManager.zs` for
   `zstack3_framesperslice2` is `[-10 -10 0 0 10 10]`. Pinned by
   `sync/tests/test_tiff_readers_corpus.py::test_memmap_dim1_ordering_matches_hStackManager_zs`.
   Reshape `(n_planes, fps)` to separate depth from repeat.

9. **If MROI is on, un-concatenate the page.** Every imaging ROI is written into
   ONE page, stacked along Y with a derived gap between tiles. `mroiEnable = 1`
   in all 25 corpus cases, so on meso4 this step is **always** required when
   there is more than one ROI. Do not implement it yourself:
   `analysis/core/mroi_layout.layout_from_si_tiff(path)` derives and VALIDATES
   the geometry and raises rather than guessing. Section 4 is the theory.

10. **Index a frame.** `page(t, f, c)` from step 6, then (if MROI)
    `layout.slice_tiles(page)` or `layout.place(page)`.

### 1.1 Reference implementation

Complete and runnable. This is the arithmetic only -- for production use the
existing readers (section 8) rather than copying this.

```python
"""Minimal correct ScanImage TIFF reader: header -> volumes -> pixels."""
import re
import numpy as np
import tifffile


class UnreadableLayout(ValueError):
    """Pages cannot be grouped into volumes. Callers must NOT degrade."""


# --- step 1: header, without matlabstr2py ---------------------------------
def read_header(path):
    with tifffile.TiffFile(path) as tif:
        sw = tif.pages[0].tags["Software"].value        # NOT scanimage_metadata
        if isinstance(sw, bytes):
            sw = sw.decode("utf-8", "replace")
        return sw, len(tif.pages), tuple(int(v) for v in tif.pages[0].shape[-2:])


def si(text, name, default=None):
    """Value of an 'SI.<...>.<name> = <value>' line, as text."""
    m = re.search(r"^SI\.[\w.]*" + re.escape(name) + r"\s*=\s*(.*)$", text, re.M)
    return m.group(1).strip() if m else default


def si_int(text, name, default=None):
    try:
        return int(float(si(text, name)))
    except (TypeError, ValueError):
        return default


def n_channels(text):
    """len(channelSave). Written as '1' or as '[1 2]'."""
    inner = (si(text, "hChannels.channelSave") or "1").strip().strip("[]").strip()
    return len([p for p in re.split(r"[\s;,]+", inner) if p]) or 1


# --- steps 2-5: the grouping ----------------------------------------------
def layout(text, n_pages):
    n_planes = si_int(text, "hStackManager.actualNumSlices", 1)     # NOT numSlices
    fps      = si_int(text, "hStackManager.framesPerSlice", 1)
    avg      = max(1, si_int(text, "hScan2D.logAverageFactor", 1))
    C        = n_channels(text)

    fpzc = si_int(text, "hStackManager.numFramesPerVolumeWithFlyback")
    if fpzc is None:                       # legacy files only; see section 2.3
        fpzc = n_planes * fps + si_int(text, "hFastZ.numDiscardFlybackFrames", 0)
    real = si_int(text, "hStackManager.numFramesPerVolume", n_planes * fps)

    # step 4 -- the refusal. A property of the FILE, not of the strategy.
    if avg > 1 and (fpzc % avg or real % avg):
        raise UnreadableLayout(
            f"logAverageFactor {avg} must divide BOTH "
            f"numFramesPerVolumeWithFlyback {fpzc} and numFramesPerVolume {real}; "
            f"it does not. "
            f"Averaged pages straddle volume boundaries -- one page blends this "
            f"volume's flyback frame with the next volume's first real frame. "
            f"Per-volume data is NOT recoverable under ANY layout.")

    pages_per_z_cycle = fpzc // avg
    real_frames       = real // avg
    pages_per_volume  = pages_per_z_cycle * C
    n_volumes         = n_pages // pages_per_volume
    return {
        "n_planes": n_planes, "frames_per_slice": fps, "log_average_factor": avg,
        "n_channels": C, "pages_per_z_cycle": pages_per_z_cycle,
        "real_frames": real_frames,
        "flyback_pages": pages_per_z_cycle - real_frames,
        "pages_per_volume": pages_per_volume, "n_volumes": n_volumes,
        # step 5 -- truncated OR wrong layout OR multi-file rollover. Say which.
        "leftover_pages": n_pages - n_volumes * pages_per_volume,
    }


# --- steps 6-8: indexing ---------------------------------------------------
def page_index(L, t, f, c=0):
    """Flat 0-based page holding volume t, real frame f, channel c.

    Flyback is the TRAILING part of the z cycle, so f in [0, real_frames) never
    selects one. Channels interleave WITHIN a frame.
    """
    if not (0 <= t < L["n_volumes"] and 0 <= f < L["real_frames"]
            and 0 <= c < L["n_channels"]):
        raise IndexError((t, f, c))
    return (t * L["pages_per_z_cycle"] + f) * L["n_channels"] + c


def depth_repeat(L, f):
    """Split dim f into (plane, framesPerSlice repeat). zs = z1 z1 z2 z2 ..."""
    reps = max(1, L["frames_per_slice"] // L["log_average_factor"])
    return divmod(f, reps)


# --- the self-check you should run once per new file ----------------------
def self_check(path, L):
    """frameNumbers[k] must equal (k // C + 1) * avg. See section 7."""
    with tifffile.TiffFile(path) as tif:
        for k, p in enumerate(tif.pages):
            m = re.search(r"frameNumbers\s*=\s*(-?\d+)", p.description or "")
            if m is None:
                return "no frameNumbers in page description"
            want = (k // L["n_channels"] + 1) * L["log_average_factor"]
            if int(m.group(1)) != want:
                return f"page {k}: frameNumbers={m.group(1)}, model says {want}"
    return "ok"


# --- putting it together ---------------------------------------------------
def read_volume(path, t):
    """(real_frames, n_channels, Y, X) for volume t, flyback excluded."""
    text, n_pages, _shape = read_header(path)
    L = layout(text, n_pages)
    with tifffile.TiffFile(path) as tif:
        return np.stack([
            np.stack([tif.pages[page_index(L, t, f, c)].asarray()
                      for c in range(L["n_channels"])])
            for f in range(L["real_frames"])])


if __name__ == "__main__":
    import sys
    p = sys.argv[1]
    text, n_pages, shape = read_header(p)
    L = layout(text, n_pages)
    print(f"{n_pages} pages of {shape} -> {L['n_volumes']} volumes x "
          f"{L['real_frames']} frames x {L['n_channels']} ch "
          f"(+{L['flyback_pages']} flyback/cycle, leftover {L['leftover_pages']})")
    print("self-check:", self_check(p, L))
```

### 1.2 Every corpus case -- check your implementation against these

All from `F:/data/meso4/si_parser_corpus`. 3 volumes requested per case except
`logavg2_fps2_z2` (4). `nSl` = `actualNumSlices`, `fps` = `framesPerSlice`,
`nfpv` = `numFramesPerVolume`, `fpzc` = `numFramesPerVolumeWithFlyback`,
`fly` = `fpzc - nfpv`, `avg` = `logAverageFactor`, `ch` = `numel(channelSave)`,
`LA` = `LineAveragingLineCount`. MEASURED (manifest) + MEASURED (corpus headers)
for every row.

| case | nSl | fps | nfpv | fpzc | fly | avg | ch | LA | tiles | pages/vol | pages | page shape | knob under test |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `baseline` | 1 | 1 | 1 | 1 | 0 | 1 | 1 | 1 | 1 | 1 | 3 | 200 x 700 | (reference config) |
| `zstack3` | 3 | 1 | 3 | 4 | 1 | 1 | 1 | 1 | 1 | 4 | 12 | 200 x 700 | z planes |
| `zstack5` | 5 | 1 | 5 | 6 | 1 | 1 | 1 | 1 | 1 | 6 | 18 | 200 x 700 | z planes |
| `multicolor2` | 1 | 1 | 1 | 1 | 0 | 1 | **2** | 1 | 1 | 2 | 6 | 200 x 700 | `channelSave` |
| `framesperslice3` | 1 | **3** | 3 | 4 | 1 | 1 | 1 | 1 | 1 | 4 | 12 | 200 x 700 | `framesPerSlice` |
| `logavg4` | 1 | 4 | 4 | 5 | 1 | **4** | 1 | 1 | 1 | **REFUSE** | 3 | 200 x 700 | `logAverageFactor` |
| `logavg2_fps2_z2` | 2 | 2 | 4 | 5 | 1 | **2** | 1 | 1 | 1 | **REFUSE** | 10 | 200 x 700 | avg divides nfpv but not fpzc |
| `zstack3_logavg2` | 3 | 2 | 6 | 7 | 1 | **2** | 1 | 1 | 1 | **REFUSE** | 10 | 200 x 700 | `logAverageFactor` + stack |
| `lineavg2` | 1 | 1 | 1 | 1 | 0 | 1 | 1 | **2** | 1 | 1 | 3 | **100** x 700 | `LineAveragingLineCount` |
| `lineavg4` | 1 | 1 | 1 | 1 | 0 | 1 | 1 | **4** | 1 | 1 | 3 | **50** x 700 | `LineAveragingLineCount` |
| `zstack3_multicolor2` | 3 | 1 | 3 | 4 | 1 | 1 | **2** | 1 | 1 | 8 | 24 | 200 x 700 | planes x channels |
| `zstack3_framesperslice2` | 3 | **2** | 6 | 7 | 1 | 1 | 1 | 1 | 1 | 7 | 21 | 200 x 700 | planes x repeats |
| `zstack3_lineavg2` | 3 | 1 | 3 | 4 | 1 | 1 | 1 | **2** | 1 | 4 | 12 | **100** x 700 | line avg + stack |
| `fb_actuator_long` | 3 | 1 | 3 | 4 | **1** | 1 | 1 | 1 | 1 | 4 | 12 | 200 x 700 | `flybackTime` 10 ms |
| `fb_actuator_20ms` | 3 | 1 | 3 | 5 | **2** | 1 | 1 | 1 | 1 | 5 | 15 | 200 x 700 | `flybackTime` 20 ms |
| `fb_actuator_multi` | 3 | 1 | 3 | 6 | **3** | 1 | 1 | 1 | 1 | 6 | 18 | 200 x 700 | `flybackTime` 35 ms |
| `fb_actuator_50ms` | 3 | 1 | 3 | 7 | **4** | 1 | 1 | 1 | 1 | 7 | 21 | 200 x 700 | `flybackTime` 50 ms |
| `fb_frame_long` | 3 | 1 | 3 | 4 | 1 | 1 | 1 | 1 | 1 | 4 | 12 | 200 x 700 | `flybackTimePerFrame` 13 ms |
| `mroi2` | 1 | 1 | 1 | 1 | 0 | 1 | 1 | 1 | **2** | 1 | 3 | **450** x 700 | ROI count |
| `mroi3` | 1 | 1 | 1 | 1 | 0 | 1 | 1 | 1 | **3** | 1 | 3 | **700** x 700 | ROI count |
| `mroi2_zstack3` | 3 | 1 | 3 | 4 | 1 | 1 | 1 | 1 | **2** | 4 | 12 | **450** x 700 | tiles x planes |
| `mroi2_lineavg2` | 1 | 1 | 1 | 1 | 0 | 1 | 1 | **2** | **2** | 1 | 3 | **225** x 700 | tiles + line avg |
| `mroi2_lineavg2_zstack3` | 3 | 1 | 3 | 4 | 1 | 1 | 1 | **2** | **2** | 4 | 12 | **225** x 700 | everything at once |
| `mroi2_flyto_long` | 1 | 1 | 1 | 1 | 0 | 1 | 1 | 1 | **2** | 1 | 3 | **498** x 700 | `flytoTimePerScanfield` 4 ms |
| `mroi3_flyto_short` | 1 | 1 | 1 | 1 | 0 | 1 | 1 | 1 | **3** | 1 | 3 | **652** x 700 | `flytoTimePerScanfield` 1 ms |

Two invariants hold in **all 25** and are worth asserting on your own parse:

- `numFramesPerVolume == actualNumSlices * framesPerSlice`. MEASURED (manifest).
  Do **not** substitute it for `numFramesPerVolumeWithFlyback` -- that is a
  different quantity (it includes flyback).
- `numDiscardFlybackFrames == numFramesPerVolumeWithFlyback - numFramesPerVolume`.
  MEASURED (corpus headers, 2026-08-02, including the flyback sweep). ScanImage
  derives it the same way -- MEASURED (SI source),
  `+scanimage/+util/+private/extractHeaderData.m:53`.

### 1.3 Three worked longhand

**`zstack3`** -- the case that makes grouping non-trivial.
```
real               = numFramesPerVolume 3
fpzc               = numFramesPerVolumeWithFlyback 4
flyback_pages      = 4 - 3                                          =  1
pages_per_volume   = (4 // logAverageFactor 1) * channels 1         =  4
n_volumes          = 12 pages // 4                                  =  3   CORRECT
```
Volume k starts at page `4*k`: pages 0, 4, 8. Page 3, 7, 11 are flyback.

**`zstack3_multicolor2`** -- channels multiply pages but not frame numbers.
```
real               = 3
fpzc               = 4
flyback_pages      = 1
pages_per_volume   = (4 // 1) * channels 2                          =  8
n_volumes          = 24 // 8                                        =  3   CORRECT
page(t=1, f=2, c=1)= (1*4 + 2) * 2 + 1                              = 13
```
Note `frameNumbers` still steps by **4** between volumes, not 8. Section 7.

**`zstack3_framesperslice2`** -- the case that catches bug B1.
```
real               = numFramesPerVolume 6   (= 3 planes x 2 frames/slice)
fpzc               = 7
flyback_pages      = 7 - 6                                          =  1   (NOT 4)
pages_per_volume   = (7 // 1) * 1                                   =  7
n_volumes          = 21 // 7                                        =  3   CORRECT
```
A reader that computes `flyback = fpzc - actualNumSlices` gets `7 - 3 = 4` and
exposes 3 of the 6 real frames per volume -- half the data, silently.

---

## 2. THE TOGGLE REFERENCE

One entry per acquisition knob: what it is, what it does physically, **what it
does to the bytes on disk**, how to detect it from the file alone, and which
corpus case exercises it.

Index:

| # | Knob | Header field(s) | Effect on the file |
|---|---|---|---|
| 2.1 | z planes | `hStackManager.actualNumSlices`, `arbitraryZs`, `zs` | more pages per volume |
| 2.2 | repeats per plane | `hStackManager.framesPerSlice` | more pages per volume; they are REAL DATA |
| 2.3 | FastZ flyback | `hFastZ.flybackTime`, `numDiscardFlybackFrames`, `numFramesPerVolumeWithFlyback` | extra TRAILING pages per volume that are not data |
| 2.4 | channels | `hChannels.channelSave` | multiplies pages; does NOT change `frameNumbers` |
| 2.5 | frame averaging | `hScan2D.logAverageFactor` | DIVIDES pages; can make the file unreadable |
| 2.6 | line averaging | `hScan2D.LineAveragingLineCount` | divides page ROWS; metadata does not follow |
| 2.7 | MROI | `hRoiManager.mroiEnable`, RoiData, `hScan2D.flytoTimePerScanfield` | tiles concatenated along Y with a gap |
| 2.8 | file rollover | `hScan2D.logFramesPerFile` | splits the acquisition across `_0000N.tif` |
| 2.9 | scanner timing | `linePeriod`, `scanFramePeriod`, `flybackTimePerFrame`, `scanZoomFactor` | moves every derived line count |
| 2.10 | volume count | `hStackManager.numVolumes` / `actualNumVolumes` | how many volumes were requested (not what is on disk) |

### 2.1 Number of z planes

**Fields:** `SI.hStackManager.actualNumSlices` (authoritative),
`SI.hStackManager.arbitraryZs` (the request, an N x 2 matrix -- one column per
remote-focus unit), `SI.hStackManager.zs` (SI's own per-frame depth vector),
`SI.hStackManager.stackDefinition` (`'arbitrary'` / `'uniform'`),
`SI.hStackManager.stackMode` (`'fast'` / `'slow'`),
`SI.hStackManager.actualStackZStepSize` (achieved) vs `stackZStepSize`
(requested).

**Physically:** the FastZ remote-focus unit steps the focal plane between frames
within one volume (`stackMode = 'fast'`, `waveformType = 'step'` in all 25 cases).

**On disk:** `actualNumSlices * framesPerSlice` real pages per volume per
channel, plus the flyback pages of 2.3.

**Detect it:** `actualNumSlices`. **Never `numSlices`** -- see section 3.

**Corpus:** `zstack3` (3 planes, `zs = [-10 0 10]`), `zstack5` (5 planes),
`baseline` (1). MEASURED (manifest).

> **`arbitraryZs` vs uniform stacks.** All 25 corpus cases use
> `stackDefinition = 'arbitrary'`, driven by `hStackManager.arbitraryZs`, which
> is an **Nplanes x 2RFUs** matrix (planes = rows). MEASURED (corpus headers):
> `zstack3` carries `arbitraryZs = [-10 -10;0 0;10 10]` and `zs = [-10 0 10]`.
> That is why `numSlices` stays at 1 while `actualNumSlices` reads 3 (section 3).
> A **uniform** stack (`stackDefinition = 'uniform'`, driven by `numSlices` +
> `stackZStepSize`) is **not in the corpus** -- OPEN whether `numSlices` and
> `actualNumSlices` agree there. Either way, read `actualNumSlices`.

> MEASURED (corpus headers): `stackZStepSize = 1` (the stale *request*) while
> `actualStackZStepSize = 10` on `zstack3`. On single-plane files
> `actualStackZStepSize = []`. Prefer `actualStackZStepSize`, fall back to
> `stackZStepSize`, and accept `None` -- which is what `MemmapTiffSI._extract_resolution` does.

### 2.2 `framesPerSlice` -- unaveraged repeats at one depth

**Field:** `SI.hStackManager.framesPerSlice`.

**Physically:** ScanImage acquires `framesPerSlice` frames at each depth before
stepping to the next. With `logAverageFactor = 1` they are all saved.

**On disk:** `framesPerSlice` consecutive pages per depth per channel. **These
are real data, not flyback.** `numFramesPerVolume` already includes them:
`numFramesPerVolume == numel(hStackManager.zs) == actualNumSlices * framesPerSlice`.
MEASURED (manifest), all 25; the same derivation SI uses -- MEASURED (SI source),
`extractHeaderData.m:40`.

**Ordering:** plane-major, repeats consecutive: `zs = [z1 z1 z2 z2 z3 z3]`.
MEASURED (manifest) for `zstack3_framesperslice2`. Pinned by
`test_tiff_readers_corpus.py::test_memmap_dim1_ordering_matches_hStackManager_zs`.
Use `divmod(f, framesPerSlice // avg)` to split depth from repeat.

**Detect it:** `numFramesPerVolume // actualNumSlices`, or read the field.

**Corpus:** `framesperslice3` (1 plane x 3 frames -> `zs = [0 0 0]`),
`zstack3_framesperslice2` (3 x 2 -> 6 real frames/volume).

> **This is bug B1.** A reader that computes
> `flyback = numFramesPerVolumeWithFlyback - actualNumSlices` misclassifies every
> repeat as flyback and silently drops it: `framesperslice3` exposed 3 of 9 real
> frames, `zstack3_framesperslice2` 9 of 18. FIXED 2026-08-02 in
> `lazy_tiff_reader` -- see section 10. Pinned by
> `test_tiff_readers_corpus.py::test_memmap_dim1_is_every_real_frame_not_the_plane_count`
> and by the discrimination check in
> `test_tiff_readers_identity.py::test_the_identity_pins_can_actually_fail` (part c).

### 2.3 FastZ flyback -- HOW MANY FLYBACK FRAMES

**Fields:** `SI.hFastZ.flybackTime` (seconds of actuator return),
`SI.hFastZ.numDiscardFlybackFrames` (the count),
`SI.hStackManager.numFramesPerVolumeWithFlyback` (**the authoritative total**),
`SI.hFastZ.enable`, `SI.hFastZ.discardFlybackFrames` (a boolean; see below),
`SI.hFastZ.waveformType`.

**Physically:** at the end of a z cycle the remote-focus actuator must travel
back from the last plane to the first. That takes `flybackTime` seconds, during
which the scanner keeps scanning and ScanImage keeps saving frames. Those frames
are the actuator in transit -- they are not usable image data.

**On disk:** they are **ordinary pages**, written to the file, carrying real
`frameNumbers` and real `frameTimestamps_sec`, positioned at the **END** of each
z cycle. They are not removed by "discard".

**THE COUNT. Two independent ways to get it, use the first:**

```
DERIVE FROM THE HEADER (always correct, always available on SI >= 2023.x):

    flyback_frames = numFramesPerVolumeWithFlyback - numFramesPerVolume
                   = numFramesPerVolumeWithFlyback - actualNumSlices * framesPerSlice

PREDICT FROM THE TIMING (what ScanImage is doing, and why the count is what it is):

    flyback_frames = ceil(hFastZ.flybackTime / hRoiManager.scanFramePeriod)
                     when numFramesPerVolume > 1, else 0
```

MEASURED (corpus headers + manifest), the full flyback sweep at
`scanFramePeriod = 0.01490064846956189 s` (14.90 ms):

| case | `flybackTime` | lag / frame period | flyback frames | `fpzc` | pages on disk |
|---|---|---|---|---|---|
| `zstack3` (default) | 3 ms | 0.201 | **1** | 4 | 12 |
| `fb_actuator_long` | 10 ms | 0.671 | **1** | 4 | 12 |
| `fb_actuator_20ms` | 20 ms | 1.342 | **2** | 5 | 15 |
| `fb_actuator_multi` | 35 ms | 2.349 | **3** | 6 | 18 |
| `fb_actuator_50ms` | 50 ms | 3.356 | **4** | 7 | 21 |

Two things this table settles:

- **A lag shorter than one frame period still costs exactly one frame.** 3 ms and
  10 ms are both a fraction of a 14.90 ms frame, and both cost 1. That is why
  `fb_actuator_long` alone could not prove the count is derived rather than
  hardcoded, and why the 20/35/50 ms points exist.
- **The count is `ceil`, not `round` and not `floor`.** 0.201 -> 1 rules out
  both.

The prediction also holds at a **second frame period**: MEASURED (corpus
headers), `fb_frame_long` has `flybackTimePerFrame = 0.013` which stretches
`scanFramePeriod` to 0.021394 s, and `ceil(0.003 / 0.021394) = 1` -- observed
`fpzc - nfpv = 1`. So the rule is not an artefact of one frame period.

**`numDiscardFlybackFrames` tracks the sweep exactly** (0 / 1 / 1 / 2 / 3 / 4 in
the cases above), MEASURED (corpus headers, all 25). It is therefore a usable
fallback for files predating `numFramesPerVolumeWithFlyback`:

```
frames_per_z_cycle = actualNumSlices * framesPerSlice + numDiscardFlybackFrames
```

INFERRED as a fallback path: it reproduces all 25 corpus cases arithmetically,
but every corpus file carries `numFramesPerVolumeWithFlyback`, so the fallback
branch was never actually exercised on a file that needed it. **The fallback
currently in `si_frames.py` omits `framesPerSlice`** and would compute 2 instead
of 4 for `framesperslice3`.

> **`SI.hFastZ.discardFlybackFrames` is not a "was it discarded" flag.**
> MEASURED (corpus headers): it reads `true` in every case that has flyback
> frames and `false` in every case that does not -- i.e. it is exactly
> `numDiscardFlybackFrames > 0` -- and in every `true` case the flyback frames
> are **still on disk**. "Discard" describes an internal display path, not the
> file. Do not branch on it to decide whether to skip pages; derive the count.

> **`SI.hFastZ.enable` is not a flyback indicator either.** MEASURED (corpus
> headers): it is `true` in **all 25 cases**, including `baseline` and every
> single-plane case with zero flyback frames. Gating a fallback on it (as
> `si_frames.parse_config` does) is harmless only because
> `numDiscardFlybackFrames` is 0 there anyway.

> **When does a flyback frame appear at all?** MEASURED (manifest), all 25:
> `flyback >= 1` for every case with `numFramesPerVolume > 1`, and `flyback == 0`
> for every case with `numFramesPerVolume == 1` -- including `framesperslice3`,
> which has one *depth* but three frames per volume, and still gets a flyback
> frame. So the trigger is frames-per-volume, not distinct depths. INFERRED as a
> general rule; the mechanism was not read out of SI's source.

**Position: TRAILING.** See step 7 of section 1 for the SI-source evidence. The
corpus is acquired dark and does not label per-page z, so this is SI's own
implementation rather than our pixel measurement -- but it is the authoritative
implementation, not a guess, and the page-set arithmetic in
`test_tiff_readers_identity.py::test_expected_page_index_matches_the_frameNumbers_in_the_header`
(unselected pages == `n_volumes * flyback * C`) is consistent with it in all 25.

### 2.4 Channels -- `channelSave`

**Field:** `SI.hChannels.channelSave` -- `1` or `[1 2]`; take `numel`.
Related: `SI.hChannels.channelsActive` (what the PMTs digitised, may exceed what
was saved).

**Physically:** meso4 has four channels: `Ch1`/`Ch2` = green PMT at RFU1/RFU2,
`Ch3`/`Ch4` = red PMT at RFU1/RFU2 (see `mesotools/CLAUDE.md`). In a `.tif` read
from Python, `Ch1` is channel-axis index **0**.

**On disk:** multiplies the page count by `numel(channelSave)`. The channel pages
of one acquired frame are **adjacent**, so the page order is frame-major /
channel-minor: `page = (t*fpzc/avg + f)*C + c`.

**Detect it:** `numel(channelSave)`; cross-check with `frameNumbers`, which
repeats `C` times (`1, 1, 2, 2, ...`).

**Does NOT change:** `frameNumbers` (the `C` channel pages of one frame share one
number), `frameTimestamps_sec` (same timestamp), `scanFrameRate`,
`scanVolumeRate`, page shape.

**Corpus:** `multicolor2` (1 plane x 2 ch = 6 pages), `zstack3_multicolor2`
(3 planes + flyback, 2 ch = 8 pages/volume, 24 pages). MEASURED (manifest).

> **The corpus has no case with more than 2 channels.** The 4-channel
> configuration the rig actually runs is untested here -- OPEN.

### 2.5 Frame averaging -- `logAverageFactor`

**Field:** `SI.hScan2D.logAverageFactor`.

**Physically:** ScanImage averages `logAverageFactor` consecutive **acquired
frames** into one saved page.

**On disk: it DIVIDES the page count. It never multiplies it.** Every frame count
in the header (`numFramesPerVolume`, `numFramesPerVolumeWithFlyback`,
`framesPerSlice`) is in **acquired frames**; divide by `avg` to get pages.

MEASURED (manifest + corpus headers):

- `logavg4`: `fps = 4`, `fpzc = 5`, `avg = 4`, 3 volumes -> 15 raw frames ->
  **3 pages**.
- `zstack3_logavg2`: `fpzc = 7`, `avg = 2`, 3 volumes -> 21 raw frames ->
  **10 pages**.
- MEASURED (plan), from an archived file: `framesPerSlice = 10`, `fpzc = 10`,
  `avg = 10` -> exactly **one** page. This is the case the first `si_frames`
  implementation computed as 10 pages/volume for a 1-page file.

**Detect it:** the field. Cross-check: `frameNumbers` on page `k` reads
`(k // C + 1) * avg`, so a step of 4 between consecutive single-channel pages
means `avg = 4`.

**Page stamping:** an averaged page carries the `frameNumbers` and
`frameTimestamps_sec` of the **LAST** raw frame of its averaging window, not the
first. Section 7.

**Corpus:** `logavg4`, `zstack3_logavg2`, `logavg2_fps2_z2`. All three are
**unreadable** -- read on.

Pinned by `sync/tests/test_si_frames_corpus.py::test_log_average_factor_divides_pages_not_frames`.

#### 2.5.1 The UNREADABLE case, and why it is unrepairable

**ScanImage averages the raw frame stream with no regard for where volumes
start.** When `logAverageFactor` does not divide `numFramesPerVolumeWithFlyback`,
the averaging window straddles the boundary between two volumes, a saved page
mixes frames from both, and **no page grouping recovers the volumes**.

Worked example, `logavg4`: `fpzc = 5`, `avg = 4`. MEASURED (corpus headers) for
the page positions; the volume starts follow from `fpzc`:

```
raw frame:   1   2   3   4 | 5   6   7   8 | 9  10  11  12 | ...
volumes:     ^vol 1 starts       ^vol 2 starts     ^vol 3 starts
             (frames 1..5)       (frames 6..10)    (frames 11..15)
pages:      [------ p0 -----][------ p1 ----][------ p2 ----]
page stamp:              fn=4              fn=8            fn=12
```

Page 0 covers raw frames 1-4, all inside volume 1. Page 1 covers raw frames 5-8:
frame 5 is volume 1's **flyback**, frames 6-8 are volume 2's first three planes.
Page 1 belongs to no volume. 5 and 4 are coprime, so the pattern never
re-synchronises.

`zstack3_logavg2` is the same failure with different numbers: `fpzc = 7`,
`avg = 2`, pages at raw frames 2, 4, 6, 8, ... while volumes start at raw frames
1, 8, 15. 21 raw frames give **10** pages, which is not a whole number of
anything.

**The decisive case, ANSWERED 2026-08-02 by acquisition.** `logavg2_fps2_z2`
(`actualNumSlices = 2`, `framesPerSlice = 2`, `avg = 2`, so `numFramesPerVolume = 4`
which `avg` DOES divide, `fpzc = 5` which it does not, 4 volumes requested).
Result: **10 pages for 4 volumes = 2.5 pages per volume**, and `frameNumbers`
reads `2, 4, 6, ..., 20` -- a uniform step of 2 across raw frames 1..20 with
**no reset at any volume boundary**. MEASURED (manifest + corpus headers).

So averaging groups the raw frame stream **continuously, flyback included**:

```
raw frames :  1  2 | 3  4 | 5  6 | 7  8 | 9 10 | ...
                            ^^^^^^
volume 1 = raw 1..5 (4 real + 1 flyback),  volume 2 = raw 6..10
page 3 averages raw frame 5 (volume 1 FLYBACK) with raw frame 6 (volume 2 real)
```

The saved page is therefore not merely un-assignable to a volume -- its pixels
are a **blend of flyback and real data from two different volumes**. There is no
post-hoc repair, which is why the refusal in step 4 must not be relaxed.

#### 2.5.2 OPERATIONAL RULE -- enforce at acquisition time

> **`logAverageFactor` MUST divide `numFramesPerVolumeWithFlyback`.** Check it
> when SETTING UP the acquisition, after ScanImage has settled the configuration,
> and refuse to grab if it fails.

The arithmetic makes this nearly unavoidable rather than a corner case: with
`flyback >= 1`, `fpzc = numFramesPerVolume + flyback`, so if `avg` divides
`numFramesPerVolume` it cannot also divide `numFramesPerVolume + 1`. **Every
averaged stack with one flyback frame lands here.**

> **Operational rule: never combine frame averaging with FastZ flyback.**
> Averaging is only safe when `flyback == 0`.

The trap that makes this easy to get wrong: ScanImage adds a flyback frame
whenever `numFramesPerVolume > 1` (2.3). So the natural setup

```
framesPerSlice   = 4        # 1 plane
logAverageFactor = 4        # "average all 4"
```

yields `numFramesPerVolumeWithFlyback = 5`, and 4 does not divide 5. That is
exactly how both original averaged corpus cases came to be unrecoverable. **Read
`numFramesPerVolumeWithFlyback` back from ScanImage after configuring, and test
`fpzc % avg == 0` against that**, not against `framesPerSlice`.

Pinned by `test_si_frames_corpus.py::test_averaging_that_straddles_volumes_is_refused`,
`test_tiff_readers_corpus.py::test_memmap_refuses_unrepresentable_averaging`
(both `allow_truncated` values),
`::test_only_the_manifest_derived_cases_are_refused` (no over- or under-refusal),
and `test_tiff_readers_identity.py::test_explicit_layout_cannot_bypass_the_unreadable_layout_refusal`.

> **The corpus contains no averaged case where `avg` DOES divide `fpzc`.** The
> divide-out path of 2.5 is pinned only by an archived file, not by the corpus.
> OPEN.

### 2.6 Line averaging -- `LineAveragingLineCount`

**Fields:** `SI.hScan2D.LineAveragingLineCount` (this rig's spelling);
`SI.hScan2D.lineAverageFactor` / `SI.hAcq.lineAverageFactor` are older SI
spellings that `mroi_layout._line_averaging` also accepts.

**Physically:** the FPGA averages `N` consecutive **scan lines** into one stored
image row. This is an **arco-local feature** (see 2.6.1).

**On disk: it divides the page ROWS by `N`.** MEASURED (manifest):

| case | LA | page rows |
|---|---|---|
| `baseline` | 1 | 200 |
| `lineavg2` | 2 | **100** |
| `lineavg4` | 4 | **50** |
| `mroi2` | 1 | 450 |
| `mroi2_lineavg2` | 2 | **225** |

**What it does NOT change:** `pixelResolutionXY` in RoiData (still declares the
raw scan-line count), `linesPerFrame` (already meaningless, section 3), the
number of pages, `scanFrameRate` / `scanFramePeriod` (MEASURED (corpus headers):
`lineavg2` and `lineavg4` both read `scanFrameRate = 67.11117318435754`,
identical to `baseline` -- the scanner still scans every line, only the storage
is reduced), or the frame-period identity of section 3, which is in **raw scan
lines**.

**Detect it:** the field. Or: `sum(pixelResolutionXY[1]) + gaps` divided by the
actual page rows.

**The rule that holds across the whole corpus:**

```
page_rows = (sum(tile_scan_lines) + gap * (n_tiles - 1)) // LineAveragingLineCount
```

where `tile_scan_lines[k] = pixelResolutionXY[1]` of tile k and `gap` is derived
per 4.2. Verified:

| case | tiles | sum + gap (scan lines) | LA | predicted rows | actual page rows |
|---|---|---|---|---|---|
| `baseline` | 1 | 200 | 1 | 200 | 200 |
| `lineavg2` | 1 | 200 | 2 | 100 | 100 |
| `lineavg4` | 1 | 200 | 4 | 50 | 50 |
| `mroi2` | 2 | 400 + 50 = 450 | 1 | 450 | 450 |
| `mroi3` | 3 | 600 + 100 = 700 | 1 | 700 | 700 |
| `mroi2_lineavg2` | 2 | 400 + 50 = 450 | 2 | 225 | 225 |

**The gap is INSIDE the division.** MEASURED: `mroi2_lineavg2` is 225 rows, not
250. `(400 + 50) / 2 = 225` matches; `400/2 + 50 = 250` does not. So the gap is
expressed in **pre-averaging scan lines**, or the tiles are concatenated before
averaging. Which of the two is INFERRED, not distinguished by any measurement.
`mroi_layout.build_layout` refuses a gap that `LA` does not divide, for exactly
this reason.

**Corpus:** `lineavg2`, `lineavg4`, `zstack3_lineavg2`, `mroi2_lineavg2`,
`mroi2_lineavg2_zstack3`.

Pinned by `test_si_frames_corpus.py::test_line_averaging_over_reports_rows_in_metadata`
(with the gap DERIVED per file) and
`analysis/tests/test_mroi_corpus.py::test_line_averaged_tiles_are_placed_at_half_height`.

#### 2.6.1 Why every ScanImage loader crashes on a line-averaged MROI file

A loader that slices MROI tile `k` at `pixelResolutionXY[1]` rows runs off the
end of the page as soon as `LineAveragingLineCount > 1`: it asks for 200 rows per
tile out of a 225-row page. This is the "LineAveraging Y-lines trap" referred to
in the `/mroi` skill.

**The crash is structural, not our misuse.** `LineAveragingLineCount` is an
**arco-local FPGA feature that upstream ScanImage's readers know nothing about.**
A whole-tree grep of `scanimage_bravo` finds it in exactly three places --
MEASURED (SI source), 2026-08-02:

| site | role |
|---|---|
| `+scanimage/+components/+scan2d/RggScan.m:93` | the property declaration |
| `+scanimage/+components/+scan2d/RggScan.m:2578` | published to the ZMQ metadata, **undivided** |
| `+scanimage/+components/+scan2d/+rggscan/@Acquisition/Acquisition.m:931-939` | validation |

**No SI loader, viewer or scanfield divides by it.** All three of SI's
un-concatenators (4.3) index at `pixelResolutionXY` rows and therefore run past
the end of a line-averaged page. There is no configuration of SI's own tooling
that reads one of our line-averaged MROI files correctly; any working reader must
be ours. `sync/si_frames.py` is immune only because it counts pages and never
touches rows.

### 2.7 MROI -- `mroiEnable` and the ROI count

**Fields:** `SI.hRoiManager.mroiEnable`; the RoiData JSON block
(`RoiGroups.imagingRoiGroup.rois[k].scanfields[0].{centerXY, sizeXY,
pixelResolutionXY}`); `SI.hScan2D.flytoTimePerScanfield`;
`SI.hRoiManager.linePeriod`; `SI.objectiveResolution` (161.275 um per DEGREE on
meso4); `SI.hScan2D.scanMode` / `scannerType`.

**Physically:** several imaging ROIs are scanned within one frame; the scanner
flies from one to the next in `flytoTimePerScanfield` seconds.

**On disk:** all tiles go into **ONE page**, stacked along **Y**, separated by
blank rows. The gap appears **between** tiles only, never before the first, hence
`(n_tiles - 1)` gaps. The page is as wide as the widest tile.

MEASURED (manifest), identical tiles of 200 scan lines each, at
`flytoTimePerScanfield = 2 ms`:

| tiles | page rows | arithmetic |
|---|---|---|
| 1 (`baseline`) | 200 | 200 |
| 2 (`mroi2`) | 450 | 200 + **50** + 200 |
| 3 (`mroi3`) | 700 | 200 + **50** + 200 + **50** + 200 |

**Detect it:** count the ROIs in RoiData. `mroiEnable` alone does not tell you --
it is `1` on this rig in **every** configuration, including single-ROI ones
(MEASURED (manifest), all 25). That is also why `linesPerFrame` and
`pixelsPerLine` can never be trusted here (section 3).

**Consequences you must handle:** `linesPerFrame` / `pixelsPerLine` are
meaningless; the gap is **not a constant** (2.9); a wrong gap mis-slices every
tile after the first and produces a perfectly plausible wrong image.

**Corpus:** `mroi2`, `mroi3`, `mroi2_zstack3`, `mroi2_lineavg2`,
`mroi2_lineavg2_zstack3`, `mroi2_flyto_long`, `mroi3_flyto_short`.

Full theory and the gap derivation: **section 4**. Implementation:
`analysis/core/mroi_layout.py`.

### 2.8 `logFramesPerFile` -- multi-file rollover

**Field:** `SI.hScan2D.logFramesPerFile`. MEASURED (corpus headers): **`Inf` in
all 25 corpus cases**, so every corpus acquisition is a single
`<stem>_00001.tif` and the rollover path has **zero corpus coverage**.

**Physically:** ScanImage closes the current TIFF and opens `<stem>_00002.tif`
after this many **frames**. Because it counts frames, nothing forces the split
onto a volume boundary.

**On disk:** the acquisition becomes several files. Each file is a structurally
complete ScanImage BigTIFF -- it carries the full BigTIFF header, the SI magic,
and identical FrameData / RoiData. Only the pages differ.

**Detect it:** `logFramesPerFile` is finite, **or** sibling files named
`<stem>_00002.tif` exist next to the one you were handed. There is no field in
file 1 that says how many files there are.

Section 5 is what happens if you ignore it.

### 2.9 Scanner timing -- `linePeriod`, `scanFramePeriod`, `flybackTimePerFrame`, `flytoTimePerScanfield`, zoom

These do not change how pages are grouped. They change every **derived line
count**, which is what makes hardcoding any of the numbers in this document a
latent bug.

| Field | Corpus value | What it drives |
|---|---|---|
| `SI.hRoiManager.linePeriod` | `4.162192309933489e-05` s, **identical in all 25** | the denominator of every transit -> line-count conversion |
| `SI.hScan2D.scannerFrequency` | 12012.9 Hz | INFERRED: `linePeriod ~= 1 / (2 * scannerFrequency)` for a bidirectional resonant scanner (`bidirectional = true`); the two agree to 3 parts in 1e4, so the relation is consistent but not exact as written |
| `SI.hRoiManager.scanFramePeriod` | 0.0149006 s (1 tile) ... 0.0357116 s (3 tiles) | the frame clock; the flyback-frame count of 2.3 divides by it |
| `SI.hRoiManager.scanFrameRate` | `1 / scanFramePeriod` | -- |
| `SI.hRoiManager.scanVolumeRate` | `scanFrameRate / numFramesPerVolumeWithFlyback` in **all 25** (MEASURED, manifest) | an independent cross-check on `fpzc` -- see section 3 |
| `SI.hScan2D.flybackTimePerFrame` | 0.0065 s (0.013 in `fb_frame_long`) | the per-FRAME scanner flyback; part of the frame-period identity |
| `SI.hScan2D.flytoTimePerScanfield` | 0.002 s (0.001 / 0.004 in the sweep) | the MROI inter-tile gap |
| `SI.hRoiManager.scanZoomFactor` | 1 in all 25 | changes the field of view, hence `linePeriod` and every derived count |

**How a zoom or density change propagates:** zoom changes the scanned angular
extent, which changes `linePeriod` and `scanFramePeriod`; those change (a) the
flyback-frame count `ceil(flybackTime / scanFramePeriod)`, (b) the MROI gap
`round_up_to_even(flytoTimePerScanfield / linePeriod)`, (c) the frame-flyback
line count. Pixel density changes `pixelResolutionXY` and hence the page shape.
**Nothing about the page-to-volume grouping changes.**

**Corpus:** `fb_frame_long` moves `flybackTimePerFrame` to 13 ms;
`mroi2_flyto_long` / `mroi3_flyto_short` move `flytoTimePerScanfield` to 4 / 1 ms;
the `fb_actuator_*` cases move `flybackTime`. **No case moves the zoom or the
pixel density**, so `linePeriod` is a single value across the whole corpus --
OPEN (section 12).

### 2.10 `numVolumes` / `actualNumVolumes`

**Fields:** `SI.hStackManager.numVolumes` (requested),
`SI.hStackManager.actualNumVolumes`.

**On disk:** nothing directly. **Take `n_volumes` from the page count**, never
from `actualNumVolumes`: ScanImage writes a default there for unbounded
acquisitions. MEASURED (plan). In the corpus `actualNumVolumes` happens to be
right (3, or 4 for `logavg2_fps2_z2`) because every case was bounded, so the
corpus cannot catch a reader that trusts it.

---

## 3. Fields you must not trust -- and what to use instead

| Field | What it looks like | Reality | Use instead |
|---|---|---|---|
| `hStackManager.numSlices` | the plane count | **1 while `actualNumSlices` is 3 or 5** in the same header. MEASURED (corpus headers): `zstack3`, `zstack5`, and in fact **all 25 read `numSlices = 1`** regardless of the real plane count, because these stacks are defined through `arbitraryZs`. MEASURED (plan): seen the other way round too -- `numSlices = 2`, `actualNumSlices = 1` -- so the sign of the disagreement is not fixed and you cannot repair one from the other. | `hStackManager.actualNumSlices` |
| `hFastZ.discardFlybackFrames` | "the flyback was discarded" | `true` whenever flyback frames exist, and they are **still written to disk**. See 2.3. | `numFramesPerVolumeWithFlyback - numFramesPerVolume` |
| `hFastZ.enable` | "FastZ is doing something" | `true` in all 25, including zero-flyback single-plane cases. See 2.3. | as above |
| `hFastZ.numDiscardFlybackFrames` | the flyback count | Actually correct in all 25 (it tracks the sweep). But it is the **legacy** source; prefer the subtraction. | `numFramesPerVolumeWithFlyback - numFramesPerVolume` |
| `hRoiManager.linesPerFrame` | page height | **512 in all 25 cases**, against real page heights of 50, 100, 200, 225, 450, 498, 652 and 700. MEASURED (manifest, `metadata_mismatch` on every case). | `tif.pages[0].shape[0]`, or `sum(pixelResolutionXY[1]) + gaps) // LA` |
| `hRoiManager.pixelsPerLine` | page width | **512 in all 25 cases**, against a real 700 columns everywhere. MEASURED (manifest). | `tif.pages[0].shape[1]`, or `pixelResolutionXY[0]` per tile |
| `scanfields[k].pixelResolutionXY[1]` | tile height in stored rows | RAW SCAN LINES. Over-reports stored rows by exactly `LineAveragingLineCount`. See 2.6. | `pixelResolutionXY[1] // LineAveragingLineCount` |
| `hStackManager.actualNumVolumes` | volumes in the file | a default for unbounded acquisitions. See 2.10. | `n_pages // pages_per_volume` |
| `hStackManager.stackZStepSize` | the z step | the stale *request*: 1 while `actualStackZStepSize` is 10. See 2.1. | `actualStackZStepSize`, falling back to `stackZStepSize` |

`mroiEnable = 1` on this rig in **every** configuration, including single-ROI
ones, so **there is no configuration on meso4 where `linesPerFrame` /
`pixelsPerLine` can be trusted.** Pinned by
`test_si_frames_corpus.py::test_lines_per_frame_is_unreliable_under_mroi` and
`test_tiff_readers_corpus.py::test_memmap_rows_come_from_the_page_not_from_metadata`.

### 3.1 Two independent self-checks

Run these on any file whose layout you have just derived. Both use header fields
that the grouping does not otherwise consume, so they are genuine witnesses
rather than restatements.

**(a) The volume-rate identity.**

```
scanVolumeRate == scanFrameRate / numFramesPerVolumeWithFlyback
```

MEASURED (manifest), exact in **all 25 cases**. It confirms `fpzc` -- the single
most load-bearing number in section 1 -- from two fields you never read. Note it
is blind to channels: `multicolor2` has the same volume rate as `baseline`.

**(b) The frame-period identity (MROI and line counts).**

```
scanFramePeriod / linePeriod
    == sum(tile_scan_lines) + (n_tiles - 1) * gap + frame_flyback_lines
```

with `gap = round_up_to_even(flytoTimePerScanfield / linePeriod)` and
`frame_flyback_lines = round_up_to_even(flybackTimePerFrame / linePeriod)`.

MEASURED (corpus headers): `scanFramePeriod / linePeriod` is **exactly integral**
in all 25 cases, and the identity closes at every one of the six distinct values
the corpus contains:

| cases | tiles | tile lines | (n-1) x gap | frame flyback | predicted | MEASURED `scanFramePeriod/linePeriod` |
|---|---|---|---|---|---|---|
| `baseline` and 13 others | 1 | 200 | 0 | 158 | 358 | **358.000** |
| `fb_frame_long` (`flybackTimePerFrame` 13 ms) | 1 | 200 | 0 | **314** | 514 | **514.000** |
| `mroi2`, `mroi2_zstack3`, `mroi2_lineavg2`, `mroi2_lineavg2_zstack3` | 2 | 400 | 50 | 158 | 608 | **608.000** |
| `mroi2_flyto_long` (fly-to 4 ms) | 2 | 400 | **98** | 158 | 656 | **656.000** |
| `mroi3_flyto_short` (fly-to 1 ms) | 3 | 600 | **52** | 158 | 810 | **810.000** |
| `mroi3` | 3 | 600 | 100 | 158 | 858 | **858.000** |

Note the identity is in **raw scan lines**: the line-averaged cases sit in the
same rows as their unaveraged twins (`mroi2_lineavg2` is 608 just like `mroi2`),
which is the direct evidence that `LineAveragingLineCount` is a storage-side
divide and not a scanning change.

Implementation: `analysis/core/mroi_layout.verify_frame_period`. Pinned by
`analysis/tests/test_mroi_corpus.py::test_frame_period_identity_holds`.

---

## 4. MROI in detail -- tiles, gaps and geometry

### 4.1 The gap is NOT a constant

MEASURED (manifest), the fly-to timing sweep -- same ROIs, same everything else:

| case | `flytoTimePerScanfield` | tiles | gap (scan lines) | page rows |
|---|---|---|---|---|
| `mroi3_flyto_short` | 1 ms | 3 | **26** | 652 |
| `mroi2` / `mroi3` | 2 ms | 2 / 3 | **50** | 450 / 700 |
| `mroi2_flyto_long` | 4 ms | 2 | **98** | 498 |

So a reader that pins 50 is right only at one setting and silently wrong at the
others. Pinned by
`analysis/tests/test_mroi_corpus.py::test_flyto_sweep_proves_the_gap_is_not_a_constant`.

### 4.2 The rule: round UP to an EVEN number of line periods

```
gap_scan_lines = 2 * ceil(flytoTimePerScanfield / linePeriod / 2)   [resonant only]
```

MEASURED (corpus headers): `flytoTimePerScanfield = 0.002`,
`linePeriod = 4.162192309933489e-05`, `scanMode = 'resonant'`.
`0.002 / 4.162192309933489e-05 = 48.0516`. ScanImage's own `round()` gives 48;
the files contain **50**. `ceil` gives 49, also wrong. Only round-up-to-even
gives 50.

What makes this conclusive rather than a curve fit is that **a second, materially
different transit goes through the same rule**: `flybackTimePerFrame / linePeriod
= 156.1677`, which the frame-period identity of 3.1(b) pins at **158**. The six
candidate rules (`round` / `ceil` / `floor`, each plain and even-rounded) round
48.0516 and 156.1677 differently, so requiring one rule to satisfy both
discriminates all six. Exactly one survives. Pinned by
`analysis/tests/test_mroi_layout.py::test_gap_rule_is_the_unique_survivor_of_both_identities`.

The 2026-08-02 sweep strengthened this further without new reasoning: the
identity now closes at **three** fly-to values (24.03 -> 26, 48.05 -> 50,
96.10 -> 98) and **two** frame-flyback values (156.17 -> 158, 312.34 -> 314).
All five round up to even; none is consistent with `round` or plain `ceil`.

Physically this is what a bidirectional resonant scanner requires: a transit must
consume a whole number of round trips, i.e. an EVEN number of line periods, so
the scan resumes travelling in the correct direction. Consistent, not
independently proven.

Implementation: `analysis/core/mroi_layout.scan_lines_for_transit`.

> **Do not hardcode 50.** Derive it, then **validate the result against the actual
> page height before slicing anything** and fail loudly on disagreement.
> `analysis/core/mroi_layout.build_layout` does both and raises `MroiLayoutError`
> rather than guessing. Pinning 50 in a *test* (as `test_si_frames_corpus.py`
> used to) is how a change becomes loud; branching on it in a *loader* is a latent
> bug. Pinned by
> `test_mroi_corpus.py::test_gap_disagreement_fails_loudly_on_a_real_corpus_file`.

### 4.3 ScanImage's own three loaders each get a DIFFERENT answer

MEASURED (SI source), 2026-08-02. All three compute the gap as
`round(flytoTimePerScanfield / linePeriod)`, times an "is this a resonant
scanner" guard -- and the guards differ:

| site | guard | gap on OUR files | error per gap |
|---|---|---|---|
| `+scanimage/+guis/+scanimagedataview/FrameScanDataView.m:361` | `strcmpi(hScan2D.scanMode,'resonant')` -> **1** | **48** | 2 rows short |
| `+scanimage/+util/getMroiFrameSequence.m:80` | `strcmp(hScan2D.scannerType,'Resonant')` -> **0**, because our files read `scannerType = 'RGG'` | **0** | the whole gap |
| `+scanimage/+util/getMroiDataFromTiff.m:100` | **no guard at all** | **48** | 2 rows short |

MEASURED (corpus headers): our files carry `scanMode = 'resonant'` **and**
`scannerType = 'RGG'`, which is why the middle row degenerates.

**The consequence is the point.** ScanImage's own viewer mis-slices our MROI
tiles by 2 rows per gap, and `getMroiFrameSequence` by the entire gap -- both
*silently*. Tile 2 comes back shifted, tile 3 shifted further. There is no crash
and no warning; the images simply are not where the loader thinks they are. This
is a different failure from the `LineAveragingLineCount` crash in 2.6.1, and
fixing one does not fix the other.

`mroi_layout.derive_gap_scan_lines` follows the two sites that agree (guarded on
resonant) and accepts `'RGG'` as resonant, which is what our files are.

### 4.4 Tile geometry

Per tile, from the RoiData block
(`RoiGroups.imagingRoiGroup.rois[k].scanfields[j]`):

| field | units | note |
|---|---|---|
| `centerXY` | DEGREES of optical scan angle | multiply by `SI.objectiveResolution` (161.275 um/deg on meso4) for microns |
| `sizeXY` | DEGREES | same |
| `pixelResolutionXY[0]` | pixels | page COLUMNS this tile uses. MEASURED (corpus headers): 700 for every tile in every MROI case, matching the page width exactly |
| `pixelResolutionXY[1]` | RAW SCAN LINES | divide by `LineAveragingLineCount` for stored rows (2.6) |
| `enable`, `zs`, `rotationDegrees`, `affine`, `pixelToRefTransform` | | `mroi_layout` drops disabled ROIs and refuses mixed `z` |

MEASURED (corpus headers), `mroi3`: three tiles, each
`centerXY = [0, 0] / [4.29561, 0] / [8.59122, 0]`,
`sizeXY = [3.9051, 1.240117811]`, `pixelResolutionXY = [700, 200]` --
i.e. identical scanfields tiled along X at 1.1x width spacing, all at `z = 0`.
That is what `sync/acquire_corpus.set_n_rois` constructs.

> **ScanImage collapses a single-element list to a bare object** in this JSON, at
> both the `rois` and the `scanfields` level. Handle both. `mroi_layout._imaging_scanfields`
> does.

### 4.5 MROI tile geometry is NOT available on the live ZMQ path

A saved TIFF is self-describing. **The live ZMQ stream is not.** MEASURED (SI
source, 2026-08-02): `RggScan.publishFrameMetadataToShm` (`RggScan.m:2546-2580`)
writes exactly eight scalars to the `shm_dict_SI` side-channel, and the wire
itself carries flat frames:

| key | source | use for MROI |
|---|---|---|
| `si_num_frames` | `hStackManager.numFramesPerVolume` | none |
| `si_num_slices` | `hStackManager.actualNumSlices` | none |
| `si_frames_per_slice` | `hStackManager.framesPerSlice` | none |
| `si_num_flyback_frames` | `hFastZ.numDiscardFlybackFrames` | none |
| `si_num_channels` | `numel(hChannels.channelsActive)` | none |
| `si_line_averaging` | `LineAveragingLineCount` | needed, and present |
| `si_pixels_per_line` | `hRoiManager.pixelsPerLine` | **actively misleading** |
| `si_lines_per_frame` | `hRoiManager.linesPerFrame` | **actively misleading** |

The last two are exactly the fields section 3 measures as 512 in all 25 cases
against real pages of 50-700 rows. So the only two published fields that look
like page geometry are the two that cannot be trusted under MROI -- and MROI is
on in every meso4 configuration. There is no per-ROI record on the wire at all,
and no `flytoTimePerScanfield` / `linePeriod` to derive the gap from.

**Consequence:** a live viewer cannot place MROI tiles. `ZmqSource.mroi_layout`
returns `None` and prints why, rather than inventing a layout. The opt-in escape
hatch is `--mroi-from <tiff acquired with the same ROI group>`, which borrows
that file's RoiData and is **refused** if its page height disagrees with the live
frame.

**Smallest change that would make it available (PROPOSED, not implemented).** In
`RggScan.publishFrameMetadataToShm`: (1) add `si_flyto_time_per_scanfield` and
`si_line_period` to the existing `metadata` struct -- two more scalars through
the mechanism already there; (2) publish the RoiData JSON once per acquisition
(it cannot change during one) to a string/array key `si_roi_data`, using the same
serialization that already produces the TIFF's RoiData block. Step 2 is the only
real work and it is re-use, not new code. With those three keys
`mroi_layout.layout_from_header` consumes the live path unchanged -- it takes
parsed header fields and a page shape, with no file I/O of its own.

---

## 5. Multi-file rollover -- the acquisition that is not all in your file

**Newly characterised 2026-08-02. Read this before trusting any frame count.**

When `SI.hScan2D.logFramesPerFile` is finite, ScanImage splits one acquisition
across `<stem>_00001.tif`, `<stem>_00002.tif`, ... It counts **frames**, so
nothing forces the split onto a volume boundary.

> **How this was characterised.** No corpus case rolls over
> (`logFramesPerFile = Inf` in all 25, MEASURED (corpus headers)), and the rig may
> not be driven from a test, so the fixture in
> `sync/tests/test_tiff_readers_identity.py` is built **OFFLINE by byte-splitting
> a corpus TIFF** into two structurally valid ScanImage BigTIFFs: the shared
> header block (BigTIFF header + SI magic + FrameData + RoiData) is copied
> verbatim into both halves, the tail half's page IFDs are rebased by the byte
> delta, and each page keeps its own `ImageDescription` and its exact pixels.
> `test_the_rollover_fixture_is_faithful` proves the split before anything is
> concluded from it. **A real `_00002.tif` from the rig has NOT been examined** --
> see the open question in section 12.

### 5.1 What every reader does today

| Reader | Volume-aligned rollover | Mid-volume rollover |
|---|---|---|
| `MemmapTiffSI` | reads file 1 correctly, **silently short** by everything in file 2 | raises `AssertionError` saying "**File may be truncated**" |
| `TiffSource` (`analysis/sources.py`) | same, **silently short** | `allow_truncated=True`, so it **silently discards** the partial volume and reports a clean stream |
| `sync.si_frames.extract_session` | **correct end to end** -- it is the only reader that takes a session rather than a file | grouping is wrong; warns "truncated file?" per file, which is the wrong diagnosis |

Five findings, all pinned in `sync/tests/test_tiff_readers_identity.py`:

1. **Neither `MemmapTiffSI` nor `TiffSource` can express a multi-file
   acquisition.** Neither takes a file list, a directory or a stem -- each opens
   exactly the path it was handed. A rolled-over acquisition is silently short by
   every volume in `_00002.tif`, with no exception, no warning, and no attribute
   a caller could check. Pinned by
   `test_neither_consumer_follows_the_rollover_to_the_second_file`, which also
   asserts the *absence* of `n_files` / `files` / `next_file` / `sibling_files`
   so that adding rollover support becomes a visible change.

2. **A mid-volume rollover is MISDIAGNOSED as truncation.** `_00001.tif` then has
   a page count that no whole number of volumes divides, and the reader's only
   vocabulary for that is "File may be truncated. Use allow_truncated=True". A
   caller cannot tell "the acquisition was interrupted" from "there is a second
   file you did not open". Pinned by
   `test_mid_volume_rollover_is_reported_as_truncation_not_rollover`.

3. **`_00002.tif` read alone regroups from ITS OWN page 0**, so every frame lands
   at the wrong `(t, plane)` -- with correct shapes, a plausible frame count and
   no error. This is the same silent-wrongness class as bugs B1/B2. Pinned by
   `test_second_file_of_a_mid_volume_rollover_is_silently_mis_grouped`, which
   checks that file 2's `[0,0,0]` really is the split-point page and really is
   not any true volume start.

4. **On a multi-channel file the rollover can fall BETWEEN the two channel pages
   of one acquired frame.** The whole grouping is then off by one page while
   `frameNumbers` -- which the `C` channel pages of a frame **share** (section 7)
   -- still reads exactly like a clean volume start. **Any contiguity or clock
   check built on `frameNumbers` is blind to it.** The reported volume is built
   from one channel page of the previous volume and one of the next, and it
   claims a real volume's frame number while doing so. Pinned by
   `test_si_frames_is_the_only_rollover_aware_reader_and_only_when_aligned`
   (the `k % C != 0` branch).

5. **`sync.si_frames.extract_session` handles a volume-aligned rollover
   correctly**: it walks every TIFF under the session, the combined volume count
   equals the acquisition's, every volume's `frameNumbers` is the one the
   manifest predicts, and it emits **no warnings**. Same test, aligned branch.

### 5.2 What to do about it

- If you are handed a path, **look for siblings** (`<stem>_0000*.tif`) before
  concluding anything about the frame count.
- For a whole session, use `sync.si_frames.extract_session` (section 8).
- If you must read pixels across a rollover today, group the concatenated page
  stream yourself: `pages_per_volume` is a property of the acquisition, not of
  the file, so volume `t` is pages `[t*ppv, (t+1)*ppv)` of the **concatenation**
  of every file in order. Nothing in the tree does this yet.

---

## 6. The explicit-layout trap

`analysis.sources.TiffSource(path, n_planes=..., n_channels=...)` (and the
matching `--n-planes` / `--n-channels` flags on `python -m analysis`) sets
`_explicit_planes` and takes a path that **never consults ScanImage's page layout
at all**:

```
page(t, p, c) = t * n_planes * n_channels + p * n_channels + c
```

There is **no flyback term**. On any file with flyback (i.e. any stack) passing
even the *correct* plane count produces, all with no exception and nothing on
stdout:

- **flyback pixels handed to the caller as ordinary image frames**;
- **real frames that are never delivered at all**, because the extra flyback
  pages push the tail past the last whole group;
- volumes after the first built from pages belonging to two different SI volumes.

Pinned by `test_tiff_readers_identity.py::test_explicit_layout_silently_mis_groups_every_flyback_file`.
Note what that test deliberately does **not** assert: that the volume count
changes. MEASURED, it often does not -- `zstack5` (fpzc 6, real 5, 18 pages) and
`zstack3_framesperslice2` (fpzc 7, real 6, 21 pages) both still report 3 volumes
because the floor divide happens to land there. A count check is exactly the kind
of assertion that misses this.

**The part that was fixed.** Until 2026-08-02 the explicit path also
short-circuited **before the refusal check of step 4 ever ran**, so an operator
passing `--n-planes 1` on a straddling-average file got a clean, silent,
contaminated stream from exactly the file the reader exists to refuse: **the same
file answering two opposite ways depending on one CLI flag** -- and it was the
flag someone reaches for when auto-detect "does not work".

`TiffSource.open()` now calls `lazy_tiff_reader.check_layout_readable`
**unconditionally**, before the explicit/auto branch (`analysis/sources.py:578-584`).
`check_layout_readable` is header-only and cheap, and it returns silently for a
non-ScanImage TIFF (there is no SI layout to contradict, so the caller's own
layout is its business). **Any reader with its own layout path must call it.**
Pinned by `test_explicit_layout_cannot_bypass_the_unreadable_layout_refusal`,
which tries several `(n_planes, n_channels)` shapes so a single lucky value
cannot pass.

Two further pins on this path, so its behaviour is known rather than assumed:

- `test_explicit_layout_is_correct_when_it_matches_the_file` -- on a
  **flyback-free** file, matching values give pixel-identical results to
  auto-detect. That is the contract the flag is for.
- `test_explicit_layout_with_wrong_values_never_refuses` -- wrong values regroup
  silently and never raise. `n_planes` one too large drops remainder pages;
  `n_planes` larger than the whole file delivers **zero** volumes and a blank
  dashboard indistinguishable from a dark recording (the exact symptom of bug
  B2); planes and channels swapped keeps the page and volume counts identical and
  changes the pixels.
- `test_explicit_layout_can_be_made_page_exact_but_keeps_flyback_as_a_plane` --
  the one explicit setting that gets the volume boundaries right is
  `n_planes = numFramesPerVolumeWithFlyback` (not the plane count). Every volume
  boundary then lands correctly, but the flyback frame arrives as an extra
  trailing plane. Worth knowing because it is the workaround an operator reaches
  for, and because it is wrong in a specific, nameable way rather than an
  unbounded one.

---

## 7. `frameNumbers`, timestamps, and the alignment anchor

All of these are **per-page** fields in the page's `ImageDescription`, not in the
`Software` tag. MEASURED (corpus headers), a full page description:

```
frameNumbers = 1
acquisitionNumbers = 1
frameNumberAcquisition = 1
frameTimestamps_sec = 0.000000000
acqTriggerTimestamps_sec = -0.000069085
nextFileMarkerTimestamps_sec = -1.000000000
endOfAcquisition = 0
endOfAcquisitionMode = 0
dcOverVoltage = 0
epoch = [2026  8  2 14 46 53.370]
auxTrigger0 = []   auxTrigger1 = []   auxTrigger2 = []   auxTrigger3 = []
I2CData = {}
```

### 7.1 `frameNumbers` -- the raw acquired-frame index

The model, true on **every** corpus file including the unreadable ones:

```
frameNumbers[k] == (k // n_channels + 1) * logAverageFactor
```

MEASURED (corpus headers). Pinned by
`test_tiff_readers_identity.py::test_frame_numbers_follow_the_page_rule_on_every_file`,
which is the test that keeps the whole `(t, f, c) -> page` model honest -- the
expected page index is derived from `si_readback` and checked against
`frameNumbers`, never against the reader.

Consequences:

- **flyback frames DO increment it** -- `zstack3` (3 planes + 1 flyback) has 12
  consecutive `frameNumbers` for 12 pages, so the flyback page is a real page
  with a real frame number.
- **channels DO NOT increment it** -- `zstack3_multicolor2` reads
  `1, 1, 2, 2, 3, 3, ...`: the `C` channel pages of one frame share one number.
- **an averaged page carries the LAST raw frame of its window**:

| case | pages | `frameNumbers` per page |
|---|---|---|
| `baseline` (avg 1) | 3 | 1, 2, 3 |
| `zstack3` (avg 1) | 12 | 1 .. 12 |
| `zstack3_multicolor2` (avg 1, 2 ch) | 24 | 1, 1, 2, 2, 3, 3, ... 12, 12 |
| `logavg4` (avg 4) | 3 | **4, 8, 12** |
| `zstack3_logavg2` (avg 2) | 10 | **2, 4, 6, ... 20** |
| `logavg2_fps2_z2` (avg 2) | 10 | **2, 4, 6, ... 20** |

**The volume-to-volume step is `numFramesPerVolumeWithFlyback`** -- never
`actualNumSlices * framesPerSlice` and never `pages_per_volume`. Getting this
wrong makes a contiguity check report a gap on every single volume. MEASURED
(manifest, `expected.frame_number_step` for all 25). Pinned by
`test_si_frames_corpus.py::test_frame_number_step_equals_frames_per_z_cycle`.

**And note its blind spot**: because the channel pages of one frame share a
number, `frameNumbers` cannot witness a page-level misalignment smaller than one
frame. That is what makes the sub-frame rollover of 5.1(4) invisible.

### 7.2 `frameTimestamps_sec` -- THE alignment anchor

**Use `frameTimestamps_sec` of a volume's FIRST page as that volume's
timestamp.** It is ScanImage's own clock, in seconds from acquisition start
(page 0 is exactly 0.0). This is the field the aligner regresses against the
volume-clock TTL train on `/vDAQ0/D0.0` (see [`../sync/WIRING.md`](../../mesotools/sync/WIRING.md)).

MEASURED (corpus headers): the per-page interval on `zstack3` is 0.014901 s =
1 / 67.111 Hz = the reported `SI.hRoiManager.scanFrameRate`, and the `C` channel
pages of one frame carry the **same** timestamp -- which is why ScanImage's own
viewer subsamples them (`obj.frameTs = header.frameTimestamps_sec(1:imgInfo.numChans:end)`,
MEASURED (SI source), `FrameScanDataView.m:376`).

**The averaging caveat.** On an averaged file the first page of a volume does not
timestamp the volume start; it timestamps the **end** of the first averaging
window. MEASURED (corpus headers): `logavg4` page 0 has
`frameTimestamps_sec = 0.044708`, which is frame 4 at the measured 0.0149 s frame
period, **not** frame 1 at t = 0. Any per-volume timestamp derived from an
averaged file is late by `(avg - 1)` frame periods unless corrected.

The clock is rigid enough to regress: pinned by
`test_si_frames_corpus.py::test_volume_timestamps_are_monotonic_and_regular`,
which requires strictly increasing volume timestamps and a volume-period spread
below 1 ms.

### 7.3 The other per-page fields

| field | what it is | use |
|---|---|---|
| `acquisitionNumbers` | which ScanImage acquisition this page belongs to | MEASURED (corpus headers): `1` on all 254 corpus pages. `si_frames.extract_session` warns when a session contains more than one, because a single volume-clock TTL train cannot be attributed to one acquisition without segmentation. |
| `frameNumberAcquisition` | frame index within the acquisition | equals `frameNumbers` throughout the corpus |
| `acqTriggerTimestamps_sec` | when the acquisition trigger fired | MEASURED (corpus headers): a real value on page 0 (`-0.000069085`), `-1.0` on every later page |
| `nextFileMarkerTimestamps_sec` | presumably the rollover marker | MEASURED (corpus headers): **`-1.0` on all 254 corpus pages**. The corpus never rolls over, so what it reads at a real rollover is **OPEN** -- but it is the obvious first thing to check if you get a real `_00002.tif`. |
| `endOfAcquisition` / `endOfAcquisitionMode` | end-of-acquisition marker | `si_frames` records `endOfAcquisition` of a volume's last page |
| `epoch = [YYYY MM DD hh mm ss.sss]` | wall clock at acquisition start | a coarse seed only (`si_frames.parse_epoch`). Nothing precise should depend on it. |
| `dcOverVoltage`, `auxTrigger0..3`, `I2CData` | hardware side-channels | unused here |

---

## 8. What already exists -- do not write a fourth reader

| You want | Use | What it gives you | What it will NOT do |
|---|---|---|---|
| **Per-volume timestamps for alignment**; a whole session; rollover-awareness | [`sync/si_frames.py`](../../mesotools/sync/si_frames.py) (`read_tiff`, `extract_session`, `--session` / `--tiff` CLI, writes `si_frames.jsonl`) | one row per VOLUME with `frame_timestamp_s`, `frame_number`, `acquisition_number`, `page_index`; walks every TIFF under a session; refuses ungroupable files and reports discarded pages instead of swallowing them | no pixels at all -- it never reads image data or touches rows, which is also why it is immune to the MROI/line-averaging traps |
| **Pixels, fast, 5-D** | [`lazy_tiff_reader.MemmapTiffSI`](../lazy_tiff_reader/memmap_tiff_si.py) | zero-copy strided view over the mapped file, `shape = (T, F, C, Y, X)` with flyback excluded and `F` = real frames per volume; `n_zplanes`, `frames_per_slice`, `resolution_xyz`, `acquisition_parameters`; raises `UnknownLayoutError` | one file only (no rollover); no MROI un-concatenation -- `Y` is the whole concatenated page |
| **Just the refusal check**, before you decide how to read | `lazy_tiff_reader.check_layout_readable(path)` | header-only, cheap, silent on non-SI TIFFs | nothing else |
| **Just the header**, without a 23 MB `matlabstr2py` parse | `lazy_tiff_reader.utils.read_si_framedata_params(path)` | `{'FrameData': {...whitelisted keys...}, 'RoiGroups': {...}, 'version': 3\|4}` | only the keys in `_SI_PARAMS` -- **add to that whitelist** if you need another field (e.g. `hStackManager.zs` and `logFramesPerFile` are NOT in it today) |
| **MROI tile geometry / placement** | [`analysis/core/mroi_layout.py`](../../mesotools/analysis/core/mroi_layout.py) (`layout_from_si_tiff`, `layout_from_header`, `build_layout`, `verify_frame_period`) | derives the gap, validates it against the real page height, `slice_tiles(page)` and `place(page)` on the last two dims (so `(Y,X)`, `(Z,Y,X)`, `(Z,C,Y,X)` all work), tile centres in degrees and microns; raises `MroiLayoutError` rather than guessing | pure geometry -- no file I/O except in `layout_from_si_tiff`; refuses tiles at differing z or differing pixel density |
| **A frame stream for the analysis graph** | `analysis.sources.TiffSource` | one VOLUME per `read()` as `(index, timestamp, ndarray)`, shaped `(Y,X)` / `(F,Y,X)` / `(F,C,Y,X)`; `allow_truncated=True` internally so short files still replay | see section 6 before passing `n_planes` / `n_channels` |

**There is exactly ONE copy of `lazy_tiff_reader`**, at `arco/lazy_tiff_reader`.
A second copy lived at `mesotools/submodules/lazy_tiff_reader` until 2026-08-02
and was deleted, because the two had diverged and **the one that actually got
imported was the older one** -- a bug fixed months earlier was still live in
production while the fix sat in the copy nothing loaded. Do not re-add it as a
submodule; `test_tiff_readers_corpus.py::test_there_is_exactly_one_reader_copy`
and `::test_reader_under_test_is_the_copy_that_actually_runs` fail if a second
tree reappears.

---

## 9. Gotchas that cost time

- **`tifffile.TiffFile.scanimage_metadata` can take minutes or time out.**
  ScanImage embeds the XROI `xroiProps` blob (full per-frame galvo/beam/FastZ
  waveforms) in `FrameData`, which balloons the `Software` tag from ~18 kB to
  **~23 MB**, and tifffile hands the whole blob to `matlabstr2py`. MEASURED
  (plan): it **timed out at 120 s** on a 342 MB file. Corpus files are much
  cheaper (MEASURED (corpus headers): 727 kB FrameData, 0.4-0.5 s), so a fast run
  on the corpus proves nothing about a real session file. For the page-to-volume
  arithmetic you need only the `Software` tag, so do a **targeted read**:
  ```python
  with tifffile.TiffFile(path) as tif:
      software = tif.pages[0].tags["Software"].value    # NOT tif.scanimage_metadata
  ```
  and regex the `SI.<...> = <value>` lines out of it. You only need
  `scanimage_metadata` for `RoiGroups` / `pixelResolutionXY`, i.e. for MROI tile
  geometry -- and even there, prefer
  `lazy_tiff_reader.utils.read_si_framedata_params`, which reads the raw blocks
  itself.

- **A single-plane file tests nothing.** Every archived file found on this rig in
  `F:/Virginia_meso` has `pages_per_volume == 1`, so the entire grouping code path
  was a no-op in every pre-corpus test. Two real bugs appeared the first time a
  genuine stack was parsed. If your test data is all single-plane, your grouping
  code is untested, not correct.

- **A plausible shape is not a correct read.** Every bug in this document
  produces a file that opens, with the right dtype and a believable shape. Two of
  them lose exactly half the frames. Assert against the page count and the
  header, never against "it looks like an image". This is why the identity tests
  compare **bytes** against the manifest-predicted page, and why
  `test_the_identity_pins_can_actually_fail` proves the three plausible wrong
  mappings really would select different pages.

- **The corpus is acquired DARK, and identity testing depends on it staying
  noisy.** MEASURED 2026-08-02: 23 of 23 cases have all pages byte-distinct
  (sensor noise, per-page std ~6.7-26). If that ever stops being true, every
  pixel-identity pin goes vacuous -- `test_every_corpus_page_is_byte_unique`
  fails loudly and `require_unique_pages` skips with a reason rather than passing.

- **ScanImage silently refuses settings.** Asking for 5 planes yielded 5 planes
  plus 1 flyback; asking for `framesPerSlice = 4` with `logAverageFactor = 4`
  yielded an unrecoverable file. Always read the configuration **back** from
  ScanImage after setting it, and derive expectations from the readback, never
  from what you asked for. `sync/acquire_corpus.py` does this and it is the whole
  reason the manifest is trustworthy.

- **ScanImage does not create `logFilePath`.** A grab against a missing directory
  writes no TIFF while `logFileCounter` still increments. `mkdir` first, set,
  then read back and confirm `exist(...,'dir')==7`.

- **`hStackManager.zs` and `hScan2D.logFramesPerFile` are not in the
  `_SI_PARAMS` whitelist**, so `read_si_framedata_params` will not return them.
  Add them to the list rather than falling back to `scanimage_metadata`.

---

## 10. Known-bad implementations (history, and SI's own)

Keep these visible so nobody reintroduces them.

### 10.1 Ours -- all FIXED 2026-08-02

| # | Bug | Symptom | Status |
|---|---|---|---|
| **B1** | `n_flyback = pages_per_z_cycle - n_zplanes` (omits `framesPerSlice`) | `framesperslice3` exposed 3 of 9 real frames; `zstack3_framesperslice2` 9 of 18. Silent. | FIXED -- `memmap_tiff_si.py` now subtracts `numFramesPerVolume`. Pinned by `test_memmap_dim1_is_every_real_frame_not_the_plane_count`. |
| **B2** | `logAverageFactor` not divided out; the resulting `AssertionError` swallowed by `allow_truncated` | `logavg4` delivered **0 frames** (blank dashboard, warning on stdout only) instead of 3; `zstack3_logavg2` delivered **1 volume of 3** | FIXED -- typed `UnknownLayoutError`, raised regardless of `allow_truncated`, re-raised by `TiffSource`. Pinned by `test_tiffsource_refuses_the_same_files_the_reader_refuses` and `test_tiffsource_never_silently_under_delivers`. |
| **B3** | Two diverged copies of `lazy_tiff_reader`; the one imported was the older | a fix landed months earlier was not running | FIXED -- submodule deleted; two tests keep it deleted. Section 8. |
| **B4** | The explicit-layout path bypassed the refusal | one file, two opposite answers, depending on a CLI flag | FIXED -- `check_layout_readable` called unconditionally. Section 6. |
| **B5** | `assert False, f"... {n_flyback} ..."` interpolated an unbound name | building the refusal message raised `NameError` instead of the refusal | FIXED. Pinned by `test_refusal_raises_a_diagnosable_error`. |

### 10.2 ScanImage's own loaders

Not ours to fix, but you will be asked "why does SI's viewer show something
different":

| site | what it gets wrong on our files |
|---|---|
| `+scanimage/+util/opentif.m:180` | `numSlicesPlusFlyback = hdr.numFramesPerVolume / numFrames` where `numFramesPerVolume` is the **with-flyback** count (undivided by `logAverageFactor`) and `numFrames = framesPerSlice / logAverageFactor`. On `logavg4` that predicts 15 images for a 3-page file and falls into the "Unexpected number of images" branch. Same class as our B2. MEASURED (SI source). |
| `FrameScanDataView.m:361`, `getMroiFrameSequence.m:80`, `getMroiDataFromTiff.m:100` | three different inter-tile gaps (48 / 0 / 48) where the file contains 50 -- section 4.3 |
| all three of the above | index tiles at `pixelResolutionXY` rows, undivided by `LineAveragingLineCount` -- run off the end of any line-averaged MROI page. Section 2.6.1 |

What SI gets **right**, and is worth citing as corroboration:
`extractHeaderData.m:52-53` derives `numFramesPerVolume = numFramesPerVolumeWithFlyback`
and `numDiscardFrames = numFramesPerVolumeWithFlyback - numFramesPerVolume`,
exactly as section 1 does; and `FrameScanDataView.m:384` appends the flyback
markers, which is our evidence that flyback is trailing.

---

## 11. How to check your reader

The corpus is executable documentation. The facts above are pinned by tests, so
drift fails a test rather than silently changing an answer.

```
corpus:   F:/data/meso4/si_parser_corpus            (25 cases + manifest.json)
tests:    mesotools/sync/tests/test_si_frames_corpus.py     page -> volume
          mesotools/sync/tests/test_tiff_readers_corpus.py  counts, shapes, refusals
          mesotools/sync/tests/test_tiff_readers_identity.py PIXEL identity, rollover,
                                                             the explicit-layout path
          mesotools/analysis/tests/test_mroi_corpus.py      MROI tiles + placement
          mesotools/analysis/tests/test_mroi_layout.py      MROI geometry (no corpus)
          mesotools/sync/tests/test_corpus_audit.py         the corpus is COMPLETE
re-acquire: python sync/acquire_corpus.py --volumes 3
            python sync/acquire_corpus.py --list
audit:      python sync/acquire_corpus.py --audit             # no rig needed
            python sync/acquire_corpus.py --migrate-manifest  # v1 -> v2, no rig
```

**Check `--audit` before trusting a green corpus run.** Every manifest-driven
test is parametrized over `manifest.json`, so a manifest listing fewer cases than
the corpus holds does not fail -- it runs fewer tests and passes. That happened:
a single `--only` run rewrote the manifest wholesale, taking it from 17 cases to
1, and the suite stayed green while covering a fraction of the data. Skipping
when the corpus is ABSENT is correct; passing when it is PARTIAL is not.

Three things now prevent a recurrence:

| guard | what it catches |
|---|---|
| `acquire_corpus.py` merges by case name, never replaces | the write that caused it |
| manifest schema v2 (`known_cases`, `acquired_at`, no top-level `volumes_per_case`) | a truncated manifest is self-detectable, and merged entries carry provenance |
| `test_si_frames_corpus.py` coverage guards + `test_corpus_audit.py` | data on disk the manifest omits, defined-but-unacquired cases, orphaned entries -- and the detector itself is tested against deliberately broken synthetic corpora |

Run with the full py311 interpreter:

```powershell
C:\Users\ScanImage\.conda\envs\py311\python.exe -m pytest sync/tests/ -q
cd analysis/tests && C:\Users\ScanImage\.conda\envs\py311\python.exe -m pytest test_mroi_corpus.py test_mroi_layout.py -v
```

> `test_mroi_corpus.py` enumerates the case DIRECTORIES rather than
> `manifest.json`, because the layout it tests is derived from each file's own
> RoiData and validated against its own page height -- so it needs no external
> expectations, and a manifest that goes stale cannot make it silently skip.

They skip cleanly when the corpus is absent, so they still run off-rig.

### 11.1 Rule -> enforcing test

| Rule (section) | Test |
|---|---|
| pages/volume, volume count, no discarded pages, per case (1) | `test_si_frames_corpus.py::test_parser_matches_scanimage_readback` |
| the page model matches the files on disk (1) | `test_tiff_readers_corpus.py::test_page_count_model_matches_the_files_on_disk` |
| `(t, f, c) -> page` is the RIGHT page, byte for byte (1.6) | `test_tiff_readers_identity.py::test_memmap_frame_is_the_page_the_manifest_predicts`, `::test_tiffsource_volume_holds_exactly_those_pages` |
| the page model is derived independently of the reader (1.6, 7.1) | `test_tiff_readers_identity.py::test_frame_numbers_follow_the_page_rule_on_every_file`, `::test_expected_page_index_matches_the_frameNumbers_in_the_header` |
| the identity pins can actually fail (9) | `test_tiff_readers_identity.py::test_the_identity_pins_can_actually_fail`, `::test_every_corpus_page_is_byte_unique` |
| flyback is trailing and never delivered (1.7, 2.3) | `test_tiff_readers_identity.py::test_memmap_never_returns_a_flyback_page` |
| `framesPerSlice` repeats are data, ordered plane-major (2.2) | `test_tiff_readers_corpus.py::test_memmap_dim1_is_every_real_frame_not_the_plane_count`, `::test_memmap_dim1_ordering_matches_hStackManager_zs`, `::test_memmap_frames_in_a_volume_are_distinct_pages` |
| `frameNumbers` steps by `fpzc` (7.1) | `test_si_frames_corpus.py::test_frame_number_step_equals_frames_per_z_cycle` |
| the SI clock is rigid enough to regress (7.2) | `test_si_frames_corpus.py::test_volume_timestamps_are_monotonic_and_regular` |
| `logAverageFactor` divides pages not frames (2.5) | `test_si_frames_corpus.py::test_log_average_factor_divides_pages_not_frames` |
| straddling averaging is REFUSED, not invented (2.5.1) | `test_si_frames_corpus.py::test_averaging_that_straddles_volumes_is_refused`, `test_tiff_readers_corpus.py::test_memmap_refuses_unrepresentable_averaging`, `::test_memmap_refusal_is_repeatable_and_identical`, `::test_only_the_manifest_derived_cases_are_refused` |
| the refusal is typed and diagnosable (1.4) | `test_tiff_readers_corpus.py::test_unknown_layout_error_is_a_value_error`, `::test_refusal_raises_a_diagnosable_error` |
| truncation is NOT "unknown layout" (1.5) | `test_tiff_readers_corpus.py::test_truncated_file_is_not_reported_as_an_unknown_layout`, `::test_allow_truncated_keeps_every_complete_volume`, `::test_tiffsource_replays_a_truncated_acquisition` |
| no consumer over- or under-delivers (8) | `test_tiff_readers_corpus.py::test_tiffsource_never_over_delivers`, `::test_tiffsource_never_silently_under_delivers`, `::test_tiffsource_and_memmap_agree` |
| line averaging over-reports rows (2.6) | `test_si_frames_corpus.py::test_line_averaging_over_reports_rows_in_metadata`, `test_mroi_corpus.py::test_line_averaged_tiles_are_placed_at_half_height` |
| `linesPerFrame` is unreliable under MROI (3) | `test_si_frames_corpus.py::test_lines_per_frame_is_unreliable_under_mroi`, `test_tiff_readers_corpus.py::test_memmap_rows_come_from_the_page_not_from_metadata` |
| the MROI gap rule is round-up-to-even (4.2) | `test_mroi_layout.py::test_gap_rule_is_the_unique_survivor_of_both_identities` |
| the gap is not a constant (4.1) | `test_mroi_corpus.py::test_flyto_sweep_proves_the_gap_is_not_a_constant` |
| the frame-period identity (3.1b) | `test_mroi_corpus.py::test_frame_period_identity_holds` |
| a wrong gap RAISES rather than mis-slicing (4.2) | `test_mroi_corpus.py::test_gap_disagreement_fails_loudly_on_a_real_corpus_file` |
| tiles land at their true `centerXY` (4.4) | `test_mroi_corpus.py::test_tiles_land_at_their_true_centerxy` |
| non-uniform tiles are refused, never assumed (4.4) | `test_mroi_layout.py::test_tiles_at_different_z_are_refused_explicitly` |
| rollover: the fixture is faithful (5) | `test_tiff_readers_identity.py::test_the_rollover_fixture_is_faithful` |
| rollover: aligned reads correctly; nothing follows to file 2 (5.1) | `::test_volume_aligned_rollover_reads_each_half_correctly`, `::test_neither_consumer_follows_the_rollover_to_the_second_file` |
| rollover: mid-volume is misdiagnosed / mis-grouped (5.1) | `::test_mid_volume_rollover_is_reported_as_truncation_not_rollover`, `::test_second_file_of_a_mid_volume_rollover_is_silently_mis_grouped`, `::test_si_frames_is_the_only_rollover_aware_reader_and_only_when_aligned` |
| the explicit-layout path (6) | `::test_explicit_layout_is_correct_when_it_matches_the_file`, `::test_explicit_layout_silently_mis_groups_every_flyback_file`, `::test_explicit_layout_cannot_bypass_the_unreadable_layout_refusal`, `::test_explicit_layout_with_wrong_values_never_refuses`, `::test_explicit_layout_can_be_made_page_exact_but_keeps_flyback_as_a_plane` |
| exactly one reader copy (8) | `test_tiff_readers_corpus.py::test_there_is_exactly_one_reader_copy`, `::test_reader_under_test_is_the_copy_that_actually_runs` |

Re-acquiring is safe: `acquire_corpus.py` refuses to run unless every beam power
is 0 (the whole corpus is acquired dark, no light on the sample), saves the full
prior configuration including the MROI ROI group via `RoiGroup.saveToFile`, and
restores it in a `finally` block. It needs `mlab` (see the `/matlab` skill) and a
live `hSI`. It **merges** into `manifest.json` rather than replacing it, so a
`--only` run cannot silently shrink the corpus.

---

## 12. Open questions -- do NOT resolve these by reasoning

Coverage the corpus does not have. Do not read a passing suite as proof of any of
these.

1. **A real multi-file rollover has never been examined.** Everything in section 5
   comes from an OFFLINE byte-split fixture. **OPEN: do `frameNumbers`,
   `frameNumberAcquisition` and `acquisitionNumbers` restart in `_00002.tif`, or
   continue?** The fixture assumes they continue (INFERRED: `frameNumbers` is
   documented as the raw acquired-frame index, and FrameData holds only
   non-varying parameters), and none of the section 5 assertions depend on it --
   but a reader that stitches files WILL depend on it. Also **OPEN: what does
   `nextFileMarkerTimestamps_sec` read at a real rollover?** It is `-1.0` on all
   254 corpus pages. Acquire with a finite `hSI.hScan2D.logFramesPerFile`.

2. **No averaged case where `logAverageFactor` divides `numFramesPerVolumeWithFlyback`.**
   All three averaged corpus cases are refused, so the divide-out path of 2.5 is
   pinned only by an archived file. Needs a `flyback == 0` acquisition with
   `avg > 1`.

3. **One line period only** (`4.162192309933489e-05` in all 25). The fly-to sweep
   varies the *transit* (1/2/4 ms) and the frame-flyback sweep varies a second
   transit (6.5/13 ms), so the rounding rule of 4.2 is now pinned at five
   distinct products -- but all share one denominator. A **zoom sweep** would be
   independent confirmation. No longer load-bearing, still worth having.

4. **No MROI case with tiles of differing size, differing pixel density,
   differing z, or different X separation.** Every tile in every case is a copy of
   the same scanfield at 1.1x width spacing at one depth.
   `analysis/core/mroi_layout.py` handles differing sizes and REFUSES differing
   density or z, but only the refusals are unit-tested; the differing-size
   placement path has never met real data. **This is the highest-value missing
   MROI case.**

5. **No case with more than 2 channels** (the rig runs 4), more than 5 planes, or
   a `stackDefinition = 'uniform'` stack. For the uniform case specifically:
   whether `numSlices` agrees with `actualNumSlices` there is OPEN. It does not
   change the rule (read `actualNumSlices`), but it would explain the disagreement.

6. **No genuinely truncated file** -- the truncation tests use a byte-truncated
   copy of a corpus file, not an acquisition that was actually interrupted.

7. **Is the gap expressed in pre-averaging scan lines, or are tiles concatenated
   before line averaging?** 2.6 shows the gap is INSIDE the division; which of
   the two mechanisms produces that is INFERRED and no measurement distinguishes
   them. It does not change the arithmetic.

8. **Why does a 1-frame volume get no flyback frame?** The pattern (flyback iff
   `numFramesPerVolume > 1`) is MEASURED in all 25, but the mechanism was not
   read out of SI's source, so the boundary case (a 1-frame volume with a long
   `flybackTime`) is INFERRED to be 0 rather than measured.

9. **`linePeriod` vs `scannerFrequency`.** `1 / (2 * 12012.9) = 4.16218e-5`
   against a header `linePeriod` of `4.162192309933489e-05` -- they agree to
   3 parts in 1e4, which is consistent with `scannerFrequency` being a rounded
   display value, but the exact relation is INFERRED.
