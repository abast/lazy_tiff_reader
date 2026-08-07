# The ScanImage TIFF metadata header — what is written, and how it is read

Companion to [`scanimage_tiff_layout.md`](scanimage_tiff_layout.md). That document
is about **pixels** — how pages map to frames, volumes, channels and MROI tiles.
This one is about the **metadata header**: the text block ScanImage writes at the
front of every TIFF, what is in it, how each value is encoded, which parts are
stock ScanImage versus added here, and exactly what `lazy_tiff_reader` does with
it.

Every number below was measured on
`F:\Virginia_meso\test\260806_dyepool_09_rfu1000_n200_00006.tif` (meso4, 5 MROI
tiles, 10 volumes) on 2026-08-07, and every encoding rule was confirmed by running
the live serialiser. Where something is inferred rather than measured it says so.

For the *meaning* of the XROI annotations and the MATLAB side that generates them,
see [`MROI_API_Project/annotations.md`](../../MROI_API_Project/annotations.md).

---

## 1. Where the header lives in the file

A ScanImage BigTIFF has a 16-byte vendor header wedged in immediately after the
standard BigTIFF header, and it contains the **lengths** of the two metadata
blocks. That is the fact everything else depends on: the blocks can be located by
arithmetic, with no TIFF tag parsing at all.

| bytes | content |
|---|---|
| 0–15 | standard BigTIFF header — byte-order mark `II`, magic `43`, first IFD offset |
| 16–19 | ScanImage magic, `0x07030301` → `117637889` little-endian |
| 20–23 | format version (`3` or `4`; this file is `4`) |
| 24–27 | `size0` — byte length of the **FrameData** block |
| 28–31 | `size1` — byte length of the **RoiData** block |
| 32 … | FrameData — MATLAB-style `SI.key = value` text, null-terminated |
| 32+`size0` … | RoiData — JSON |
| then | the image data |

Measured on the reference file:

```
file total            :    9,713,812 bytes
size0 = FrameData     :      736,998 bytes   (522 lines of text)
size1 = RoiData       :        6,940 bytes   (JSON)
image data            :    8,969,842 bytes
```

So the header is ~7.6% of this file. On a short acquisition it can dominate: the
header size is fixed by the *configuration*, not by the number of frames.

---

## 2. How a value is formatted

FrameData is a flat list of lines, one assignment per line:

```
SI.hRoiManager.linePeriod = 4.162355199999999e-05
SI.hScan2D.scanMode = 'resonant'
SI.hFastZ.enable = true
SI.hChannels.channelSave = 1
```

Nesting is expressed by **dots in the key**, not by structure — a nested struct
becomes one line per leaf field. The encoding is done by
`C:\Program Files\Vidrio\arco\scanimage_bravo\+most\+util\toString.m`, driven by
`structOrObj2Assignments.m`. Confirmed by running the real serialiser on each type:

| MATLAB value | header text |
|---|---|
| `pi` | `3.141592653589793` |
| `[1 2 3]` | `[1 2 3]` |
| `[1 2; 3 4]` | `[1 2;3 4]` (semicolon = row break) |
| `[]` or `0x0` empty | `[]` |
| `zeros(1,0)` | `zeros(1,0)` ← **a function call, not a literal** |
| `true` | `true` |
| `[true false]` | `[true false]` |
| `'resonant'` | `'resonant'` |
| `'it''s'` (contains a quote) | `'$it''s'` ← **`$` prefix, quotes doubled** |
| `{'ROI1','ROI2'}` | `{'ROI1' 'ROI2'}` (space-separated) |
| `{'a';'b'}` | `{'a';'b'}` |
| `{1,[2 3]}` | `{1 [2 3]}` |
| `{}` | `{}` |
| `int32(7)` | `7` (integer type is not recorded) |
| `ones(2,2,2)` (3-D) | `[2 2 2]&[1 1 1 1 1 1 1 1]` ← **shape`&`data** |
| a struct | recursed into, one line per leaf |
| anything else | `<unencodeable value>` |

Three of these bite a naive parser, and are called out again in § 9:
`zeros(m,n)`, the `$`-prefixed char, and the `shape&data` 3-D form.

### 2.1 The precision trap: nested numerics lose digits

`SI.m` asks for 16 significant figures — `mdlGetHeaderString(...,16)` at
`C:\Program Files\Vidrio\arco\scanimage_bravo\+scanimage\SI.m` lines 932 and 934.
**A numeric nested inside a struct never gets them.** The recursion in
`structOrObj2Assignments.m` lines 70–80 calls itself without passing
`numericPrecision`, so it falls back to `[]` at line 23, and `toString.m` line 16
defaults to 6.

Verified with one value in both positions, precision 16 requested:

```
SI.top               = 1786120369.86042   <- top level, 16 digits as asked
SI.nested.same_value = 1.78612e+09        <- nested, 6 digits
SI.as_char           = '1786120369.860'   <- char, exact
```

Visible in any real header — top level keeps its digits, nested does not:

| field | nesting | digits |
|---|---|---|
| `SI.hRoiManager.linePeriod = 4.162355199999999e-05` | top | 16 |
| `SI.xroiProps.ao_volts_raw.B = [1.41008 1.02525 …]` | nested | 6 |
| `SI.xroiProps.powerbox_mask = [0.179085 -0.0561576 …]` | nested | 6 |

Consequences, in order of how much they matter:

- **`ao_volts_raw` — the recorded AO waveform voltages — are stored to 6
  significant figures.** For a ±10 V DAC with 16-bit resolution (~0.3 mV steps)
  that is adequate, but it is an accident, not a decision. Do not treat
  `ao_volts_raw` as an exact record of what was played.
- It is what made `annotations_for.built_at_posix` useless as a timestamp: a
  10-digit posix value quantised to 1e4 s, i.e. ±5000 s. Fixed by writing that
  field as `char`, which sidesteps the mechanism entirely — see `annotations.md`.
- **This is not "the header writer uses `%g`."** A top-level property does get its
  16 digits. The loss is specific to struct nesting, and `xroiProps` is a struct,
  which is why everything under it is affected.

Fixing it at source means passing `numericPrecision` through both recursive calls
in a **shared vendor serialiser** used by every header field — a far wider blast
radius than any one field justifies, in a file that will conflict on the next
upstream ScanImage pull.

---

## 3. What is actually in there

522 lines, 50 top-level groups, and a wildly lopsided byte distribution — two
groups are 97.3% of the block:

| group | lines | bytes | share |
|---|---|---|---|
| `SI.xroiProps` | 55 | 504,054 | **68.39%** |
| `SI.hBeams` | 21 | 213,026 | **28.90%** |
| `SI.hScan2D` | 111 | 6,205 | 0.84% |
| `SI.hUserFunctions` | 29 | 1,647 | 0.22% |
| `SI.hStackManager` | 35 | 1,372 | 0.19% |
| `SI.hPhotostim` | 32 | 1,172 | 0.16% |
| `SI.hChannels` | 19 | 1,122 | 0.15% |
| `SI.hMotionManager` | 22 | 1,112 | 0.15% |
| `SI.hDisplay` | 24 | 945 | 0.13% |
| `SI.hRoiManager` | 23 | 872 | 0.12% |
| …38 more groups | ~100 | ~5,400 | 0.73% |

Note the line counts versus the byte counts. `SI.hScan2D` has **five times** as
many lines as `SI.hBeams` and 1/34th the bytes. Everything expensive is a handful
of very long lines:

```
SI.hBeams.powerBoxZs          212,075 bytes   <- STOCK ScanImage
SI.xroiProps.ao_volts_raw.B   180,887 bytes   <- added for XROI
SI.xroiProps.powerbox_mask    155,550 bytes   <- added for XROI
```

**`SI.hBeams.powerBoxZs` is stock ScanImage and is 29% of the header on its own.**
The header would be large even without the XROI additions.

---

## 4. What stock ScanImage writes

Nothing in the header is opted into. `most.Model` walks every property that is
**public `GetAccess` and not `Hidden`** — see `mdlGetHeaderableProps` at
`C:\Program Files\Vidrio\arco\scanimage_bravo\+most\Model.m` line 772 — and
serialises all of it. So the header is "every visible property of the SI model and
its subsystems", not a curated set.

The chain is:

```
SI.zzzInitializeLogging                     SI.m:1641   builds the byte buffer
  -> obj.getHeaderString(modelProps)        SI.m:922
     -> obj.mdlGetHeaderString(...,16)      SI.m:932/934  (also strips
                                                          'scanimage.SI.' -> 'SI.')
        -> most.util.structOrObj2Assignments Model.m:543
           -> most.util.toString                         per-value encoding
```

Stock content, by kind:

- **Geometry and timing** — `SI.hRoiManager.*` (`linePeriod`, `scanFramePeriod`,
  `scanFrameRate`, `scanVolumeRate`, `mroiEnable`), `SI.hScan2D.*`
  (`flytoTimePerScanfield`, `flybackTimePerFrame`, `scanMode`, `scannerType`,
  `logAverageFactor`, `LineAveragingLineCount`), `SI.objectiveResolution`.
- **Stack / volume structure** — `SI.hStackManager.*` (`actualNumSlices`,
  `actualNumVolumes`, `framesPerSlice`, `numFramesPerVolumeWithFlyback`,
  `stackDefinition`, `stackMode`), `SI.hFastZ.*`.
- **Channels and detectors** — `SI.hChannels.*` (`channelSave` decides what is on
  disk), `SI.hPmts.*`, `SI.hShutters.*`.
- **Beams** — `SI.hBeams.*`, including the 212 KB `powerBoxZs`.
- **Everything else the GUI happens to own** — `hDisplay`, `hUserFunctions`,
  `hPhotostim`, `hMotors`, `hTileManager`, `hCycleManager`, `hMotionManager`,
  `hIntegrationRoiManager`, `hCameraManager`, `hWSConnector`, `hConfigurationSaver`,
  `hCoordinateSystems`, `hWaveformManager`.
- **Version and session scalars** — `SI.VERSION_MAJOR/MINOR/UPDATE/COMMIT`,
  `SI.TIFF_FORMAT_VERSION`, `SI.LINE_FORMAT_VERSION`, `SI.acqState`, `SI.name`,
  `SI.errorMsg`, `SI.warnMsg`, `SI.userInfo`.

Then, in the **separate RoiData JSON block**, stock ScanImage writes the MROI
scanfield definitions — the geometric source of truth for tiles:

```
RoiGroups
  imagingRoiGroup        ver, classname, name, UserData, roiUuid, roiUuiduint64,
                         rois[5]  ->  each with scanfields:
                                        pixelResolutionXY  e.g. [630, 102]
                                        centerXY           e.g. [15.12855945, 0]
                                        sizeXY             e.g. [3.9051, 0.62005]
  photostimRoiGroups     null on this file
  integrationRoiGroup    present, empty rois
```

`pixelResolutionXY` is what `load_mroi_geometry` uses for per-tile height, and it
is why `linesPerFrame` / `pixelsPerLine` must never be trusted under MROI — see
`scanimage_tiff_layout.md`.

---

## 5. What the XROI additions write

`xroiProps` is **not** stock. It was added to the vendor fork by commit
`35b375b` ("Add XROI properties to hSI", abast, 2025-08-13), with
`xroiPropsExcluded` added later in `30c5874` (Virginia Ruetten, 2026-06-28) to keep
runtime handles *out* of the header. The declaration is a single line:

```33:33:c:\Program Files\Vidrio\arco\scanimage_bravo\+scanimage\SI.m
        xroiProps = struct('use_XROI_tasks', false);
```

Because it is a public, non-`Hidden` property of the SI model, the header writer
picks it up automatically. **Nobody chose to put annotations in the TIFF** — they
are there as a side effect of that visibility, and so are the waveform blobs.

Breakdown of the 504,054 bytes:

| field | lines | bytes | what it is |
|---|---|---|---|
| `ao_volts_raw` | 4 | 330,616 | the raw AO waveforms actually played (B, and per-actuator siblings) |
| `powerbox_mask` | 1 | 155,551 | per-sample power-box mask |
| `Annotations.Beam` | 10 | 5,557 | line/ROI index annotations, beam actuator |
| `Annotations.Galvo` | 8 | 5,554 | same, galvo actuator |
| `Annotations.FastZ` | 8 | 5,554 | same, FastZ actuator |
| `mroi_specs` | 9 | 397 | tile spec summary |
| `annotations_for` | 6 | 367 | fingerprint of what the annotation describes |
| `Annotations.{ROINames, LogicalRoiNames}` | 2 | 153 | tile names |
| `Annotations.{PlaneIndices, RequestedZs, AppliedZs, ScanimageZs}` | 4 | 203 | plane/Z bookkeeping |
| `use_XROI_tasks`, `has_powerbox`, `recompute_seq` | 3 | 102 | flags and a rebuild counter |

**The annotations are not the bloat.** The three annotation actuator groups plus
all their metadata come to ~17 KB — 3.4% of `xroiProps`. `ao_volts_raw` and
`powerbox_mask` are 96.4% of it. If the header size ever needs to come down, those
two are the target and the annotations are not.

### 5.1 The annotation fields

Every index is a **waveform sample index**, 1-based (MATLAB), per actuator.

| field | shape on the reference file | meaning |
|---|---|---|
| `ROIStart` / `ROIEnd` | 5 = one per tile | sample span of each scanfield, **including** its flyto dead time |
| `LineStart` / `LineEnd` | 510 = 5 tiles × 102 lines | sample span of each acquired line |
| `FrameFlyBackStart` / `FrameFlyBackEnd` | 1 | sample span of the frame flyback |
| `ROITransitionStart` / `ROITransitionEnd` | 4 = tiles − 1 | the flyto transit between adjacent tiles |
| `LineBlankingIndices` | Beam only | samples where the beam is blanked within a line |
| `FlybackBlankingIndices` | Beam only, 5 | samples where the beam is blanked at flyback |

**Gotcha: the sample rate differs per actuator, so the same line has different
index values in each group.** On the reference file `Beam.LineStart` steps by 14
samples per line while `Galvo.LineStart` and `FastZ.LineStart` step by 17. Never
mix fields across actuators, and never assume a shared period — derive it from the
actuator you are using (`annotation_line_period` does this by taking the median
`diff` of `LineStart`).

`annotations_for` is a fingerprint added so a reader can tell whether the
annotation describes *this* page: `roi_names`, `n_rois`, `lines_per_frame`,
`beam_line_period_samples`, `built_at_posix`, `built_at_iso`. The two timestamps
are `char`, deliberately — see § 2.1.

---

## 6. How `lazy_tiff_reader` reads it

All of the below is
`C:\Program Files\Vidrio\arco\lazy_tiff_reader\lazy_tiff_reader\utils\read_si_framedata_params.py`.

### 6.1 Why it does not use `tifffile`

`tifffile.scanimage_metadata` hands the entire FrameData blob to `matlabstr2py`,
which builds a Python object for every line — including the 212 KB, 181 KB and
156 KB blobs. On a large enough header it never returns. The module docstring
records the case that forced this (a 163 MB blob).

### 6.2 Step 1 — locate and read the blocks (`_read_si_blocks`, lines 193–234)

Confirm `II` and BigTIFF magic `43`; seek byte 16; unpack `magic, version, size0,
size1`; reject unless the magic matches and version is 3 or 4; then seek byte 32
and issue exactly **two reads**, `size0` bytes and `size1` bytes. No IFD walk, no
page iteration, no image data touched.

### 6.3 Step 2 — FrameData is grepped, never parsed

FrameData is treated as a flat text file. One regex, assembled once, is run over
the raw bytes with `re.MULTILINE`:

```268:280:C:\Program Files\Vidrio\arco\lazy_tiff_reader\lazy_tiff_reader\utils\read_si_framedata_params.py
    wanted = b'|'.join(re.escape(key).encode() for key in _SI_PARAMS)
    pattern = (rb'^(' + wanted + rb'|' + _ANNOTATION_PATTERN
               + rb'|' + _FINGERPRINT_PATTERN
               + rb')[ \t]*=[ \t]*([^\n\r]*)')

    frame_data: Dict[str, Any] = {}
    for m in re.finditer(pattern, frame_raw, re.MULTILINE):
        key = m.group(1).decode('ascii')
        frame_data[key] = _parse_si_value(
            m.group(2).rstrip(),
            unwrap_single='.Annotations.' not in key,
        )
```

The alternation has **three arms, and they work differently**:

- **`_SI_PARAMS` (lines 50–106) is an explicit whitelist** of 29 literal key
  names — the shape, timing and MROI-geometry keys. A key not on this list is
  invisible to the reader even though it is in the file. To expose one, add it
  here. (27 of the 29 are present in the reference file; the two absent ones are
  alternate spellings of the line-averaging key.)
- **`_ANNOTATION_PATTERN` (line 113) is generic:**
  `SI\.xroiProps\.Annotations\.[A-Za-z0-9_]+\.[A-Za-z0-9_]+`. Any actuator and any
  field matches, so a new actuator appears without a code change.
- **`_FINGERPRINT_PATTERN` (line 125) captures `annotations_for.*`**, likewise
  generically. Added 2026-08-07; before that the fingerprint was in the file but
  unreachable through the parser.

`unwrap_single` is switched **off** for annotations. Without that, a one-element
annotation would collapse to a scalar and callers that index it would break — the
`IndexError: too many indices for array: array is 0-dimensional` class of bug.

Cost and selectivity on the reference file: **5.8 ms**, 26 annotation matches, and
it decodes **16,639 of 736,998 bytes (2.3%)**. The other 720,359 bytes — including
all three big blobs — are never converted to Python objects. That selectivity *is*
the performance; it is not incidental.

### 6.4 Step 3 — RoiData is parsed properly, in full

```285:291:C:\Program Files\Vidrio\arco\lazy_tiff_reader\lazy_tiff_reader\utils\read_si_framedata_params.py
    if roi_raw:
        try:
            roi_meta = json.loads(roi_raw.split(b'\x00', 1)[0].decode('utf-8'))
        except (ValueError, UnicodeDecodeError):
            roi_meta = None
        if isinstance(roi_meta, dict):
            result.update(roi_meta)
```

Only 6,940 bytes, and all of it is wanted, so it gets a real JSON parse. The
`split(b'\x00', 1)[0]` matters — the block is null-terminated and the trailing
padding is not valid JSON.

### 6.5 What comes back

```python
{'FrameData': {...59 keys...},   # 27 whitelisted params + 26 annotation fields
                                 # + 6 annotations_for fingerprint fields
 'RoiGroups': {...},             # absent if the file has no RoiData block
 'version': 4}
```

The fingerprint arrives already typed, so a caller can compare it directly against
`load_mroi_geometry`. Measured on the two reference files:

```
_00006 -> {"roi_names": ["ROI1","ROI2","ROI3","ROI4","ROI5"], "n_rois": 5,
           "lines_per_frame": 510, "beam_line_period_samples": 14,
           "built_at_posix": "1786120733.194",
           "built_at_iso": "2026-08-07T12:38:53.194-04:00"}
_00007 -> {"roi_names": ["ROI1","ROI2","ROI3","ROI4"],        "n_rois": 4,
           "lines_per_frame": 408, ...}
```

`roi_names` is a **list**, never collapsed to a scalar for a one-ROI group — a
cell-array branch was added to `_parse_si_value` for this. The two timestamps are
`str` because the header stores them as `char` (§ 2.1); callers convert.

`None` for a non-ScanImage file, which is the signal the callers use to fall back.

### 6.6 Who consumes it

Two layers, and only one of them interprets the annotations.

**`MemmapTiffSI`** (`memmap_tiff_si.py`) calls the parser at line 146, exposes the
whole dict unmodified through its `metadata` property (lines 427–429), and uses
only shape and count keys — the 13 in `_extract_acquisition_parameters` (lines
324–338) plus the T/Z/C keys. It **never reads an annotation**; it carries them for
callers. On the reference file it yields `(10, 1, 1, 710, 630)` `int16`.

**`mroi_si_tiff`**
(`C:\Program Files\Vidrio\arco\lazy_tiff_reader\lazy_tiff_reader\mroi_si_tiff.py`
— moved into this repo from `mesoscope.utils` on 2026-08-07)
is the only layer that turns annotations into row masks. It takes this parser as
its *primary* path, with `tifffile` reserved for non-ScanImage files:

```52:58:C:\Program Files\Vidrio\arco\lazy_tiff_reader\lazy_tiff_reader\mroi_si_tiff.py
@lru_cache(maxsize=8)
def _read_si_metadata_cached(path: str, _stat: Tuple[int, int]) -> Dict[str, Any]:
    sm = read_si_framedata_params(path)
    if sm is None:
        with tifffile.TiffFile(path) as tif:
            return dict(tif.scanimage_metadata or {})
    return sm
```

Results are cached on `(path, size, mtime)`, because geometry and line masks are
requested separately but come from the same header scan.

---

## 7. What the reader does not capture

Three gaps, in descending order of how likely they are to bite.

*(A fourth — `annotations_for` being unreachable — was **fixed on 2026-08-07** by
adding `_FINGERPRINT_PATTERN`. Before that the fingerprint was in the file but no
pattern matched it, because `_ANNOTATION_PATTERN` needs a capital `Annotations`
followed by a dot and `annotations_for` has neither. A file written by a
pre-fingerprint writer still has no `annotations_for` at all, and its freshness
cannot be verified — absent must not be read as fresh.)*

1. **An empty numeric array arrives as the string `'zeros(1,0)'`.** MATLAB's
   `mat2str` emits a *function call* for an empty with a non-standard shape —
   `mat2str(zeros(1,0))` is `'zeros(1,0)'`, while `mat2str([])` is `'[]'`.
   `_parse_si_value` has no case for it, so it falls through to return-as-string.
   Live on `Annotations.Beam.LineBlankingIndices` in every file checked. Latent
   only because nothing reads that field yet: `_annotation_field` raises
   `ValueError: invalid literal for int() with base 10: 'zeros(1,0)'` the moment
   anyone does. Fix in `_parse_si_value`, not the call site — any future empty
   annotation field arrives the same way.

2. **A char containing a quote gets a `$` prefix.** `toString.m` lines 71–74 turn
   `it's` into `'$it''s'`. `_parse_si_value` strips the outer quotes and returns
   `$it''s`, leaving both the sentinel and the doubled quote. No current key hits
   this; a user-supplied string field (`SI.userInfo`, `SI.name`, a filename) could.

3. **A 3-D array is encoded `shape&data`** — `ones(2,2,2)` becomes
   `[2 2 2]&[1 1 1 1 1 1 1 1]` via `most.util.array2Str`. `_parse_si_value` returns
   that whole thing as a string. No whitelisted key is 3-D today.

Also worth knowing: **integer type is not preserved** (`int32(7)` and `7` are
indistinguishable in the header), and a property the serialiser cannot encode
becomes the literal text `<unencodeable value>`.

---

## 8. Checking a file yourself

To see the raw header text for a key, search the **whole file**, not the first few
hundred KB. The `xroiProps` block sits near the *end* of FrameData — byte 736,675
on the reference file — because the big blobs come first. A short-window probe
reports `annotations_for` as absent, which reads as "the writer is broken" rather
than "I did not look far enough". That mistake cost a wrong conclusion once.

```python
import re, struct
from pathlib import Path

p = Path(r"F:\Virginia_meso\test\260806_dyepool_09_rfu1000_n200_00006.tif")
with open(p, "rb") as f:
    f.seek(16)
    magic, version, size0, size1 = struct.unpack("<IIII", f.read(16))
    f.seek(32)
    frame_raw = f.read(size0)

for line in frame_raw.split(b"\n"):
    if b"annotations_for" in line:
        print(line.decode("ascii", "replace"))
```

For the parsed view, and the annotation-versus-geometry cross-check, use
`mroi_si_tiff.read_si_metadata` + `annotation_describes_geometry` rather than
hand-rolling it; the validation rules and their rationale are in
`annotations.md` § 8.

---

## 9. Summary

- The header is two blocks at a known offset, with their lengths in a fixed
  16-byte slot: **FrameData** as `SI.key = value` text, **RoiData** as JSON.
- Nothing is curated. Every public non-`Hidden` SI property is serialised, which
  is why the header is 737 KB and why the XROI annotations are in it at all.
- **97.3% of FrameData is two groups**, and the single largest line
  (`SI.hBeams.powerBoxZs`, 212 KB) is stock ScanImage, not an XROI addition.
- **Numerics nested in a struct are written at 6 significant figures** even though
  16 are requested, because the serialiser drops the precision argument when it
  recurses. This affects everything under `xroiProps`, including `ao_volts_raw`.
- `lazy_tiff_reader` **greps FrameData and parses RoiData** — two reads, one regex,
  2.3% of the bytes decoded, 5.8 ms. Params come from a literal whitelist;
  annotations from a generic pattern.
- It **carries** annotations but never interprets them. `mroi_si_tiff` (in this
  repo since 2026-08-07) is the only layer that converts them into row masks — see
  § 10 for how the two layers divide the work.

---

## 10. The two readers — `memmap_tiff_si` and `mroi_si_tiff`

There are **two reader layers**, and they answer two different questions. Both now
live in this repo — `mroi_si_tiff` moved here from `mesoscope` on 2026-08-07
(§ 10.3). This section says what each does and how the seam between them was
chosen.

### 10.1 `memmap_tiff_si.MemmapTiffSI` — pages to a 5-D array

`C:\Program Files\Vidrio\arco\lazy_tiff_reader\lazy_tiff_reader\memmap_tiff_si.py`

**Question it answers: which page is which frame?** It turns a flat stream of TIFF
pages into `(T, Z, C, Y, X)` and hands back zero-copy views, using
`numpy.lib.stride_tricks.as_strided` over one memory map. Nothing is copied for
full-frame access.

It knows nothing about ROIs. `Y` is the whole page height — 710 rows on the
reference file, blank flyto gaps included — because at this level a page is just a
rectangle.

Almost all of its difficulty is **page-to-volume arithmetic**, and the comments
record which parts were got wrong before:

- `numFramesPerVolumeWithFlyback` is the authority for on-disk pages per volume;
  `SI.hFastZ.discardFlybackFrames` is **not** reliable — it can read `True` while
  flyback frames are still written.
- `framesPerSlice` repeats are **real data**, not flyback. Using `n_zplanes` here
  instead of `n_zplanes * framesPerSlice` silently exposed 1 frame in 3 on a
  `framesPerSlice=3` acquisition.
- `logAverageFactor` means one page per N *acquired frames*, so every frame count
  from the header must be divided by it. Omitting this meant the divide never
  happened.
- `n_volumes` is computed from the **actual page count**, never from
  `actualNumVolumes`, which holds garbage for infinite acquisitions.

It also **refuses** in one specific case rather than guessing, via
`check_layout_readable` and `UnknownLayoutError`: when `logAverageFactor` does not
divide the volume, averaged pages straddle volume boundaries and per-volume data is
not recoverable under *any* grouping. The docstring for that error is worth reading
before anyone tries to "handle" it — `allow_truncated` deliberately does not
suppress it, and a caller that catches `ValueError` broadly must let this one
through.

`GappedMemmap` (`gapped_memmap.py`) is the sibling for the same job with a
different access strategy: exact byte offsets per page, shape always `(T, C, Y, X)`,
and efficient spatial sub-ROI reads (`data[50, 1, :100, :100]` reads only that
region). Its `method='scanimage'` path generates page offsets from the stride
pattern instead of iterating pages, which is where its speed comes from.

### 10.2 `mroi_si_tiff` — a page into tiles and masks

`C:\Program Files\Vidrio\arco\lazy_tiff_reader\lazy_tiff_reader\mroi_si_tiff.py`
*(moved into this repo from `mesoscope/mesoscope/utils/` on 2026-08-07 — see
§ 10.3)*

**Question it answers: what is actually inside one page?** It is the only layer
with any concept of an ROI. Four jobs:

1. **Tile layout.** `load_mroi_geometry` → `MroiGeometry` (`dims_xy`,
   `center_xy_um`, `size_xy_um`, `sf_skip`, `total_y`, `line_averaging`,
   `pixel_size_um`, `objective_resolution`). `sf_skip_lines` derives the flyto gap as
   `round(flytoTimePerScanfield / linePeriod)` and then *validates* it against
   `sum(heights) + (n-1)*gap == total_y`, inferring from geometry on mismatch.
   `roi_y_slices` yields `(roi_index, y0, y1)` bands and errors unless they exactly
   fill the page.
2. **Row masks.** Which stored rows are real: `annotation_line_period`,
   `acq_line_mask_from_annotations`, `annotation_describes_geometry`,
   `line_masks_for_geometry`, plus `erode_line_mask` / `shrink_line_mask`. This is
   where the annotations from § 5 are consumed — the only place in either repo.
3. **Spatial reassembly.** On disk the tiles are stacked along Y in *acquisition*
   order, which has nothing to do with where they were in the sample: two ROIs
   imaged side by side come back one above the other. `mroi_mosaic_layout` places
   each tile on a shared canvas from its recorded micron centre and size,
   `assemble_mosaic` pastes the pixels, `split_rois` is the alternative that keeps
   them as separate images.
4. **Choosing a reader.** `open_si_stack` returns a **pair** —
   `(reader, "gapped" | "memmap")` — picking `GappedMemmap` when `_prefer_gapped`
   detects that `MemmapTiffSI` would collapse multi-frame time into `T=1`.
   `stack_tyx(reader, layout)` then gives a uniform `(T, Y, X)` view either way,
   which is why the layout tag has to be carried alongside the reader.

Like the layer below, it **refuses rather than guesses** — `sf_skip_lines` raises
on irreconcilable Y arithmetic, `line_masks_for_geometry` discards annotations that
fail validation (with `StaleAnnotationWarning`) instead of applying them, and
`mroi_mosaic_layout` rejects tiles whose pixel sizes differ by more than 2% rather
than silently resampling them onto one canvas.

### 10.3 One repo, two layers (the seam was moved 2026-08-07)

The split is **not** duplicated logic — verified: exactly one file each of
`memmap_tiff_si.py`, `gapped_memmap.py` and `mroi_si_tiff.py` exists across both
trees. The header parse has one implementation (`read_si_framedata_params`) and both
layers use it; the geometry and mask code has one implementation and only
`mroi_si_tiff` has it. The layering is sound: pages→array below, page→tiles above.

What *was* wrong was where the seam fell: `mroi_si_tiff` lived in the `mesoscope`
repo even though it imported nothing from `mesoscope` — its only project
dependency was this package. That put the reader-selection logic
(`open_si_stack` / `_prefer_gapped`, which exists to detect `MemmapTiffSI`
mis-parsing multi-frame time as `T=1`) in a different repo from the classes it
selects between, and let the two trees version independently.

**Resolved 2026-08-07: `mroi_si_tiff` moved into this repo**, keeping the
two-layer split. What changed:

- `lazy_tiff_reader/mroi_si_tiff.py` is the one copy; its imports became
  relative. The MROI API is exported from the package root
  (`from lazy_tiff_reader import load_mroi_geometry, open_si_stack, ...`).
- Its synthetic-geometry test builders moved into
  `lazy_tiff_reader/testing.py` (importable, unlike a `tests/` dir), because
  `mesoscope`'s reformat tests build fake MROI pages with them too.
- **No alias was left behind.** `mesoscope.utils.mroi_si_tiff` is gone, not
  shimmed — two import paths for one module is exactly how the stale-copy
  incident of 2026-08-02 happened (a `PYTHONPATH` prepend made an older
  duplicate win for months). The four `mesoscope` importers were updated
  instead.
- `mesoscope` keeps what is genuinely its own: `reformat_si_tiff`,
  `si_temporal_noise`, and the reformat round-trip tests.

What the move deliberately does **not** fix: the lowest-level default is still a
page, not an image. `MemmapTiffSI(...)` on `_00006.tif` gives
`(10, 1, 1, 710, 630)`; those 710 rows are five 102-row tiles plus **200 blank
ones** (measured — `load_mroi_geometry` on the same file reports `n_rois=5`,
heights `[102]*5`, `total_y=710`). That is the correct answer to "which page is
which frame" and the wrong answer to "give me the data". The rule stands: **for
anything on this rig, start from the MROI layer** (`open_si_stack`,
`load_mroi_geometry`, `line_masks_for_geometry`) — reaching for `MemmapTiffSI`
directly gets you a page, and a page is not an image.

One caveat carried over from the old location: the MROI geometry arithmetic is
general ScanImage MROI, but its calibration (`objectiveResolution`, the
resonant-only gap rule, never trusting `linesPerFrame` under MROI) is documented
against meso4 configurations. Now that this repo owns it, this repo owns that
generality question too.
