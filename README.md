# lazy_tiff_reader

Zero-copy, memory-mapped access to ScanImage BigTIFF acquisitions.

`MemmapTiffSI` builds a strided numpy view directly over the mapped file, so
full-frame access copies nothing. `GappedMemmap` does arbitrary indexing with
explicit byte-offset computation.

```python
from lazy_tiff_reader import MemmapTiffSI

mm = MemmapTiffSI("acquisition_00001.tif")
mm.shape          # (T, Z, C, Y, X)
frame = mm[0, 0, 0]      # (Y, X) view -- no copy
volume = mm[3]           # (Z, C, Y, X)
```

---

## THERE IS EXACTLY ONE COPY OF THIS REPO

It lives at `arco/lazy_tiff_reader`. A second copy existed as
`mesotools/submodules/lazy_tiff_reader` until 2026-08-02 and was deleted, because
the two had diverged and **the one that actually got imported was the older
one** -- so a bug fixed months earlier was still live in production while the fix
sat in the copy nothing loaded.

Do not re-add it as a submodule. `mesotools` has a test
(`sync/tests/test_tiff_readers_corpus.py::test_there_is_exactly_one_reader_copy`)
that fails if a second tree reappears, because `activate.ps1` used to *prepend*
the submodule to `PYTHONPATH`, which made the shadow copy win over everything.

---

## Three things that will surprise you

Read [`docs/scanimage_tiff_layout.md`](docs/scanimage_tiff_layout.md) before
changing any arithmetic here -- it ships with this repo, so a standalone clone
has it. (It is mirrored from `mesotools/docs/`, which is canonical because the
corpus and the tests live there; `mesotools/sync/sync_layout_doc.py` resyncs it
and a test fails if the two drift.) It is the authoritative, measurement-backed
reference for how ScanImage pages map to volumes, verified against a 25-case
acquisition corpus. Summary of the traps this reader exists to handle:

### 1. `shape[1]` is REAL FRAMES per volume, not the plane count

With `framesPerSlice > 1` there are several unaveraged repeats at each depth, and
they are real data. Dimension 1 counts all of them:

```
shape[1] == actualNumSlices * framesPerSlice   (after frame averaging)
```

Ordering is plane-major with the repeats consecutive -- for 3 planes x 2 frames
the order is `z1 z1 z2 z2 z3 z3`, matching `hStackManager.zs`. Use
`mm.n_zplanes` and `mm.frames_per_slice` to separate depth from repeat.

Until 2026-08-02 those repeats were misclassified as flyback and silently
dropped: a `framesPerSlice=3` acquisition exposed 3 frames of 9, with no error.

### 2. `logAverageFactor` counts ACQUIRED frames, not pages

ScanImage writes one page per `logAverageFactor` acquired frames, so every frame
count in the header is in frames and must be divided before it is a page count.

### 3. Some files cannot be read at all -- and say so

```python
from lazy_tiff_reader import MemmapTiffSI, UnknownLayoutError

try:
    mm = MemmapTiffSI(path)
except UnknownLayoutError as e:
    ...   # do NOT fall back to a flat page stream
```

`UnknownLayoutError` (a `ValueError`) means the file is a ScanImage TIFF whose
page layout cannot be determined -- in practice, frame averaging whose factor
does not divide the volume. MEASURED: averaging is applied to the raw frame
stream **continuously, flyback included**, so one saved page physically averages
one volume's flyback frame with the next volume's first real frame. Those pixels
are contaminated; there is no best-effort answer.

**Callers that degrade to a flat page stream on failure MUST let this propagate.**
Degrading presents pages as if they were volumes, which is exactly the silent
wrongness this reader exists to prevent.

It is raised regardless of `allow_truncated`. That flag means "the acquisition
stopped mid-volume", never "I cannot determine the layout" -- collapsing the two
is how an unreadable file became a quietly short one.

> Operationally: **never combine frame averaging with FastZ flyback.** With one
> flyback frame it is arithmetically impossible for the factor to divide both the
> real and the with-flyback frame counts, so every such stack is unreadable.

---

## Verifying a change

Do NOT verify against a single file. Almost every archived ScanImage TIFF on this
rig has `pages_per_volume == 1`, which makes the grouping arithmetic a no-op --
that is precisely why two silent bugs survived for months.

Verify against the corpus, which sweeps z planes, flyback, channels,
framesPerSlice, frame averaging, line averaging, ROI count and the scanner timing
knobs:

```
cd mesotools
python -m pytest sync/tests/ -q            # 641 passed
python sync/acquire_corpus.py --list       # what the 25 cases cover
python sync/acquire_corpus.py --volumes 3  # re-acquire (rig only; acquires DARK)
```

The corpus lives at `F:/data/meso4/si_parser_corpus` with a machine-readable
`manifest.json` recording ScanImage's own readback per case.

## This repo's own tests

```
python -m pytest tests/ -q
```

Two tests fail on a machine without the `C:\nearline\spruston` share mounted
(`test_gapped_memmap.py`); that is a missing fixture, not a regression.
