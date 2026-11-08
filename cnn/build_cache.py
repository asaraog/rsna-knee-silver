"""Decode slices out of the zip once into a compact local array. Caching is a lever.

Training off /workspace every epoch is what strangled us: MooseFS is network
storage, so 15 slices/study starved the GPU to 16% and epoch time went 70s -> 428s.
Decoding once to the pod's LOCAL disk makes epochs seconds instead of minutes.

Every choice that shapes what the model sees is a flag here, and the filename
encodes the config so caches coexist and can be compared head to head:

  python3 build_cache.py --slices 8 --img 256
  python3 build_cache.py --slices 16 --img 224 --fluid 1
  python3 build_cache.py --slices 8 --img 256 --crop-mm 140 --keep-frac 0.5

Size is n_study * n_plane * slices * img^2 bytes, linear in slices and quadratic
in resolution. Local disk is ~30 GB, so:
    slices=8  img=256 -> 6.9 GB
    slices=16 img=224 -> 10.6 GB
    slices=8  img=384 -> 15.6 GB
    slices=16 img=320 -> 21.7 GB

Writes to CACHE_DIR: <name>.npy (uint8 images), <name>.mask.npy, <name>.json.

Directory mode (--root DIR) reads the same tree out of a directory instead of the
zip, for external data such as OAI. DIR must mirror the zip's layout:
<split>_images/<StudyInstanceUID>/<SeriesInstanceUID>/<slice>.dcm (4 path parts,
top dir starting with the split name, files ending .dcm). --contrast-slots, the
default slots and any --fluid filter need <split>_series.csv and <split>.csv in
DIR with the competition columns (SeriesInstanceUID, Anatomical_Plane,
Fluid_Sensitive; StudyInstanceUID). --physics-slots without those CSVs takes each
series' plane from ImageOrientationPatient and the study list from the tree.
Geometry, slot logic and the cache name are untouched, so a directory-built cache
is interchangeable with a zip-built one under the same flags. Idea: a teammate's
duck-typed DirArchive.

  python3 build_cache.py --root /data/oai --slices 16 --img 384 --crop-mm 130 \
      --centroid-crop --contrast-slots --name oai_cs16
"""
import argparse
import io
import re
import json
import os
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn.functional as F

ZIP = "/workspace/rsna-knee-abnormality-detection.zip"
CACHE_DIR = os.environ.get("CACHE_DIR", "/root/cache")   # LOCAL disk, not /workspace
ANTIALIAS = os.environ.get("ANTIALIAS", "0") == "1"
_local = __import__("threading").local()
_ARCHIVE = None          # DirArchive once --root is given; None means the zip


class DirArchive:
    """Quacks like zipfile.ZipFile over a directory tree (after a teammate's
    directory reader). Every read in this file goes through namelist(),
    read() and open(), so this is the only seam directory mode needs.

    open() matters: the header helpers stream 8 KB with zf().open(name); a shim
    with only read() would silently turn each of those into a whole-file read."""

    def __init__(self, root):
        self.root = root
        self._names = None

    def namelist(self):
        """Relative POSIX paths of every file under root, walked once, sorted so
        the middle-of-series header probes are deterministic across filesystems."""
        if self._names is None:
            names = []
            for dp, _, fns in os.walk(self.root, followlinks=True):
                rel = os.path.relpath(dp, self.root)
                for fn in fns:
                    p = fn if rel == "." else os.path.join(rel, fn)
                    names.append(p.replace(os.sep, "/"))
            self._names = sorted(names)
        return self._names

    def _path(self, name):
        p = os.path.join(self.root, *name.split("/"))
        if not os.path.isfile(p):
            raise KeyError(f"There is no item named {name!r} in the archive")
        return p

    def read(self, name):
        with io.open(self._path(name), "rb") as f:
            return f.read()

    def open(self, name, mode="r"):
        return io.open(self._path(name), "rb")

    def close(self):
        pass


def open_archive():
    """zipfile.ZipFile(ZIP) by default; the DirArchive installed by --root otherwise."""
    return zipfile.ZipFile(ZIP) if _ARCHIVE is None else _ARCHIVE


def zf():
    z = getattr(_local, "z", None)
    if z is None:
        z = _local.z = open_archive()
    return z


_IN_TAG = re.compile(rb"\x20\x00\x13\x00IS(..)", re.S)   # (0020,0013), explicit VR
_LAT_TAG = re.compile(rb"\x20\x00\x60\x00CS(..)", re.S)  # (0020,0060) Laterality
_IPP_TAG = re.compile(rb"\x20\x00\x32\x00DS(..)", re.S)  # ImagePositionPatient
_IOP_TAG = re.compile(rb"\x20\x00\x37\x00DS(..)", re.S)  # ImageOrientationPatient
_ROW_TAG = re.compile(rb"\x28\x00\x10\x00US(..)", re.S)  # Rows
_COL_TAG = re.compile(rb"\x28\x00\x11\x00US(..)", re.S)  # Columns
_PXS_TAG = re.compile(rb"\x28\x00\x30\x00DS(..)", re.S)  # PixelSpacing
_TR_TAG = re.compile(rb"\x18\x00\x80\x00DS(..)", re.S)   # RepetitionTime
_TE_TAG = re.compile(rb"\x18\x00\x81\x00DS(..)", re.S)   # EchoTime
_SD_TAG = re.compile(rb"\x08\x00\x3e\x10LO(..)", re.S)   # SeriesDescription
_FS_RX = re.compile(r"\bfs\b|fatsat|fat.sat|stir|spair|spir|\bwe\b|tirm|fatsup", re.I)


def series_class(name):
    """(is_fluid, is_fatsat, is_t1) from the header physics, byte-scanned.

    The provided CSV's one bit cannot separate a true T1 from a non-fat-sat T2;
    TR/TE can: TR<800 -> T1-weighted, TR>=800 & TE>60 -> T2, else PD."""
    try:
        with zf().open(name) as f:
            head = f.read(8192)
        tr = _ds_vals(head, _TR_TAG)
        te = _ds_vals(head, _TE_TAG)
        m = _SD_TAG.search(head)
        desc = ""
        if m:
            ln = int.from_bytes(m.group(1), "little")
            desc = head[m.end():m.end()+ln].decode("ascii", "ignore").lower()
        fs = bool(_FS_RX.search(desc))
        t1 = bool(tr and tr[0] < 800)
        fluid = not t1
        return fluid, fs, t1
    except Exception:
        return None, None, None


def _ds_vals(head, rx):
    m = rx.search(head)
    if not m:
        return None
    ln = int.from_bytes(m.group(1), "little")
    try:
        return [float(v) for v in
                head[m.end():m.end()+ln].decode("ascii", "ignore").strip().split("\\")]
    except Exception:
        return None


def geo_side(name, lim_mm=20.0):
    """'L'/'R'/None from scanner geometry: patient +x is the patient's LEFT, so the
    FOV center's x-coordinate tells the side when it sits clearly off the midline.
    Sagittal: the slice position IS x. Coronal/axial: center = origin + row_dir*cols/2,
    used only when the row direction is cleanly x-aligned. Anything ambiguous -> None."""
    try:
        with zf().open(name) as f:
            head = f.read(8192)
        ipp = _ds_vals(head, _IPP_TAG)
        iop = _ds_vals(head, _IOP_TAG)
        if not ipp or not iop or len(iop) < 6:
            return None
        rx_ = iop[0]
        if abs(rx_) < 0.2:                      # sagittal-like: row not along x
            cx = ipp[0]
        elif abs(rx_) > 0.9:                    # coronal/axial with x-aligned rows
            mcol = _COL_TAG.search(head)
            pxs = _ds_vals(head, _PXS_TAG)
            if not mcol or not pxs:
                return None
            cols = int.from_bytes(head[mcol.end():mcol.end()+2], "little")
            cx = ipp[0] + rx_ * cols * pxs[0] / 2.0
        else:
            return None
        if cx > lim_mm:
            return "L"
        if cx < -lim_mm:
            return "R"
        return None
    except Exception:
        return None


def geo_side_fov(name, lim_mm=20.0):
    """'L'/'R'/None from the TRUE FOV centre, valid for any slice orientation.

    geo_side() has an orientation branch: sagittal-like rows (|row_x| < 0.2) use
    ipp[0], x-aligned rows (|row_x| > 0.9) add row_dir*cols*dx/2, and anything
    between returns None. Oblique sagittals prescribed 12-23 degrees off-axis sit
    exactly in that gap (|row_x| 0.2-0.9), so ~25% of untagged RSNA sagittals and
    21% of OAI sagittals were silently left un-mirrored (found in a teammate's audit, 2026-09-24).
    The general form needs no branch:
        c = IPP + row_dir * cols*col_spacing/2 + col_dir * rows*row_spacing/2
    with row_dir = iop[0:3], col_dir = iop[3:6], and PixelSpacing (0028,0030) =
    [row spacing, column spacing] per DICOM C.7.6.2.1.1 (identical for the square
    pixels every knee series here has). Patient +x is the patient's LEFT; same
    +-lim_mm dead zone as geo_side(). Measured: resolves 100% of OAI sagittals
    (geo_side: 78.8%) with 100% agreement to the true side, and 97.0% of tagged
    RSNA sagittals at 98.9% agreement with the Laterality tag, identical to
    geo_side() wherever both resolve.

    OPT-IN via --geo-lat-fov (cache suffix _gf). Never wired into the _gl path:
    a checkpoint trained on a _gl cache saw the oblique knees un-mirrored and
    must keep seeing them that way at inference."""
    try:
        with zf().open(name) as f:
            head = f.read(8192)
        ipp = _ds_vals(head, _IPP_TAG)
        iop = _ds_vals(head, _IOP_TAG)
        pxs = _ds_vals(head, _PXS_TAG)
        mrow = _ROW_TAG.search(head)
        mcol = _COL_TAG.search(head)
        if (not ipp or len(ipp) < 3 or not iop or len(iop) < 6
                or not pxs or len(pxs) < 2 or not mrow or not mcol):
            return None
        rows = int.from_bytes(head[mrow.end():mrow.end()+2], "little")
        cols = int.from_bytes(head[mcol.end():mcol.end()+2], "little")
        if not rows or not cols:
            return None
        cx = (ipp[0] + iop[0] * cols * pxs[1] / 2.0
              + iop[3] * rows * pxs[0] / 2.0)
        if cx > lim_mm:
            return "L"
        if cx < -lim_mm:
            return "R"
        return None
    except Exception:
        return None


def header_plane(name):
    """'Sagittal' / 'Coronal' / 'Axial' / None from ImageOrientationPatient: the
    slice normal (row x col) points along patient x, y or z respectively. Only
    used by --root mode without a series CSV; the zip path never calls this."""
    try:
        with zf().open(name) as f:
            head = f.read(8192)
        iop = _ds_vals(head, _IOP_TAG)
        if not iop or len(iop) < 6:            # implicit VR or tag past 8 KB
            d = pydicom.dcmread(io.BytesIO(zf().read(name)),
                                stop_before_pixels=True, force=True)
            iop = [float(v) for v in getattr(d, "ImageOrientationPatient", [])]
        if len(iop) < 6:
            return None
        normal = np.cross(np.asarray(iop[:3]), np.asarray(iop[3:6]))
        return ("Sagittal", "Coronal", "Axial")[int(np.argmax(np.abs(normal)))]
    except Exception:
        return None


def laterality(name):
    """'L' / 'R' / None from the header, byte-scanned like instnum."""
    try:
        with zf().open(name) as f:
            head = f.read(8192)
        m = _LAT_TAG.search(head)
        if m:
            ln = int.from_bytes(m.group(1), "little")
            v = head[m.end():m.end() + ln].decode("ascii", "ignore").strip().upper()
            if v[:1] in ("L", "R"):
                return v[0]
    except Exception:
        pass
    return None


def instnum(name):
    """InstanceNumber by byte-scanning the first 8 KB for tag (0020,0013).

    Filenames are SOPInstanceUIDs, so sorted() is spatially meaningless: measured
    Spearman rho vs InstanceNumber is ~0 across sampled series. Every file in this
    corpus is Explicit VR Little Endian (verified), so the tag is the literal bytes
    20 00 13 00 'IS' <len:2>. A pydicom parse here made the build ~10x slower.
    """
    try:
        with zf().open(name) as f:
            head = f.read(8192)
        m = _IN_TAG.search(head)
        if m:
            ln = int.from_bytes(m.group(1), "little")
            val = head[m.end():m.end() + ln].decode("ascii", "ignore").strip()
            if val.lstrip("+-").isdigit():
                return int(val)
    except Exception:
        pass
    try:                                   # implicit VR or tag past 8 KB: rare
        d = pydicom.dcmread(io.BytesIO(zf().read(name)),
                            stop_before_pixels=True, force=True)
        return int(getattr(d, "InstanceNumber", 0) or 0)
    except Exception:
        return 0


def centroid_off(name, lim_mm=35.0):
    """Per-series crop offset (dy, dx) in mm, from the tissue-mask centroid of one
    slice. Keys on the leg/air silhouette, which pathology does not move; a shift
    beyond lim_mm is treated as a failed locate (metal void, fat-sat failure) and
    falls back to (0,0) - i.e. exactly the old center-crop behavior."""
    try:
        ds = pydicom.dcmread(io.BytesIO(zf().read(name)), force=True)
        a = _pixels(ds).astype(np.float32)
        px = getattr(ds, "PixelSpacing", None)
        if not px:
            return (0.0, 0.0)
        px = float(px[0])
        t = a > np.percentile(a, 60)
        ys, xs = np.nonzero(t)
        if len(ys) < 100:
            return (0.0, 0.0)
        h, w = a.shape
        dy, dx = (ys.mean() - h / 2) * px, (xs.mean() - w / 2) * px
        if abs(dy) > lim_mm or abs(dx) > lim_mm:
            return (0.0, 0.0)
        return (dy, dx)
    except Exception:
        return (0.0, 0.0)



def slice_step_mm(name):
    """SpacingBetweenSlices, else SliceThickness. None when neither is present."""
    try:
        d = pydicom.dcmread(io.BytesIO(zf().read(name)), stop_before_pixels=True, force=True)
        v = getattr(d, "SpacingBetweenSlices", None) or getattr(d, "SliceThickness", None)
        return float(v) if v else None
    except Exception:
        return None

def _pixels(ds):
    """ds.pixel_array, with a fallback for DICOMs whose PixelData is 8-bit while the
    header declares BitsAllocated=16. Full-corpus scan 2026-08-25: exactly 2 of 819,635
    training files do this (SOP ...761543626 and ...685746320), both at exactly 1.00
    bytes/pixel against an expected 2.00. pydicom raises on them, and read_one used to
    swallow that and drop the slice - so the pixels were recoverable but silently lost.
    Confirmed on the forum as a distributed-dataset defect, not a bad download."""
    try:
        return ds.pixel_array
    except Exception:
        r, c = int(ds.Rows), int(ds.Columns)
        spp = int(getattr(ds, "SamplesPerPixel", 1) or 1)
        raw = ds.PixelData
        if len(raw) >= r * c * spp:
            return np.frombuffer(raw[:r * c * spp], np.uint8).reshape(r, c)
        raise


def read_one(name, img, crop_mm, off_mm=(0.0, 0.0)):
    try:
        ds = pydicom.dcmread(io.BytesIO(zf().read(name)), force=True)
        a = _pixels(ds).astype(np.float32)
        a = a * float(getattr(ds, "RescaleSlope", 1) or 1) \
            + float(getattr(ds, "RescaleIntercept", 0) or 0)
        px = getattr(ds, "PixelSpacing", None)
        if px and crop_mm:
            pxf = float(px[0])
            want = int(round(crop_mm / pxf))
            h, w = a.shape
            if 16 < want < min(h, w):
                half = want // 2
                cy = h // 2 + int(round(off_mm[0] / pxf))
                cx = w // 2 + int(round(off_mm[1] / pxf))
                cy = min(max(cy, half), h - half)   # window stays in bounds
                cx = min(max(cx, half), w - half)
                a = a[cy - half:cy + half, cx - half:cx + half]
        lo, hi = np.percentile(a, [1, 99])
        a = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)
        t = torch.from_numpy(np.ascontiguousarray(a))[None, None]
        # 56% of studies are downsampled here (census 2026-08-22: crop is 411px at the
        # median, 520px at p75, against a 384px output). Plain bilinear has no low-pass
        # prefilter, so thin structures - fracture lines, meniscal tears - alias. antialias
        # applies proper area weighting on downsample and is a no-op when upsampling.
        t = F.interpolate(t, (img, img), mode="bilinear", align_corners=False,
                          antialias=ANTIALIAS)[0, 0]
        return (t * 255).round().clamp(0, 255).to(torch.uint8).numpy()
    except Exception:
        return None


_ID_DTYPE = {"SeriesInstanceUID": str, "StudyInstanceUID": str}


def dir_tables(z, a, names, planes):
    """plane_of, fluid_of, studies for --root mode; the zip path never gets here.

    With <split>_series.csv and <split>.csv in the root (competition columns)
    this is the zip path exactly, except the id columns are forced to str: OAI
    series ids are all-digit barcodes, pandas infers int64, every lookup against
    the str parsed out of the file path misses, and the build 'succeeds' with 0
    slots and a file of zeros (a teammate hit this). Without the CSVs, --physics-slots
    takes the study list from the tree and each series' plane from its header;
    everything else needs the CSVs because Fluid_Sensitive only exists there."""
    scsv, ccsv = f"{a.split}_series.csv", f"{a.split}.csv"
    have = set(names)
    if scsv in have and ccsv in have:
        ser = pd.read_csv(io.BytesIO(z.read(scsv)), dtype=_ID_DTYPE)
        plane_of = dict(zip(ser.SeriesInstanceUID, ser.Anatomical_Plane))
        fluid_of = dict(zip(ser.SeriesInstanceUID, ser.Fluid_Sensitive))
        studies = list(pd.read_csv(io.BytesIO(z.read(ccsv)), dtype=_ID_DTYPE).StudyInstanceUID)
        return plane_of, fluid_of, studies
    if not a.physics_slots or a.fluid >= 0:
        raise SystemExit(f"--root {a.root}: {scsv} and {ccsv} not found in the root; "
                         "add them (competition columns) or pass --physics-slots "
                         "with --fluid -1 so planes come from the DICOM headers")
    plane_of, seen = {}, set()
    for n in names:
        if not n.endswith(".dcm"):
            continue
        parts = n.split("/")
        if len(parts) != 4 or not parts[0].startswith(a.split):
            continue
        _, st, se, _ = parts
        seen.add(st)
        if se not in plane_of:                  # one header read per series
            plane_of[se] = header_plane(n)
    studies = sorted(seen)
    n_pl = sum(1 for v in plane_of.values() if v in planes)
    print(f"--root {a.root}: no CSVs; {len(studies)} studies from the tree, "
          f"{n_pl}/{len(plane_of)} series given a plane by ImageOrientationPatient",
          flush=True)
    return plane_of, {}, studies


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=None,
                   help="directory of DICOMs to read instead of the competition zip; "
                        "same tree layout as the zip (see the runbook above)")
    p.add_argument("--slices", type=int, default=8, help="slices kept per plane")
    p.add_argument("--slab-mm", dest="slab_mm", type=float, default=0.0,
                   help="sample a fixed physical slab (mm) instead of a fraction "
                        "of the stack; 0 keeps the legacy keep-frac behaviour")
    p.add_argument("--img", type=int, default=256, help="pixels; quadratic cost")
    p.add_argument("--crop-mm", type=float, default=160.0,
                   help="physical extent of the centre crop; 0 disables")
    p.add_argument("--slice-power", dest="slice_power", type=float, default=1.0,
                   help="exponent on the slice-position curve. 1.0 = uniform (the "
                        "default nobody ever chose); >1 concentrates slices toward the "
                        "centre of the stack; <1 spreads them toward the edges.")
    p.add_argument("--keep-frac", type=float, default=0.60,
                   help="central fraction of each stack to sample from")
    p.add_argument("--planes", default="Sagittal,Coronal,Axial")
    p.add_argument("--fluid", type=int, default=-1,
                   help="-1 any series, 1 fluid-sensitive only, 0 non-fluid only. "
                        "Note Fluid_Sensitive and Fat_Suppression are the SAME column "
                        "in the provided CSVs, so this is one bit, not two")
    p.add_argument("--split", default="train", choices=["train", "test"])
    p.add_argument("--workers", type=int, default=64)
    p.add_argument("--name", default=None)
    p.add_argument("--true-select", action="store_true",
                   help="within each (plane, Fluid_Sensitive) slot, prefer the series "
                        "whose header physics best matches the slot's intent (true "
                        "fat-sat fluid for f=1, true T1 for f=0) instead of just the "
                        "densest stack. Same slot semantics; better occupants.")
    p.add_argument("--pos-order", dest="pos_order", action="store_true",
                   help="order sagittal slices by patient-x position (IPP), not InstanceNumber; with --lat-norm, "
                        "cache index 0 = medial for every study. Cache name suffix _po.")
    p.add_argument("--geo-lat", action="store_true",
                   help="when the Laterality tag is absent, read the side from scanner "
                        "geometry (FOV center x vs patient midline, 20mm dead zone). "
                        "Extends lat-norm from ~half of studies to nearly all.")
    p.add_argument("--geo-lat-fov", action="store_true",
                   help="like --geo-lat, but the side comes from the true FOV centre "
                        "(IPP + row_dir*cols*dx/2 + col_dir*rows*dy/2) instead of "
                        "geo_side()'s two orientation branches, so oblique sagittals "
                        "(|row_x| 0.2-0.9, ~25%% of untagged RSNA sagittals) resolve "
                        "too. Cache suffix _gf, not _gl. Checkpoints trained on a _gf "
                        "cache need GEO_LAT_FOV=True in the kernel; _gl ones must not.")
    p.add_argument("--centroid-crop", action="store_true",
                   help="center the crop on the tissue centroid instead of the image "
                        "center (one offset per series, 35mm clamp). Measured: median "
                        "offset 12.8mm, 21%% of series >20mm; mandatory for crops "
                        "tighter than ~140mm.")
    p.add_argument("--lat-norm", action="store_true",
                   help="mirror right knees onto a left-knee convention: coronal and "
                        "axial slices flip horizontally, sagittal slice order reverses. "
                        "4 of 12 labels are medial/lateral pairs; without this the model "
                        "must infer sidedness from anatomy. Unknown laterality = no flip.")
    p.add_argument("--physics-slots", dest="physics_slots", action="store_true",
                   help="slots = plane x RECOVERED contrast class, read from the DICOM "
                        "headers instead of the CSV. Fluid_Sensitive and Fat_Suppression "
                        "are byte-identical in the provided CSVs - one bit, not two - so "
                        "a fat-suppressed fluid-sensitive series (bone bruise screams) "
                        "and a non-suppressed one (fat also bright) land in the same "
                        "slot and cannot be told apart. TR/TE and SeriesDescription "
                        "never lost the distinction: TR<800 -> T1, else fluid-sensitive, "
                        "and the description carries the fat-sat marker. Three classes "
                        "per plane (t1, fluid+fatsat, fluid-nofatsat) = 9 slots.")
    p.add_argument("--contrast-slots", action="store_true",
                   help="slots = plane x Fluid_Sensitive (6 slots instead of 3), so the "
                        "model sees both contrasts of each plane instead of whichever "
                        "series happened to be densest. Uses the provided CSV bit, which "
                        "is coarse (it cannot split T1 from non-fat-sat T2) but exists "
                        "for the test set too, so inference needs no header recovery.")
    a = p.parse_args()
    if a.root:
        global _ARCHIVE
        if not os.path.isdir(a.root):
            raise SystemExit(f"--root {a.root}: not a directory")
        _ARCHIVE = DirArchive(a.root)

    planes = a.planes.split(",")
    if a.physics_slots:
        slot_keys = [(pl, c) for pl in planes for c in ("t1", "fluid_fs", "fluid_nofs")]
    else:
        slot_keys = ([(pl, f) for pl in planes for f in (1, 0)] if a.contrast_slots
                 else [(pl, None) for pl in planes])
    name = a.name or (f"{a.split}_s{a.slices}_i{a.img}_c{int(a.crop_mm)}"
                      f"_k{int(a.keep_frac*100)}_f{a.fluid}"
                      + ("" if a.slice_power == 1.0 else f"_sp{int(a.slice_power*10)}")
                      + ("_ps" if a.physics_slots else "")
                      + ("_cs" if a.contrast_slots and not a.physics_slots else "")
                      + ("_lat" if a.lat_norm else "")
                      + ("_gf" if a.geo_lat_fov else ("_gl" if a.geo_lat else ""))
                      + ("_ts" if a.true_select else "")
                      + ("_cc" if a.centroid_crop else "")
                      + ("_po" if a.pos_order else ""))
    os.makedirs(CACHE_DIR, exist_ok=True)
    out = os.path.join(CACHE_DIR, name)

    t0 = time.time()
    z = open_archive()                # zipfile.ZipFile(ZIP), or the --root DirArchive
    names = z.namelist()
    if a.root:
        plane_of, fluid_of, studies = dir_tables(z, a, names, planes)
    else:
        ser = pd.read_csv(io.BytesIO(z.read(f"{a.split}_series.csv")))
        plane_of = dict(zip(ser.SeriesInstanceUID, ser.Anatomical_Plane))
        fluid_of = dict(zip(ser.SeriesInstanceUID, ser.Fluid_Sensitive))
        studies = list(pd.read_csv(io.BytesIO(z.read(f"{a.split}.csv"))).StudyInstanceUID)
    print(f"{len(studies)} studies, zip parsed in {time.time()-t0:.1f}s", flush=True)

    # study -> plane -> series -> sorted slice names
    tree = {}
    for n in names:
        if not n.endswith(".dcm"):
            continue
        parts = n.split("/")
        if len(parts) != 4 or not parts[0].startswith(a.split):
            continue
        _, st, se, _ = parts
        pl = plane_of.get(se)
        if pl not in planes:
            continue
        if a.fluid >= 0 and int(fluid_of.get(se, -1)) != a.fluid:
            continue
        if a.physics_slots:
            key = (pl, None)          # class assigned below, after files are grouped
        elif a.contrast_slots:
            key = (pl, int(fluid_of.get(se, -1)))
        else:
            key = (pl, None)
        tree.setdefault(st, {}).setdefault(key, {}).setdefault(se, []).append(n)

    if a.physics_slots:
        # One header read per series (middle slice) decides its class. Doing it here,
        # after grouping, costs one read per series rather than one per file.
        def classify(files):
            fluid, fs, t1 = series_class(files[len(files) // 2])
            if fluid is None:
                return "fluid_nofs"          # unreadable header: the commonest class
            if t1:
                return "t1"
            return "fluid_fs" if fs else "fluid_nofs"

        for st, byslot in list(tree.items()):
            regrouped = {}
            for (pl, _), series in byslot.items():
                for se, files in series.items():
                    regrouped.setdefault((pl, classify(files)), {})[se] = files
            tree[st] = regrouped

    sidx = {s: i for i, s in enumerate(studies)}
    shape = (len(studies), len(slot_keys), a.slices, a.img, a.img)
    gb = np.prod(shape) / 1024 ** 3
    free = os.statvfs(CACHE_DIR)
    if gb > free.f_bavail * free.f_frsize / 1024 ** 3 - 1:
        raise SystemExit(f"only {free.f_bavail*free.f_frsize/1024**3:.1f} GB free on {CACHE_DIR}")
    # disk-backed, not in RAM: a 384px cache is 62 GB and the 4090 pods have 31 GB RAM
    arr = np.lib.format.open_memmap(f"{out}.npy", mode="w+", dtype=np.uint8, shape=shape)
    mask = np.zeros((len(studies), len(slot_keys)), np.float32)
    print(f"cache {shape} = {gb:.1f} GB -> {out}.npy (memmap)", flush=True)

    def pick_series(series, key):
        if not a.true_select or key[1] is None:
            return max(series.values(), key=len)
        want_fluid = bool(key[1])
        best, best_score = None, (-1, -1)
        for files in series.values():
            fluid, fs, t1 = series_class(files[len(files)//2])
            if fluid is None:
                score = 0
            elif want_fluid:
                score = 2 if (fluid and fs) else (1 if fluid else 0)
            else:
                score = 2 if t1 else (1 if not fs else 0)
            if (score, len(files)) > best_score:
                best_score, best = (score, len(files)), files
        return best

    jobs = []
    for st, byslot in tree.items():
        if st not in sidx:
            continue
        for pi, key in enumerate(slot_keys):
            series = byslot.get(key)
            if series:
                jobs.append((sidx[st], pi, pick_series(series, key)))
    print(f"{len(jobs)} (study,plane) slots", flush=True)

    done = [0]
    pad = (1 - a.keep_frac) / 2
    def work(job):
        si, pi, files = job
        # spatial order first; only then does "central keep_frac" mean the joint
        files = [f for _, f in sorted((instnum(f), f) for f in files)]
        if a.pos_order and slot_keys[pi][0] == "Sagittal" and len(files) > 1:
            # InstanceNumber is NOT anatomical: RSNA census 2026-09-27, 633 sagittal series, InstanceNumber runs
            # along patient +x in 295 and along -x in 338, independent of knee side. Put every sagittal stack in
            # ascending patient-x order (x = patient LEFT) so that, after the --lat-norm reversal of right knees
            # below, cache index 0 is the MEDIAL end for every study (left knee: medial = -x).
            try:
                x0 = _ds_vals(zf().open(files[0]).read(8192), _IPP_TAG)
                x1 = _ds_vals(zf().open(files[-1]).read(8192), _IPP_TAG)
                if x0 and x1 and x0[0] > x1[0]:
                    files = files[::-1]
            except Exception:
                pass
        n = len(files)
        lo, hi = int(pad * (n - 1)), int((1 - pad) * (n - 1))
        if a.slab_mm:
            # Through-plane normalisation. Census 2026-08-22: slice step ranges 2.2-5.0mm
            # and the fixed-16 sample spans 48-85mm depending on the scanner - and for the
            # ~40% of series with under ~28 slices, linspace+round repeats indices, so a
            # duplicated slice gets double weight in the study mean. Sampling a fixed
            # physical slab at a fixed step fixes both: same anatomy, same through-plane
            # resolution, no repeats. Series too thin to fill the slab are clamped and the
            # shortfall is left to the mask rather than padded with duplicates.
            step = slice_step_mm(files[n // 2]) or 3.5
            half_n = (a.slab_mm / 2.0) / step            # slices from centre
            c = (n - 1) / 2.0
            want = np.linspace(c - half_n, c + half_n, a.slices)
            pos = [int(round(min(max(v, 0), n - 1))) for v in want]
        elif hi <= lo:
            pos = [n // 2] * a.slices
        elif a.slice_power == 1.0:
            pos = [int(round(x)) for x in np.linspace(lo, hi, a.slices)]
        else:
            # Same slice budget, spent differently. t**p with p>1 pulls sample points
            # toward the middle of the stack (dense at the joint line, sparse at the
            # edges); p<1 pushes them outward. Uniform spacing was never a decision,
            # it was the default, and it competes directly with --keep-frac: one asks
            # to look wider, this asks to look harder at the centre.
            t = np.linspace(-1.0, 1.0, a.slices)
            t = np.sign(t) * np.abs(t) ** a.slice_power
            mid, half = (lo + hi) / 2.0, (hi - lo) / 2.0
            pos = [int(round(mid + v * half)) for v in t]
        off = centroid_off(files[n // 2]) if a.centroid_crop else (0.0, 0.0)
        side = laterality(files[n // 2]) if a.lat_norm else None
        if side is None and a.geo_lat_fov:
            side = geo_side_fov(files[n // 2])
        elif side is None and a.geo_lat:
            side = geo_side(files[n // 2])
        flip_x = False
        if a.lat_norm and side == "R":
            pl = slot_keys[pi][0]
            if pl in ("Coronal", "Axial"):
                flip_x = True          # mirror in-plane: M/L is an image axis here
            else:
                pos = pos[::-1]        # sagittal: M/L is the slice axis; reverse order
        for k, x in enumerate(pos):
            im = read_one(files[x], a.img, a.crop_mm, off)
            if im is not None:
                arr[si, pi, k] = im[:, ::-1] if flip_x else im
        mask[si, pi] = 1.0
        done[0] += 1
        if done[0] % 1000 == 0:
            el = time.time() - t0
            print(f"  {done[0]}/{len(jobs)} slots", flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(work, jobs))

    arr.flush()
    np.save(f"{out}.mask.npy", mask)
    json.dump({"studies": studies, "planes": planes,
               "slots": [f"{pl}|f{f}" for pl, f in slot_keys],
               "lat_norm": a.lat_norm,
               "geo_lat": a.geo_lat,
               "geo_lat_fov": a.geo_lat_fov,
               "pos_order": a.pos_order,
               "true_select": a.true_select,
               "centroid_crop": a.centroid_crop,
               "slices": a.slices, "img": a.img,
               "crop_mm": a.crop_mm, "keep_frac": a.keep_frac, "slab_mm": a.slab_mm,
               "antialias": ANTIALIAS, "fluid": a.fluid,
               "split": a.split}, open(f"{out}.json", "w"))
    cov = mask.mean(0)
    print(f"\nwrote {out}.npy ({gb:.1f} GB) in {time.time()-t0:.0f}s", flush=True)
    for (pl, f), c in zip(slot_keys, cov):
        print(f"  {pl:10s} f={f}  present in {c:.1%} of studies")
    print(f"  studies with zero planes: {int((mask.sum(1)==0).sum())}")


if __name__ == "__main__":
    main()
